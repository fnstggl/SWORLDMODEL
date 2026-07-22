"""Phase 11 — plan lineage, checkpointing, rollback, deterministic replay, anti-thrashing (spec §18/§19).

Immutable plan lineage (nodes + edges) with cycle / A→B→A oscillation detection. Atomic recompile
transactions: snapshot the source ensemble → build the candidate world OFF-PATH → validate + verify invariants
and integrity hashes → activate atomically, else ROLL BACK to the source snapshot (the only valid world is
never partially mutated). Integrity uses the same sha256 discipline as Phase 8's ``PersistentCheckpoint``;
checkpoints persist a verifiable SUMMARY atomically (closing Phase 8's non-atomic-write gap) while the live
object graph is held for exact rollback (``WorldState`` has no serializer, so deepcopy is the snapshot).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from swm.world_model_v2.phase11._serial import content_hash, atomic_write_json
from swm.world_model_v2.phase11.contracts import PlanLineageNode, PlanLineageEdge


@dataclass
class Checkpoint:
    checkpoint_id: str = ""
    sim_time: float = 0.0
    plan_hash: str = ""
    ensemble: dict = field(default_factory=dict)           # LIVE deepcopy: {worlds, weights, pending}
    summary: dict = field(default_factory=dict)            # serializable digest (entity ids, weights, sizes)
    integrity_hash: str = ""

    def compute_integrity(self) -> str:
        return content_hash(self.summary, length=32)

    def seal(self) -> "Checkpoint":
        self.integrity_hash = self.compute_integrity()
        return self

    def verify(self) -> bool:
        return bool(self.integrity_hash) and self.integrity_hash == self.compute_integrity()

    def persist(self, path: str) -> str:
        return atomic_write_json(path, {"checkpoint_id": self.checkpoint_id, "sim_time": self.sim_time,
                                        "plan_hash": self.plan_hash, "summary": self.summary,
                                        "integrity_hash": self.integrity_hash})


def _ensemble_summary(worlds, weights, pending):
    return {"n_particles": len(worlds),
            "entity_ids": sorted({e for w in worlds for e in getattr(w, "entities", {})}),
            "weights": [round(float(x), 6) for x in weights],
            "pending_counts": [len(p) for p in pending],
            "sim_times": sorted({round(float(getattr(getattr(w, "clock", None), "now", 0.0)), 3) for w in worlds})}


def snapshot(worlds, weights, pending, plan, sim_time, *, cid: str = "") -> Checkpoint:
    """Freeze a deep, integrity-hashed copy of the ensemble (the rollback reference)."""
    summary = _ensemble_summary(worlds, weights, pending)
    cp = Checkpoint(checkpoint_id=cid or content_hash({"t": sim_time, "s": summary}, length=12),
                    sim_time=sim_time, plan_hash=(plan.plan_hash() if hasattr(plan, "plan_hash") else "plan"),
                    ensemble={"worlds": [w.clone(branch_id=getattr(w, "branch_id", "b")) for w in worlds],
                              "weights": list(weights),
                              "pending": [list(p) for p in pending]},
                    summary=summary)
    return cp.seal()


@dataclass
class LineageGraph:
    """Immutable append-only plan lineage. Detects cycles + A→B→A oscillation to stop thrashing."""
    nodes: dict = field(default_factory=dict)              # plan_id -> PlanLineageNode.as_record()
    edges: list = field(default_factory=list)              # [PlanLineageEdge.as_record()]
    _sequence: list = field(default_factory=list)          # ordered plan_hash activations

    def add_node(self, node: PlanLineageNode):
        self.nodes[node.plan_id] = node.as_record()
        return self

    def add_edge(self, edge: PlanLineageEdge):
        self.edges.append(edge.as_record())
        return self

    def activate(self, plan_hash: str):
        self._sequence.append(plan_hash)

    def has_cycle(self) -> bool:
        """A plan hash reappearing in the activation sequence = a cycle / oscillation."""
        seen, order = set(), self._sequence
        return len(order) != len(set(order))

    def oscillation(self, next_plan_hash: str, *, window: int = 4) -> bool:
        """A→B→A within a short window without new evidence — refuse to re-activate a recently active plan."""
        return next_plan_hash in self._sequence[-window:]

    def depth(self) -> int:
        return len(self._sequence)

    def as_dict(self):
        return {"nodes": list(self.nodes.values()), "edges": self.edges, "sequence": list(self._sequence),
                "depth": self.depth(), "has_cycle": self.has_cycle()}


class RollbackError(RuntimeError):
    pass


@dataclass
class RecompileTransaction:
    """Atomic recompile: build the candidate ensemble OFF-PATH, validate + verify, then activate-or-rollback.
    The source snapshot is never mutated in place, so a mid-migration failure leaves the active world intact."""
    source: Checkpoint = None

    def run(self, build_candidate, verify_invariants):
        """``build_candidate() -> (worlds, weights, pending, migration_report)`` runs off-path (may raise).
        ``verify_invariants(candidate) -> (ok, reasons)`` checks the migrated ensemble. On any failure we roll
        back to ``self.source`` and report. Returns a dict with the activated ensemble or the rollback."""
        if self.source is None or not self.source.verify():
            raise RollbackError("source checkpoint missing or corrupt — refusing to migrate")
        try:
            worlds, weights, pending, report = build_candidate()
        except Exception as e:  # noqa: BLE001 — build failure → rollback, active world untouched
            return {"activated": False, "rolled_back": True, "reason": f"candidate build failed: {e}",
                    "worlds": self.source.ensemble["worlds"], "weights": self.source.ensemble["weights"],
                    "pending": self.source.ensemble["pending"], "report": {}}
        ok, reasons = verify_invariants({"worlds": worlds, "weights": weights, "pending": pending,
                                         "report": report})
        if not ok:
            return {"activated": False, "rolled_back": True, "reason": "invariant check failed: " + "; ".join(reasons),
                    "worlds": self.source.ensemble["worlds"], "weights": self.source.ensemble["weights"],
                    "pending": self.source.ensemble["pending"], "report": report}
        return {"activated": True, "rolled_back": False, "reason": "atomic activation ok",
                "worlds": worlds, "weights": weights, "pending": pending, "report": report}


def standard_invariants(candidate) -> tuple:
    """The migration gates as a pass/fail check on a migrated ensemble (spec §27.5)."""
    r = candidate.get("report", {}) or {}
    reasons = []
    if r.get("time_reversal_count", 0) != 0:
        reasons.append("time reversal detected")
    if r.get("duplicate_event_rate", 0.0) not in (0, 0.0):
        reasons.append("duplicate events detected")
    if r.get("lost_valid_event_rate", 0.0) not in (0, 0.0):
        reasons.append("valid events lost")
    if not candidate.get("worlds"):
        reasons.append("empty ensemble after migration")
    ws = candidate.get("weights", [])
    if ws and abs(sum(ws) - 1.0) > 1e-6:
        reasons.append("posterior weights not normalized")
    return (not reasons, reasons)
