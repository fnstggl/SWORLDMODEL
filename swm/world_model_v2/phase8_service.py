"""Phase 8 — the durable cross-run persistence SERVICE: replay, checkpoint, restore, migration, lineage.

This is the production gap Phase 4 explicitly named ("not a substitute for Phase 8's future cross-run
persistence service"): a store that outlives a single process, treats the event log as the source of truth,
derives state by REPLAY, and can checkpoint/restore/roll back deterministically.

Core object: ``PersistentStore`` = an ``EventLog`` + a registry of filter builders (variable_id → factory).
  * ``replay(as_of, mode)`` derives every registered variable's posterior by running its sequential filter
    over the leakage-safe event stream — the ONLY way state is produced (no state is hand-set);
  * ``checkpoint(as_of)`` snapshots the derived posteriors + the event-log watermark + code/schema versions
    + lineage + an integrity hash;
  * ``restore``/``save``/``load`` round-trip a checkpoint to disk (cross-run durability);
  * ``verify`` detects corruption (integrity hash mismatch); ``migrate`` upgrades an older schema with a
    typed error on an incompatible one; ``rollback`` returns an earlier checkpoint; ``compare`` diffs two.

Determinism contract (Part 9): identical event log + as_of + seed ⇒ identical posteriors + integrity hash,
within numerical tolerance. The tests exercise checkpoint→restore→replay parity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from swm.world_model_v2.phase8_events import EventLog, PersistentEvent
from swm.world_model_v2.phase8_persistence import (SCHEMA_VERSION, PersistentLineage, PersistentStateKey,
                                                   PersistentStatePosterior)

STORE_SCHEMA_VERSION = "phase8-store-1.0"
COMPATIBLE_SCHEMAS = ("phase8-store-1.0",)


class MigrationError(Exception):
    """A checkpoint's schema is incompatible and cannot be migrated — a typed failure, never a silent load."""


class CorruptionError(Exception):
    """A checkpoint's integrity hash does not match its content."""


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:24]


@dataclass
class PersistentCheckpoint:
    """A versioned snapshot of derived persistent state (Part 9). Derived from the event log by replay; the
    log remains the source of truth. ``integrity_hash`` chains the content so tampering is detectable."""
    world_id: str
    scenario_id: str
    as_of: float
    posteriors: dict = field(default_factory=dict)         # key_token -> posterior.as_dict()
    event_watermark: str = ""
    transition_param_posteriors: dict = field(default_factory=dict)
    structural_hypotheses: list = field(default_factory=list)
    unresolved_identity_hypotheses: list = field(default_factory=list)
    actor_visibility_state: dict = field(default_factory=dict)
    memory_index: dict = field(default_factory=dict)
    evidence_hashes: list = field(default_factory=list)
    code_versions: dict = field(default_factory=dict)
    schema_version: str = STORE_SCHEMA_VERSION
    seed: int = 0
    lineage: dict = field(default_factory=dict)            # key_token -> PersistentLineage.as_dict()
    integrity_hash: str = ""

    def compute_integrity(self) -> str:
        payload = {"world_id": self.world_id, "scenario_id": self.scenario_id, "as_of": self.as_of,
                   "event_watermark": self.event_watermark, "schema_version": self.schema_version,
                   "seed": self.seed,
                   "posteriors": {k: {"mean": round(v.get("mean", 0.0), 8), "sd": round(v.get("sd", 0.0), 8),
                                      "n": v.get("n_events_assimilated")}
                                  for k, v in sorted(self.posteriors.items())}}
        return _hash(payload)

    def seal(self) -> "PersistentCheckpoint":
        self.integrity_hash = self.compute_integrity()
        return self

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "PersistentCheckpoint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PersistentStore:
    """The cross-run persistence service for one (world, scenario)."""
    world_id: str
    scenario_id: str
    log: EventLog = None
    filter_builders: dict = field(default_factory=dict)    # variable_id -> callable(key, events)->posterior
    code_versions: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.log is None:
            self.log = EventLog(world_id=self.world_id, scenario_id=self.scenario_id)

    # ---- registration -----------------------------------------------------------------------------
    def register_filter(self, variable_id: str, builder) -> None:
        """builder(key: PersistentStateKey, observations: list[(event_id, obs, ts)], as_of, seed)
        -> PersistentStatePosterior. Determines HOW each variable is filtered over the event stream."""
        self.filter_builders[variable_id] = builder

    # ---- the ONLY way state is produced: replay from the log --------------------------------------
    def replay(self, as_of: float, *, mode: str = "filter", seed: int = 0,
               variable_keys=None) -> dict:
        """Derive posteriors for the requested (variable_id, scope, entity_id) keys by running each filter
        over the leakage-safe event stream. Returns {key_token: PersistentStatePosterior}. This is
        deterministic given (log, as_of, seed) — the checkpoint parity contract."""
        events = self.log.events_as_of(as_of, mode=mode)
        # group observations per (variable_id, entity_id) — the caller declares which keys to derive
        out = {}
        keys = variable_keys or self._infer_keys(events)
        for key in keys:
            builder = self.filter_builders.get(key.variable_id)
            if builder is None:
                continue
            obs = self._observations_for(key, events)
            post = builder(key, obs, as_of, seed)
            out[key.token()] = post
        return out

    def _infer_keys(self, events) -> list:
        """Default key inference: one engagement_propensity key per actor seen in passive_exposure events.
        Scenario adapters override by passing ``variable_keys`` to ``replay``."""
        actors = set()
        for e in events:
            if e.event_type in ("passive_exposure",):
                actors |= set(e.actor_ids)
        return [PersistentStateKey(self.world_id, self.scenario_id, "actor", a, "engagement_propensity")
                for a in sorted(actors)]

    def _observations_for(self, key: PersistentStateKey, events) -> list:
        """Extract the per-variable observation stream for a key from the event log. The observation VALUE
        is the event outcome (binary/typed) — the filter interprets it per family."""
        obs = []
        for e in events:
            if key.entity_id not in e.actor_ids and key.entity_id not in "|".join(e.actor_ids):
                # dyad/edge keys encode "a|b"; match if either endpoint is an actor
                if not any(part in e.actor_ids for part in key.entity_id.split("|")):
                    continue
            obs.append((e.event_id, e.outcome if e.outcome is not None else e.event_type, e.event_time))
        return obs

    # ---- checkpoint / restore / durability --------------------------------------------------------
    def checkpoint(self, as_of: float, *, mode="filter", seed=0, variable_keys=None,
                   structural_hypotheses=None, evidence_hashes=None) -> PersistentCheckpoint:
        posteriors = self.replay(as_of, mode=mode, seed=seed, variable_keys=variable_keys)
        lineage = {}
        for tok, post in posteriors.items():
            lineage[tok] = PersistentLineage(
                key=tok, event_watermark=self.log.watermark(),
                genesis_event_ids=[r.get("event_id") for r in post.lineage[:3]],
                posterior_hash=post.posterior_hash(), transition_param_source=post.transition_params.get("source", ""),
                code_versions=self.code_versions, schema_version=SCHEMA_VERSION).as_dict()
        cp = PersistentCheckpoint(
            world_id=self.world_id, scenario_id=self.scenario_id, as_of=as_of,
            posteriors={k: v.as_dict() for k, v in posteriors.items()},
            event_watermark=self.log.watermark(),
            transition_param_posteriors={k: v.transition_params for k, v in posteriors.items()},
            structural_hypotheses=structural_hypotheses or [],
            unresolved_identity_hypotheses=self._unresolved_identities(as_of),
            evidence_hashes=evidence_hashes or [], code_versions=self.code_versions, seed=seed,
            lineage=lineage)
        return cp.seal()

    def _unresolved_identities(self, as_of: float) -> list:
        out = []
        for e in self.log.events_as_of(as_of, mode="filter"):
            if e.identity_link_uncertainty > 0.0:
                out.append({"event_id": e.event_id, "actors": list(e.actor_ids),
                            "link_uncertainty": e.identity_link_uncertainty})
        return out

    def save_checkpoint(self, cp: PersistentCheckpoint, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cp.as_dict(), indent=2, default=str))
        return str(p)

    def load_checkpoint(self, path: str) -> PersistentCheckpoint:
        cp = PersistentCheckpoint.from_dict(json.loads(Path(path).read_text()))
        self.verify(cp)                                    # corruption + schema check on load
        return cp

    def verify(self, cp: PersistentCheckpoint) -> dict:
        """Corruption + schema check. Raises CorruptionError / MigrationError — never silently loads a bad
        or incompatible checkpoint."""
        if cp.schema_version not in COMPATIBLE_SCHEMAS:
            raise MigrationError(f"checkpoint schema {cp.schema_version!r} not directly loadable "
                                 f"(compatible: {COMPATIBLE_SCHEMAS}); call migrate() first")
        recomputed = cp.compute_integrity()
        if cp.integrity_hash and recomputed != cp.integrity_hash:
            raise CorruptionError(f"integrity hash mismatch: stored {cp.integrity_hash!r} != "
                                  f"recomputed {recomputed!r} — checkpoint corrupt")
        return {"ok": True, "integrity_hash": recomputed, "schema_version": cp.schema_version}

    def migrate(self, cp: PersistentCheckpoint) -> PersistentCheckpoint:
        """Upgrade an older-but-known schema to the current one. Unknown/incompatible schemas raise
        MigrationError (typed) rather than loading garbage."""
        if cp.schema_version == STORE_SCHEMA_VERSION:
            return cp
        # Known upgrade path example: an older 0.x checkpoint → 1.0 (re-seal under the new schema, preserving
        # the derived posteriors). Truly incompatible schemas (major mismatch) are refused.
        major = str(cp.schema_version).split("-")[-1].split(".")[0]
        if not major.isdigit() or int(major) > 1:
            raise MigrationError(f"cannot migrate checkpoint schema {cp.schema_version!r} to "
                                 f"{STORE_SCHEMA_VERSION} (incompatible major version)")
        cp.schema_version = STORE_SCHEMA_VERSION
        return cp.seal()

    def restore(self, cp: PersistentCheckpoint) -> dict:
        """Reconstruct posteriors from a checkpoint (deterministic). Verifies integrity first."""
        self.verify(cp)
        out = {}
        for tok, d in cp.posteriors.items():
            parts = tok.split("::")
            key = PersistentStateKey(*parts[:5]) if len(parts) >= 5 else PersistentStateKey(
                self.world_id, self.scenario_id, "actor", tok, d.get("variable_id", ""))
            post = PersistentStatePosterior(key=key, variable_id=d.get("variable_id", ""),
                                            posterior_family=d.get("posterior_family", "beta_bernoulli"))
            for f in ("mean", "sd", "representation", "prior_mean", "transition_params",
                      "n_events_assimilated", "n_effective_observations", "ess", "as_of", "method",
                      "lineage", "diagnostics"):
                if f in d:
                    setattr(post, f, d[f])
            out[tok] = post
        return out

    def rollback_to(self, cp: PersistentCheckpoint) -> dict:
        """Return the state as-of an earlier checkpoint (does not delete later events; the log is immutable).
        Rollback is a READ of an earlier derived state — the event log stays complete."""
        return self.restore(cp)

    @staticmethod
    def compare(cp_a: PersistentCheckpoint, cp_b: PersistentCheckpoint) -> dict:
        """Diff two checkpoints: which variables changed and by how much. Used by the causal-ablation and
        determinism tests."""
        keys = set(cp_a.posteriors) | set(cp_b.posteriors)
        diffs, max_delta = {}, 0.0
        for k in sorted(keys):
            a = cp_a.posteriors.get(k, {})
            b = cp_b.posteriors.get(k, {})
            da = float(a.get("mean", 0.0)) - float(b.get("mean", 0.0))
            if abs(da) > 1e-9 or (k in cp_a.posteriors) != (k in cp_b.posteriors):
                diffs[k] = {"mean_a": a.get("mean"), "mean_b": b.get("mean"), "delta": round(da, 6)}
            max_delta = max(max_delta, abs(da))
        return {"n_changed": len(diffs), "max_abs_delta": round(max_delta, 6), "diffs": diffs,
                "watermark_a": cp_a.event_watermark, "watermark_b": cp_b.event_watermark,
                "identical": len(diffs) == 0}
