"""Run-scoped shared artifacts — compile once, reference everywhere, never share mutable state.

Full fidelity re-derives several QUESTION-LEVEL artifacts once per structural model (resolution
criterion, scheduled/calendar facts, evidence canonicalization) and sends complete copies into many
prompts. The lean profile compiles each shared artifact ONCE per run into this registry; structural
models and particles hold REFERENCES to the immutable payloads plus their own branch-local state.

Contract per artifact: stable content hash, version, provenance, as_of, one owner (the stage that
built it), and an explicit invalidation rule with dependency-aware cascade — invalidating an
artifact invalidates its dependents, never silently serves a stale composite. Mutable branch state
(worlds, queues, actor states, StateDeltas) is structurally unregisterable: payloads are frozen at
registration (deep-copied then treated as read-only) and a registered payload is served as the same
shared object ONLY through `get`, which is documented read-only; anything a branch mutates must be
its own copy."""
from __future__ import annotations

import copy
import hashlib
import json
import time as _time
from dataclasses import dataclass, field, asdict

ARTIFACTS_VERSION = "lean.artifacts.v1"

#: canonical artifact names the lean runtime shares across structural models / particles
SHARED_ARTIFACT_NAMES = (
    "resolution_criterion", "question_type", "evidence_bundle", "evidence_accepted",
    "evidence_rejected", "canonical_evidence_facts", "calendar_facts", "recurrence_facts",
    "grounded_prior_inputs", "institution_composition", "institution_rules", "actor_roster",
    "authority_graph", "scenario_schema", "structural_model", "temporal_model",
    "outcome_definition", "operator_registry_bindings", "actor_dossiers",
    "organization_dossiers", "action_language_schema", "consequence_compiler_config",
)


def content_hash(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class ArtifactRecord:
    name: str
    content_hash: str
    version: str
    provenance: dict
    as_of: str
    owner: str                                  # the stage that built it (single writer)
    invalidation_rule: str                      # human-readable condition that voids it
    depends_on: list = field(default_factory=list)
    created_at: float = 0.0
    invalidated: bool = False
    invalidation_reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class RunSharedArtifacts:
    """One registry per lean run. Register once; read many. Re-registration with identical
    content is a no-op (recorded as a dedup hit); re-registration with different content requires
    an explicit `supersede=True` and invalidates dependents."""

    def __init__(self, *, as_of: str = "", run_id: str = ""):
        self.as_of = str(as_of)
        self.run_id = str(run_id)
        self._records: dict[str, ArtifactRecord] = {}
        self._payloads: dict[str, object] = {}
        self._dependents: dict[str, set] = {}
        self.dedup_hits: dict[str, int] = {}
        self.reads: dict[str, int] = {}

    # ---- write side -----------------------------------------------------------------
    def register(self, name: str, payload, *, owner: str, version: str = "1",
                 provenance: dict = None, invalidation_rule: str = "as_of boundary change",
                 depends_on: list = None, supersede: bool = False) -> ArtifactRecord:
        frozen = copy.deepcopy(payload)
        h = content_hash(frozen)
        existing = self._records.get(name)
        if existing is not None and not existing.invalidated:
            if existing.content_hash == h:
                self.dedup_hits[name] = self.dedup_hits.get(name, 0) + 1
                return existing
            if not supersede:
                raise ValueError(
                    f"artifact {name!r} already registered with different content "
                    f"({existing.content_hash} != {h}); pass supersede=True to replace "
                    f"(dependents will be invalidated)")
            self.invalidate(name, reason=f"superseded by {owner}")
        rec = ArtifactRecord(name=name, content_hash=h, version=str(version),
                             provenance=dict(provenance or {}), as_of=self.as_of, owner=owner,
                             invalidation_rule=invalidation_rule,
                             depends_on=list(depends_on or []), created_at=_time.time())
        self._records[name] = rec
        self._payloads[name] = frozen
        for dep in rec.depends_on:
            self._dependents.setdefault(dep, set()).add(name)
        return rec

    def invalidate(self, name: str, *, reason: str):
        """Dependency-aware cascade: a stale component can never survive inside a composite."""
        rec = self._records.get(name)
        if rec is None or rec.invalidated:
            return
        rec.invalidated, rec.invalidation_reason = True, str(reason)[:200]
        for dep_name in sorted(self._dependents.get(name, ())):
            self.invalidate(dep_name, reason=f"dependency {name} invalidated: {reason}"[:200])

    # ---- read side ------------------------------------------------------------------
    def get(self, name: str, default=None):
        """READ-ONLY shared payload (the whole point is not copying it per particle). Callers
        must never mutate; anything branch-local must be copied by the branch itself."""
        rec = self._records.get(name)
        if rec is None or rec.invalidated:
            return default
        self.reads[name] = self.reads.get(name, 0) + 1
        return self._payloads[name]

    def hash_of(self, name: str) -> str:
        rec = self._records.get(name)
        return "" if rec is None or rec.invalidated else rec.content_hash

    def has(self, name: str) -> bool:
        rec = self._records.get(name)
        return rec is not None and not rec.invalidated

    def manifest(self) -> dict:
        return {"version": ARTIFACTS_VERSION, "run_id": self.run_id, "as_of": self.as_of,
                "artifacts": {n: r.as_dict() for n, r in sorted(self._records.items())},
                "dedup_hits": dict(self.dedup_hits), "reads": dict(self.reads),
                "n_live": sum(1 for r in self._records.values() if not r.invalidated)}
