"""Run-scoped actor-decision equivalence cache — one call per genuinely distinct situation.

Thirty particles may remain thirty distinct worlds while only the materially different actor
decision situations inside them spend provider calls: the cache key is the deterministic
`DecisionRelevantContext` signature (exact equality after projection — lean_context) plus the
actor prompt version, the backend fingerprint and the behavioral replicate index. Nothing fuzzy,
no LLM judge, no embeddings.

What is shared: ONE immutable, validated decision template (the raw one-call response + parsed
decision snapshot + hashes). What is never shared: mutable actor state, memory, event queues,
world objects, institution state, StateDeltas, consequence executions — the receiving branch
deep-copies the decision, reruns the deterministic memory stages on ITS OWN state, and every
downstream check (perceived feasibility, revision, actual feasibility at execute, authority,
consequence execution) runs per branch exactly as before.

Failures are never cached: provider errors, parse failures, validation failures, truncations and
escalations all bypass storage. Concurrent identical requests collapse to one provider call
(single-flight); waiters are released on failure and may retry."""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field, asdict

from swm.world_model_v2.lean_context import (DecisionEquivalenceCertificate,
                                             DecisionRelevantContext)

CACHE_VERSION = "lean.decision.cache.v1"


@dataclass
class ActorDecisionTemplate:
    """The immutable first-occurrence record. `response` is the raw provider text (the shared
    artifact); `qd_snapshot` is the parsed decision as a dict (deep-copied on every reuse)."""
    context_hash: str
    actor_id: str
    cohort_id: str
    prompt_hash: str
    response_hash: str
    response: str
    qd_snapshot: dict
    model_fingerprint: str
    prompt_version: str
    replicate_index: int
    source_branch: str
    context: dict                                   # full projection (audit/explain)
    validation_record: dict = field(default_factory=dict)

    def as_manifest_row(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if k not in ("response", "qd_snapshot", "context")}


@dataclass
class DecisionReuseRecord:
    actor_id: str
    context_hash: str
    source_branch: str
    receiving_branch: str
    cohort_id: str
    revalidation: dict = field(default_factory=dict)
    certificate: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class SingleFlightDecision:
    """Per-key single flight: exactly one provider call for concurrent identical contexts; the
    losing threads wait for the winner's stored template. On failure the key is released so a
    controlled retry can happen; failures are never handed to waiters as results."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}

    def begin(self, key: str):
        """Returns ("leader", None) for the thread that must do the work, or ("waiter", event)
        for threads that should wait for the leader."""
        with self._lock:
            ev = self._inflight.get(key)
            if ev is None:
                self._inflight[key] = threading.Event()
                return "leader", None
            return "waiter", ev

    def finish(self, key: str):
        with self._lock:
            ev = self._inflight.pop(key, None)
        if ev is not None:
            ev.set()


class DecisionEquivalenceCache:
    """Run-scoped (cross-run reuse is DISABLED by default: a cache instance lives inside one
    lean run and is never persisted). Thread-safe; single-flight built in."""

    def __init__(self, *, behavioral_replicates_per_decision_context: int = 1):
        self.behavioral_replicates = max(1, int(behavioral_replicates_per_decision_context))
        self._templates: dict[str, ActorDecisionTemplate] = {}
        self._lock = threading.RLock()
        self.single_flight = SingleFlightDecision()
        self.reuse_records: list[DecisionReuseRecord] = []
        self.hits = 0
        self.misses = 0
        self.invalidated_hits = 0                 # hits whose receiving-branch revalidation failed
        self.stores = 0
        self.failures_not_cached = 0

    @staticmethod
    def key_for(ctx: DecisionRelevantContext) -> str:
        # prompt version / fingerprint / replicate index are fields OF the projection —
        # the signature already covers them; the key is the signature itself.
        return ctx.signature()

    def get(self, key: str) -> ActorDecisionTemplate | None:
        with self._lock:
            t = self._templates.get(key)
            if t is not None:
                self.hits += 1
            else:
                self.misses += 1
            return t

    def peek(self, key: str) -> bool:
        with self._lock:
            return key in self._templates

    def store(self, key: str, template: ActorDecisionTemplate):
        """Only validated successes may be stored (the caller enforces WHAT counts as success;
        this layer enforces immutability + no-overwrite)."""
        with self._lock:
            if key not in self._templates:
                self._templates[key] = template
                self.stores += 1

    def record_failure(self):
        self.failures_not_cached += 1

    def reuse(self, template: ActorDecisionTemplate, *, receiving_branch: str,
              revalidation: dict) -> tuple:
        """Deep-copy the immutable template for the receiving branch + build the equivalence
        certificate. `revalidation` carries the receiving-branch check results (trigger match is
        structural — equal projection — and feasibility/authority re-run downstream)."""
        qd_copy = copy.deepcopy(template.qd_snapshot)
        cert = DecisionEquivalenceCertificate(
            context_hash=template.context_hash, actor_id=template.actor_id,
            cohort_id=template.cohort_id, source_branch=template.source_branch,
            receiving_branch=receiving_branch,
            matched_components=sorted(template.context.keys()),
            revalidation=dict(revalidation))
        rec = DecisionReuseRecord(
            actor_id=template.actor_id, context_hash=template.context_hash,
            source_branch=template.source_branch, receiving_branch=receiving_branch,
            cohort_id=template.cohort_id, revalidation=dict(revalidation),
            certificate=cert.as_dict())
        with self._lock:
            self.reuse_records.append(rec)
            if not revalidation.get("ok", True):
                self.invalidated_hits += 1
        return qd_copy, cert

    # ---- audit ----------------------------------------------------------------------
    def explain_equivalence(self, context_hash: str = None) -> str:
        """Human-readable audit of every reuse (or of one context): who served whom, on exactly
        which matched projection, what was ignored and why it was non-material, and how the
        receiving branch revalidated. No chain-of-thought is exposed — only projection fields."""
        rows = [r for r in self.reuse_records
                if context_hash is None or r.context_hash == context_hash]
        if not rows:
            return "no decision reuse recorded" + (f" for context {context_hash}" if context_hash
                                                   else "")
        out = []
        for r in rows:
            cert = r.certificate
            out.append("\n".join([
                f"DECISION REUSE {r.context_hash[:16]}… actor={r.actor_id} "
                f"cohort={r.cohort_id or '(none)'}",
                f"  source branch {r.source_branch} → receiving branch {r.receiving_branch}",
                f"  matched components (exact equality after deterministic projection): "
                + ", ".join(cert.get("matched_components", [])),
                "  ignored differences (never projected, with reasons):",
                *[f"    - {d['field']}: {d['why']}"
                  for d in cert.get("ignored_differences", [])],
                f"  receiving-branch revalidation: {r.revalidation}"]))
        return "\n".join(out)

    def manifest(self) -> dict:
        with self._lock:
            per_ctx: dict[str, int] = {}
            for r in self.reuse_records:
                per_ctx[r.context_hash] = per_ctx.get(r.context_hash, 0) + 1
            top = sorted(per_ctx.items(), key=lambda kv: -kv[1])[:10]
            return {"version": CACHE_VERSION,
                    "behavioral_replicates_per_decision_context": self.behavioral_replicates,
                    "unique_decision_contexts": len(self._templates),
                    "hits": self.hits, "misses": self.misses, "stores": self.stores,
                    "invalidated_hits": self.invalidated_hits,
                    "failures_not_cached": self.failures_not_cached,
                    "reuses": len(self.reuse_records),
                    "largest_context_reuse": [{"context_hash": h, "n_reuses": n}
                                              for h, n in top],
                    "templates": [t.as_manifest_row() for t in
                                  list(self._templates.values())[:60]],
                    "cross_run_persistence": "disabled (run-scoped by construction)"}
