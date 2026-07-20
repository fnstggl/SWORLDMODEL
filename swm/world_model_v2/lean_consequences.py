"""Consequence-program reuse — compile an identical validated action once per equivalent context.

The consequence compilers (causal_boundary.CausalActionCompiler on the default
generated_actor_mediated_world path; semantic_consequences.SemanticConsequenceCompiler on the
semantic path) build their compile prompt DETERMINISTICALLY from exactly the §12 key inputs:
scenario schema (record/event/mechanism types + version), the actor and their chosen action's
exact content/target/timing/conditions, the relevant world-object projection the compiler reads,
and the compiler prompt version. Byte-identical compile context ⟺ byte-identical prompt.

Reuse is therefore implemented at the narrowest safe point: the PROVIDER RESPONSE for an
identical (backend fingerprint, compile prompt) is reused, and the entire deterministic
compile-validate pipeline — parse, op normalization, schema validation, directness/authority
critique, causal-boundary checks, program binding — RERUNS on every receiving branch. That is
strictly more conservative than cloning a compiled program: nothing validated is assumed, only
the provider text is shared. Execution (delivery, attention, institutional processing, physical
results, other-actor reactions, downstream events) stays branch-specific because it happens
entirely outside the compiler.

Only compile-stage calls are cached. Nothing about execution results is ever stored."""
from __future__ import annotations

import hashlib
import threading

CONSEQUENCE_CACHE_VERSION = "lean.consequence.cache.v1"


class ConsequenceProgramCache:
    """`llm(prompt) -> str` wrapper installed as the runtime's consequence backend. Response-level
    exact-key memo + per-key reuse manifest. Failures (exceptions, empty responses) are never
    stored; concurrent identical compiles single-flight to one provider call."""

    def __init__(self, llm, *, fingerprint: str = "", ledger=None, stage: str =
                 "consequence_compile"):
        self._llm = llm
        self._fingerprint = fingerprint or f"{type(llm).__name__}"
        self._ledger = ledger
        self.stage = stage
        self._store: dict[str, str] = {}
        self._meta: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}
        self.compile_calls = 0
        self.reuses = 0
        self.failures_not_cached = 0

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{CONSEQUENCE_CACHE_VERSION}\x00{self._fingerprint}\x00{prompt}"
                              .encode()).hexdigest()[:40]

    def __call__(self, prompt: str) -> str:
        k = self._key(prompt)
        with self._lock:
            if k in self._store:
                self.reuses += 1
                self._meta[k]["reuses"] += 1
                if self._ledger is not None:
                    self._ledger.record(stage=self.stage, model_id="", cached=True)
                return self._store[k]
            ev = self._inflight.get(k)
            if ev is None:
                self._inflight[k] = threading.Event()
                leader = True
            else:
                leader = False
        if not leader:
            ev.wait(timeout=300)
            with self._lock:
                if k in self._store:
                    self.reuses += 1
                    self._meta[k]["reuses"] += 1
                    if self._ledger is not None:
                        self._ledger.record(stage=self.stage, model_id="", cached=True)
                    return self._store[k]
            # leader failed — fall through to a controlled retry as the new leader
            with self._lock:
                self._inflight.setdefault(k, threading.Event())
        try:
            out = self._llm(prompt)
        except Exception:
            with self._lock:
                self.failures_not_cached += 1
                ev2 = self._inflight.pop(k, None)
            if ev2 is not None:
                ev2.set()                        # release waiters; they may retry
            if self._ledger is not None:
                self._ledger.errors += 1
            raise
        with self._lock:
            self.compile_calls += 1
            if isinstance(out, str) and out.strip():
                self._store[k] = out
                self._meta[k] = {"reuses": 0, "prompt_chars": len(prompt),
                                 "prompt_head": prompt[:120]}
            else:
                self.failures_not_cached += 1
            ev2 = self._inflight.pop(k, None)
        if ev2 is not None:
            ev2.set()
        if self._ledger is not None:
            self._ledger.record(stage=self.stage, model_id="", cached=False, prompt=prompt,
                                response=out if isinstance(out, str) else "")
        return out

    def manifest(self) -> dict:
        with self._lock:
            top = sorted(self._meta.items(), key=lambda kv: -kv[1]["reuses"])[:10]
            return {"version": CONSEQUENCE_CACHE_VERSION,
                    "compile_calls": self.compile_calls, "reuses": self.reuses,
                    "distinct_programs": len(self._store),
                    "failures_not_cached": self.failures_not_cached,
                    "reuse_design": "provider response shared on exact (fingerprint, prompt) "
                                    "equality; parse/normalize/validate/authority/boundary "
                                    "checks and ALL execution rerun per receiving branch",
                    "top_reused": [{"key": k[:16], "reuses": m["reuses"],
                                    "prompt_head": m["prompt_head"]} for k, m in top]}
