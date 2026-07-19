"""Content-addressed LLM-call caching + per-stage metering for the structural-ensemble runtime.

COST OPTIMIZATION THAT CANNOT REDUCE ACCURACY (Section 0/14 of the ensemble contract): a response is
reused ONLY when the complete semantic input is identical — the full prompt text plus the backend's
result-determining configuration (model id, temperature, max_tokens where the backend exposes them).
"Merely similar" prompts never share. Two different actor views therefore never share a response
(their prompts differ); one actor with the byte-identical view, prompt and particle scope across two
structural models does share (which is both cheaper and better controlled — the decision becomes a
common random number across models).

Three wrappers, all preserving the `llm(prompt) -> str` protocol:

  MeteredLLM      — counts calls per (stage, model_id) with no behavior change; the cost manifest reads it.
  CachedLLM       — content-addressed exact-input cache for ensemble-stage calls (generation, critics,
                    compilation, equivalence judging). Deterministic-input stages only.
  ScopedActorCache — cross-model actor-decision sharing: responses are keyed by (particle scope, prompt,
                    occurrence index) so model B's k-th identical call at particle i reuses model A's k-th
                    response at particle i — and two DIFFERENT particles (or different views) never share.
                    Within-model behavioral variance is untouched: inside one model every occurrence still
                    triggers its own backend call the first time it is seen.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


class NullStore(dict):
    """A cache store that never retains — the COST-BENCHMARK-ONLY cache-off arm (Section 27). Disabling
    the cache only ever costs more; it can never change results (reuse requires identical inputs)."""

    def __contains__(self, key):  # noqa: D105
        return False

    def __setitem__(self, key, value):  # noqa: D105
        return None


def backend_fingerprint(llm) -> str:
    """Result-determining backend identity, as far as the callable exposes it. Honest limitation: a bare
    function exposes nothing, so its fingerprint is its qualified name — reuse then assumes the backend is
    deterministic for identical prompts (recorded in the cost manifest, never hidden)."""
    parts = [type(llm).__name__, getattr(llm, "__qualname__", "")]
    for attr in ("model", "model_id", "temperature", "max_tokens", "seed"):
        v = getattr(llm, attr, None)
        if v is not None and not callable(v):
            parts.append(f"{attr}={v}")
    return "|".join(str(p) for p in parts if p)


def _key(fingerprint: str, prompt: str) -> str:
    return hashlib.sha256(f"{fingerprint}\x00{prompt}".encode()).hexdigest()[:40]


@dataclass
class CallLedger:
    """Shared mutable ledger: exact call counts by stage and structural model + cache accounting."""
    calls_by_stage: dict = field(default_factory=dict)       # stage -> n backend calls
    calls_by_model: dict = field(default_factory=dict)       # model_id -> n backend calls
    cache_hits_by_stage: dict = field(default_factory=dict)
    prompt_chars: int = 0
    response_chars: int = 0
    errors: int = 0

    def record(self, *, stage: str, model_id: str, cached: bool, prompt: str = "", response: str = ""):
        if cached:
            self.cache_hits_by_stage[stage] = self.cache_hits_by_stage.get(stage, 0) + 1
            return
        self.calls_by_stage[stage] = self.calls_by_stage.get(stage, 0) + 1
        if model_id:
            self.calls_by_model[model_id] = self.calls_by_model.get(model_id, 0) + 1
        self.prompt_chars += len(prompt)
        self.response_chars += len(response or "")

    def total_calls(self) -> int:
        return sum(self.calls_by_stage.values())

    def total_cache_hits(self) -> int:
        return sum(self.cache_hits_by_stage.values())

    def as_dict(self) -> dict:
        return {"llm_calls_by_stage": dict(self.calls_by_stage),
                "llm_calls_by_model": dict(self.calls_by_model),
                "cache_hits_by_stage": dict(self.cache_hits_by_stage),
                "total_llm_calls": self.total_calls(), "total_cache_hits": self.total_cache_hits(),
                "prompt_chars": self.prompt_chars, "response_chars": self.response_chars,
                "errors": self.errors}


class MeteredLLM:
    """`llm(prompt) -> str` pass-through that records every backend call in the shared ledger."""

    def __init__(self, llm, *, ledger: CallLedger, stage: str, model_id: str = ""):
        self._llm = llm
        self._ledger = ledger
        self.stage = stage
        self.model_id = model_id
        # surface the inner backend's result-determining config so fingerprints stay truthful
        for attr in ("model", "model_id_attr", "temperature", "max_tokens"):
            if hasattr(llm, attr) and attr != "model_id_attr":
                try:
                    setattr(self, attr, getattr(llm, attr))
                except Exception:  # noqa: BLE001
                    pass

    def __call__(self, prompt: str) -> str:
        try:
            out = self._llm(prompt)
        except Exception:
            self._ledger.errors += 1
            raise
        self._ledger.record(stage=self.stage, model_id=self.model_id, cached=False,
                            prompt=prompt, response=out if isinstance(out, str) else "")
        return out


class CachedLLM:
    """Exact-input content-addressed cache for deterministic ensemble stages. The key is
    (backend fingerprint, full prompt) — nothing less. Never used for within-model actor decisions
    (those intentionally carry behavioral variance); see ScopedActorCache for the cross-model case."""

    def __init__(self, llm, *, ledger: CallLedger = None, stage: str = "ensemble", model_id: str = "",
                 store: dict = None):
        self._llm = llm
        self._ledger = ledger or CallLedger()
        self.stage = stage
        self.model_id = model_id
        self._store = store if store is not None else {}
        self._fingerprint = backend_fingerprint(llm)

    @property
    def store(self) -> dict:
        return self._store

    def with_stage(self, stage: str, model_id: str = "") -> "CachedLLM":
        """Same cache store + ledger, different accounting labels."""
        return CachedLLM(self._llm, ledger=self._ledger, stage=stage, model_id=model_id,
                         store=self._store)

    def __call__(self, prompt: str) -> str:
        k = _key(self._fingerprint, prompt)
        if k in self._store:
            self._ledger.record(stage=self.stage, model_id=self.model_id, cached=True)
            return self._store[k]
        try:
            out = self._llm(prompt)
        except Exception:
            self._ledger.errors += 1
            raise
        if isinstance(out, str):
            self._store[k] = out
        self._ledger.record(stage=self.stage, model_id=self.model_id, cached=False,
                            prompt=prompt, response=out if isinstance(out, str) else "")
        return out


class ScopedActorCache:
    """Cross-structural-model actor-decision sharing under exact CRN alignment.

    The runtime sets `.scope` to the particle index before rolling each branch (every structural model
    uses the same base seed and the same per-index rollout seed law, so particle i is the SAME common
    random number across models). Key = (scope, backend fingerprint, prompt, occurrence#): model B's k-th
    byte-identical call at particle i reuses model A's k-th response at that particle. Any differing
    causal input — a different view, private state, prompt, particle or backend — produces a different
    key and a fresh call. Occurrence sequencing preserves each model's own intra-branch call order
    exactly, so within-model behavior is byte-identical to the uncached run."""

    def __init__(self, llm, *, ledger: CallLedger = None, stage: str = "actor", model_id: str = ""):
        self._llm = llm
        self._ledger = ledger or CallLedger()
        self.stage = stage
        self.model_id = model_id
        self.scope = None                       # particle index; None disables sharing (plain metering)
        self._responses = {}                    # (scope, key) -> [response per occurrence]
        self._occurrence = {}                   # (scope, key) -> occurrences seen in the CURRENT branch
        self._fingerprint = backend_fingerprint(llm)

    def enter_branch(self, scope):
        """Called by the runtime before each branch roll; resets occurrence counters for the scope."""
        self.scope = scope
        self._occurrence = {k: 0 for k in self._occurrence if k[0] != scope}

    def __call__(self, prompt: str) -> str:
        if self.scope is None:
            out = self._llm(prompt)
            self._ledger.record(stage=self.stage, model_id=self.model_id, cached=False,
                                prompt=prompt, response=out if isinstance(out, str) else "")
            return out
        k = (self.scope, _key(self._fingerprint, prompt))
        idx = self._occurrence.get(k, 0)
        self._occurrence[k] = idx + 1
        seq = self._responses.setdefault(k, [])
        if idx < len(seq):
            self._ledger.record(stage=self.stage, model_id=self.model_id, cached=True)
            return seq[idx]
        out = self._llm(prompt)
        if isinstance(out, str):
            seq.append(out)
        self._ledger.record(stage=self.stage, model_id=self.model_id, cached=False,
                            prompt=prompt, response=out if isinstance(out, str) else "")
        return out
