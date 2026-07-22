"""The ONE external-call gateway — every provider call in a Lean V2 run flows through here.

Responsibilities, in order, for each call:
    1. budget admission (BudgetLedger.allow_call — hard caps, recorded refusals);
    2. real task-tier routing (deterministic work never reaches this module at all; light-tier
       stages may run a cheaper backend; STRONG_ONLY_STAGES are pinned to the strong backend
       by the Lean V1 routing contract, reused verbatim);
    3. execution with ONE bounded retry on provider exception (globally capped);
    4. stage-labeled ledger row (chars, latency, tier, retries).

Raises BudgetExhausted when admission fails — callers treat it as "finish with what exists",
never as "relaunch"."""
from __future__ import annotations

import time

from swm.world_model_v2.lean_routing import STRONG_ONLY_STAGES, TaskRoutingPolicy
from swm.world_model_v2.lean_v2.budget import BudgetLedger


class BudgetExhausted(RuntimeError):
    """The hard consumer budget refused a call. Carries the dimension; the orchestrator
    finalizes with completed state + labeled forecast + disclosure."""
    def __init__(self, dimension: str, stage: str):
        super().__init__(f"consumer budget exhausted ({dimension}) at stage {stage}")
        self.dimension = dimension
        self.stage = stage


class LLMGateway:
    """`call(stage, prompt)` -> str. The only path to a provider inside lean_v2."""

    def __init__(self, *, strong_llm, light_llm=None, ledger: BudgetLedger,
                 policy: TaskRoutingPolicy = None, backend_fingerprint: str = ""):
        if strong_llm is None:
            raise ValueError("lean_v2 requires an LLM backend for its consequential calls")
        self.strong = strong_llm
        self.light = light_llm or strong_llm
        self.light_is_distinct = light_llm is not None
        self.ledger = ledger
        self.policy = policy or TaskRoutingPolicy(tiers_share_model_family=light_llm is None)
        self.backend_fingerprint = str(backend_fingerprint or getattr(strong_llm, "model", "")
                                       or type(strong_llm).__name__)
        self.rows: list = []                    # full per-call ledger (audit)

    def tier_for(self, stage: str) -> str:
        if stage in STRONG_ONLY_STAGES:
            return "strong"
        return self.policy.tier_for(stage)

    def call(self, stage: str, prompt: str) -> str:
        tier = self.tier_for(stage)
        backend = self.strong if tier == "strong" else self.light
        allowed, why = self.ledger.allow_call(stage, len(prompt))
        if not allowed:
            raise BudgetExhausted(why, stage)
        t = time.time()
        retried = False
        try:
            reply = backend(prompt)
        except Exception as first:  # noqa: BLE001 — ONE bounded retry, globally capped
            if not self.ledger.retries_allowed():
                raise BudgetExhausted("provider_retry_budget_exhausted", stage) from first
            retried = True
            reply = backend(prompt)              # a second failure propagates to the stage runner
        reply = reply if isinstance(reply, str) else ""
        latency = time.time() - t
        self.ledger.record_call(stage, prompt_chars=len(prompt), reply_chars=len(reply),
                                latency_s=latency, tier=tier, retried=retried)
        self.rows.append({"stage": stage, "tier": tier, "prompt_chars": len(prompt),
                          "reply_chars": len(reply), "latency_s": round(latency, 3),
                          "retried": retried})
        return reply

    def manifest(self) -> dict:
        by = {}
        for r in self.rows:
            k = f"{r['stage']}|{r['tier']}"
            by[k] = by.get(k, 0) + 1
        return {"backend_fingerprint": self.backend_fingerprint,
                "light_tier_distinct_backend": self.light_is_distinct,
                "strong_only_stages": list(STRONG_ONLY_STAGES),
                "calls_by_stage_tier": dict(sorted(by.items())),
                "n_calls": len(self.rows),
                "retried_calls": sum(1 for r in self.rows if r["retried"])}
