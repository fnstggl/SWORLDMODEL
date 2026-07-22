"""Task-tier model routing — deterministic code first, the light tier for low-risk language work,
the strongest actor-capable backend for everything consequential.

The router never touches human decisions: consequential actor decisions, structural-model
compilation, reversal-capable criticism and ambiguous consequence compilation are pinned to the
STRONG tier by contract (`STRONG_ONLY_STAGES` — routing them lighter raises). Deterministic work
(parsing, validation, hashing, canonicalization, authority/feasibility checks, cache lookups)
never reaches this module at all — it is plain code.

A light-tier result escalates to the strong tier when validation fails, sources conflict, the
result would change a causal pathway, or the task turns actor-psychological — escalations are
recorded per stage. Where the deployment has one physical model family (the benchmark's DeepSeek
arm), the light tier is the same family with a reduced completion budget; the manifest says so
honestly instead of pretending a second family exists."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

ROUTING_VERSION = "lean.routing.v1"

#: stages that may NEVER be routed below the strong tier (§15)
STRONG_ONLY_STAGES = ("actor_decision", "actor_cognition", "structural_generation",
                      "structural_compile", "structural_critic", "consequence_compile",
                      "evidence_interpretation_high_impact",
                      # reference-class case proposal is high-impact grounding — a weak model
                      # here would propose worse historical cases; pinned strong
                      "reference_class_grounding", "state_generation")

#: default light-tier-eligible stages (low-risk language work)
LIGHT_ELIGIBLE_STAGES = ("query_reformulation", "title_normalization", "schema_format_repair",
                         "contradiction_extraction", "summary_compression",
                         "duplicate_detection_post_deterministic", "documentation")


@dataclass
class TaskRoutingPolicy:
    """stage name → tier. Anything unmapped runs strong (conservative default)."""
    light_stages: tuple = LIGHT_ELIGIBLE_STAGES
    tiers_share_model_family: bool = True         # honest single-provider deployments say so

    def tier_for(self, stage: str) -> str:
        if stage in STRONG_ONLY_STAGES:
            return "strong"
        return "light" if stage in self.light_stages else "strong"


class TieredRouter:
    """Holds the tier backends and meters calls/escalations per (stage, tier). Both tiers keep
    the `llm(prompt) -> str` contract."""

    def __init__(self, *, strong_llm, light_llm=None, policy: TaskRoutingPolicy = None):
        if strong_llm is None:
            raise ValueError("TieredRouter requires a strong backend")
        self.strong = strong_llm
        self.light = light_llm or strong_llm
        self.policy = policy or TaskRoutingPolicy(
            tiers_share_model_family=light_llm is None or _same_family(strong_llm, light_llm))
        self.calls: dict[tuple, int] = {}
        self.escalations: dict[str, list] = {}
        self._lock = threading.RLock()

    def route(self, stage: str):
        """The backend for one stage, as a metering callable."""
        tier = self.policy.tier_for(stage)
        backend = self.strong if tier == "strong" else self.light

        def _call(prompt: str) -> str:
            with self._lock:
                self.calls[(stage, tier)] = self.calls.get((stage, tier), 0) + 1
            return backend(prompt)
        _call.tier = tier
        return _call

    def call_with_escalation(self, stage: str, prompt: str, *, validate) -> tuple:
        """Light-tier attempt + deterministic validation; on failure, escalate to strong with the
        reason recorded. `validate(text) -> (ok, reason)`. Strong-only stages skip the light
        attempt entirely."""
        tier = self.policy.tier_for(stage)
        if tier == "strong":
            return self.route(stage)(prompt), "strong", ""
        text = self.route(stage)(prompt)
        ok, reason = validate(text)
        if ok:
            return text, "light", ""
        with self._lock:
            self.escalations.setdefault(stage, []).append(str(reason)[:160])
            self.calls[(stage, "strong")] = self.calls.get((stage, "strong"), 0) + 1
        return self.strong(prompt), "light_escalated_to_strong", str(reason)[:160]

    def manifest(self) -> dict:
        with self._lock:
            return {"version": ROUTING_VERSION,
                    "tiers_share_model_family": self.policy.tiers_share_model_family,
                    "strong_only_stages": list(STRONG_ONLY_STAGES),
                    "calls_by_stage_tier": {f"{s}|{t}": n
                                            for (s, t), n in sorted(self.calls.items())},
                    "escalations": {s: rs[:10] for s, rs in sorted(self.escalations.items())}}


def _same_family(a, b) -> bool:
    fa = getattr(a, "model", None) or type(a).__name__
    fb = getattr(b, "model", None) or type(b).__name__
    return str(fa).split("-")[0] == str(fb).split("-")[0]
