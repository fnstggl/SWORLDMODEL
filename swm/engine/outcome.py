"""The native-typed Forecast — the answer object that always answers the actual question.

"Who wins NY-10" → a distribution over NAMED candidates. "Best headline" → ranked ACTUAL headline texts.
"Does Thiel reply to this email" → a scenario-specific p for that person and that message. Never a scalar
about an abstract logit. Every forecast carries its grounding report (what was established, from where,
what's MISSING), its calibration verdict (grade-or-abstain), and a per-persona audit — trust it or
knowingly distrust it, but never mistake it for more than it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Forecast:
    question: str
    mechanism: str                       # "grounded_agents:<process>" | "resolved_by_evidence" | "abstained"
    answer_space: dict = field(default_factory=dict)
    distribution: dict = field(default_factory=dict)     # option -> p (NATIVE: named options / binary)
    interval: dict = field(default_factory=dict)         # option -> [lo, hi] (interaction spread)
    ranked_artifacts: list = field(default_factory=list)  # artifact mode: [{text, p_engage, ...}]
    headline: str = ""
    calibration: dict = field(default_factory=dict)      # grade-or-abstain verdict (calibrate.py)
    grounding: dict = field(default_factory=dict)        # SceneDossier.as_report()
    audit: list = field(default_factory=list)            # per-persona probs + WHY (branch 0)
    rounds: list = field(default_factory=list)           # the real dates simulated
    abstain: bool = False
    abstain_reason: str = ""
    n_personas: int = 0
    n_llm_calls: int = 0
    detail: dict = field(default_factory=dict)           # mode-specific extras (individual per_state, ...)

    def top(self):
        if not self.distribution:
            return None
        return max(self.distribution.items(), key=lambda kv: kv[1])

    def as_dict(self) -> dict:
        return {"question": self.question, "mechanism": self.mechanism,
                "answer_space": self.answer_space, "distribution": self.distribution,
                "interval": self.interval, "ranked_artifacts": self.ranked_artifacts,
                "headline": self.headline, "calibration": self.calibration,
                "grounding": self.grounding, "audit": self.audit, "rounds": self.rounds,
                "abstain": self.abstain, "abstain_reason": self.abstain_reason,
                "n_personas": self.n_personas, "n_llm_calls": self.n_llm_calls, "detail": self.detail}


def build_headline(f: Forecast) -> str:
    """One sentence that answers the actual question — with the honesty flags inline."""
    tag = ""
    if f.calibration.get("abstain_confident"):
        tag = f"  [{f.calibration.get('grade', 'ungraded')} — hypothesis, not a calibrated forecast]"
    if f.abstain:
        return f"ABSTAINED: {f.abstain_reason}"
    if f.mechanism == "resolved_by_evidence":
        return f.headline
    if f.ranked_artifacts:
        best = f.ranked_artifacts[0]
        return (f"Best of {len(f.ranked_artifacts)}: \"{best['text']}\" "
                f"(simulated engagement {best['p_engage']:.0%} vs runner-up "
                f"{f.ranked_artifacts[1]['p_engage']:.0%})" if len(f.ranked_artifacts) > 1 else
                f"Best: \"{best['text']}\"") + tag
    t = f.top()
    if t is None:
        return "no distribution produced" + tag
    others = sorted(((o, p) for o, p in f.distribution.items() if o != t[0]),
                    key=lambda kv: -kv[1])[:3]
    rest = ", ".join(f"{o} {p:.0%}" for o, p in others)
    iv = f.interval.get(t[0])
    ivs = f" (branch spread {iv[0]:.0%}–{iv[1]:.0%})" if iv else ""
    return f"{t[0]}: {t[1]:.0%}{ivs}" + (f"; then {rest}" if rest else "") + tag
