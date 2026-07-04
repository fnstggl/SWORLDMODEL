"""Incentives / stakes state (spec: "incentives/stakes" in both regimes).

A response is shaped by *why the audience would act*: the stakes of the topic, the reward structure
of the platform (karma, virality, money), and the controversy/identity load that drives engagement.
These are not content features — they are properties of the situation that modulate how strongly a
given content signal converts to a response.

This module keeps them explicit and, crucially, ablatable: each incentive is a named scalar in
[0,1] (or a signed controversy axis) with a documented source. `IncentiveState` is attached to a
`PopulationState` (aggregate) or read per-action (individual). The transition model may use them as
features or as multipliers on the base propensity; ablation decides whether they earn their place.

No claim that these are causal — they are decision-relevant covariates, kept because they help
held-out prediction, dropped if they don't.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IncentiveState:
    """Situational incentive covariates. All roughly in [0,1] unless noted.

    - stakes:        how consequential the topic is (money/safety/status on the line)
    - controversy:   how identity/tribe-activating (drives engagement both ways)
    - novelty:       how new/surprising vs. saturated the topic is
    - reward_gradient: platform payoff for engaging (karma/virality/monetary)
    - effort_cost:   cost to the responder of engaging (long read, hard ask) — suppresses response
    """
    stakes: float = 0.0
    controversy: float = 0.0
    novelty: float = 0.0
    reward_gradient: float = 0.5
    effort_cost: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def as_features(self) -> dict[str, float]:
        return {
            "inc_stakes": self.stakes,
            "inc_controversy": self.controversy,
            "inc_novelty": self.novelty,
            "inc_reward_gradient": self.reward_gradient,
            "inc_effort_cost": self.effort_cost,
            **{f"inc_{k}": v for k, v in self.extra.items()},
        }

    def engagement_multiplier(self) -> float:
        """A bounded, monotone summary: high stakes/controversy/novelty/reward raise the odds of a
        response; effort cost lowers them. Used ONLY as an optional feature/prior, never as the
        probability itself (that comes from the calibrated head)."""
        drive = (0.9 * self.stakes + 1.1 * self.controversy + 0.8 * self.novelty
                 + 0.7 * (self.reward_gradient - 0.5))
        supp = 0.9 * self.effort_cost
        # map net drive to a multiplier in ~[0.5, 2.0]
        net = drive - supp
        return max(0.5, min(2.0, 1.0 + 0.6 * net))


# --- cheap, transparent extractors for the HN-style aggregate domain -------------------------
# (LLM extractors can replace these; they never see the outcome, so they don't leak.)

_STAKE_KW = ("lawsuit", "acquire", "acquisition", "layoff", "breach", "raise", "ipo", "ban",
             "recall", "outage", "vulnerability", "exploit", "death", "war", "crash", "bankrupt")
_CONTROVERSY_KW = ("vs", "why you", "the truth about", "controversial", "ban", "woke", "drama",
                   "harmful", "unethical", "scam", "lawsuit", "boycott", "backlash", "rant")
_NOVELTY_KW = ("first", "new", "introducing", "announcing", "breakthrough", "novel", "release",
               "launch", "show hn", "we built", "i built", "open source")


def incentives_from_title(title: str, *, domain: str = "", is_text: bool = False,
                          title_len_chars: int = 0) -> IncentiveState:
    """Heuristic incentive extraction from a headline. Bounded, deterministic, leakage-free."""
    t = title.lower()
    def hits(kws: tuple[str, ...]) -> float:
        return min(1.0, sum(1 for k in kws if k in t) / 2.0)
    length = title_len_chars or len(title)
    return IncentiveState(
        stakes=hits(_STAKE_KW),
        controversy=hits(_CONTROVERSY_KW),
        novelty=hits(_NOVELTY_KW),
        reward_gradient=0.6 if t.startswith(("show hn", "ask hn")) else 0.5,
        # long link-posts to heavy domains cost more attention; short Show-HNs cost less
        effort_cost=min(1.0, max(0.0, (length - 60) / 120.0)) * (0.5 if is_text else 1.0),
    )
