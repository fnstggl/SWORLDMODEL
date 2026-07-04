"""Actors and stakeholder segments — the explicit agents a real simulation represents (Phase 2/5).

A simulation is only a simulation if it represents actors/segments explicitly and evolves their
state as they react. This module is those actors:

- `ActorState`      — base latent state (preferences, incentives, attention, stance, exposure,
                      memory, uncertainty).
- `SegmentAgent`    — a representative community segment (technical readers, casual front-page
                      browsers, ...) with a size weight and feature sensitivities. Aggregate
                      simulation runs over ~8 weighted segments, not millions of people.
- `IndividualActorState` — a named recipient/entity for individualized simulation (email reply,
                      objection, conversion), carrying relationship + thread state.

Segments react to an action via a `ReactionModel` (reactions.py); the engine (engine.py) updates
their attention/exposure/stance after each step. The final outcome is read from the *distribution of
simulated trajectories*, never from a classifier over the initial state.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActorState:
    actor_id: str
    preferences: dict[str, float] = field(default_factory=dict)  # feature -> affinity in [-1,1]
    incentives: dict[str, float] = field(default_factory=dict)
    attention: float = 1.0            # how available/active right now
    stance: float = 0.0               # trust/support toward the actor/source, [-1,1]
    exposure: float = 0.0             # cumulative exposure to the current action
    social_influence_susceptibility: float = 0.5
    memory: list = field(default_factory=list)
    uncertainty: float = 1.0

    def copy(self) -> "ActorState":
        return ActorState(self.actor_id, dict(self.preferences), dict(self.incentives),
                          self.attention, self.stance, self.exposure,
                          self.social_influence_susceptibility, list(self.memory), self.uncertainty)


@dataclass
class SegmentAgent:
    """A weighted community segment — the unit of AGGREGATE simulation.

    `weight` is the segment's relative size / share of the attention pool. `affinity` maps action
    feature keys to this segment's sensitivity (how much a unit of that feature raises its reaction
    propensity). `base_rate` is its baseline upvote/engage propensity when exposed. Attention and
    exposure evolve during a trajectory."""
    segment_id: str
    weight: float
    affinity: dict[str, float] = field(default_factory=dict)
    base_rate: float = 0.05
    attention: float = 1.0
    social_susceptibility: float = 0.5
    exposure: float = 0.0

    def copy(self) -> "SegmentAgent":
        return SegmentAgent(self.segment_id, self.weight, dict(self.affinity), self.base_rate,
                            self.attention, self.social_susceptibility, self.exposure)


@dataclass
class IndividualActorState(ActorState):
    """A named recipient for individualized simulation (email/CRM). Adds relationship + thread state
    on top of the base actor latent state."""
    sender_id: str = ""
    relationship_stance: float = 0.0      # warmth/standing with the sender, [-1,1]
    prior_replies: int = 0
    prior_ignores: int = 0
    thread_len: int = 0
    last_contact_ts: float | None = None
    fatigue: float = 0.0                  # rises with repeated outreach; suppresses response

    @property
    def responsiveness_prior(self) -> float:
        n = self.prior_replies + self.prior_ignores
        return (self.prior_replies + 1.0) / (n + 2.0)   # Laplace-smoothed personal reply rate


# ------------------------------------------------------------------ HN community segments (default)
# Deliberately a small, interpretable set. Weights are relative shares of the attention pool; they
# are re-normalized by the engine. Affinities are prior sensitivities, refined by fitting.
def default_hn_segments() -> list[SegmentAgent]:
    return [
        SegmentAgent("casual_frontpage", weight=0.42, base_rate=0.030, social_susceptibility=0.85,
                     affinity={"audience_fit": 1.2, "hn_native": 0.6, "novelty": 0.5,
                               "emotional_valence": 0.3, "controversy": 0.4}),
        SegmentAgent("technical", weight=0.16, base_rate=0.06, social_susceptibility=0.4,
                     affinity={"technical_depth": 1.4, "specificity": 0.7, "hn_native": 0.8,
                               "source_credibility": 0.5}),
        SegmentAgent("ai_ml", weight=0.12, base_rate=0.06, social_susceptibility=0.6,
                     affinity={"topic_ai": 1.6, "technical_depth": 0.7, "novelty": 0.6}),
        SegmentAgent("startup_biz", weight=0.10, base_rate=0.05, social_susceptibility=0.6,
                     affinity={"topic_business": 1.4, "novelty": 0.6, "hn_native": 0.4}),
        SegmentAgent("oss_maintainers", weight=0.07, base_rate=0.06, social_susceptibility=0.35,
                     affinity={"cat_Show": 1.2, "hn_native": 1.0, "technical_depth": 0.6}),
        SegmentAgent("security", weight=0.06, base_rate=0.06, social_susceptibility=0.4,
                     affinity={"topic_security": 1.8, "technical_depth": 0.6, "controversy": 0.4}),
        SegmentAgent("science", weight=0.05, base_rate=0.055, social_susceptibility=0.5,
                     affinity={"topic_science": 1.6, "source_credibility": 0.7, "novelty": 0.5}),
        SegmentAgent("politics_policy", weight=0.02, base_rate=0.05, social_susceptibility=0.7,
                     affinity={"topic_politics": 1.6, "controversy": 1.0, "emotional_valence": -0.3}),
    ]
