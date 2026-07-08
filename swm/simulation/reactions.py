"""Reaction models: p(reaction | actor/segment state, action, context, exposure) (Phase 3).

A segment exposed to an action reacts — ignore / read / upvote / reply / share / oppose / support /
convert — with some intensity. The reaction model computes that propensity from the segment's latent
affinities, the action's features, the current trajectory state (social proof, novelty), and author
reputation. The engine turns propensity + exposed count into a SAMPLED reaction each step; the final
probability comes from the trajectory distribution, so the reaction model is a *transition
parameter*, never the outcome classifier.

Two models (both required by the spec):
- `HeuristicReactionModel` — transparent rules over affinity·features, social proof, novelty.
- `LearnedReactionModel` — the same functional form but with per-segment scale + global sensitivity
  parameters FIT from historical trajectories (final-score-matching, via policies.fit_*), so the
  reaction propensities are learned from data rather than hand-set.

LLM involvement is confined to producing the action FEATURES consumed here (novelty, controversy,
technical_depth, audience_fit, ...) — it is never the reaction probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.simulation.actors import SegmentAgent
from swm.simulation.trajectory_state import TrajectoryState

REACTION_TYPES = ("ignore", "read", "upvote", "reply", "share", "oppose", "support", "convert")


@dataclass
class Reaction:
    actor_id: str
    content_ref: str
    reaction_type: str
    intensity: float
    timestamp: float
    features: dict = field(default_factory=dict)
    uncertainty: float = 1.0


def _affinity_score(segment: SegmentAgent, action_feats: dict[str, float]) -> float:
    """Dot product of the segment's affinities with the action features (bounded, logistic-ish)."""
    z = 0.0
    for k, w in segment.affinity.items():
        z += w * float(action_feats.get(k, 0.0))
    return z


@dataclass
class HeuristicReactionModel:
    """Transparent first-pass reaction propensity. Global knobs are the transition parameters that
    policies.fit tunes; defaults are sane priors."""
    affinity_gain: float = 1.0
    social_proof_gain: float = 1.2
    novelty_gain: float = 1.0
    author_rep_gain: float = 0.6

    def upvote_propensity(self, segment: SegmentAgent, action_feats: dict[str, float],
                          tstate: TrajectoryState, author_rep: float) -> float:
        """Per-exposed-person probability this segment upvotes at the current step, in [0,1]."""
        aff = _affinity_score(segment, action_feats) * self.affinity_gain
        social = 1.0 + self.social_proof_gain * tstate.social_proof * segment.social_susceptibility
        novelty = tstate.novelty ** self.novelty_gain
        rep = 1.0 + self.author_rep_gain * author_rep     # author_rep ~ centered log-reputation
        base_logit = math.log(max(1e-6, segment.base_rate) / (1 - max(1e-6, segment.base_rate)))
        p = 1.0 / (1.0 + math.exp(-(base_logit + aff)))
        return max(0.0, min(0.95, p * social * novelty * rep * segment.attention))

    def reaction_type(self, segment: SegmentAgent, action_feats: dict[str, float]) -> str:
        """Dominant qualitative reaction mode (diagnostic; the score uses upvotes)."""
        contro = float(action_feats.get("controversy", 0.0))
        if contro > 0.6 and segment.social_susceptibility > 0.6:
            return "oppose" if action_feats.get("emotional_valence", 0.5) < 0.4 else "support"
        if float(action_feats.get("cat_Show", 0.0)) or float(action_feats.get("hn_native", 0.0)) > 0.6:
            return "upvote"
        return "upvote"


@dataclass
class LearnedReactionModel(HeuristicReactionModel):
    """Same functional form, but with per-segment base-rate multipliers and global gains FIT from
    historical final-score outcomes (see policies.fit_reaction_params). 'Learned from trajectories'
    in the sense that the parameters are chosen to make simulated trajectory outcomes match observed
    scores — we have no per-reaction labels, so final-outcome matching is the honest training signal."""
    segment_scale: dict[str, float] = field(default_factory=dict)

    def upvote_propensity(self, segment, action_feats, tstate, author_rep):
        p = super().upvote_propensity(segment, action_feats, tstate, author_rep)
        return max(0.0, min(0.95, p * self.segment_scale.get(segment.segment_id, 1.0)))
