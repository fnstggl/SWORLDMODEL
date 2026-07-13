"""Claim-class observation models + dependence-corrected likelihood — Phase 3 (Parts C, D).

These convert a QUALITATIVE `ClaimTag` (from the LLM) into a NUMERIC likelihood using FIXED, registered
parameters — the LLM never supplies a probability. Two production models drive the two consumed posteriors:

  DirectionalRateModel      P(claim direction | outcome base-rate r) — a noisy binary "vote" with
                            sensitivity/specificity keyed on evidential strength, flattened toward
                            uninformative (0.5) by low source reliability and strategic-statement discount.
  StructuralDetectionModel  P(claim supports/opposes hypothesis h | h is the true structure) — a detection
                            model with detect/false rates keyed on strength; the likelihood ratio moves
                            structural mass toward the hypotheses the evidence supports.

DEPENDENCE CORRECTION (Part D): claims are collapsed by their Phase-2 `dependence_group` BEFORE any
likelihood is multiplied, so twenty syndicated copies of one report contribute ONE effective observation,
not twenty. The collapsed group keeps the most-reliable member's direction and the max strength.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: evidential-strength → (sensitivity, specificity) of the claim as a signal of the true direction.
#: strong evidence discriminates well; weak evidence is near-uninformative. FIXED — not LLM-minted.
_STRENGTH_SENS_SPEC = {"weak": (0.58, 0.58), "moderate": (0.72, 0.72), "strong": (0.85, 0.85)}
#: structural detection: (P(support h | h true), P(support h | h false)) by strength.
_STRENGTH_DETECT_FALSE = {"weak": (0.55, 0.45), "moderate": (0.70, 0.30), "strong": (0.85, 0.15)}
STRATEGIC_DISCOUNT = 0.6            # a strategic public statement is worth 60% of a costly signal


def _effective_reliability(tag) -> float:
    r = max(0.0, min(1.0, tag.reliability))
    if tag.is_strategic:
        r *= STRATEGIC_DISCOUNT
    return r


@dataclass
class DirectionalRateModel:
    """P(claim | outcome_rate r). A supports_yes claim is a noisy 'yes vote'; supports_no a 'no vote';
    neutral is uninformative (flat). sens/spec come from strength; reliability flattens toward 0.5."""
    name: str = "directional_rate"

    def likelihood(self, tag, r: float) -> float:
        if tag.outcome_direction == "neutral":
            return 1.0                                         # uninformative → no reweight
        sens, spec = _STRENGTH_SENS_SPEC.get(tag.strength, (0.72, 0.72))
        rel = _effective_reliability(tag)
        # blend the discriminating signal toward 0.5 by (1-reliability): an unreliable source ≈ coin flip
        sens = 0.5 + rel * (sens - 0.5)
        spec = 0.5 + rel * (spec - 0.5)
        if tag.outcome_direction == "supports_yes":
            p = r * sens + (1.0 - r) * (1.0 - spec)            # P(observe a yes-vote | rate r)
        else:                                                  # supports_no
            p = (1.0 - r) * sens + r * (1.0 - spec)
        return max(1e-6, min(1.0 - 1e-6, p))


@dataclass
class StructuralDetectionModel:
    """P(claim supports/opposes h | h true). Multiplying across dependence groups yields the structural
    posterior. A claim that supports h raises h's likelihood; one that opposes h lowers it."""
    name: str = "structural_detection"

    def loglik_for_hypothesis(self, tag, hid: str) -> float:
        detect, false = _STRENGTH_DETECT_FALSE.get(tag.strength, (0.70, 0.30))
        rel = _effective_reliability(tag)
        detect = 0.5 + rel * (detect - 0.5)
        false = 0.5 + rel * (false - 0.5)
        ll = 0.0
        if hid in tag.supports_hypotheses:
            ll += math.log(detect)
        elif tag.supports_hypotheses:                          # supports some OTHER hypothesis
            ll += math.log(max(1e-6, false))
        if hid in tag.opposes_hypotheses:
            ll += math.log(max(1e-6, 1.0 - detect))
        return ll


# --------------------------------------------------------------------------- dependence correction (Part D)
def collapse_by_dependence(tags: list) -> list:
    """Collapse tags sharing a Phase-2 dependence_group into ONE effective tag (most-reliable member's
    direction; max strength; supports/opposes unioned). Ungrouped tags (empty group) pass through as
    singletons. This is applied BEFORE any likelihood multiplication so syndicated copies count once."""
    from collections import defaultdict
    groups = defaultdict(list)
    singletons = []
    for t in tags:
        if t.dependence_group:
            groups[t.dependence_group].append(t)
        else:
            singletons.append(t)
    _order = {"weak": 0, "moderate": 1, "strong": 2}
    collapsed = []
    for gid, members in groups.items():
        best = max(members, key=lambda t: t.reliability)
        rep = ClaimTagLike(
            claim_id=f"depgroup:{gid}", outcome_direction=best.outcome_direction,
            supports_hypotheses=sorted({h for m in members for h in m.supports_hypotheses}),
            opposes_hypotheses=sorted({h for m in members for h in m.opposes_hypotheses}),
            strength=max((m.strength for m in members), key=lambda s: _order.get(s, 1)),
            is_strategic=all(m.is_strategic for m in members), reliability=best.reliability,
            dependence_group=gid, n_collapsed=len(members))
        collapsed.append(rep)
    return collapsed + singletons


@dataclass
class ClaimTagLike:
    """A collapsed dependence-group representative (duck-types ClaimTag for the models)."""
    claim_id: str
    outcome_direction: str = "neutral"
    supports_hypotheses: list = None
    opposes_hypotheses: list = None
    strength: str = "moderate"
    is_strategic: bool = False
    reliability: float = 0.8
    dependence_group: str = ""
    n_collapsed: int = 1

    def __post_init__(self):
        self.supports_hypotheses = self.supports_hypotheses or []
        self.opposes_hypotheses = self.opposes_hypotheses or []
