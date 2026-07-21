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


# ============================================================ Phase 9: typed NETWORK observation models (Part J)
# Each maps a typed evidence class to P(observation | edge exists) and P(observation | edge absent) on a
# specific relation LAYER — a detection model, exactly like StructuralDetectionModel. Different evidence classes
# carry different discriminating power (a logged message is near-certain evidence of a communication edge; a
# social follow is weak evidence of an influence edge; an *absence* of expected interaction is negative
# evidence). FIXED tables — the LLM supplies only the qualitative class/strength, never these numbers. A
# distinct likelihood per relation layer (the spec forbids one likelihood for every layer).
#
# schema: evidence_class -> {"layer", "detect" P(obs|edge), "false" P(obs|no edge), "polarity" +1 present / -1 absent}
EDGE_OBS_MODELS = {
    # near-definitional records (a logged message / official filing IS the relationship): tiny false rate
    "direct_communication_record":  {"layer": "communication", "detect": 0.92, "false": 0.008, "polarity": 1},
    "repeated_interaction":         {"layer": "communication", "detect": 0.90, "false": 0.06, "polarity": 1},
    "org_chart_relationship":       {"layer": "reporting",     "detect": 0.94, "false": 0.02, "polarity": 1},
    "formal_authority_record":      {"layer": "authority",     "detect": 0.95, "false": 0.015, "polarity": 1},
    "voting_alignment":             {"layer": "alliance",      "detect": 0.80, "false": 0.30, "polarity": 1},
    "resource_transfer":            {"layer": "resource",      "detect": 0.92, "false": 0.02, "polarity": 1},
    "endorsement":                  {"layer": "influence",     "detect": 0.75, "false": 0.20, "polarity": 1},
    "coattendance":                 {"layer": "affiliation",   "detect": 0.65, "false": 0.30, "polarity": 1},
    "social_follow":                {"layer": "influence",     "detect": 0.62, "false": 0.35, "polarity": 1},
    "content_exposure":             {"layer": "exposure",      "detect": 0.78, "false": 0.22, "polarity": 1},
    "public_statement_support":     {"layer": "alliance",      "detect": 0.70, "false": 0.28, "polarity": 1},
    "private_statement_support":    {"layer": "trust",         "detect": 0.82, "false": 0.18, "polarity": 1},
    "co_membership":                {"layer": "membership",    "detect": 0.88, "false": 0.12, "polarity": 1},
    "conflict_record":              {"layer": "conflict",      "detect": 0.90, "false": 0.08, "polarity": 1},
    "absence_of_expected_interaction": {"layer": "communication", "detect": 0.30, "false": 0.85, "polarity": -1},
    "edge_expiration":              {"layer": "communication", "detect": 0.20, "false": 0.80, "polarity": -1},
}
#: relation layers with distinct causal semantics (Part F) — >=10 required.
RELATION_LAYERS = ("communication", "friendship", "affiliation", "authority", "reporting", "trust",
                   "influence", "exposure", "resource", "alliance", "conflict", "membership",
                   "coordination", "jurisdiction")
#: strength bucket → multiplier that sharpens (strong) or flattens (weak) the detect/false gap toward 0.5.
_EDGE_STRENGTH_SHARPEN = {"weak": 0.5, "moderate": 0.8, "strong": 1.0}


@dataclass
class EdgeObservation:
    """One typed observation bearing on a directed edge (src→dst) in a relation layer. `present` False =
    negative/absence evidence. reliability from source type (NOT LLM-minted); strength from the LLM tag."""
    src: str
    dst: str
    evidence_class: str
    strength: str = "moderate"
    reliability: float = 0.8
    dependence_group: str = ""
    n_collapsed: int = 1
    claim_id: str = ""

    @property
    def layer(self) -> str:
        return EDGE_OBS_MODELS.get(self.evidence_class, {}).get("layer", "communication")


def _edge_rates(evidence_class: str, strength: str, reliability: float):
    """Effective (detect, false) for an edge observation, sharpened by strength and flattened toward 0.5 by
    (1-reliability). Returns the pair used in the Bernoulli-edge likelihood."""
    m = EDGE_OBS_MODELS.get(evidence_class, {"detect": 0.7, "false": 0.3})
    detect, false = m["detect"], m["false"]
    sharp = _EDGE_STRENGTH_SHARPEN.get(strength, 0.8)
    rel = max(0.0, min(1.0, reliability))
    # move detect/false toward 0.5 by (1 - sharp*rel): weak/unreliable evidence barely discriminates
    k = sharp * rel
    detect = 0.5 + k * (detect - 0.5)
    false = 0.5 + k * (false - 0.5)
    return max(1e-6, min(1 - 1e-6, detect)), max(1e-6, min(1 - 1e-6, false))


def edge_loglik(obs: EdgeObservation, exists: bool) -> float:
    """log P(observation | edge exists?) for a typed edge observation. Absence-polarity observations invert:
    seeing an *absence* is likely when the edge does NOT exist."""
    detect, false = _edge_rates(obs.evidence_class, obs.strength, obs.reliability)
    polarity = EDGE_OBS_MODELS.get(obs.evidence_class, {}).get("polarity", 1)
    # p_obs_given_edge / p_obs_given_no_edge; for absence evidence, the model's detect/false already encode
    # P(see-absence | edge) < P(see-absence | no edge), so no extra inversion of the pair is needed.
    p = detect if exists else false
    if polarity < 0:
        # the observation IS the absence; detect=P(absence|edge) is low, false=P(absence|no edge) is high
        p = detect if exists else false
    return math.log(max(1e-9, p))


def collapse_edge_observations(obs_list: list) -> list:
    """Dependence-collapse edge observations sharing (edge, layer, dependence_group) into ONE effective
    observation (max strength, most-reliable member) — syndicated reports of the same relationship count once
    (Part D, network variant)."""
    from collections import defaultdict
    groups, singletons = defaultdict(list), []
    for o in obs_list:
        if o.dependence_group:
            groups[(o.src, o.dst, o.layer, o.dependence_group)].append(o)
        else:
            singletons.append(o)
    order = {"weak": 0, "moderate": 1, "strong": 2}
    out = []
    for key, members in groups.items():
        best = max(members, key=lambda o: o.reliability)
        out.append(EdgeObservation(
            src=best.src, dst=best.dst, evidence_class=best.evidence_class,
            strength=max((m.strength for m in members), key=lambda s: order.get(s, 1)),
            reliability=best.reliability, dependence_group=key[3], n_collapsed=len(members),
            claim_id=f"edepgroup:{key[3]}"))
    return out + singletons
