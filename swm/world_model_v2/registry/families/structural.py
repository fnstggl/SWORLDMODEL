"""Structural mechanism families — Phase 6 continuation (deepen research-encoded families into executables).

These are the structural forms the priority matrix flagged as `structural_candidate_only` / `research_encoded`.
Each is a REAL executable transition; where a coefficient is a mathematical form (friendship paradox,
position propensity, Gamson proportionality) it is exact by construction; where it is an empirical shape
(weak-tie inverted-U, punishment-sustains-cooperation) the FORM is implemented and the magnitude carries
broad uncertainty. Only `position_bias_propensity` (Joachims 2017 eq. 7, η=1) carries a core-verified pack.

Families:
  position_bias_propensity   p(examine|rank)=(1/rank)^η        Joachims-Swaminathan-Schnabel 2017 (WSDM) eq.7
  friendship_paradox_targeting  E[deg(friend)] ≥ E[deg]        Feld 1991 (AJS) — exact for var(deg)>0
  weak_tie_transmission_shape   inverted-U in tie strength     Rajkumar et al. 2022 (Science) — shape only
  altruistic_punishment_cooperation  punishment sustains coop  Fehr & Gächter 2000 (AER) — form (magnitude broad)
  coalition_payoff_gamson    portfolio share ≈ seat share      Gamson 1961; Browne & Franklin 1973 — slope≈1
"""
from __future__ import annotations

import math


# ------------------------------------------------------------------ position bias (Joachims 2017 eq. 7)
def position_bias_propensity(rank: int, eta: float = 1.0) -> float:
    """Examination propensity by rank: p(examine | rank) = (1/rank)^η  (Joachims, Swaminathan & Schnabel
    2017, WSDM, eq. 7; core-verified from the primary PDF). η=1 is the default; η>1 underestimates small
    propensities (harmful), η<1 is conservative. Real-world estimated propensities decay to ≈0.12 by rank
    ~21. Supported: click-through under a fixed ranked list where clicks confound rank×relevance. Forbidden:
    treating a click as relevance/quality without inverse-propensity correction; a non-ranked feed."""
    r = max(1, int(rank))
    return (1.0 / r) ** max(0.0, eta)


def click_probability(rank: int, relevance: float, eta: float = 1.0) -> float:
    """P(click) = P(examine|rank) · P(click|examined, relevance). The examination/relevance split IS the
    mechanism (a click at rank 3 is worth more than a click at rank 1 for the same relevance)."""
    return position_bias_propensity(rank, eta) * max(0.0, min(1.0, relevance))


# ------------------------------------------------------------------ friendship-paradox targeting (Feld 1991)
def friendship_paradox_gain(degrees: list) -> dict:
    """Feld 1991: the average friend has degree E[d²]/E[d] ≥ E[d] (equality iff degree variance 0). Sampling
    a random node's random friend (nomination targeting) reaches a higher-degree node than sampling a random
    node — WITHOUT observing the graph. Returns mean node degree, mean friend degree, and the multiplicative
    gain. Kim et al. 2015 (Lancet) is the field-experimental confirmation that nomination beats random."""
    degs = [d for d in degrees if d is not None]
    if not degs:
        return {"mean_degree": 0.0, "mean_friend_degree": 0.0, "gain": 1.0}
    n = len(degs)
    e_d = sum(degs) / n
    e_d2 = sum(d * d for d in degs) / n
    mean_friend = e_d2 / e_d if e_d > 0 else 0.0
    return {"mean_degree": round(e_d, 3), "mean_friend_degree": round(mean_friend, 3),
            "gain": round(mean_friend / e_d, 3) if e_d > 0 else 1.0}


def nomination_seed_expected_degree(degrees: list) -> float:
    """Expected degree of a friendship-nomination seed = E[d²]/E[d] (the targeting mechanism's readout)."""
    return friendship_paradox_gain(degrees)["mean_friend_degree"]


# ------------------------------------------------------------------ weak-tie transmission (Rajkumar 2022 shape)
def weak_tie_transmission_shape(tie_strength: float, *, peak: float = 0.25, spread: float = 0.18) -> float:
    """Inverted-U novel-information transmission as a function of tie strength ∈ [0,1] (Rajkumar et al. 2022,
    Science: moderately weak ties maximize job mobility). Implemented as a Gaussian bump peaked at a
    moderately-weak strength. SHAPE only — the peak location/height are NOT a transportable coefficient
    (broad uncertainty); the causal magnitude is LinkedIn-specific and reverses in less-digital industries."""
    s = max(0.0, min(1.0, tie_strength))
    return math.exp(-((s - peak) ** 2) / (2 * spread ** 2))


# ------------------------------------------------------------------ altruistic punishment (Fehr-Gächter form)
def altruistic_punishment_cooperation(has_punishment: bool, period: int, n_periods: int = 10, *,
                                      coop_with: float = 0.9, coop_without: float = 0.15) -> float:
    """Cooperation fraction over repeated public-goods rounds. Fehr & Gächter 2000 (AER): WITHOUT punishment
    cooperation decays toward free-riding; WITH costly peer punishment it rises and is sustained near-full.
    FORM implemented (rise-with / decay-without); the endpoint magnitudes are BROAD priors (the paper's
    scanned tables were not core-verified this run — kept research-encoded, not a validated pack). Forbidden:
    treating the endpoints as calibrated; anti-social punishment cultures violate the sign."""
    frac = period / max(1, n_periods - 1)
    if has_punishment:
        return coop_without + (coop_with - coop_without) * frac       # rises and sustains
    return max(0.02, coop_without * (1.0 - 0.6 * frac))               # decays toward free-riding


# ------------------------------------------------------------------ Gamson coalition payoff
def coalition_payoff_gamson(seat_share: float, *, slope: float = 1.0, intercept: float = 0.0) -> float:
    """Gamson's law (Gamson 1961; Browne & Franklin 1973 replication): a coalition partner's portfolio share
    is ≈ proportional to its seat contribution (slope ≈ 1, R²≈0.9, with a small-party bonus). Structural
    proportionality; the exact slope/intercept are research-encoded (not core-verified this run)."""
    return max(0.0, min(1.0, intercept + slope * max(0.0, min(1.0, seat_share))))
