"""Social / relationship / influence mechanism families — executable transitions.

Trust and opinion dynamics are candidate STRUCTURAL FAMILIES, not universal laws (the audit is explicit:
DeGroot/bounded-confidence/threshold are hypotheses whose parameters and applicability must be validated).
Each is executable and carries its published form + limits in the registry record.

Families:
  trust_formation / trust_violation / trust_repair
                         asymmetric trust dynamics (gains slow, losses fast; repair partial) —
                         form after Slovic 1993 ("asymmetry principle") + trust-game evidence
  reciprocity            direct reciprocity update on an edge (tit-for-tat-like, continuous)
  degroot_influence      DeGroot 1974 consensus: opinion ← Σ_j w_ij·opinion_j (row-stochastic)
  bounded_confidence     Hegselmann-Krause 2002: average only over opinions within ε
  threshold_adoption     Granovetter 1978 threshold: adopt iff active-neighbor fraction ≥ threshold
  latent_expressed       latent vs expressed opinion (preference falsification, Kuran 1995): expressed =
                         latent unless social pressure exceeds private conviction
"""
from __future__ import annotations


# ------------------------------------------------------------------ trust (asymmetric)
def trust_update(trust: float, outcome: str, *, gain: float = 0.05, loss: float = 0.20,
                 repair: float = 0.08) -> float:
    """Asymmetric trust dynamics. outcome ∈ {cooperated, defected, repaired}. Losses steeper than gains
    (Slovic 1993 asymmetry); repair partial. Bounds [0,1]. Rates are reference-class priors — must be
    refit per relationship where data exists."""
    if outcome == "cooperated":
        return min(1.0, trust + gain * (1.0 - trust))
    if outcome == "defected":
        return max(0.0, trust - loss * trust - loss * 0.5)
    if outcome == "repaired":
        return min(1.0, trust + repair * (1.0 - trust))
    return trust


# ------------------------------------------------------------------ reciprocity
def reciprocity_update(edge_strength: float, other_action_value: float, *, rate: float = 0.15,
                       baseline: float = 0.0) -> float:
    """Continuous direct reciprocity: move edge strength toward the kindness of the other's last action
    (value − baseline), bounded [0,1]. rate labels adaptation speed."""
    return min(1.0, max(0.0, edge_strength + rate * (other_action_value - baseline)))


# ------------------------------------------------------------------ DeGroot consensus
def degroot_step(opinions: dict, weights: dict) -> dict:
    """One DeGroot round: o_i ← Σ_j w_ij·o_j. weights[i] is a dict {j: w_ij} row-stochastic over known
    j (self-weight included). DeGroot 1974 — LIMIT: assumes fixed, trusting, row-stochastic influence;
    empirically people are NOT naive DeGroot updaters (Chandrasekhar et al. 2020). Candidate family."""
    out = {}
    for i, oi in opinions.items():
        w = weights.get(i, {i: 1.0})
        z = sum(w.values()) or 1.0
        out[i] = sum(wij * opinions.get(j, oi) for j, wij in w.items()) / z
    return out


# ------------------------------------------------------------------ bounded confidence (H-K)
def bounded_confidence_step(opinions: dict, *, eps: float = 0.2, neighbors: dict = None) -> dict:
    """Hegselmann-Krause 2002: each agent averages only opinions within ε of its own. `neighbors` restricts
    the comparison set per agent (None = fully mixed). eps is the confidence bound (must be fit/varied)."""
    out = {}
    for i, oi in opinions.items():
        pool = neighbors.get(i, list(opinions)) if neighbors else list(opinions)
        close = [opinions[j] for j in pool if abs(opinions[j] - oi) <= eps]
        out[i] = sum(close) / len(close) if close else oi
    return out


# ------------------------------------------------------------------ threshold adoption
def threshold_adopt(active: bool, active_neighbor_frac: float, threshold: float) -> bool:
    """Granovetter 1978 threshold model: adopt (and stay adopted) iff the active-neighbor fraction meets
    the individual threshold. Heterogeneous thresholds (a distribution) produce cascade/tipping behavior."""
    return active or (active_neighbor_frac >= threshold)


# ------------------------------------------------------------------ latent vs expressed opinion
def expressed_opinion(latent: float, *, social_pressure: float, conviction: float) -> float:
    """Preference falsification (Kuran 1995): expressed opinion tracks latent unless social pressure in
    the opposite direction exceeds private conviction, in which case expression is pulled toward the
    perceived norm. Returns the expressed value in the same [0,1] scale as latent."""
    if social_pressure <= conviction:
        return latent
    norm = 1.0 - latent                                       # pressure pushes toward the opposite pole
    pull = min(1.0, (social_pressure - conviction))
    return latent + pull * (norm - latent)
