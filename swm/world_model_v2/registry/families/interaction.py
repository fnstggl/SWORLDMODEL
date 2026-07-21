"""Bargaining / coalition / participation / platform / network-evolution families — executable transitions.

All are real functional forms with published provenance; each is a state transition the world's operators
call. Parameters are labeled priors/fitted; none is a universal law (recorded per the registry contract).
"""
from __future__ import annotations

import math


# ------------------------------------------------------------------ bargaining (Rubinstein)
def rubinstein_split(delta_a: float, delta_b: float) -> float:
    """Rubinstein 1982 alternating-offers unique SPE share to the first mover:
    x* = (1−δ_b)/(1−δ_a·δ_b). δ are per-round discount factors (patience). Returns proposer's share [0,1]."""
    return (1.0 - delta_b) / (1.0 - delta_a * delta_b)


def concession_offer(initial: float, reservation: float, t: float, deadline: float, *, beta: float = 1.0) -> float:
    """Time-dependent concession tactic (Faratin, Sierra & Jennings 1998): offer decays from `initial`
    toward `reservation` as the deadline approaches, at rate governed by β (β<1 boulware/tough,
    β>1 conceder). t, deadline in the same units."""
    frac = min(1.0, max(0.0, t / max(1e-9, deadline)))
    return initial + (reservation - initial) * (frac ** (1.0 / max(1e-6, beta)))


# ------------------------------------------------------------------ coalition (weighted voting power)
def banzhaf_power(weights: dict, quota: float) -> dict:
    """Banzhaf 1965 voting power index: fraction of swing coalitions each member is pivotal in.
    Exact enumeration (use for small committees; O(2^n)). quota = votes needed to pass."""
    members = list(weights)
    n = len(members)
    swings = {m: 0 for m in members}
    total_swings = 0
    for mask in range(1 << n):
        s = sum(weights[members[i]] for i in range(n) if mask & (1 << i))
        for i in range(n):
            if mask & (1 << i):
                without = s - weights[members[i]]
                if s >= quota > without:           # member i is pivotal
                    swings[members[i]] += 1
                    total_swings += 1
    if total_swings == 0:
        return {m: 1.0 / n for m in members}
    return {m: swings[m] / total_swings for m in members}


def coalition_forms(support_shares: dict, quota: float) -> tuple:
    """Greedy minimal-winning-coalition formation by descending support share. Returns (members, total)."""
    ordered = sorted(support_shares.items(), key=lambda kv: -kv[1])
    members, tot = [], 0.0
    for m, s in ordered:
        members.append(m)
        tot += s
        if tot >= quota:
            break
    return members, tot


# ------------------------------------------------------------------ participation (turnout / donation / mobilization)
def turnout_probability(base_rate: float, *, cost: float = 0.0, benefit: float = 0.0,
                        duty: float = 0.0, mobilized: float = 0.0) -> float:
    """Calculus-of-voting (Riker & Ordeshook 1968) as a bounded logit adjustment on a base turnout rate:
    p = σ(logit(base) − k_c·cost + k_b·benefit + k_d·duty + k_m·mobilized). Coefficients are labeled
    reference-class priors; must be refit per electorate."""
    z = math.log(min(1 - 1e-6, max(1e-6, base_rate)) / (1 - min(1 - 1e-6, max(1e-6, base_rate))))
    z += -1.0 * cost + 0.8 * benefit + 1.2 * duty + 0.6 * mobilized
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def donation_amount(capacity: float, *, ask: float, affinity: float, prior_gifts: float) -> float:
    """Bounded donation response: a fraction of capacity increasing in the ask (anchoring), affinity, and
    donor history, saturating at capacity. Reference-class form; refit per campaign."""
    frac = 1.0 / (1.0 + math.exp(-(0.6 * affinity + 0.4 * math.log1p(prior_gifts) - 0.5)))
    return min(capacity, frac * ask)


# ------------------------------------------------------------------ platform (examination / position bias / ranking)
def position_examination(rank: int, *, gamma: float = 0.8) -> float:
    """Cascade/position-bias examination probability (Craswell et al. 2008): P(examine|rank)=γ^rank.
    γ is the persistence of attention down a ranked list (fitted per platform)."""
    return gamma ** max(0, rank)


def click_probability(rank: int, relevance: float, *, gamma: float = 0.8) -> float:
    """Examination × relevance click model: P(click)=P(examine|rank)·relevance."""
    return position_examination(rank, gamma=gamma) * min(1.0, max(0.0, relevance))


def rank_by_score(items: dict) -> list:
    """Platform ranking/allocation: order items by score (desc). items: {id: score}. Returns [(id, rank)]."""
    return [(i, r) for r, (i, _) in enumerate(sorted(items.items(), key=lambda kv: -kv[1]))]


# ------------------------------------------------------------------ network rewiring / co-evolution
def rewire_probability(homophily: float, tie_age_days: float, *, decay_days: float = 90.0,
                       homophily_weight: float = 1.0) -> float:
    """Edge co-evolution: probability an edge persists/forms given trait similarity (homophily) and tie
    age (older ties decay unless reinforced). Snijders SAOM-style structural form; parameters labeled."""
    age_factor = math.exp(-max(0.0, tie_age_days) / decay_days)
    z = homophily_weight * (homophily - 0.5) + math.log(max(1e-6, age_factor))
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
