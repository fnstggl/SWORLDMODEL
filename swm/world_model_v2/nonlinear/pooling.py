"""Hierarchical partial pooling — Phase 7, Part 3.

When a mechanism's parameter varies by group (actor, community, platform, institution, time period, dataset),
the two extremes are both wrong: one global estimate ignores real heterogeneity; a separate unconstrained fit
per group overfits sparse groups. Partial pooling shrinks each group toward the population mean by an amount
set by how much data the group has and how much groups actually differ — the empirical-Bayes / random-effects
answer. Sparse groups borrow strength; data-rich groups keep their signal.

This is a lightweight, dependency-free empirical-Bayes estimator for group means (Gaussian and Beta-Binomial),
which is the case that recurs across the Phase-7 testbeds (per-test CTR in Upworthy, per-name adoption, per-
segment churn). Full hierarchical MCMC is available in the offline layer (`nonlinear/fit.py`, PyMC-optional);
this runtime-side estimator returns the pooled estimate, per-group posteriors, shrinkage, ESS, and — honestly —
which groups were too sparse to escape the prior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PooledEstimate:
    grand_mean: float
    between_var: float                   # τ² — how much groups genuinely differ (0 → full pooling)
    within_var: float                    # σ² — noise scale
    groups: dict = field(default_factory=dict)   # gid -> {raw, n, shrunk, shrinkage, se}
    n_groups: int = 0
    method: str = "empirical_bayes_gaussian"

    def estimate(self, gid) -> float:
        g = self.groups.get(gid)
        return g["shrunk"] if g else self.grand_mean

    def transfer_estimate(self) -> float:
        """Out-of-group / new-group prediction = the population mean (no group data → full shrinkage)."""
        return self.grand_mean

    def as_dict(self):
        return {"grand_mean": round(self.grand_mean, 6), "between_var": round(self.between_var, 8),
                "within_var": round(self.within_var, 8), "n_groups": self.n_groups, "method": self.method,
                "groups": {str(k): {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                    for kk, vv in v.items()} for k, v in self.groups.items()}}


def pool_gaussian(group_stats: dict) -> PooledEstimate:
    """Empirical-Bayes shrinkage for group means.

    group_stats: {gid: {"mean": ȳ_g, "n": n_g, "var": s²_g (optional within-group variance)}}.
    Uses a method-of-moments τ² (DerSimonian–Laird style): shrink factor B_g = τ²/(τ² + σ²/n_g);
    shrunk_g = B_g·ȳ_g + (1−B_g)·grand_mean. B_g→0 for sparse/noisy groups (full pooling)."""
    gids = list(group_stats)
    ns = {g: max(1.0, float(group_stats[g].get("n", 1))) for g in gids}
    ys = {g: float(group_stats[g]["mean"]) for g in gids}
    ntot = sum(ns.values())
    grand = sum(ys[g] * ns[g] for g in gids) / (ntot or 1.0)
    # within-group variance: use provided, else a pooled Bernoulli-style estimate
    within = {}
    for g in gids:
        v = group_stats[g].get("var")
        if v is None:
            p = min(1 - 1e-9, max(1e-9, ys[g]))
            v = p * (1 - p)                 # Bernoulli default
        within[g] = float(v)
    sigma2 = sum(within[g] * ns[g] for g in gids) / (ntot or 1.0)
    # between-group variance via method of moments
    if len(gids) > 1:
        wsum = sum(ns[g] for g in gids)
        q = sum(ns[g] * (ys[g] - grand) ** 2 for g in gids)
        denom = wsum - sum(ns[g] ** 2 for g in gids) / wsum
        tau2 = max(0.0, (q - (len(gids) - 1) * sigma2) / (denom or 1.0)) if denom > 0 else 0.0
    else:
        tau2 = 0.0
    groups = {}
    for g in gids:
        se2 = sigma2 / ns[g]
        B = tau2 / (tau2 + se2) if (tau2 + se2) > 0 else 0.0
        shrunk = B * ys[g] + (1 - B) * grand
        groups[g] = {"raw": ys[g], "n": ns[g], "shrunk": shrunk, "shrinkage": round(1 - B, 4),
                     "se": math.sqrt(se2), "escaped_prior": B > 0.5}
    return PooledEstimate(grand_mean=grand, between_var=tau2, within_var=sigma2, groups=groups,
                          n_groups=len(gids))


def pool_beta_binomial(group_counts: dict) -> PooledEstimate:
    """Empirical-Bayes shrinkage for group RATES with a Beta prior fit by moment matching across groups.

    group_counts: {gid: {"k": successes, "n": trials}}. Returns Beta-Binomial posterior means
    (k+α)/(n+α+β); sparse groups shrink to the population rate α/(α+β)."""
    gids = list(group_counts)
    rates, ns = {}, {}
    for g in gids:
        n = max(1.0, float(group_counts[g]["n"]))
        rates[g] = float(group_counts[g]["k"]) / n
        ns[g] = n
    ntot = sum(ns.values())
    grand = sum(rates[g] * ns[g] for g in gids) / (ntot or 1.0)
    # moment-match Beta(α,β): mean m=grand; var v across groups (weighted)
    if len(gids) > 1:
        v = sum(ns[g] * (rates[g] - grand) ** 2 for g in gids) / (ntot or 1.0)
        m = grand
        v = max(1e-9, min(v, m * (1 - m) - 1e-9)) if 0 < m < 1 else 1e-6
        strength = max(1.0, m * (1 - m) / v - 1.0) if 0 < m < 1 else 50.0
    else:
        strength = 50.0
    alpha, beta = grand * strength, (1 - grand) * strength
    groups = {}
    for g in gids:
        k = float(group_counts[g]["k"])
        n = ns[g]
        post = (k + alpha) / (n + alpha + beta)
        B = n / (n + alpha + beta)
        groups[g] = {"raw": rates[g], "n": n, "shrunk": post, "shrinkage": round(1 - B, 4),
                     "se": math.sqrt(post * (1 - post) / (n + alpha + beta)), "escaped_prior": B > 0.5}
    return PooledEstimate(grand_mean=grand, between_var=1.0 / (strength + 1), within_var=grand * (1 - grand),
                          groups=groups, n_groups=len(gids), method="empirical_bayes_beta_binomial")
