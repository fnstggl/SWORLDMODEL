"""Reference World E — heterogeneous audience response (Upworthy Research Archive, CC-BY).

The portfolio's POPULATION-HETEROGENEITY test, on randomized ground truth: each Upworthy test showed
2-4 headline variants to randomized traffic; the CTR winner is causal. Task: rank the variants (pick the
winner) on held-out tests.

Structure through the universal machinery:
  * every headline is INTERPRETED once by the universal actor-cognition boundary (an audience member
    reading an incoming item: benefit_of_action, needs_clarification≈curiosity gap, urgency, relevance,
    risk, intent, …) → typed dims, no LLM-minted CTR
  * a calibration layer FITTED ON TRAIN maps dims + surface features → click propensity
  * the POPULATION world: audience particles with heterogeneous preference weights over the dims
    (Dirichlet-ish, prior-backed); a headline's simulated CTR = mean particle click propensity; the
    winner = argmax. Ablation no_population → rank by the point fitted scalar (mean-audience).
"""
from __future__ import annotations

import math
import random

SURFACE = ("len_words", "has_number", "has_question", "has_quote", "has_you", "all_caps_word")


def surface_features(h: str) -> list:
    words = h.split()
    return [min(1.0, len(words) / 20.0),
            1.0 if any(c.isdigit() for c in h) else 0.0,
            1.0 if "?" in h else 0.0,
            1.0 if ('"' in h or "'" in h) else 0.0,
            1.0 if any(w.lower() in ("you", "your", "you're") for w in words) else 0.0,
            1.0 if any(w.isupper() and len(w) > 2 for w in words) else 0.0]


def fit_ctr_layer(samples, *, l2=0.02, iters=500, lr=0.2):
    """Linear-in-features CTR model fitted on TRAIN variants: target = within-test CTR z-score (removes
    test-level traffic effects; ranking is within-test). Returns predict(features)->score."""
    if not samples:
        return (lambda x: 0.0), {"w": [], "n": 0}
    k = len(samples[0][0])
    w = [0.0] * k
    n = len(samples)
    for _ in range(iters):
        g = [0.0] * k
        for x, z in samples:
            e = sum(wi * xi for wi, xi in zip(w, x)) - z
            for i in range(k):
                g[i] += e * x[i]
        w = [wi - lr * (gi / n + l2 * wi) for wi, gi in zip(w, g)]
    return (lambda x: sum(wi * xi for wi, xi in zip(w, x))), {"w": [round(v, 4) for v in w], "n": n}


def zscores(vals):
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 or 1.0
    return [(v - mu) / sd for v in vals]


def population_rank(variants_feats, w_fitted, *, n_particles=200, seed=0, heterogeneity=True,
                    dim_sd=0.8):
    """The population world: each audience particle perturbs the FITTED dim weights multiplicatively
    (lognormal, sd=dim_sd — a labeled broad prior for taste heterogeneity) and "clicks" its argmax
    variant (a choosing audience member, not an averaging one); a variant's score = share of particles
    it wins. heterogeneity=False → exactly the point fitted scalar (the ablation nests cleanly).
    Returns scores per variant (higher = predicted winner)."""
    if not heterogeneity:
        return [sum(wi * xi for wi, xi in zip(w_fitted, f)) for f in variants_feats]
    rng = random.Random(seed)
    k = len(variants_feats[0])
    scores = [0.0] * len(variants_feats)
    for _ in range(n_particles):
        m = [math.exp(rng.gauss(0.0, dim_sd)) for _ in range(k)]
        appeals = [sum(wi * mi * xi for wi, mi, xi in zip(w_fitted, m, f)) for f in variants_feats]
        best = max(range(len(appeals)), key=lambda i: appeals[i])
        scores[best] += 1.0
    return [s / n_particles for s in scores]
