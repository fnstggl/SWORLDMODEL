"""Mechanism ingestion pipeline — Phase 6 tooling.

register_published_mechanism() → encode transition → attach studies/datasets → add packs →
fit_pack_from_data() → compare_functional_forms() → posterior_predictive_check() → held-out validation →
transfer test → promote/quarantine/reject. Every step writes records; failures are preserved.

The fitting utilities here are the shared pure-Python estimation core the family modules use:
  * fit_bernoulli_hazard: cloglog/log-linear hazard GLM by Newton-damped gradient (survival mechanism fits)
  * fit_logistic: plain logistic GLM
  * profile_frailty: 1-D profile likelihood for lognormal frailty sd (Gauss-Hermite marginalization)
  * compare_forms: train-fit candidates, score on validation split, report ALL (no silent winner-only)
"""
from __future__ import annotations

import math
import random

from swm.world_model_v2.registry.record import (Citation, MechanismRecord, ParameterPack,
                                                RegistryError, ValidationRecord, now_iso)

# ------------------------------------------------------------------ estimation core (pure python)
def _dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


def _clip(z, lo=-30.0, hi=30.0):
    return max(lo, min(hi, z))


def fit_logistic(X, Y, *, iters=500, lr=0.3, l2=1e-3):
    k = len(X[0])
    n = len(X)
    base = sum(Y) / max(1, n)
    w = [0.0] * k
    b = math.log(max(1e-6, base) / max(1e-6, 1 - base))
    for _ in range(iters):
        gw, gb = [0.0] * k, 0.0
        for x, y in zip(X, Y):
            q = 1 / (1 + math.exp(-_clip(_dot(w, x) + b)))
            e = q - y
            for i in range(k):
                gw[i] += e * x[i]
            gb += e
        w = [wi - lr * (gi / n + l2 * wi) for wi, gi in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def fit_bernoulli_hazard(X, Y, W_days, *, iters=800, lr=0.25, l2=1e-3):
    """Window-outcome hazard GLM (cloglog link): λ_i = exp(θ·x_i), P_i = 1 − exp(−λ_i·W).
    Maximizes Bernoulli likelihood by gradient ascent (damped). Returns θ (x must include an intercept
    term as x[0]=1). This is the survival-mechanism fit: the SAME λ integrates step-by-step in rollout."""
    k = len(X[0])
    n = len(X)
    base = max(1e-6, sum(Y) / max(1, n))
    th = [0.0] * k
    th[0] = math.log(-math.log(max(1e-9, 1 - base)) / W_days)   # intercept init from base rate
    for it in range(iters):
        g = [0.0] * k
        for x, y in zip(X, Y):
            lam = math.exp(_clip(th[0] * x[0] + _dot(th[1:], x[1:])))
            H = lam * W_days
            p = 1 - math.exp(-min(50.0, H))
            p = min(1 - 1e-9, max(1e-9, p))
            # dL/dθ_j = x_j · H · (y/p − (1−y)/(1−p)) · exp(−H)   [d p/dθ_j = x_j·H·exp(−H)]
            common = H * math.exp(-min(50.0, H)) * (y / p - (1 - y) / (1 - p))
            for j in range(k):
                g[j] += common * x[j]
        step = lr / n
        th = [t + step * gj - step * l2 * t for t, gj in zip(th, g)]
    return th


def hazard_lambda(theta, x):
    return math.exp(_clip(_dot(theta, x)))


_GH_NODES = [(-2.3506049736745, 0.019111580500770), (-1.3358490740137, 0.13383774880098),
             (-0.4360774119276, 0.44648878212421), (0.4360774119276, 0.44648878212421),
             (1.3358490740137, 0.13383774880098), (2.3506049736745, 0.019111580500770)]


def marginal_window_p(lam, W_days, sigma):
    """E_ε[1 − exp(−ε·λ·W)] with ε ~ LN(−σ²/2, σ) (mean-1 frailty), 6-node Gauss-Hermite."""
    if sigma <= 1e-9:
        return 1 - math.exp(-min(50.0, lam * W_days))
    tot = 0.0
    for z, w in _GH_NODES:
        eps = math.exp(sigma * math.sqrt(2.0) * z - sigma * sigma / 2.0)
        tot += w * (1 - math.exp(-min(50.0, eps * lam * W_days)))
    return tot / math.sqrt(math.pi)


def profile_frailty(theta, X, Y, W_days, *, grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)):
    """Profile log-likelihood over lognormal frailty sd — susceptibility heterogeneity, FITTED not assumed.
    Returns (best_sigma, {sigma: ll})."""
    lls = {}
    for s in grid:
        ll = 0.0
        for x, y in zip(X, Y):
            p = marginal_window_p(hazard_lambda(theta, x), W_days, s)
            p = min(1 - 1e-9, max(1e-9, p))
            ll += y * math.log(p) + (1 - y) * math.log(1 - p)
        lls[s] = ll
    best = max(lls, key=lls.get)
    return best, {str(k): round(v, 2) for k, v in lls.items()}


def brier(preds, ys):
    return sum((p - y) ** 2 for p, y in zip(preds, ys)) / max(1, len(ys))


def compare_forms(forms: dict, val_rows, y_key="y") -> dict:
    """forms: {name: predict_fn(row)->p}. Scores EVERY candidate on the validation split; reports all
    (a silent winner-only report hides the comparison the registry is supposed to preserve)."""
    ys = [r[y_key] for r in val_rows]
    out = {}
    for name, fn in forms.items():
        preds = [fn(r) for r in val_rows]
        out[name] = {"brier": round(brier(preds, ys), 6),
                     "pred_rate": round(sum(preds) / max(1, len(preds)), 5)}
    ranked = sorted(out, key=lambda n: out[n]["brier"])
    return {"scores": out, "ranked": ranked, "winner": ranked[0]}


# ------------------------------------------------------------------ the pipeline verbs
def register_published_mechanism(store, rec: MechanismRecord, citations: list) -> MechanismRecord:
    """Step 1-3: register a family from literature. Requires ≥1 citation with explicit limits; enters at
    status=proposed (or implemented if its code_ref already resolves and it has a test_ref)."""
    if not citations:
        raise RegistryError("a published mechanism needs at least one citation (with limits)")
    for c in citations:
        if not isinstance(c, Citation) or not c.limits.strip():
            raise RegistryError("every citation must state its transport limits explicitly")
    rec.citations = list(citations)
    rec.status = "proposed"
    store.register(rec)
    if rec.executable() and rec.test_ref:
        store.set_status(rec.family_id, "implemented", reason="executable transition + tests present")
    return rec


def fit_pack_from_data(store, family_id: str, *, pack: ParameterPack,
                       validation: ValidationRecord | None = None) -> ParameterPack:
    """Step 5-6: attach a pack fitted from a local dataset (fit itself happens in the family module —
    this records it, enforces uncertainty labeling, and optionally attaches the fit's validation)."""
    p = store.add_pack(family_id, pack)
    if validation is not None:
        store.add_validation(family_id, validation, pack_id=pack.pack_id)
    return p


def record_failure(store, family_id: str, vr: ValidationRecord, *, quarantine: bool = False,
                   reason: str = ""):
    """Step 11-12: failures are recorded, never deleted; optionally quarantine the family."""
    vr.passed = False
    store.add_validation(family_id, vr)
    if quarantine:
        store.set_status(family_id, "quarantined", reason=reason or f"failed {vr.kind} on {vr.dataset}")
    return vr


def promote(store, family_id: str, target: str, *, reason: str):
    """Step 12: lifecycle promotion with enforced gates (see MechanismRecord.promotion_blockers)."""
    return store.set_status(family_id, target, reason=reason)


# ------------------------------------------------------------------ misc shared utilities
def paired_bootstrap_delta(ys, pa, pb, *, n_boot=1000, seed=5):
    """Paired Brier delta (arm A − arm B) with bootstrap CI95 — negative = A better."""
    d = [(a - y) ** 2 - (b - y) ** 2 for a, b, y in zip(pa, pb, ys)]
    rng = random.Random(seed)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": round(sum(d) / n, 6), "ci95": [round(bs[int(0.025 * n_boot)], 6),
                                                   round(bs[int(0.975 * n_boot) - 1], 6)], "n": n}
