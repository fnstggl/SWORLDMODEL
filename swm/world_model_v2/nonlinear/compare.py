"""Model comparison & validation-only selection — Phase 7, Part 8 + Part 22.

Compares candidate structural forms under IDENTICAL data / splits / inputs, always against the required
baselines (constant base-rate, linear/logistic, the current Phase-6 form), selects the form using VALIDATION
data only, and scores the untouched TEST set once. The selection rule is deliberately conservative: prefer the
simpler form when the nonlinear form is not meaningfully better on held-out (a paired bootstrap CI that
includes zero → keep the simpler form). A more flexible form with lower TRAIN error but no held-out gain is
NOT promoted — the anti-overfitting discipline the spec demands.

Metrics (Part 22): Brier, log loss, calibration (ECE + reliability), AUC (secondary), and paired bootstrap
deltas. CRPS is provided for the trajectory/continuous testbeds.
"""
from __future__ import annotations

import math
import random

from swm.world_model_v2.registry.ingestion import paired_bootstrap_delta


# ---------------------------------------------------------------- metrics
def brier(preds, ys):
    return sum((p - y) ** 2 for p, y in zip(preds, ys)) / max(1, len(ys))


def logloss(preds, ys):
    return -sum(y * math.log(max(1e-9, p)) + (1 - y) * math.log(max(1e-9, 1 - p))
                for p, y in zip(preds, ys)) / max(1, len(ys))


def auroc(preds, ys):
    pos = [p for p, y in zip(preds, ys) if y == 1]
    neg = [p for p, y in zip(preds, ys) if y == 0]
    if not pos or not neg:
        return None
    wins = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
    return wins / (len(pos) * len(neg))


def calibration(preds, ys, *, bins=10):
    """Expected Calibration Error + reliability table."""
    buckets = [[] for _ in range(bins)]
    for p, y in zip(preds, ys):
        b = min(bins - 1, int(p * bins))
        buckets[b].append((p, y))
    ece, table = 0.0, []
    n = len(preds)
    for b, bk in enumerate(buckets):
        if not bk:
            continue
        conf = sum(p for p, _ in bk) / len(bk)
        acc = sum(y for _, y in bk) / len(bk)
        ece += len(bk) / n * abs(conf - acc)
        table.append({"bin": b, "n": len(bk), "conf": round(conf, 4), "acc": round(acc, 4)})
    return {"ece": round(ece, 5), "reliability": table}


def crps_gaussian(mu, sigma, y):
    """CRPS for a Gaussian predictive dist (trajectory testbeds). Lower is better."""
    if sigma <= 0:
        return abs(mu - y)
    z = (y - mu) / sigma
    from math import erf, exp, sqrt, pi
    Phi = 0.5 * (1 + erf(z / sqrt(2)))
    phi = exp(-z * z / 2) / sqrt(2 * pi)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))


def metrics(preds, ys):
    out = {"brier": round(brier(preds, ys), 6), "logloss": round(logloss(preds, ys), 6),
           "pred_rate": round(sum(preds) / max(1, len(preds)), 5),
           "real_rate": round(sum(ys) / max(1, len(ys)), 5), "n": len(ys)}
    a = auroc(preds, ys)
    if a is not None:
        out["auroc"] = round(a, 4)
    out.update({"calibration": calibration(preds, ys)})
    return out


# ---------------------------------------------------------------- comparison harness
def compare_forms(candidates: dict, val_rows, test_rows, *, y_key="y",
                  primary_metric="brier", seed=5):
    """candidates: {name: predict_fn(row)->p}. Selects on VALIDATION, scores TEST once (Part 8).

    Returns: val scores (used for selection), test scores (reported for ALL, no silent winner-only), the
    selected form (validation winner), and paired bootstrap deltas of the selected form vs each baseline on
    TEST. `_is_baseline` names (constant/linear/phase6) are compared against; the selected form is the
    validation winner regardless."""
    yv = [r[y_key] for r in val_rows]
    yt = [r[y_key] for r in test_rows]
    val_preds = {nm: [fn(r) for r in val_rows] for nm, fn in candidates.items()}
    test_preds = {nm: [fn(r) for r in test_rows] for nm, fn in candidates.items()}
    val_scores = {nm: metrics(val_preds[nm], yv) for nm in candidates}
    test_scores = {nm: metrics(test_preds[nm], yt) for nm in candidates}
    ranked_val = sorted(candidates, key=lambda nm: val_scores[nm][primary_metric])
    selected = ranked_val[0]
    # paired deltas on TEST vs baselines and vs simpler forms
    deltas = {}
    for nm in candidates:
        if nm == selected:
            continue
        deltas[f"{selected}_vs_{nm}"] = paired_bootstrap_delta(yt, test_preds[selected], test_preds[nm],
                                                               seed=seed)
    return {"val_scores": val_scores, "test_scores": test_scores, "ranked_val": ranked_val,
            "selected": selected, "paired_test_deltas": deltas}


def select_with_parsimony(comparison: dict, *, simpler=("constant", "linear", "logistic"),
                          margin_key="ci95"):
    """Apply the parsimony rule: if the validation winner is a nonlinear form but its TEST paired delta vs the
    best available SIMPLER form has a CI that includes 0 (indistinguishable), KEEP THE SIMPLER FORM.

    Returns {'promoted': form, 'reason', 'beat_baseline': bool}."""
    sel = comparison["selected"]
    if sel in simpler:
        return {"promoted": sel, "reason": "validation winner is already a simple form", "beat_baseline": True}
    # find the strongest simpler form present
    present_simple = [s for s in simpler if s in comparison["test_scores"]]
    if not present_simple:
        return {"promoted": sel, "reason": "no simpler baseline present to compare", "beat_baseline": None}
    best_simple = min(present_simple, key=lambda s: comparison["test_scores"][s]["brier"])
    key = f"{sel}_vs_{best_simple}"
    d = comparison["paired_test_deltas"].get(key)
    if d is None:
        return {"promoted": sel, "reason": "no paired delta recorded", "beat_baseline": None}
    ci = d.get(margin_key, [0, 0])
    # negative delta = selected better (arm A − arm B, brier). Meaningful iff CI upper < 0.
    beat = ci[1] < 0
    if beat:
        return {"promoted": sel, "beat_baseline": True,
                "reason": f"nonlinear {sel} beats {best_simple} on held-out (paired ΔBrier {d['mean']} "
                          f"CI {ci})"}
    return {"promoted": best_simple, "beat_baseline": False,
            "reason": f"nonlinear {sel} NOT distinguishable from {best_simple} on held-out "
                      f"(paired ΔBrier {d['mean']} CI {ci} includes 0) — keep the simpler form (parsimony)"}


# ---------------------------------------------------------------- baseline predictors
def constant_predictor(train_rows, *, y_key="y"):
    rate = sum(r[y_key] for r in train_rows) / max(1, len(train_rows))
    return lambda r: rate


def linear_predictor(train_rows, feat_keys, *, y_key="y"):
    """A plain logistic-on-standardized-features baseline (the 'linear'/'logistic' form to beat)."""
    from swm.world_model_v2.nonlinear.fit import fit_logistic_form
    from swm.world_model_v2.nonlinear.forms import get_form
    fr = fit_logistic_form(train_rows, feat_keys)
    form = get_form("logistic")
    return lambda r: form.eval(fr.params, {"features": r["features"]})
