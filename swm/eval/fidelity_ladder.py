"""Fidelity ladder — the no-cheat test of the thesis "more pressuring variables → better prediction".

Add variables one at a time and score held-out accuracy at each rung, under three weighting schemes:

  - NAIVE       — an (almost) unregularized logistic: every added variable gets a free point weight. This is
                  the scheme that made "more variables hurt" — noise compounds, correlated features fight.
  - CALIBRATED  — L2-to-prior shrinkage (BayesianLogistic): a useless/noisy variable auto-shrinks to ~zero
                  weight (harmless); a useful one earns its weight. The claim: under this scheme, adding
                  variables does NOT hurt.
  - UNCERTAINTY — CALIBRATED plus integrating over the Laplace posterior on the weights, so an unknown weight
                  widens the prediction instead of biasing it. The claim: this improves CALIBRATION (ECE),
                  especially data-poor.

The ladder settles the disagreement empirically: if NAIVE degrades as variables are added while CALIBRATED
holds/improves, then the binding constraint was never variable COUNT — it was weight calibration + shrinkage
+ uncertainty, and "model every pressuring variable" is right *provided each carries a calibrated weight*.
"""
from __future__ import annotations

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
from swm.variables.bayes_logistic import BayesianLogistic, variance_contribution


def _acc(y, p):
    return sum(1 for yi, pi in zip(y, p) if (pi >= 0.5) == (yi == 1)) / len(y) if y else float("nan")


def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4), "acc": round(_acc(y, p), 4)}


ARMS = {
    "naive": {"l2": 0.02, "integrate": False},          # ~unregularized: every variable gets a free weight
    "calibrated": {"l2": 1.0, "integrate": False},      # L2-to-prior shrinkage
    "uncertainty": {"l2": 1.0, "integrate": True},      # + integrate weight-posterior uncertainty
}


def _feats(specs_k, r):
    """Flatten a row's features across specs; each spec's fn may return a scalar or a list (a one-hot block
    for a categorical variable), so 'adding a variable' can add a whole block of columns."""
    out = []
    for _, fn in specs_k:
        v = fn(r)
        if isinstance(v, (list, tuple)):
            out.extend(float(x) for x in v)
        else:
            out.append(float(v))
    return out


def _cols(specs_k, sample_row):
    names = []
    for nm, fn in specs_k:
        v = fn(sample_row)
        if isinstance(v, (list, tuple)):
            names.extend(f"{nm}[{i}]" for i in range(len(v)))
        else:
            names.append(nm)
    return names


def run_ladder(train, test, feature_specs, *, prior_w0=None, seed=0, arms=ARMS, extras=True):
    """`train`/`test`: rows with a `y` in {0,1}. `feature_specs`: ordered list of (name, fn(row)->float|list);
    the ladder adds them left-to-right (a categorical variable's fn returns a one-hot block, added as a unit).
    `prior_w0`: optional per-column prior weight mean (the elasticity prior). Returns {arm:[rung dicts]} + a
    final weight report / variance triage at full fidelity."""
    ytr = [int(r["y"]) for r in train]
    yte = [int(r["y"]) for r in test]
    curves = {a: [] for a in arms}
    final_extras = {}
    for k in range(1, len(feature_specs) + 1):
        specs_k = feature_specs[:k]
        Xtr = [_feats(specs_k, r) for r in train]
        Xte = [_feats(specs_k, r) for r in test]
        w0 = (list(prior_w0[:len(Xtr[0])]) if prior_w0 is not None else None)
        for aname, cfg in arms.items():
            m = BayesianLogistic(l2=cfg["l2"], w0=w0).fit(Xtr, ytr)
            if cfg["integrate"]:
                preds = [m.predict_dist(x, n_samples=120, seed=seed)["p"] for x in Xte]
            else:
                preds = [m.predict_proba(x) for x in Xte]
            rung = {"k": k, "n_cols": len(Xtr[0])}
            rung.update(_score(yte, preds))
            curves[aname].append(rung)
            if extras and k == len(feature_specs) and aname == "calibrated":
                cols = _cols(specs_k, train[0])
                final_extras["weight_report"] = m.weight_report(cols)
                tri = variance_contribution(Xtr, m.w)[:10]
                final_extras["variance_triage"] = [{"name": cols[j], "share": round(s, 4)} for j, s in tri]
    return {"curves": curves, "final": final_extras, "verdict": _verdict(curves)}


def average_ladders(results):
    """Average several ladder `curves` (e.g. one per question) rung-by-rung, per arm — a robust curve."""
    if not results:
        return {}
    arms = results[0]["curves"].keys()
    out = {}
    for a in arms:
        maxk = min(len(r["curves"][a]) for r in results)
        rungs = []
        for k in range(maxk):
            ll = sum(r["curves"][a][k]["log_loss"] for r in results) / len(results)
            ece = sum(r["curves"][a][k]["ece"] for r in results) / len(results)
            acc = sum(r["curves"][a][k]["acc"] for r in results) / len(results)
            rungs.append({"k": k + 1, "log_loss": round(ll, 4), "ece": round(ece, 4), "acc": round(acc, 4)})
        out[a] = rungs
    return {"curves": out, "verdict": _verdict(out)}


def _verdict(curves) -> dict:
    """Did adding variables hurt (log-loss rose from the best rung to full fidelity) per arm?"""
    out = {}
    for a, curve in curves.items():
        if not curve:
            continue
        best = min(c["log_loss"] for c in curve)
        full = curve[-1]["log_loss"]
        one = curve[0]["log_loss"]
        out[a] = {"one_var": one, "full": full, "best": best,
                  "full_minus_best": round(full - best, 4),
                  "adding_vars_helped": full <= one + 1e-9,
                  "adding_vars_hurt_vs_best": round(full - best, 4) > 0.01}
    return out
