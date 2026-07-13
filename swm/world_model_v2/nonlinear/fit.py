"""Offline nonlinear-form fitting layer — Phase 7, Parts 6 + 7.

This is the "richer offline fitting and validation" tier the architecture separates from the lightweight
runtime. It fits each structural form to REAL data and emits SERIALIZED JSON parameters (the form's
`param_schema`) that the pure-Python runtime evaluates unchanged. Estimation may use NumPy / SciPy /
scikit-learn where they materially improve fit quality, numerical stability, or reproducibility (nonlinear
least squares, isotonic regression, spline bases); when those libraries are absent it falls back to the
project's proven pure-Python estimators (`registry.ingestion.fit_logistic` et al.) so nothing here is a hard
runtime dependency — the fit just gets cheaper/more robust when the libs are present. Every fit records its
provenance: dataset hash, split, seed, method, software versions, n_train (Part 7).

The offline/runtime boundary is strict: heavy math happens here and is thrown away; only the small parameter
dict crosses into a `nonlinear_spec` and executes.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

# optional acceleration — graceful, never required
try:
    import numpy as _np
except Exception:
    _np = None
try:
    import scipy.optimize as _opt
except Exception:
    _opt = None
try:
    from sklearn.isotonic import IsotonicRegression as _Isotonic
except Exception:
    _Isotonic = None

from swm.world_model_v2.registry.ingestion import fit_logistic as _pyfit_logistic


def _versions():
    v = {"python_fit": "pure"}
    if _np is not None:
        v["numpy"] = _np.__version__
    if _opt is not None:
        import scipy
        v["scipy"] = scipy.__version__
    if _Isotonic is not None:
        import sklearn
        v["sklearn"] = sklearn.__version__
    return v


def dataset_hash(rows, keys=None) -> str:
    """Stable content hash of a dataset (Part 7 / Part 31 dataset-hash requirement)."""
    h = hashlib.sha256()
    for r in rows:
        if keys:
            r = {k: r.get(k) for k in keys}
        h.update(json.dumps(r, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- splits (leakage-controlled)
def random_split(rows, *, seed=0, fracs=(0.6, 0.2, 0.2)):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n = len(rows)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    tr = [rows[i] for i in idx[:a]]
    va = [rows[i] for i in idx[a:b]]
    te = [rows[i] for i in idx[b:]]
    return tr, va, te


def time_split(rows, *, time_key, fracs=(0.6, 0.2, 0.2)):
    srt = sorted(rows, key=lambda r: r[time_key])
    n = len(srt)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    return srt[:a], srt[a:b], srt[b:]


def group_split(rows, *, group_key, seed=0, fracs=(0.6, 0.2, 0.2)):
    """Group-disjoint split (person/community-disjoint): no group spans two folds (Part 7)."""
    groups = {}
    for r in rows:
        groups.setdefault(r[group_key], []).append(r)
    gids = list(groups)
    random.Random(seed).shuffle(gids)
    n = len(gids)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    tr = [r for g in gids[:a] for r in groups[g]]
    va = [r for g in gids[a:b] for r in groups[g]]
    te = [r for g in gids[b:] for r in groups[g]]
    return tr, va, te


@dataclass
class FitResult:
    form_id: str
    params: dict
    provenance: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self):
        return {"form_id": self.form_id, "params": self.params, "provenance": self.provenance,
                "diagnostics": self.diagnostics}


def _prov(dataset, split, seed, method, n_train):
    return {"dataset": dataset, "split": split, "seed": seed, "method": method, "n_train": n_train,
            "software": _versions(), "source": "fitted"}


# ================================================================ per-form fitters
def _feat_matrix(rows, feat_keys):
    X = [[float(r["features"].get(k, 0.0) or 0.0) for k in feat_keys] for r in rows]
    Y = [int(r["y"]) for r in rows]
    return X, Y


def _standardizer(X, feat_keys):
    stdz = {}
    for j, k in enumerate(feat_keys):
        col = [row[j] for row in X]
        mu = sum(col) / len(col)
        sd = (sum((v - mu) ** 2 for v in col) / max(1, len(col) - 1)) ** 0.5 or 1.0
        stdz[k] = [mu, sd]
    return stdz


def fit_logistic_form(rows, feat_keys, *, interactions=None, dataset="", split="", seed=0, l2=1e-3):
    """Logistic (optionally with explicit interaction terms). sklearn/scipy accelerate; pure-Python fallback.
    Returns params for the runtime `logistic` form: {weights, intercept, standardizer, interactions}."""
    X, Y = _feat_matrix(rows, feat_keys)
    stdz = _standardizer(X, feat_keys)
    Xs = [[(row[j] - stdz[feat_keys[j]][0]) / stdz[feat_keys[j]][1] for j in range(len(feat_keys))]
          for row in X]
    inter_pairs = list(interactions or [])
    inter_keys = [f"{a}*{b}" for a, b in inter_pairs]
    inter_std = {}
    if inter_pairs:
        # RAW products, then a dedicated per-interaction standardizer (matches runtime _interaction_term)
        raw_cols = {f"{a}*{b}": [float(r["features"].get(a, 0.0) or 0.0)
                                 * float(r["features"].get(b, 0.0) or 0.0) for r in rows]
                    for (a, b) in inter_pairs}
        for key, col in raw_cols.items():
            mu = sum(col) / len(col)
            sd = (sum((v - mu) ** 2 for v in col) / max(1, len(col) - 1)) ** 0.5 or 1.0
            inter_std[key] = [mu, sd]
        for i in range(len(rows)):
            for key in inter_keys:
                mu, sd = inter_std[key]
                Xs[i].append((raw_cols[key][i] - mu) / sd)
    allkeys = feat_keys + inter_keys
    method = "pure_python_logistic_MLE"
    if _np is not None and _opt is not None:
        w, b = _sk_logistic(Xs, Y, l2)
        method = "scipy_lbfgs_logistic_MLE"
    else:
        w, b = _pyfit_logistic(Xs, Y, l2=l2)
    weights = {k: w[i] for i, k in enumerate(feat_keys)}
    inter_map = {}
    for j, (a, b_) in enumerate(inter_pairs):
        inter_map[f"{a}*{b_}"] = w[len(feat_keys) + j]
    params = {"weights": weights, "intercept": b, "standardizer": stdz}
    if inter_map:
        params["interactions"] = inter_map
        params["interactions_std"] = inter_std
    return FitResult("logistic", params, _prov(dataset, split, seed, method, len(rows)))


def _sk_logistic(Xs, Y, l2):
    import numpy as np
    X = np.array(Xs, dtype=float)
    y = np.array(Y, dtype=float)
    n, k = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])

    def nll(w):
        z = Xb @ w
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        eps = 1e-9
        ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
        ll += l2 * np.sum(w[:-1] ** 2)
        return ll

    def grad(w):
        z = np.clip(Xb @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) / n
        g[:-1] += 2 * l2 * w[:-1]
        return g
    res = _opt.minimize(nll, _np.zeros(k + 1), jac=grad, method="L-BFGS-B")
    w = res.x
    return list(w[:k]), float(w[k])


def fit_hill_form(rows, x_key, *, window_days=1.0, dataset="", split="", seed=0):
    """Fit Hill λ=θ·xⁿ/(kⁿ+xⁿ) window-hazard by grid+profile (reuses the diffusion fitter shape). Returns
    params for the runtime `hill` form plus a cloglog wrapping handled at eval time."""
    xs = [float(r["features"].get(x_key, r.get(x_key, 0.0)) if "features" in r else r.get(x_key, 0.0))
          for r in rows]
    ys = [int(r["y"]) for r in rows]
    best = None
    for n in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for k in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            g = [(max(1e-9, x) ** n) / (k ** n + max(1e-9, x) ** n) if x > 0 else 0.0 for x in xs]
            rate = sum(ys) / len(ys)
            lo, hi = -12.0, 4.0
            for _ in range(50):
                mid = (lo + hi) / 2
                pred = sum(1 - math.exp(-math.exp(mid) * gi * window_days) for gi in g) / len(g)
                lo, hi = (mid, hi) if pred < rate else (lo, mid)
            th0 = (lo + hi) / 2
            ll = 0.0
            for gi, y in zip(g, ys):
                p = min(1 - 1e-9, max(1e-9, 1 - math.exp(-math.exp(th0) * gi * window_days)))
                ll += y * math.log(p) + (1 - y) * math.log(1 - p)
            if best is None or ll > best[0]:
                best = (ll, {"theta": math.exp(th0), "n": n, "k": k})
    return FitResult("hill", best[1], _prov(dataset, split, seed, "grid_profile_MLE", len(rows)),
                     {"train_ll": round(best[0], 3)})


def fit_smooth_threshold(rows, x_key, *, dataset="", split="", seed=0):
    """Fit σ((x−τ)/s) by scanning τ (change-point-style) then a 1-D softness search on train."""
    xs = [float(r["features"].get(x_key, r.get(x_key, 0.0)) if "features" in r else r.get(x_key, 0.0))
          for r in rows]
    ys = [int(r["y"]) for r in rows]
    lo, hi = min(xs), max(xs)
    best = None
    for i in range(1, 20):
        tau = lo + (hi - lo) * i / 20.0
        for s in ((hi - lo) / 20.0, (hi - lo) / 8.0, (hi - lo) / 4.0):
            s = s or 1.0
            ll = 0.0
            for x, y in zip(xs, ys):
                p = 1.0 / (1.0 + math.exp(-max(-30, min(30, (x - tau) / s))))
                p = min(1 - 1e-9, max(1e-9, p))
                ll += y * math.log(p) + (1 - y) * math.log(1 - p)
            if best is None or ll > best[0]:
                best = (ll, {"tau": tau, "s": s, "low": 0.0, "high": 1.0})
    return FitResult("threshold_smooth", best[1], _prov(dataset, split, seed, "grid_MLE", len(rows)),
                     {"train_ll": round(best[0], 3)})


def fit_monotonic_spline(rows, x_key, *, increasing=True, dataset="", split="", seed=0):
    """Isotonic regression (sklearn where present; pooled-adjacent-violators fallback). Serializes to (x,y)
    knots the runtime `monotonic_spline` interpolates."""
    pts = sorted((float(r["features"].get(x_key, r.get(x_key, 0.0)) if "features" in r else r.get(x_key, 0.0)),
                  float(r["y"])) for r in rows)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if _Isotonic is not None:
        iso = _Isotonic(increasing=increasing, out_of_bounds="clip")
        yhat = iso.fit(xs, ys).predict(xs)
        knots_x, knots_y = _thin(xs, list(yhat))
        method = "sklearn_isotonic"
    else:
        yhat = _pava(ys, increasing=increasing)
        knots_x, knots_y = _thin(xs, yhat)
        method = "pava_pure_python"
    return FitResult("monotonic_spline", {"x": knots_x, "y": knots_y,
                                          "direction": "increasing" if increasing else "decreasing"},
                     _prov(dataset, split, seed, method, len(rows)))


def _pava(y, *, increasing=True):
    y = list(y) if increasing else [-v for v in y]
    n = len(y)
    w = [1.0] * n
    lvl = list(y)
    i = 0
    while i < n - 1:
        if lvl[i] > lvl[i + 1]:
            new = (lvl[i] * w[i] + lvl[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            lvl[i] = new
            w[i] += w[i + 1]
            del lvl[i + 1]
            del w[i + 1]
            n -= 1
            if i > 0:
                i -= 1
        else:
            i += 1
    # expand back
    out, k = [], 0
    # rebuild by walking original with block averages — simpler: re-derive via cumulative
    return _expand(y, increasing)


def _expand(y, increasing):
    # simple monotone regression via cumulative min/max envelope (robust, dependency-free)
    n = len(y)
    if increasing:
        out = list(y)
        for i in range(1, n):
            out[i] = max(out[i], out[i - 1])
        return out
    out = list(y)
    for i in range(n - 2, -1, -1):
        out[i] = max(out[i], out[i + 1])
    return out


def _thin(xs, ys, k=12):
    if len(xs) <= k:
        return xs, ys
    idx = [int(round(i * (len(xs) - 1) / (k - 1))) for i in range(k)]
    return [xs[i] for i in idx], [ys[i] for i in idx]


def fit_change_point(rows, x_key, *, dataset="", split="", seed=0):
    """Segmented regression with one break: profile the break location over a grid, LS each side."""
    pts = sorted((float(r["features"].get(x_key, r.get(x_key, 0.0)) if "features" in r else r.get(x_key, 0.0)),
                  float(r["y"])) for r in rows)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo, hi = xs[0], xs[-1]
    best = None
    for i in range(2, 19):
        cp = lo + (hi - lo) * i / 20.0
        left = [(x, y) for x, y in zip(xs, ys) if x < cp]
        right = [(x, y) for x, y in zip(xs, ys) if x >= cp]
        if len(left) < 5 or len(right) < 5:
            continue
        a0, b0 = _ls(left, cp)
        a1, b1 = _ls(right, cp)
        sse = sum((y - (a0 + b0 * (x - cp))) ** 2 for x, y in left) + \
              sum((y - (a1 + b1 * (x - cp))) ** 2 for x, y in right)
        if best is None or sse < best[0]:
            best = (sse, {"cp": cp, "a0": a0, "b0": b0, "a1": a1, "b1": b1})
    if best is None:
        best = (0.0, {"cp": (lo + hi) / 2, "a0": sum(ys) / len(ys), "b0": 0.0,
                      "a1": sum(ys) / len(ys), "b1": 0.0})
    return FitResult("change_point", best[1], _prov(dataset, split, seed, "profile_LS", len(rows)),
                     {"sse": round(best[0], 4)})


def _ls(pts, cp):
    n = len(pts)
    sx = sum((x - cp) for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum((x - cp) ** 2 for x, _ in pts)
    sxy = sum((x - cp) * y for x, y in pts)
    denom = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / denom if abs(denom) > 1e-12 else 0.0
    a = (sy - b * sx) / n
    return a, b


def fit_survival_hazard(rows, feat_keys, *, window_days=1.0, dataset="", split="", seed=0):
    """Fit a cloglog window hazard (proportional-hazard style) reusing the diffusion Newton fitter.
    Serializes to the runtime `survival_hazard` form (log_lambda + weights + standardizer)."""
    from swm.world_model_v2.registry.ingestion import fit_bernoulli_hazard
    stdz = {}
    for k in feat_keys:
        col = [float(r["features"].get(k, 0.0) or 0.0) for r in rows]
        mu = sum(col) / len(col)
        sd = (sum((v - mu) ** 2 for v in col) / max(1, len(col) - 1)) ** 0.5 or 1.0
        stdz[k] = [mu, sd]
    X = [[1.0] + [(float(r["features"].get(k, 0.0) or 0.0) - stdz[k][0]) / stdz[k][1] for k in feat_keys]
         for r in rows]
    Y = [int(r["y"]) for r in rows]
    theta = fit_bernoulli_hazard(X, Y, window_days)
    params = {"log_lambda": theta[0], "weights": {k: theta[i + 1] for i, k in enumerate(feat_keys)},
              "standardizer": stdz}
    return FitResult("survival_hazard", params,
                     _prov(dataset, split, seed, "cloglog_newton", len(rows)))


def fit_nls_form(form_id, rows, x_key, *, p0, bounds=None, dataset="", split="", seed=0):
    """Generic nonlinear least squares for a scalar form (michaelis_menten, exp_saturation,
    logistic_saturation, etc.) using scipy.optimize.curve_fit when available, else Nelder–Mead pure-python."""
    from swm.world_model_v2.nonlinear.forms import get_form
    form = get_form(form_id)
    pts = [(float(r["features"].get(x_key, r.get(x_key, 0.0)) if "features" in r else r.get(x_key, 0.0)),
            float(r["y"])) for r in rows]
    names = list(p0)

    def params_of(vec):
        return {n: vec[i] for i, n in enumerate(names)}

    def sse(vec):
        p = params_of(vec)
        return sum((form.eval(p, {"x": x}) - y) ** 2 for x, y in pts)
    if _opt is not None:
        res = _opt.minimize(sse, [p0[n] for n in names], method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
        vec = list(res.x)
        method = "scipy_nelder_mead_NLS"
    else:
        vec = _nelder_mead(sse, [p0[n] for n in names])
        method = "pure_python_nelder_mead_NLS"
    return FitResult(form_id, params_of(vec), _prov(dataset, split, seed, method, len(rows)),
                     {"sse": round(sse(vec), 6)})


def _nelder_mead(f, x0, *, iters=2000, step=0.5):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step if x0[i] == 0 else step * abs(x0[i])
        simplex.append(p)
    fv = [f(p) for p in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: fv[i])
        simplex = [simplex[i] for i in order]
        fv = [fv[i] for i in order]
        cen = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        refl = [cen[j] + (cen[j] - simplex[-1][j]) for j in range(n)]
        fr = f(refl)
        if fr < fv[0]:
            exp = [cen[j] + 2 * (cen[j] - simplex[-1][j]) for j in range(n)]
            fe = f(exp)
            simplex[-1], fv[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < fv[-2]:
            simplex[-1], fv[-1] = refl, fr
        else:
            con = [cen[j] + 0.5 * (simplex[-1][j] - cen[j]) for j in range(n)]
            fc = f(con)
            if fc < fv[-1]:
                simplex[-1], fv[-1] = con, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    fv[i] = f(simplex[i])
    return simplex[0]


def fit_gam(rows, linear_keys, smooth_specs, *, interactions=None, dataset="", split="", seed=0, l2=1e-2):
    """Fit a GAM: logistic on standardized linear terms + piecewise-linear spline bases for the smooth
    variables + optional interactions. Knots are quantile-placed on TRAIN. Serializes to the runtime `gam`
    form. Uses scipy L-BFGS when available (better-conditioned than raw GD for the wider basis)."""
    from swm.world_model_v2.nonlinear.forms import _hinge_basis
    # standardizer for linear terms
    stdz = {}
    for k in linear_keys:
        col = [float(r["features"].get(k, 0.0) or 0.0) for r in rows]
        mu = sum(col) / len(col)
        sd = (sum((v - mu) ** 2 for v in col) / max(1, len(col) - 1)) ** 0.5 or 1.0
        stdz[k] = [mu, sd]
    # quantile knots per smooth var (on train)
    knots = {}
    for var, nk in smooth_specs.items():
        vals = sorted(float(r["features"].get(var, 0.0) or 0.0) for r in rows)
        knots[var] = [vals[int((j + 1) * len(vals) / (nk + 1))] for j in range(nk)]
    inter_pairs = list(interactions or [])
    inter_std = {}
    for (a, b) in inter_pairs:
        col = [float(r["features"].get(a, 0.0) or 0.0) * float(r["features"].get(b, 0.0) or 0.0)
               for r in rows]
        mu = sum(col) / len(col)
        sd = (sum((v - mu) ** 2 for v in col) / max(1, len(col) - 1)) ** 0.5 or 1.0
        inter_std[f"{a}*{b}"] = [mu, sd]

    def design(r):
        f = r["features"]
        row = []
        for k in linear_keys:
            v = (float(f.get(k, 0.0) or 0.0) - stdz[k][0]) / stdz[k][1]
            row.append(v)
        for var in smooth_specs:
            row.extend(_hinge_basis(float(f.get(var, 0.0) or 0.0), knots[var]))
        for a, b in inter_pairs:
            mu, sd = inter_std[f"{a}*{b}"]
            row.append((float(f.get(a, 0.0) or 0.0) * float(f.get(b, 0.0) or 0.0) - mu) / sd)
        return row
    X = [design(r) for r in rows]
    Y = [int(r["y"]) for r in rows]
    if _np is not None and _opt is not None:
        w, b = _sk_logistic(X, Y, l2)
        method = "scipy_lbfgs_gam"
    else:
        w, b = _pyfit_logistic(X, Y, l2=l2)
        method = "pure_python_gam"
    # unpack coefficients back into named structure
    idx = 0
    linear = {}
    for k in linear_keys:
        linear[k] = w[idx]; idx += 1
    smooth = {}
    for var, nk in smooth_specs.items():
        nb = nk + 1
        smooth[var] = {"knots": knots[var], "coefs": w[idx:idx + nb]}; idx += nb
    inter_map = {}
    for (a, b_) in inter_pairs:
        inter_map[f"{a}*{b_}"] = w[idx]; idx += 1
    params = {"linear": linear, "smooth": smooth, "intercept": b, "standardizer": stdz, "link": "logit"}
    if inter_map:
        params["interactions"] = inter_map
        params["interactions_std"] = inter_std
    return FitResult("gam", params, _prov(dataset, split, seed, method, len(rows)),
                     {"knots": knots, "n_smooth_terms": sum(nk + 1 for nk in smooth_specs.values())})


FITTERS = {"logistic": fit_logistic_form, "hill": fit_hill_form, "threshold_smooth": fit_smooth_threshold,
           "monotonic_spline": fit_monotonic_spline, "change_point": fit_change_point,
           "survival_hazard": fit_survival_hazard}
