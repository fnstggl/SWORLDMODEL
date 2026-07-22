"""Phase 11 — predictive diagnostics that drive trigger detection.

The signals here are all derived from the CURRENT plan's posterior predictive distribution over an observable
and the realised observation — the same principle as ``posterior.posterior_predictive_check`` (low predictive
density ⇒ the model family, not just the weights, may be wrong). We compute, per observation:

  * ``residual``  = −log predictive density of the observed value (surprise, in nats)
  * ``tail_prob`` = predictive probability of an outcome at least as extreme (calibration p-value)
  * ``impossible`` = the observation is outside the current outcome/support (density ~ 0)

and, over a sequence: sustained-failure runs, ESS collapse, and PIT-based calibration error. These are the
raw evidence for the trigger detectors; NONE of them fires a recompile on its own (fusion + persistence do).
"""
from __future__ import annotations

import math

_EPS = 1e-9


def _mean_sd(xs):
    n = len(xs) or 1
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(max(0.0, v))


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def surprise(predictive, observed, *, support=None, impossible_z: float = 6.0,
             impossible_density: float = 1e-4) -> dict:
    """Surprise of ``observed`` under ``predictive``.

    ``predictive`` is EITHER a discrete distribution ``{value: prob}`` OR a list of predictive samples
    (continuous). ``support`` optionally bounds the continuous outcome space ``(lo, hi)``; an observed value
    outside it is *impossible* under the current plan. Returns residual/tail_prob/density/impossible/z.
    """
    # ---- discrete predictive distribution ----
    if isinstance(predictive, dict):
        z = sum(predictive.values()) or 1.0
        p = predictive.get(observed)
        if p is None:
            # try string/ীnormalised key match
            p = predictive.get(str(observed))
        prob = (p / z) if p is not None else 0.0
        in_support = p is not None
        dens = prob
        residual = -math.log(max(_EPS, prob))
        # tail prob for an ordered discrete space: mass on outcomes no more likely than the observed
        tail = sum(q / z for q in predictive.values() if (q / z) <= prob + _EPS) if in_support else 0.0
        return {"residual": round(residual, 4), "tail_prob": round(tail, 5),
                "predictive_density": round(dens, 5), "impossible": (not in_support) or prob < _EPS,
                "z": None, "kind": "discrete", "support_size": len(predictive)}
    # ---- continuous predictive samples ----
    xs = [float(x) for x in (predictive or []) if x is not None]
    if not xs:
        return {"residual": 0.0, "tail_prob": 1.0, "predictive_density": 1.0, "impossible": False,
                "z": None, "kind": "empty"}
    m, sd = _mean_sd(xs)
    sd = max(sd, 1e-6)
    obs = float(observed)
    z = (obs - m) / sd
    dens = math.exp(-0.5 * z * z) / (sd * math.sqrt(2 * math.pi))
    residual = -math.log(max(_EPS, dens))
    tail = 2.0 * min(_norm_cdf(z), 1.0 - _norm_cdf(z))
    impossible = abs(z) >= impossible_z or dens < impossible_density
    if support is not None:
        lo, hi = support
        if obs < lo - _EPS or obs > hi + _EPS:
            impossible = True
    return {"residual": round(residual, 4), "tail_prob": round(max(0.0, tail), 5),
            "predictive_density": round(dens, 6), "impossible": bool(impossible),
            "z": round(z, 3), "kind": "continuous", "pred_mean": round(m, 4), "pred_sd": round(sd, 4)}


def sustained_failure(residual_history, *, threshold: float, min_run: int = 3) -> dict:
    """A single high residual is noise; a RUN of them is structural. Returns the current trailing run of
    residuals above ``threshold`` and whether it meets ``min_run`` (the persistence requirement)."""
    run = 0
    for r in reversed(residual_history):
        if r >= threshold:
            run += 1
        else:
            break
    mean_r = sum(residual_history) / len(residual_history) if residual_history else 0.0
    return {"run_length": run, "sustained": run >= min_run, "mean_residual": round(mean_r, 4),
            "n": len(residual_history), "threshold": threshold}


def ess_diagnostic(weights, *, collapse_frac: float = 0.1) -> dict:
    """Effective sample size of a weight vector. Collapse (ess_frac < collapse_frac) is a NUMERICAL failure
    (all mass on a few particles) distinct from evidential concentration — a particle-collapse trigger."""
    n = len(weights) or 1
    z = sum(weights) or 1.0
    ws = [w / z for w in weights]
    ess = 1.0 / max(_EPS, sum(w * w for w in ws))
    frac = ess / n
    return {"ess": round(ess, 3), "n": n, "ess_frac": round(frac, 4), "collapsed": frac < collapse_frac}


def calibration_error(pit_values) -> dict:
    """Probability-integral-transform calibration: PITs of a well-calibrated model are ~Uniform(0,1).
    Returns a simple ECE (mean abs deviation of the empirical CDF from the diagonal) + KS statistic."""
    xs = sorted(float(p) for p in pit_values if p is not None)
    n = len(xs)
    if n == 0:
        return {"ece": None, "ks": None, "n": 0}
    ks = max(max(abs((i + 1) / n - x), abs(i / n - x)) for i, x in enumerate(xs))
    # 10-bin ECE
    bins = [0] * 10
    for x in xs:
        bins[min(9, int(x * 10))] += 1
    ece = sum(abs(c / n - 0.1) for c in bins) / 10.0
    return {"ece": round(ece, 4), "ks": round(ks, 4), "n": n}


def regime_shift(pre_residuals, post_residuals, *, min_each: int = 3) -> dict:
    """Mechanism-regime change: the SAME observable now follows a different transition. Detected as a
    significant jump in mean residual between a pre-window and a post-window (Welch-style standardized gap)."""
    if len(pre_residuals) < min_each or len(post_residuals) < min_each:
        return {"shift": False, "reason": "insufficient_window", "gap": 0.0}
    m0, s0 = _mean_sd(pre_residuals)
    m1, s1 = _mean_sd(post_residuals)
    se = math.sqrt(s0 * s0 / len(pre_residuals) + s1 * s1 / len(post_residuals)) or 1e-6
    gap = (m1 - m0) / se
    return {"shift": gap >= 2.0, "gap": round(gap, 3), "pre_mean": round(m0, 3), "post_mean": round(m1, 3)}


def parameter_drift(pre_values, post_values, *, min_each: int = 3) -> dict:
    """Structure valid but a PARAMETER moved: the observable's LEVEL shifts (mean move) while remaining inside
    the model's support (i.e. residuals do NOT explode). A standardized mean shift that is material but not a
    support violation → refit, not restructure."""
    if len(pre_values) < min_each or len(post_values) < min_each:
        return {"drift": False, "reason": "insufficient_window", "shift": 0.0}
    m0, s0 = _mean_sd(pre_values)
    m1, s1 = _mean_sd(post_values)
    pooled = math.sqrt((s0 * s0 + s1 * s1) / 2.0) or 1e-6
    shift = abs(m1 - m0) / pooled                     # standardized effect size (Cohen's d style)
    return {"drift": shift >= 0.8, "shift": round(shift, 3), "pre_mean": round(m0, 4),
            "post_mean": round(m1, 4)}
