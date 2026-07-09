"""Grounded inference — turn an LLM *guess* into an as-close-to-*measured* estimate (the three pillars).

The lesson from EXP-085: a MEASURED variable (a senator's real ideology) beat a GUESSED one, because the
measurement was a low-bias, calibrated, high-resolution estimate of the latent state. Measurement is just
inference with perfect evidence. So the goal is to make an inference low-bias AND honestly-uncertain, in any
situation — even with no ground-truth data on THIS entity. Three pillars, composed as a Bayesian estimator
with a MEASURED prior and a CALIBRATED likelihood:

  Pillar 1 — EVIDENCE.  Never infer cold. Condition on a dossier (history, public footprint, retrieved
             context). Handled upstream (the caller assembles the dossier); this module consumes it.
  Pillar 2 — REFERENCE-CLASS ANCHORING.  The number starts at the MEASURED base rate of the tightest
             reference class (party, cohort, ...), not the LLM's vibe. `reference_prior`.
  Pillar 3 — CALIBRATION + SHRINKAGE + ENSEMBLE.  `ensemble_infer` (spread = free uncertainty),
             `fit_calibration`/`apply_calibration` (correct the LLM's systematic bias, learned where truth
             exists), `shrink` (empirical-Bayes pull toward the base rate unless evidence earns the
             deviation — the antidote to over-individuation).

Everything here is pure-python and deterministic given the chat function (injected seam), so it is unit-
testable and the LLM backend is swappable.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass
class Estimate:
    value: float
    sd: float                       # calibrated 1-sigma uncertainty (honest error bar)
    provenance: str = ""


# ---- Pillar 3: ensemble / self-consistency -------------------------------------------------------------
def ensemble_infer(chat_fn, prompt, parse, k=3):
    """Sample the LLM `k` times and parse each to a number. Returns (mean, spread, samples) where spread is
    a FREE uncertainty estimate — agreement => tight, disagreement => wide. None if nothing parsed."""
    vals = []
    for _ in range(k):
        try:
            v = parse(chat_fn(prompt))
            if v is not None:
                vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return None
    return fmean(vals), (pstdev(vals) if len(vals) > 1 else None), vals


# ---- Pillar 2: reference-class prior -------------------------------------------------------------------
def reference_prior(base_rates, class_key, fallback):
    """The MEASURED base rate (mean, sd) for an entity's reference class — the outside view. `base_rates`
    maps class_key -> (mean, sd); `fallback` (mean, sd) is used for an unknown class."""
    return base_rates.get(class_key, fallback)


# ---- Pillar 3: calibration learned where truth exists --------------------------------------------------
def fit_calibration(raw, truth):
    """Least-squares map raw_estimate -> truth, learned on entities where truth IS known (e.g. senators with
    real ideology). Corrects the LLM's systematic bias/scale. Returns (a, b) for a*raw + b, plus the RMSE of
    the residual (a measured reliability => the calibrated estimate's error bar)."""
    n = len(raw)
    if n < 2:
        return (1.0, 0.0, None)
    mx, my = fmean(raw), fmean(truth)
    sxx = sum((x - mx) ** 2 for x in raw)
    sxy = sum((x - mx) * (y - my) for x, y in zip(raw, truth))
    a = sxy / sxx if sxx > 1e-12 else 0.0
    b = my - a * mx
    resid = [y - (a * x + b) for x, y in zip(raw, truth)]
    rmse = (sum(r * r for r in resid) / n) ** 0.5
    return (a, b, rmse)


def apply_calibration(cal, raw):
    a, b = cal[0], cal[1]
    return a * raw + b


# ---- Pillar 3: shrink to the base rate (empirical Bayes) ----------------------------------------------
def shrink(estimate, est_sd, prior_mean, prior_sd):
    """Precision-weighted posterior of the LLM estimate and the measured base rate. An UNCERTAIN estimate
    (large est_sd) is pulled hard toward the base rate (the outside view); a CONFIDENT, well-evidenced one
    keeps its deviation. Returns (posterior_mean, posterior_sd)."""
    if est_sd is None or est_sd <= 1e-9:
        est_sd = prior_sd                      # no spread info -> treat as no better than the class
    if prior_sd <= 1e-9:
        return estimate, est_sd
    we, wp = 1.0 / est_sd ** 2, 1.0 / prior_sd ** 2
    post = (we * estimate + wp * prior_mean) / (we + wp)
    return post, (1.0 / (we + wp)) ** 0.5


def grounded_estimate(*, llm_mean, llm_spread, cal, class_prior, evidence_strength=1.0):
    """Compose the pillars into one estimate: calibrate the LLM's number (Pillar 3), then shrink it toward
    the measured reference-class base rate (Pillar 2) by its calibrated uncertainty (Pillar 3). Higher
    `evidence_strength` (Pillar 1 dossier richness) narrows the estimate's error bar, so specific evidence
    earns a larger deviation from the base rate."""
    prior_mean, prior_sd = class_prior
    if llm_mean is None:
        return Estimate(prior_mean, prior_sd, "base_rate_only")
    val = apply_calibration(cal, llm_mean) if cal else llm_mean
    # calibrated reliability (rmse) is the floor of the error bar; ensemble spread and evidence tighten it
    base_sd = cal[2] if (cal and cal[2]) else (llm_spread or prior_sd)
    est_sd = base_sd / max(1e-6, evidence_strength ** 0.5)
    if llm_spread is not None:
        est_sd = (est_sd ** 2 + llm_spread ** 2) ** 0.5
    post, post_sd = shrink(val, est_sd, prior_mean, prior_sd)
    return Estimate(post, post_sd, "grounded(pillars 2+3)")
