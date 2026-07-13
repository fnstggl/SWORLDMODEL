"""Phase 3B — repaired posterior consumption (production module).

The diagnostic backtest showed the raw Phase-3 posterior OVERRIDES the Phase-2 terminal with a worse, over-
confident estimate. This module owns the repaired inference math, shared by the offline fitter and the serving
path so they are byte-identical:

  1) calibrated rate posterior  — the real DirectionalRateModel likelihoods, tempered by `gamma` (shrinkage),
     mixed with a flat no-information model (`no_info_mix`), optionally flattened by `post_temp`, on top of a
     reference-class prior when one applies. Fights the over-concentration that made Phase-3 over-confident.
  2) learned stack + gate       — the repaired forecast is a frozen logistic combination of the Phase-2
     terminal and the calibrated Phase-3 rate; below a support threshold it FALLS BACK to Phase-2. This lets
     the system conclude "this evidence does not justify moving the Phase-2 forecast."

All parameters are LOADED from a frozen JSON (`experiments/results/phase3b/repair_params.json`) fit on the
dev split; this module never fits. Identity defaults reproduce a pure Phase-2 fallback when no params exist.
"""
from __future__ import annotations
import json, math
from pathlib import Path

from swm.world_model_v2.phase3_latent_spec import ClaimTag
from swm.world_model_v2.phase3_observation import DirectionalRateModel, collapse_by_dependence
from swm.world_model_v2.phase3_priors import reference_class_prior
from swm.world_model_v2.phase3b_reference_priors import reference_data_for

_RATE_MODEL = DirectionalRateModel()
_GRID = [(i + 0.5) / 200.0 for i in range(200)]
_EPS = 1e-6
_PARAMS_PATH = Path("experiments/results/phase3b/repair_params.json")


def _clip(p):
    return min(1 - _EPS, max(_EPS, p))


def logit(p):
    p = _clip(p)
    return math.log(p / (1 - p))


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def _as_tags(tags_or_rows):
    out = []
    for t in tags_or_rows:
        if isinstance(t, ClaimTag):
            out.append(t); continue
        out.append(ClaimTag(
            claim_id=t["claim_id"], outcome_direction=t.get("outcome_direction", "neutral"),
            supports_hypotheses=list(t.get("supports_hypotheses", [])),
            opposes_hypotheses=list(t.get("opposes_hypotheses", [])),
            strength=t.get("strength", "moderate"), is_strategic=bool(t.get("is_strategic", False)),
            dependence_group=t.get("dependence_group", ""), reliability=float(t.get("reliability", 0.8))))
    return out


def calibrated_rate_posterior(tags_or_rows, alpha, beta, *, gamma=1.0, no_info_mix=0.0, post_temp=1.0,
                              use_dependence=True):
    """Deterministic grid posterior over the outcome rate with calibration. Returns (mean, sd, n_effective)."""
    a0 = float(alpha) if alpha else 1.0
    b0 = float(beta) if beta else 1.0
    tags = _as_tags(tags_or_rows)
    eff = collapse_by_dependence(tags) if use_dependence else tags
    logdens = [(a0 - 1) * math.log(r) + (b0 - 1) * math.log(1 - r) for r in _GRID]
    for t in eff:
        if t.outcome_direction == "neutral":
            continue
        for i, r in enumerate(_GRID):
            like = _RATE_MODEL.likelihood(t, r)
            if no_info_mix > 0:
                like = (1 - no_info_mix) * like + no_info_mix * 1.0
            logdens[i] += gamma * math.log(max(_EPS, like))
    if post_temp != 1.0:
        logdens = [x / post_temp for x in logdens]
    m = max(logdens)
    w = [math.exp(x - m) for x in logdens]
    z = sum(w) or 1.0
    w = [x / z for x in w]
    mean = sum(r * wi for r, wi in zip(_GRID, w))
    var = sum(wi * (r - mean) ** 2 for r, wi in zip(_GRID, w))
    n_eff = sum(1 for t in eff if t.outcome_direction != "neutral")
    return mean, math.sqrt(max(0.0, var)), n_eff


def reference_prior_ab(qid, question, as_of, domain, lean="neutral"):
    """(alpha, beta, reference_data) for a question's reference class, or (None, None, None)."""
    rd = reference_data_for(qid, question, as_of, domain)
    if not rd:
        return None, None, None
    spec = reference_class_prior(rd["reference_class"], rd["successes"], rd["total"],
                                 transport_risk=rd["transport_risk"], lean=lean)
    return spec.alpha, spec.beta, rd


def load_params(path=None):
    p = Path(path) if path else _PARAMS_PATH
    if p.exists():
        return json.loads(p.read_text())
    # identity / safe fallback: pure Phase-2 (no Phase-3 influence)
    return {"rate_calibration": {"use_ref_prior": False, "gamma": 1.0, "no_info_mix": 0.0, "post_temp": 1.0},
            "blend": {"w_phase2": 1.0}, "gate": {"min_effective_obs": 999}}


def combine(p_phase2, p3_cal, n_effective, params):
    """Repaired forecast: gate to Phase-2 when support is thin, else a CONVEX blend of the Phase-2 terminal
    and the calibrated Phase-3 rate (w in [0,1] so Phase-2 can never be inverted). The blend is in logit
    space; the Phase-3 over-responsiveness is already tamed upstream by the likelihood shrinkage (gamma) that
    pulls p3_cal toward the per-question prior. No global shrink-toward-0.5 (that would leak the base rate)."""
    gate = (params.get("gate") or {}).get("min_effective_obs", 999)
    if (n_effective or 0) < gate:
        return _clip(p_phase2), "gate_phase2_fallback"
    w = float((params.get("blend") or {}).get("w_phase2", 1.0))
    w = min(1.0, max(0.0, w))
    p = sigmoid(w * logit(p_phase2) + (1 - w) * logit(p3_cal))
    return _clip(p), ("phase2_only" if w >= 0.999 else "blended")


def repaired_from_capture_row(r, params):
    """Compute the repaired forecast for a captured/decomposed row (offline path used by fit + eval)."""
    cal = params["rate_calibration"]
    a0, b0 = r["prior"]["alpha"] or 1.0, r["prior"]["beta"] or 1.0
    ref = None
    if cal.get("use_ref_prior"):
        ra, rb, rd = reference_prior_ab(r["qid"], r["question"], r["as_of"], r["domain"],
                                        r.get("outcome_lean", "neutral"))
        if ra is not None:
            a0, b0, ref = ra, rb, rd
    p3_cal, sd, n_eff = calibrated_rate_posterior(
        r["tags"], a0, b0, gamma=cal["gamma"], no_info_mix=cal["no_info_mix"], post_temp=cal["post_temp"])
    p, mode = combine(r["p_phase2"], p3_cal, n_eff, params)
    return {"repaired_p": p, "p3_calibrated": round(p3_cal, 4), "p3_cal_sd": round(sd, 4),
            "n_effective": n_eff, "mode": mode, "reference_class": (ref or {}).get("reference_class")}
