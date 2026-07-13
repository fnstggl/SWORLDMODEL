"""Phase 3 accuracy — offline arm computation shared by the fitter and the evaluator.

Given a captured row (tags, prior, p_phase2, p_phase3, causal_proposal, causal_claim_map, structural) and a set
of frozen params, computes every forecasting arm deterministically (no network):

  prior_only        the reference/lean prior mean
  phase2            the Phase-2 evidence terminal (posterior ignored)
  phase3_raw        the raw generic outcome-rate posterior consumed
  phase3_repaired   the Phase-3B calibrated-rate + convex-blend + gate repair
  fitted_generic    the FITTED hierarchical observation model's generic rate (Part 2)
  causal            the scenario-specific causal-latent rate (Part 3), using fitted or hand-set LRs
  causal_struct     causal latents combined with the structural-carrying p_phase3
  selector          picks among {phase2, repaired, causal} by pre-outcome support features (Part 4)
"""
from __future__ import annotations
import math

from swm.world_model_v2.phase3b_repair import repaired_from_capture_row, logit, sigmoid, _clip
from swm.world_model_v2 import phase3_fitted_obs as fo
from swm.world_model_v2 import phase3_causal_latents as cl


def tag_by_claim(row):
    return {t["claim_id"]: t for t in row.get("tags", [])}


def _blend(pa, pb, w):
    """convex logit blend: w*logit(pa) + (1-w)*logit(pb)."""
    return sigmoid(w * logit(pa) + (1 - w) * logit(pb))


def causal_rate(row, fitted_params=None):
    """Scenario-specific causal-latent rate, or None if no latents were proposed."""
    prop = row.get("causal_proposal") or {}
    if not prop.get("latents"):
        return None, {}
    lr_lookup = (lambda tag: fo.fitted_lr(tag, fitted_params)) if fitted_params else None
    rate, post = cl.causal_latent_rate(prop, row.get("causal_claim_map") or {}, tag_by_claim(row),
                                       lr_lookup=lr_lookup)
    return rate, post


def structural_entropy(row):
    """Entropy of the structural posterior (normalized 0..1); high => uncertain structure."""
    sp = row.get("structural_posterior") or {}
    vals = [v for v in sp.values() if isinstance(v, (int, float)) and v > 0]
    if len(vals) < 2:
        return 1.0
    s = sum(vals) or 1.0
    p = [v / s for v in vals]
    h = -sum(pi * math.log(pi) for pi in p)
    return h / math.log(len(p))


def all_arms(row, params):
    """Compute every arm for a row given frozen `params`. Returns {arm: prob-or-None}."""
    p2 = row.get("p_phase2")
    p3 = row.get("p_phase3")
    prior = (row.get("prior") or {}).get("mean")
    fitted = params.get("fitted_obs")
    repair = params.get("repair_params") or {}
    cal = params.get("causal") or {}

    arms = {"prior_only": prior, "phase2": p2, "phase3_raw": p3}

    # repaired (Phase-3B) — needs a row shaped for phase3b_repair
    try:
        rrow = {"qid": row["qid"], "question": row["question"], "as_of": row["as_of"], "domain": row["domain"],
                "outcome": row.get("outcome"), "outcome_lean": row.get("outcome_lean", "neutral"),
                "prior": row["prior"], "tags": row["tags"], "p_phase2": p2}
        arms["phase3_repaired"] = repaired_from_capture_row(rrow, repair)["repaired_p"] if p2 is not None else None
    except Exception:  # noqa: BLE001
        arms["phase3_repaired"] = p2

    # fitted generic
    arms["fitted_generic"] = fo.predict_rate(row, fitted) if fitted else None

    # causal (use fitted LR if the config selected it) + Platt recalibration (fixes the systematic
    # conjunction-mechanism bias) + optional safety blend toward Phase-2. All params frozen from training.
    cr, _ = causal_rate(row, fitted_params=fitted if cal.get("use_fitted_lr") else None)
    if cr is not None:
        platt = cal.get("platt")
        if platt:
            cr = sigmoid(platt["A"] + platt["B"] * logit(cr))
        if cal.get("blend_with_phase2_w") is not None and p2 is not None:
            cr = _blend(p2, cr, cal["blend_with_phase2_w"])
    arms["causal"] = cr

    # causal + structural (structural enters via p_phase3, which consumes the structural posterior)
    if cr is not None and p3 is not None:
        w = cal.get("struct_blend_w", 0.5)
        arms["causal_struct"] = _blend(cr, p3, w)
    else:
        arms["causal_struct"] = cr if cr is not None else p3

    # ensemble of the two evidence-informed arms with a demonstrated edge
    if arms.get("fitted_generic") is not None and p2 is not None:
        arms["ensemble"] = _blend(p2, arms["fitted_generic"], 0.5)
    else:
        arms["ensemble"] = p2

    # selector — pre-outcome features only, safe Phase-2 default
    arms["selector"] = _select(row, arms, params)
    return arms


def _select(row, arms, params):
    """Pick an arm by a FROZEN named policy using ONLY pre-outcome support features. Safe default: Phase-2."""
    sel = params.get("selector") or {}
    policy = sel.get("policy", "phase2")
    p2 = arms.get("phase2")
    if p2 is None:
        return arms.get("phase3_raw")
    n_eff = row.get("n_effective_observations") or 0
    n_lat = len((row.get("causal_proposal") or {}).get("latents", []))
    thr = sel.get("min_effective", 0)
    if policy == "phase2":
        return p2
    if policy == "fitted_gated":
        return arms.get("fitted_generic") if (n_eff >= thr and arms.get("fitted_generic") is not None) else p2
    if policy == "ensemble_gated":
        return arms.get("ensemble") if (n_eff >= thr and arms.get("ensemble") is not None) else p2
    if policy == "repaired":
        return arms.get("phase3_repaired") if arms.get("phase3_repaired") is not None else p2
    if policy == "causal_gated":
        if (n_lat >= sel.get("min_latents", 99) and n_eff >= thr and arms.get("causal") is not None):
            return arms["causal"]
        return p2
    return p2
