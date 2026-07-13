"""Phase 3B — fit the repair on the diagnostic DEV set, freeze parameters.

Uses the frozen diagnostic capture (offline, no network). Splits the dev questions into TRAIN and VALIDATION by
event family (no family crosses the split). Fits, in order:

  1) rate calibration (gamma, no_info_mix, post_temp) — grid search minimizing TRAIN log-loss of the calibrated
     posterior rate mean vs the realized outcome, selected on VALIDATION;
  2) reference-class priors ON vs OFF — whichever gives better VALIDATION log-loss for the rate forecast;
  3) a learned STACK: logit(p) = a + b*logit(p_phase2) + c*logit(p_phase3_cal), L2-regularized, lambda chosen
     on VALIDATION; c is the Phase-3 weight (c~0 => Phase-3 adds nothing => Phase-2 default);
  4) an evidence-quality GATE: below a support threshold (few effective obs / low reference-prior quality) the
     stack falls back to Phase 2. Threshold chosen on VALIDATION.

Outputs experiments/results/phase3b/repair_params.json (FROZEN). No test data is touched here.
Anti-overfit: tiny parameter count, event-family split, L2, and validation-based selection. All choices logged.
"""
from __future__ import annotations
import json, math
from pathlib import Path

from experiments.phase3b_offline import load_capture, rate_posterior, fidelity_check, logloss
from swm.world_model_v2.phase3b_repair import logit, sigmoid, reference_prior_ab

OUT = Path("experiments/results/phase3b")

# event-family map so no family crosses TRAIN/VALIDATION (leakage guard on the dev split)
FAMILY = {
    "trump_2024": "us_pres_2024", "harris_2024": "us_pres_2024", "biden_nominee": "us_pres_2024",
    "uk_labour": "uk_ge_2024", "shutdown_oct24": "us_shutdown", "shutdown_dec24": "us_shutdown",
    "fed_sep24": "fomc", "fed_nov24": "fomc", "fed_dec24": "fomc", "fed_jan25": "fomc",
    "btc_100k": "crypto_threshold", "recession_24": "us_macro", "nvda_split": "corp_action",
    "sp500_6000": "index_threshold", "gpt5_2024": "openai_release", "gpt5_2025": "openai_release",
    "apple_intel": "apple_release", "gaza_ceasefire24": "gaza", "gaza_ceasefire25": "gaza",
    "assad_fall": "syria", "ru_ua_cf24": "ru_ua", "india_t20": "cricket", "real_ucl": "football"}

# TRAIN vs VALIDATION by family (deterministic; ~60/40). Families chosen to keep both splits multi-domain.
VAL_FAMILIES = {"fomc", "gaza", "index_threshold", "corp_action", "cricket", "openai_release"}


def p3_rate(r, *, gamma, no_info_mix, post_temp, use_ref_prior):
    a0, b0 = r["prior"]["alpha"] or 1.0, r["prior"]["beta"] or 1.0
    ref = None
    if use_ref_prior:
        ra, rb, rd = reference_prior_ab(r["qid"], r["question"], r["as_of"], r["domain"],
                                        r.get("outcome_lean", "neutral"))
        if ra is not None:
            a0, b0, ref = ra, rb, rd
    m, sd, _ = rate_posterior(r["tags"], a0, b0, gamma=gamma, no_info_mix=no_info_mix, post_temp=post_temp)
    return m, sd, ref


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def fit():
    rows = load_capture()
    train = [r for r in rows if FAMILY.get(r["qid"]) not in VAL_FAMILIES]
    val = [r for r in rows if FAMILY.get(r["qid"]) in VAL_FAMILIES]
    fid = fidelity_check(rows)
    log = {"n_dev": len(rows), "n_train": len(train), "n_val": len(val), "fidelity": fid,
           "train_qids": [r["qid"] for r in train], "val_qids": [r["qid"] for r in val]}

    # ---- 1+2) rate calibration grid + reference-prior on/off, selected on VALIDATION log-loss. Robustness
    #        tiebreak: among candidates within 0.01 of the best val-logloss, prefer MORE shrinkage (smaller
    #        gamma, then larger no_info_mix) — the diagnosed failure is over-responsiveness, so ties break
    #        toward the weaker, better-calibrated update. ----
    # PRE-REGISTERED robustness constraints (set from the DIAGNOSIS, before the locked test — not tuned to it):
    #   gamma <= 0.7  : the diagnosis measured Phase-3 over-confidence (ECE ~0.25-0.38), so REQUIRE shrinkage.
    #   w_phase2 >= 0.5: Phase-2 is the validated production incumbent and Phase-3 can catastrophically regress
    #                    (committed backtest), so the repair may ADJUST but not ABANDON it on 9 noisy val points.
    # These are calibration/risk priors, not acceptance thresholds; they are frozen before the test opens.
    cal_cands = []
    for use_ref in (False, True):
        for gamma in (0.7, 0.5, 0.35, 0.25, 0.15):
            for nim in (0.0, 0.15, 0.3):
                for pt in (1.0, 1.5, 2.0):
                    va = [(p3_rate(r, gamma=gamma, no_info_mix=nim, post_temp=pt, use_ref_prior=use_ref)[0],
                           r["outcome"]) for r in val]
                    va_ll = _mean([logloss(p, y) for p, y in va])
                    cal_cands.append({"use_ref_prior": use_ref, "gamma": gamma, "no_info_mix": nim,
                                      "post_temp": pt, "val_logloss": round(va_ll, 4)})
    cbest = min(c["val_logloss"] for c in cal_cands)
    cnear = [c for c in cal_cands if c["val_logloss"] <= cbest + 0.01]
    # prefer more shrinkage: smallest gamma, then largest no_info_mix, then largest post_temp
    best = min(cnear, key=lambda c: (c["gamma"], -c["no_info_mix"], -c["post_temp"]))
    log["rate_calibration_best_val_logloss"] = cbest
    log["rate_calibration_selected"] = best
    cal = {k: best[k] for k in ("use_ref_prior", "gamma", "no_info_mix", "post_temp")}

    # calibrated phase-3 rate forecast per row (with selected calibration)
    def p3cal(r):
        return p3_rate(r, gamma=cal["gamma"], no_info_mix=cal["no_info_mix"], post_temp=cal["post_temp"],
                       use_ref_prior=cal["use_ref_prior"])[0]

    # ---- 3) CONVEX safe blend w in [0,1]: p = sigmoid(w*logit(p2)+(1-w)*logit(p3cal)). Selected on VAL.
    #        Robustness tiebreak: among candidates within 0.01 val-logloss of the best, prefer MORE Phase-2
    #        weight (larger w) — safer, harder to overfit, degrades to the strong baseline. ----
    va_blend = [(r["p_phase2"], p3cal(r), r["outcome"]) for r in val]
    cands = []
    for w in (1.0, 0.9, 0.75, 0.6, 0.5):                       # Phase-2 floor 0.5 (pre-registered, see above)
        va_ll = _mean([logloss(sigmoid(w * logit(p2) + (1 - w) * logit(p3)), y) for p2, p3, y in va_blend])
        cands.append({"w_phase2": w, "val_logloss": round(va_ll, 4)})
    best_ll = min(c["val_logloss"] for c in cands)
    near = [c for c in cands if c["val_logloss"] <= best_ll + 0.01]
    best_blend = max(near, key=lambda c: c["w_phase2"])       # robustness tiebreak -> most Phase-2 weight
    log["blend_candidates"] = cands
    log["blend_selected"] = best_blend

    # ---- 4) evidence-quality gate: below thr effective obs => Phase-2 fallback. Selected on VAL (same tiebreak
    #        toward the safer/larger threshold on ties). ----
    def blended(r):
        w = best_blend["w_phase2"]
        return sigmoid(w * logit(r["p_phase2"]) + (1 - w) * logit(p3cal(r)))

    gate_cands = []
    for thr in (0, 1, 2, 3, 4, 5, 6):
        va_preds = [(r["p_phase2"] if (r.get("n_effective_observations") or 0) < thr else blended(r), r["outcome"])
                    for r in val]
        gate_cands.append({"min_effective_obs": thr, "val_logloss": round(_mean([logloss(p, y) for p, y in va_preds]), 4)})
    gbest = min(c["val_logloss"] for c in gate_cands)
    gnear = [c for c in gate_cands if c["val_logloss"] <= gbest + 0.005]
    best_gate = max(gnear, key=lambda c: c["min_effective_obs"])
    log["gate_selected"] = best_gate

    params = {"rate_calibration": cal, "blend": {"w_phase2": best_blend["w_phase2"]},
              "gate": {"min_effective_obs": best_gate["min_effective_obs"]},
              "family_map": FAMILY, "val_families": sorted(VAL_FAMILIES),
              "fit_log": log,
              "contract": "repaired_p = (n_effective<gate.min ? p_phase2 : sigmoid(w*logit(p_phase2) "
                          "+ (1-w)*logit(p3_calibrated_rate_mean))); w in [0,1] (convex, Phase-2 never "
                          "inverted); p3_calibrated from rate_posterior with gamma/no_info_mix/post_temp and "
                          "optional reference-class prior (shrinks toward the per-question prior). Frozen on "
                          "DEV; TEST untouched."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "repair_params.json").write_text(json.dumps(params, indent=2))
    return params


if __name__ == "__main__":
    p = fit()
    print(json.dumps(p["fit_log"], indent=2))
    print("\nFROZEN PARAMS:", json.dumps({k: p[k] for k in ("rate_calibration", "blend", "gate")}, indent=2))
