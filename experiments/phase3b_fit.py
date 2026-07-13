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

    # ---- 1+2) rate calibration grid + reference-prior on/off, selected on VALIDATION log-loss ----
    best = None
    for use_ref in (False, True):
        for gamma in (1.0, 0.7, 0.5, 0.35, 0.25, 0.15):
            for nim in (0.0, 0.15, 0.3):
                for pt in (1.0, 1.5, 2.0):
                    tr = [(p3_rate(r, gamma=gamma, no_info_mix=nim, post_temp=pt, use_ref_prior=use_ref)[0],
                           r["outcome"]) for r in train]
                    va = [(p3_rate(r, gamma=gamma, no_info_mix=nim, post_temp=pt, use_ref_prior=use_ref)[0],
                           r["outcome"]) for r in val]
                    tr_ll = _mean([logloss(p, y) for p, y in tr])
                    va_ll = _mean([logloss(p, y) for p, y in va])
                    cand = {"use_ref_prior": use_ref, "gamma": gamma, "no_info_mix": nim, "post_temp": pt,
                            "train_logloss": round(tr_ll, 4), "val_logloss": round(va_ll, 4)}
                    if best is None or va_ll < best["val_logloss"] - 1e-9:
                        best = cand
    log["rate_calibration_selected"] = best
    cal = {k: best[k] for k in ("use_ref_prior", "gamma", "no_info_mix", "post_temp")}

    # calibrated phase-3 rate forecast per row (with selected calibration)
    def p3cal(r):
        return p3_rate(r, gamma=cal["gamma"], no_info_mix=cal["no_info_mix"], post_temp=cal["post_temp"],
                       use_ref_prior=cal["use_ref_prior"])[0]

    # ---- 3) learned stack logit(p)=a+b*logit(p2)+c*logit(p3cal), L2, lambda on VALIDATION ----
    def fit_stack(data, lam, iters=4000, lr=0.15):
        a, b, c = 0.0, 1.0, 0.0
        n = len(data)
        for _ in range(iters):
            ga = gb = gc = 0.0
            for p2, p3, y in data:
                x2, x3 = logit(p2), logit(p3)
                pred = sigmoid(a + b * x2 + c * x3)
                e = pred - y
                ga += e; gb += e * x2; gc += e * x3
            a -= lr * ga / n
            b -= lr * (gb / n + lam * b)
            c -= lr * (gc / n + lam * c)
        return a, b, c

    tr_stack = [(r["p_phase2"], p3cal(r), r["outcome"]) for r in train]
    va_stack = [(r["p_phase2"], p3cal(r), r["outcome"]) for r in val]
    best_stack = None
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
        a, b, c = fit_stack(tr_stack, lam)
        va_ll = _mean([logloss(sigmoid(a + b * logit(p2) + c * logit(p3)), y) for p2, p3, y in va_stack])
        cand = {"lam": lam, "a": round(a, 4), "b": round(b, 4), "c": round(c, 4), "val_logloss": round(va_ll, 4)}
        if best_stack is None or va_ll < best_stack["val_logloss"] - 1e-9:
            best_stack = cand
    log["stack_selected"] = best_stack

    # ---- 4) evidence-quality gate: fall back to Phase 2 when support is thin. threshold on VALIDATION ----
    # support score = n_effective_observations (a simple, pre-outcome evidence-quantity feature)
    def stacked(r):
        a, b, c = best_stack["a"], best_stack["b"], best_stack["c"]
        return sigmoid(a + b * logit(r["p_phase2"]) + c * logit(p3cal(r)))

    best_gate = None
    for thr in (0, 1, 2, 3, 4, 5, 6):
        # below thr effective obs => Phase 2; else stacked
        va_preds = [(r["p_phase2"] if (r.get("n_effective_observations") or 0) < thr else stacked(r), r["outcome"])
                    for r in val]
        va_ll = _mean([logloss(p, y) for p, y in va_preds])
        cand = {"min_effective_obs": thr, "val_logloss": round(va_ll, 4)}
        if best_gate is None or va_ll < best_gate["val_logloss"] - 1e-9:
            best_gate = cand
    log["gate_selected"] = best_gate

    params = {"rate_calibration": cal, "stack": {k: best_stack[k] for k in ("a", "b", "c")},
              "gate": {"min_effective_obs": best_gate["min_effective_obs"]},
              "family_map": FAMILY, "val_families": sorted(VAL_FAMILIES),
              "fit_log": log,
              "contract": "repaired_p = (n_effective<gate.min ? p_phase2 : sigmoid(a + b*logit(p_phase2) "
                          "+ c*logit(p3_calibrated_rate_mean))); p3_calibrated from rate_posterior with "
                          "gamma/no_info_mix/post_temp and optional reference-class prior. Frozen on DEV; "
                          "TEST untouched."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "repair_params.json").write_text(json.dumps(params, indent=2))
    return params


if __name__ == "__main__":
    p = fit()
    print(json.dumps(p["fit_log"], indent=2))
    print("\nFROZEN PARAMS:", json.dumps({k: p[k] for k in ("rate_calibration", "stack", "gate")}, indent=2))
