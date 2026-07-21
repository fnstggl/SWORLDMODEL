"""Phase 3 accuracy — fit ALL new components on the frozen DEV set, freeze params.

Uses the enriched 23-question diagnostic capture (dev artifact) split TRAIN/VALIDATION by event family. Fits,
in order, on TRAIN and selects on VALIDATION:

  1) fitted hierarchical observation models (phase3_fitted_obs) — compared vs hand-set/global-gamma/no-info;
  2) causal-latent config: fitted-vs-hand-set LR, Platt recalibration (fixes conjunction bias), and an
     optional safety blend toward Phase-2 and structural blend weight;
  3) a production selector over {phase2, repaired, causal} using pre-outcome support features only, with a
     safe Phase-2 default.

Writes experiments/results/phase3acc/accuracy_params.json (FROZEN). No locked-test data is touched.
"""
from __future__ import annotations
import json, math
from pathlib import Path

from swm.world_model_v2 import phase3_fitted_obs as fo
from swm.world_model_v2.phase3b_repair import load_params as load_repair, logit, sigmoid, _clip
from experiments.phase3acc_arms import all_arms, causal_rate

OUT = Path("experiments/results/phase3acc")
DEV = OUT / "dev_enriched.json"

# reuse the Phase-3B event-family map + val split (dev questions are the same 23)
from experiments.phase3b_fit import FAMILY, VAL_FAMILIES


def _ll(p, y):
    p = _clip(p); return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _mean(xs):
    return sum(xs) / len(xs) if xs else 9.9


def load_dev():
    d = json.loads(DEV.read_text())
    return [r for r in d["rows"] if r.get("status", "").startswith("completed") and r.get("outcome") in (0, 1)]


def _platt(pairs, iters=2000, lr=0.1):
    """Fit logit(p_cal)=A+B*logit(p_raw) by GD on (raw_prob, y). Returns {A,B}."""
    A, B = 0.0, 1.0
    n = max(1, len(pairs))
    for _ in range(iters):
        ga = gb = 0.0
        for praw, y in pairs:
            x = logit(praw); pred = sigmoid(A + B * x); e = pred - y
            ga += e; gb += e * x
        A -= lr * ga / n; B -= lr * gb / n
    return {"A": round(A, 4), "B": round(B, 4)}


def fit():
    rows = load_dev()
    train = [r for r in rows if FAMILY.get(r["qid"]) not in VAL_FAMILIES]
    val = [r for r in rows if FAMILY.get(r["qid"]) in VAL_FAMILIES]
    log = {"n_dev": len(rows), "n_train": len(train), "n_val": len(val)}

    # 1) fitted observation models on TRAIN
    fitted = fo.fit(train)
    # compare generic rates on VAL: fitted vs hand-set(phase3 raw) vs prior
    def score_pred(getp, data):
        return _mean([_ll(getp(r), r["outcome"]) for r in data if getp(r) is not None])
    log["fitted_vs_baselines_val_logloss"] = {
        "fitted_generic": round(score_pred(lambda r: fo.predict_rate(r, fitted), val), 4),
        "phase3_raw": round(score_pred(lambda r: r.get("p_phase3"), val), 4),
        "phase2": round(score_pred(lambda r: r.get("p_phase2"), val), 4),
        "prior_only": round(score_pred(lambda r: (r.get("prior") or {}).get("mean"), val), 4)}

    # 2) causal config: choose fitted-vs-handset LR by TRAIN raw causal log-loss, then Platt on TRAIN,
    #    then choose safety blend + struct blend on VAL.
    def raw_causal(r, use_fitted):
        cr, _ = causal_rate(r, fitted_params=fitted if use_fitted else None)
        return cr
    tr_c = {uf: [(raw_causal(r, uf), r["outcome"]) for r in train if raw_causal(r, uf) is not None]
            for uf in (False, True)}
    lr_choice = min((False, True), key=lambda uf: _mean([_ll(p, y) for p, y in tr_c[uf]]))
    platt = _platt(tr_c[lr_choice])
    log["causal_lr_choice_use_fitted"] = lr_choice
    log["causal_platt"] = platt

    # select safety blend w (toward Phase-2) and struct blend on VAL
    base_cal = {"use_fitted_lr": lr_choice, "platt": platt}
    best = None
    for wb in (None, 0.75, 0.5, 0.25):                        # None => pure causal
        for ws in (0.3, 0.5, 0.7):
            cal = dict(base_cal, blend_with_phase2_w=wb, struct_blend_w=ws)
            params = {"fitted_obs": fitted, "repair_params": load_repair(), "causal": cal,
                      "selector": {"min_latents": 99, "min_effective_for_causal": 99,
                                   "min_effective_for_repaired": 99}}
            vll = _mean([_ll(all_arms(r, params)["causal"], r["outcome"]) for r in val
                         if all_arms(r, params)["causal"] is not None])
            vll_s = _mean([_ll(all_arms(r, params)["causal_struct"], r["outcome"]) for r in val
                           if all_arms(r, params)["causal_struct"] is not None])
            cand = {"blend_with_phase2_w": wb, "struct_blend_w": ws, "causal_val_ll": round(vll, 4),
                    "causal_struct_val_ll": round(vll_s, 4)}
            key = min(vll, vll_s)
            if best is None or key < best["_key"] - 1e-9:
                best = dict(cand, _key=key)
    causal_cfg = dict(base_cal, blend_with_phase2_w=best["blend_with_phase2_w"],
                      struct_blend_w=best["struct_blend_w"])
    log["causal_blend_selected"] = {k: best[k] for k in ("blend_with_phase2_w", "struct_blend_w",
                                                          "causal_val_ll", "causal_struct_val_ll")}

    # 3) selector: pick the best FROZEN named policy on VAL (safe default Phase-2). Only pre-outcome features.
    repair = load_repair()
    sel_cands = [{"policy": "phase2"}, {"policy": "repaired"}]
    for thr in (0, 3, 5, 8):
        sel_cands.append({"policy": "fitted_gated", "min_effective": thr})
        sel_cands.append({"policy": "ensemble_gated", "min_effective": thr})
    for thr in (3, 5, 8):
        for ml in (2, 3):
            sel_cands.append({"policy": "causal_gated", "min_effective": thr, "min_latents": ml})
    best_sel = None
    for sc in sel_cands:
        params = {"fitted_obs": fitted, "repair_params": repair, "causal": causal_cfg, "selector": sc}
        vll = _mean([_ll(all_arms(r, params)["selector"], r["outcome"]) for r in val
                     if all_arms(r, params)["selector"] is not None])
        cand = dict(sc, val_logloss=round(vll, 4))
        if best_sel is None or vll < best_sel["val_logloss"] - 1e-9:
            best_sel = cand
    log["selector_candidates"] = sorted(
        [dict(sc, val_logloss=round(_mean([_ll(all_arms(r, {"fitted_obs": fitted, "repair_params": repair,
         "causal": causal_cfg, "selector": sc})["selector"], r["outcome"]) for r in val
         if all_arms(r, {"fitted_obs": fitted, "repair_params": repair, "causal": causal_cfg,
         "selector": sc})["selector"] is not None]), 4)) for sc in sel_cands],
        key=lambda c: c["val_logloss"])[:5]
    log["selector_selected"] = best_sel

    params = {"fitted_obs": fitted, "repair_params": repair, "causal": causal_cfg,
              "selector": {k: v for k, v in best_sel.items() if k != "val_logloss"},
              "fit_log": log,
              "contract": "All params fit on the 23-question DEV set (event-family train/val split), frozen "
                          "before the 93-question locked test. Causal rate is Platt-recalibrated (fixes the "
                          "conjunction-mechanism bias). Selector uses only pre-outcome features; safe default "
                          "is Phase-2."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "accuracy_params.json").write_text(json.dumps(params, indent=2))
    return params


if __name__ == "__main__":
    p = fit()
    print(json.dumps(p["fit_log"], indent=2))
