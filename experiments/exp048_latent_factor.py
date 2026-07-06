"""EXP-048: does modelling the correlation structure beat shrinking it? (the estimation frontier)

The binding constraint the project keeps hitting: correlated variables (party ≈ ideology) get
double-counted, so naive estimators are confidently wrong. EXP-041's pooled logistic *shrinks* the
collinear dummies; this tests the first-principles alternative — *decompose* them into orthogonal latent
value factors and estimate on those, so redundant variables collapse into one axis counted once.

Head-to-head on the OpinionQA individual-prediction task (no-cheat, respondents split train/test), at the
full 11-variable richness where double-counting bit hardest (EXP-040/041):
  - NB (independence, double-counts)         — the original failure mode
  - PooledLogistic (shrinkage, EXP-041)      — the current best estimator
  - LatentFactor (decorrelate, EXP-048)      — model the structure, K value factors

Reported overall and on DATA-POOR questions (train n < 25) where estimation efficiency matters most, plus
a sweep over K (how many latent axes the demographics really carry). Writes JSON.
Run: python -m experiments.exp048_latent_factor
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.datasets_opinionqa import load
from experiments.exp040_grounded_simulation import ATTRS, _QModel, _split
from swm.variables.latent_factor_readout import LatentFactorReadout
from swm.variables.pooled_readout import PooledLogisticReadout

RESULT = "experiments/results/exp048_latent_factor.json"


def _ll_acc(pred_fn, rows):
    ll, correct, tot = 0.0, 0, 0
    for r in rows:
        p1 = pred_fn(r)
        pa = p1 if r["answer_idx"] == 1 else (1 - p1)
        ll += -math.log(min(1 - 1e-9, max(1e-9, pa)))
        correct += int((p1 >= 0.5) == (r["answer_idx"] == 1))
        tot += 1
    return round(ll / max(1, tot), 4), round(correct / max(1, tot), 4), tot


def _nb_models(rows, n_opt):
    by = {}
    for r in rows:
        by.setdefault(r["qid"], []).append(r)
    return {q: _QModel(rs, n_opt[q]) for q, rs in by.items() if len(rs) >= 12}


def run():
    recs = load()
    n_opt = {r["qid"]: r["n_opt"] for r in recs}
    tr, te = _split(recs, salt=0)

    nb = _nb_models(tr, n_opt)
    pooled = PooledLogisticReadout(attrs=ATTRS, tau=80.0).fit(tr)
    # choose K (how many latent value axes the demographics carry) on a TRAIN-INTERNAL holdout — no leak
    tr_fit, tr_val = _split(tr, test_frac=0.3, salt=7)
    k_sweep = {}
    best_k, best_ll = 5, 1e9
    for k in (3, 5, 8, 12):
        lf = LatentFactorReadout(attrs=ATTRS, k=k, tau=80.0).fit(tr_fit)
        ll = _ll_acc(lambda r: lf.predict(r["qid"], r["demo"]),
                     [r for r in tr_val if r["qid"] in lf._models])[0]
        k_sweep[k] = ll
        if ll < best_ll:
            best_ll, best_k = ll, k
    best_lf = LatentFactorReadout(attrs=ATTRS, k=best_k, tau=80.0).fit(tr)   # refit on full train at best K

    te_common = [r for r in te if r["qid"] in nb and r["qid"] in pooled._models and r["qid"] in best_lf._models]
    overall = {
        "NB": _ll_acc(lambda r: nb[r["qid"]].predict(r["demo"], ATTRS, alpha=10.0)[1], te_common),
        "PooledLogistic": _ll_acc(lambda r: pooled.predict(r["qid"], r["demo"]), te_common),
        "LatentFactor": _ll_acc(lambda r: best_lf.predict(r["qid"], r["demo"]), te_common),
    }
    # data-poor subset (train n < 25) — where estimation efficiency should matter most
    poor_q = {q for q, e in best_lf._models.items() if e[2] < 25}
    te_poor = [r for r in te_common if r["qid"] in poor_q]
    poor = {
        "n_questions": len(poor_q), "n_test": len(te_poor),
        "PooledLogistic": _ll_acc(lambda r: pooled.predict(r["qid"], r["demo"]), te_poor)[0],
        "LatentFactor": _ll_acc(lambda r: best_lf.predict(r["qid"], r["demo"]), te_poor)[0],
    }

    out = {"dataset": "OpinionQA", "n_test": len(te_common), "best_k": best_k, "k_sweep": k_sweep,
           "overall": {m: list(v) for m, v in overall.items()}, "data_poor": poor,
           "latent_beats_pooled_overall": overall["LatentFactor"][0] < overall["PooledLogistic"][0],
           "latent_beats_pooled_datapoor": poor["LatentFactor"] < poor["PooledLogistic"]}

    print(f"EXP-048 latent-factor (decorrelate) vs shrinkage — OpinionQA, n_test={len(te_common)}, best K={best_k}")
    print(f"  K sweep (latent-factor log-loss): {k_sweep}")
    print("  INDIVIDUAL log-loss (accuracy) at all 11 variables:")
    for m in ("NB", "PooledLogistic", "LatentFactor"):
        print(f"    {m:<16} log_loss {overall[m][0]}  accuracy {overall[m][1]}")
    print(f"  DATA-POOR questions (n<25, {poor['n_questions']} q): "
          f"pooled {poor['PooledLogistic']}  latent {poor['LatentFactor']}")
    print(f"  -> latent-factor beats shrinkage: overall {out['latent_beats_pooled_overall']}, "
          f"data-poor {out['latent_beats_pooled_datapoor']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
