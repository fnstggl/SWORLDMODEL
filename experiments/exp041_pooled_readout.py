"""EXP-041: the estimator that makes "map more variables" monotone (Part 2 of the north-star build).

EXP-040 found grounded simulation beats the composite BUT naive Bayes double-counts correlated variables
(party ≈ ideology) and overfits — "more variables" helped only after hand-tuning shrinkage. This builds
and validates the fix as a reusable component: a CORRELATION-AWARE, PARTIALLY-POOLED readout.

Three estimators compared on OpinionQA individual prediction, no-cheat (respondents split train/test),
across the same variable-richness ladder as EXP-040:
  - NB              : shrinkage naive Bayes (EXP-040's estimator, independence assumption)
  - Logistic        : correlation-aware logistic over one-hot variables (shares credit among collinear dummies)
  - PooledLogistic  : + n-adaptive partial pooling toward the question marginal (tau tuned by empirical
                      Bayes on a TRAIN-internal hold-out — no hand-tuning, no test leakage)

Claims tested:
  C1. Correlation-aware logistic makes "more variables" MONOTONE without hand-tuning (fixes NB's collapse).
  C2. Partial pooling beats independent per-question fitting, concentrated on DATA-POOR questions.
  C3. The pooled readout is the best individual simulator of the person's answer.
Run: python -m experiments.exp041_pooled_readout
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.datasets_opinionqa import load
from experiments.exp040_grounded_simulation import ATTRS, LEVELS, _QModel, _split
from swm.variables.pooled_readout import PooledLogisticReadout

RESULT = "experiments/results/exp041_pooled_readout.json"


def _nb_models(rows, n_opt):
    by = {}
    for r in rows:
        by.setdefault(r["qid"], []).append(r)
    return {q: _QModel(rs, n_opt[q]) for q, rs in by.items() if len(rs) >= 12}


def _ll_acc(pred_fn, rows):
    ll, correct, tot = 0.0, 0, 0
    for r in rows:
        p1 = pred_fn(r)
        pa = p1 if r["answer_idx"] == 1 else (1 - p1)
        ll += -math.log(min(1 - 1e-9, max(1e-9, pa)))
        correct += int((p1 >= 0.5) == (r["answer_idx"] == 1))
        tot += 1
    return round(ll / max(1, tot), 4), round(correct / max(1, tot), 4), tot


def run():
    recs = load()
    n_opt = {r["qid"]: r["n_opt"] for r in recs}
    tr, te = _split(recs, salt=0)

    # tune the pooling strength tau on a train-internal hold-out (empirical Bayes, leakage-free)
    tr_fit, tr_val = _split(tr, test_frac=0.3, salt=1)
    tr_val_q = {r["qid"] for r in tr_fit if True}
    best_tau, best = 20.0, 1e9
    for tau in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
        m = PooledLogisticReadout(attrs=ATTRS, tau=tau).fit(tr_fit)
        ll, _, _ = _ll_acc(lambda r: m.predict(r["qid"], r["demo"]),
                           [r for r in tr_val if r["qid"] in m._models])
        if ll < best:
            best, best_tau = ll, tau

    # fit all three estimators on train, evaluate on test, across the variable-richness ladder
    nb = _nb_models(tr, n_opt)
    te_nb = [r for r in te if r["qid"] in nb]
    ladder = {}
    for name, use in LEVELS.items():
        row = {"n_vars": len(use)}
        # NB at a fixed moderate smoothing (EXP-040's estimator)
        row["NB"] = _ll_acc(lambda r: nb[r["qid"]].predict(r["demo"], use, alpha=10.0)[1], te_nb)[:2]
        # correlation-aware logistic (no pooling: tau=0) and pooled logistic (tau tuned)
        if use:
            lg = PooledLogisticReadout(attrs=use, tau=0.0).fit(tr)
            pl = PooledLogisticReadout(attrs=use, tau=best_tau).fit(tr)
            te_l = [r for r in te if r["qid"] in lg._models]
            row["Logistic"] = _ll_acc(lambda r: lg.predict(r["qid"], r["demo"]), te_l)[:2]
            row["PooledLogistic"] = _ll_acc(lambda r: pl.predict(r["qid"], r["demo"]), te_l)[:2]
        else:
            base = _ll_acc(lambda r: nb[r["qid"]].p_a[1], te_nb)[:2]
            row["Logistic"] = row["PooledLogistic"] = base
        ladder[name] = row

    # C2: pooling's win on DATA-POOR questions (train n < 25), full variable set
    full = ATTRS
    lg = PooledLogisticReadout(attrs=full, tau=0.0).fit(tr)
    pl = PooledLogisticReadout(attrs=full, tau=best_tau).fit(tr)
    poor_q = {q for q, e in pl._models.items() if e[3] < 25}
    te_poor = [r for r in te if r["qid"] in poor_q and r["qid"] in lg._models]
    poor = {"n_questions": len(poor_q), "n_test": len(te_poor),
            "independent_logistic": _ll_acc(lambda r: lg.predict(r["qid"], r["demo"]), te_poor)[0],
            "pooled_logistic": _ll_acc(lambda r: pl.predict(r["qid"], r["demo"]), te_poor)[0]}

    def _mono(key):
        seq = [ladder[n][key][0] for n in LEVELS]
        return all(b <= a + 1e-9 for a, b in zip(seq, seq[1:]))

    out = {"dataset": "OpinionQA", "tuned_tau": best_tau, "n_test": len(te_nb),
           "ladder": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                      for k, v in ladder.items()},
           "monotone_in_variables": {"NB": _mono("NB"), "Logistic": _mono("Logistic"),
                                     "PooledLogistic": _mono("PooledLogistic")},
           "data_poor_pooling": poor}

    print(f"EXP-041 correlation-aware pooled readout — OpinionQA, tuned tau={best_tau}, n_test={len(te_nb)}")
    print("  INDIVIDUAL log-loss (accuracy) by variable richness — 3 estimators:")
    print(f"    {'variables':<28}{'NB':>18}{'Logistic':>18}{'PooledLogistic':>18}")
    for name in LEVELS:
        r = ladder[name]
        def fmt(k):
            ll, ac = r[k]; return f"{ll} ({ac})"
        print(f"    {name:<28}{fmt('NB'):>18}{fmt('Logistic'):>18}{fmt('PooledLogistic'):>18}")
    print(f"  C1 monotone in #variables: NB={out['monotone_in_variables']['NB']}  "
          f"Logistic={out['monotone_in_variables']['Logistic']}  "
          f"PooledLogistic={out['monotone_in_variables']['PooledLogistic']}")
    print(f"  C2 data-poor questions (n<25, {poor['n_questions']} q): independent logistic "
          f"{poor['independent_logistic']} -> pooled {poor['pooled_logistic']}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
