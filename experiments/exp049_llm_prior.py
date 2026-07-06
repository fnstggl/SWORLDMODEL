"""EXP-049: LLM-informed priors — does world knowledge fix data-poverty? (estimation frontier, piece #1)

The synthesis. An LLM knows how attributes relate to outcomes without any dataset (conservatives favor the
death penalty; the secular favor legal marijuana). `swm/variables/llm_prior.py` turns that into a PRIOR on
the effect coefficients; the data updates it (logistic regularized toward the prior, not toward zero). The
decisive test is SAMPLE EFFICIENCY: as training data shrinks, a world-knowledge prior should carry the
prediction where a from-scratch data-only estimator collapses.

GSS individual prediction (predict a respondent's binary answer to an item from demographics), no-cheat
(respondents split train/test). Three estimators, swept over training size N (respondents):
  - prior_only          — zero-shot LLM world knowledge, NO fitting (the N=0 anchor)
  - data_only           — the pooled logistic (EXP-041), one-hot demographics, no prior
  - prior_informed      — logistic on the prior-signed features, coefficients shrunk toward the LLM prior

Claim: prior_informed dominates data_only at small N (grounding beats data-poverty) and stays ≥ it at
large N; prior_only alone is already far above the marginal (the LLM has a real social world model).
Writes JSON. Run: python -m experiments.exp049_llm_prior
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.datasets_gss import load
from experiments.exp045_population_rollout import ATTRS
from swm.variables.llm_prior import LLMPriorReadout
from swm.variables.pooled_readout import PooledLogisticReadout

RESULT = "experiments/results/exp049_llm_prior.json"
SIZES = [50, 200, 1000, 5000]           # training-respondent sweep (caps at 5000 for tractable fits)


def _rows(recs):
    out = []
    for r in recs:
        for item, ans in r["answers"].items():
            out.append({"uid": r["uid"], "item": item, "qid": item,
                        "answer": ans, "answer_idx": ans, "demo": r["demo"]})
    return out


def _split(recs, salt=0, test_frac=0.3):
    tr, te = [], []
    for r in recs:
        if (hash((salt, r["uid"])) % 1000) / 1000.0 < test_frac:
            te.append(r)
        else:
            tr.append(r)
    return tr, te


def _subsample(recs, n, salt=1):
    if n is None or n >= len(recs):
        return recs
    return sorted(recs, key=lambda r: hash((salt, r["uid"])))[:n]


def _score(pred_fn, rows):
    ll, correct, tot = 0.0, 0, 0
    for r in rows:
        p = pred_fn(r)
        pa = p if r["answer"] == 1 else (1 - p)
        ll += -math.log(min(1 - 1e-9, max(1e-9, pa)))
        correct += int((p >= 0.5) == (r["answer"] == 1))
        tot += 1
    return round(ll / max(1, tot), 4), round(correct / max(1, tot), 4)


def run():
    recs = load()
    tr_recs, te_recs = _split(recs)
    te = _rows(te_recs)
    # marginal baseline (per-item majority from the full train)
    from collections import defaultdict
    ic = defaultdict(lambda: [0, 0])
    for r in _rows(tr_recs):
        ic[r["item"]][r["answer"]] += 1
    marg = {it: (c[1] + 1) / (c[0] + c[1] + 2) for it, c in ic.items()}
    base = _score(lambda r: marg.get(r["item"], 0.5), te)

    prior_only = LLMPriorReadout(attrs=ATTRS, prior_only=True).fit(_rows(tr_recs[:200]))
    prior_pt = _score(lambda r: prior_only.predict(r["item"], r["demo"]), te)

    curve = []
    for n in SIZES:
        sub = _subsample(tr_recs, n)
        rows = _rows(sub)
        data = PooledLogisticReadout(attrs=ATTRS, tau=20.0, epochs=120).fit(rows, min_q=8)
        prior = LLMPriorReadout(attrs=ATTRS, l2=2.0, epochs=120).fit(rows)
        d_ll, d_acc = _score(lambda r: data.predict(r["item"], r["demo"]), te)
        p_ll, p_acc = _score(lambda r: prior.predict(r["item"], r["demo"]), te)
        curve.append({"n_respondents": n or len(tr_recs), "n_train_rows": len(rows),
                      "data_only": [d_ll, d_acc], "prior_informed": [p_ll, p_acc],
                      "prior_wins": p_ll < d_ll})

    small = curve[0]
    out = {"dataset": "GSS", "n_test_rows": len(te), "attrs": len(ATTRS),
           "marginal_baseline": list(base), "prior_only_zeroshot": list(prior_pt),
           "sample_efficiency": curve,
           "prior_wins_at_smallest_N": small["prior_wins"],
           "prior_only_beats_marginal": prior_pt[0] < base[0]}

    print(f"EXP-049 LLM-informed priors on GSS — {len(te)} test answers, {len(ATTRS)} attributes")
    print(f"  marginal baseline        log_loss {base[0]}  acc {base[1]}")
    print(f"  prior_only (zero-shot)   log_loss {prior_pt[0]}  acc {prior_pt[1]}   "
          f"<- LLM world knowledge, NO data")
    print("  SAMPLE EFFICIENCY (log-loss / accuracy), data_only vs prior_informed:")
    print(f"    {'N_resp':<8}{'rows':<8}{'data_only':<22}{'prior_informed':<22}{'prior wins'}")
    for c in curve:
        print(f"    {c['n_respondents']:<8}{c['n_train_rows']:<8}"
              f"{str(c['data_only']):<22}{str(c['prior_informed']):<22}{c['prior_wins']}")
    print(f"  -> prior_only beats marginal: {out['prior_only_beats_marginal']}; "
          f"prior_informed wins at smallest N: {out['prior_wins_at_smallest_N']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
