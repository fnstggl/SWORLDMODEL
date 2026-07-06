"""EXP-050: the unified GroundedReadout — does composing the three estimation pieces compound? + end-to-end.

Unifies EXP-048 (latent factors), EXP-049 (LLM world-knowledge prior), and reliability weighting into one
estimator (swm/variables/grounded_readout.py), and wires it into an assembled end-to-end simulator
(swm/api/grounded_simulate.py). Three things validated on GSS, no-cheat:

  A. COMPOUNDING — does factors+prior together beat each piece alone (and beat plain data)? Ablation on
     individual prediction at a data-poor training size (where estimation quality matters most).
  B. RELIABILITY — inject noisy INFERRED variables (corrupted copies, provenance "llm"); does reliability
     weighting (down-weighting inferred inputs) recover the accuracy that uniform weighting loses to noise?
  C. END-TO-END — the assembled GroundedSimulator takes a real question + a held-out population and returns
     a calibrated support share (MAE vs the true held-out share) with an auditable value-factor breakdown.

Run: python -m experiments.exp050_grounded_readout
"""
from __future__ import annotations

import json
import math
import zlib
from collections import defaultdict
from pathlib import Path


def _h(key):
    return (zlib.crc32(str(key).encode()) % 100000) / 100000.0     # deterministic across runs

from experiments.datasets_gss import load
from experiments.exp045_population_rollout import ATTRS
from swm.api.grounded_simulate import GroundedSimulator
from swm.variables.grounded_readout import GroundedReadout

RESULT = "experiments/results/exp050_grounded_readout.json"
N_TRAIN = 150          # data-poor regime where estimation quality separates the estimators


def _rows(recs):
    out = []
    for r in recs:
        for item, ans in r["answers"].items():
            out.append({"qid": item, "answer_idx": ans, "demo": r["demo"]})
    return out


def _split(recs, salt=0, frac=0.3):
    tr, te = [], []
    for r in recs:
        (te if _h((salt, r["uid"])) < frac else tr).append(r)
    return tr, te


def _sub(recs, n, salt=1):
    return recs if n >= len(recs) else sorted(recs, key=lambda r: _h((salt, r["uid"])))[:n]


def _ll_acc(pred, rows):
    ll, correct = 0.0, 0
    for r in rows:
        p = pred(r); pa = p if r["answer_idx"] == 1 else 1 - p
        ll += -math.log(min(1 - 1e-9, max(1e-9, pa)))
        correct += int((p >= 0.5) == (r["answer_idx"] == 1))
    return round(ll / len(rows), 4), round(correct / len(rows), 4)


def _corrupt(recs, k=4, seed=3):
    """Add k NOISY inferred copies of demographics (shuffled labels) with provenance 'llm' — the analog of
    unreliable LLM-inferred variables mixed in with the grounded ones."""
    import random
    rng = random.Random(seed)
    levels = {a: list({r["demo"].get(a, "") for r in recs}) for a in ATTRS[:k]}
    out = []
    for r in recs:
        d = dict(r["demo"])
        for a in ATTRS[:k]:
            d[f"noisy_{a}"] = d.get(a) if rng.random() < 0.5 else rng.choice(levels[a])  # 50% corrupted
        out.append({**r, "demo": d})
    return out


def run():
    recs = load()
    tr_full, te_recs = _split(recs)
    tr_recs = _sub(tr_full, N_TRAIN)                 # data-poor subset for the estimator ablation (A/B)
    tr, te = _rows(tr_recs), _rows(te_recs)

    # A. ablation: do the pieces help, and does the self-configuring GROUNDED capture the best?
    def fit_variant(**kw):
        return GroundedReadout(attrs=ATTRS, k=3, **kw).fit(tr)
    variants = {
        "plain (no factors,no prior)": fit_variant(use_factors=False, use_prior=False),
        "factors_only (EXP-048)": fit_variant(use_factors=True, use_prior=False),
        "prior_only (EXP-049)": fit_variant(use_factors=False, use_prior=True),
    }
    grounded = GroundedReadout(attrs=ATTRS, k=3).fit_auto(tr)     # self-configures on a train-internal holdout
    variants["GROUNDED (auto-config)"] = grounded
    ablation = {name: list(_ll_acc(lambda r: m.predict(r["qid"], r["demo"]), te))
                for name, m in variants.items()}
    chosen = grounded.chosen

    # B. reliability: real + noisy-inferred variables; uniform vs reliability-weighted
    NA = ATTRS + [f"noisy_{a}" for a in ATTRS[:4]]
    prov = {**{a: "data" for a in ATTRS}, **{f"noisy_{a}": "llm" for a in ATTRS[:4]}}
    trc, tec = _rows(_corrupt(tr_recs)), _rows(_corrupt(te_recs))
    unif = GroundedReadout(attrs=NA, provenance=prov, k=4, use_reliability=False).fit(trc)
    relw = GroundedReadout(attrs=NA, provenance=prov, k=4, use_reliability=True).fit(trc)
    reliability = {
        "uniform_weighting": _ll_acc(lambda r: unif.predict(r["qid"], r["demo"]), tec)[0],
        "reliability_weighting": _ll_acc(lambda r: relw.predict(r["qid"], r["demo"]), tec)[0],
    }

    # C. end-to-end: assembled simulator predicts held-out population support shares per question
    #    (trained on the FULL training population — a real forecast, not the data-poor ablation subset)
    sim = GroundedSimulator(attrs=ATTRS).fit(_rows(tr_full))
    te_by_item = defaultdict(list)
    for r in te:
        te_by_item[r["qid"]].append(r)
    maes, example = [], None
    for item, rs in te_by_item.items():
        if len(rs) < 30:
            continue
        true_share = sum(r["answer_idx"] for r in rs) / len(rs)
        fc = sim.simulate_population(item, [r["demo"] for r in rs])
        maes.append(abs(fc.p_outcome - true_share))
        if item == "grass" or (example is None and item in ("cappun", "homosex", "abany")):
            example = {"question": item, "predicted_share": round(fc.p_outcome, 4),
                       "true_share": round(true_share, 4), "n": fc.n,
                       "confidence": round(fc.confidence, 3),
                       "value_drivers": [[i, round(c, 4)] for i, c in fc.value_drivers]}
    e2e = {"n_questions": len(maes), "mean_abs_error_vs_true_share": round(sum(maes) / len(maes), 4),
           "worked_example": example}

    g = ablation["GROUNDED (auto-config)"][0]
    plain = ablation["plain (no factors,no prior)"][0]
    best_piece = min(ablation[k][0] for k in
                     ("plain (no factors,no prior)", "factors_only (EXP-048)", "prior_only (EXP-049)"))
    out = {"dataset": "GSS", "n_train_respondents": N_TRAIN, "n_test_rows": len(te),
           "A_ablation": ablation, "grounded_chosen_config": chosen,
           "B_reliability": reliability, "C_end_to_end": e2e,
           "grounded_beats_plain": g < plain,
           "grounded_matches_best_piece": g <= best_piece + 0.002,
           "reliability_helps": reliability["reliability_weighting"] < reliability["uniform_weighting"]}

    print(f"EXP-050 unified GroundedReadout — GSS, {N_TRAIN} train respondents, {len(te)} test answers")
    print("  A. ABLATION (individual log-loss / acc; lower log-loss better):")
    for name in variants:
        print(f"     {name:<30} {ablation[name]}")
    print(f"     -> GROUNDED auto-chose {chosen}; beats plain: {out['grounded_beats_plain']} "
          f"({g} vs {plain}); matches best piece: {out['grounded_matches_best_piece']}")
    print(f"  B. RELIABILITY (real+noisy-inferred vars): uniform {reliability['uniform_weighting']} "
          f"-> reliability-weighted {reliability['reliability_weighting']}  helps={out['reliability_helps']}")
    print(f"  C. END-TO-END population forecast: MAE vs true share {e2e['mean_abs_error_vs_true_share']} "
          f"over {e2e['n_questions']} questions")
    if example:
        print(f"     worked example [{example['question']}]: predicted {example['predicted_share']} "
              f"vs true {example['true_share']} (n={example['n']}, conf {example['confidence']})")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
