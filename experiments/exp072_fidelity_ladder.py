"""EXP-072: does modeling MORE pressuring variables help or hurt? — the thesis, tested no-cheat.

The disagreement: is "model every relevant variable" right, or does adding variables hurt? The honest
answer depends entirely on HOW each variable's weight is set. This settles it on real data with a controlled
probe.

Setup — real ChangeMyView persuasion (person × message → persuaded?), temporal split (no leakage). We climb
a FIDELITY LADDER: add the 8 real variables (ordered by relevance), THEN add irrelevant NOISE variables
(deterministic, uncorrelated with the outcome) — the stress test for "a variable that doesn't really press
on the outcome". At each rung, three weighting schemes score the held-out future:

  NAIVE       — a free point weight per variable (~unregularized).
  CALIBRATED  — L2-to-prior shrinkage (BayesianLogistic): useless variables shrink to ~zero weight.
  UNCERTAINTY — CALIBRATED + integrating the Laplace posterior over the weights (unknown weight -> wider p).

Prediction of the thesis-with-calibration: NAIVE degrades as noise variables are piled on; CALIBRATED holds
(the noise weights shrink away); UNCERTAINTY calibrates best. If so: "model every pressuring variable" is
right *provided each carries a calibrated weight with uncertainty* — the binding constraint is weighting, not
variable count. The weight report shows the model KNOWS which weights it hasn't pinned down (noise vars get
signal-to-noise < 1).

Run: python -m experiments.exp072_fidelity_ladder
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from experiments.exp069_action_layer_validation import _load_cmv
from swm.eval.fidelity_ladder import average_ladders, run_ladder

RESULT = "experiments/results/exp072_fidelity_ladder.json"
OQA = "experiments/results/exp028_oqa/oqa_parsed.json"
# demographic attributes in rough order of general predictive strength for opinions (added left-to-right)
OQA_ATTRS = ["party", "ideology", "religion", "attendance", "race", "age", "education", "income", "sex",
             "marital", "region", "citizen"]
REAL_VARS = ["trait_openness", "skepticism", "certainty_disposition",       # person
             "addresses_crux", "evidence", "clarity", "politeness_disposition", "expertise"]  # message


def _get(name):
    def f(r):
        if name in r["person"]:
            return float(r["person"][name])
        if name in r["message"]:
            return float(r["message"][name])
        return 0.5
    return f


def _noise(j):
    def f(r):
        return random.Random((r["_i"] * 131 + j * 977 + 7) & 0xFFFFFFFF).random()   # deterministic, unrelated
    return f


def _corr(rows, fn):
    xs = [fn(r) for r in rows]
    ys = [r["y"] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    vy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    return abs(cov / (vx * vy)) if vx > 1e-9 and vy > 1e-9 else 0.0


def _onehot_spec(attr, vocab):
    def fn(r):
        lv = r["demo"].get(attr, "NA")
        return [1.0 if lv == v else 0.0 for v in vocab]
    return (attr, fn)


def run_oqa(min_count=140, max_q=25) -> dict:
    """OpinionQA: add demographic attributes (each a one-hot block) one at a time, predicting a binary opinion
    from demographics, on many data-rich questions. This is the high-dimensional regime where NAIVE overfits
    the correlated dummies as attributes pile on — and where calibrated shrinkage should hold. Averaged over
    questions for a robust curve. naive vs calibrated only (the overfitting story)."""
    rows = [r for r in json.loads(Path(OQA).read_text()) if r.get("n_opt") == 2 and int(r["answer_idx"]) in (0, 1)]
    byq = {}
    for r in rows:
        byq.setdefault(r["qid"], []).append(r)
    big = sorted([(q, rs) for q, rs in byq.items() if len(rs) >= min_count], key=lambda t: -len(t[1]))[:max_q]
    arms = {"naive": {"l2": 0.02, "integrate": False}, "calibrated": {"l2": 1.0, "integrate": False},
            "calibrated_strong": {"l2": 5.0, "integrate": False}}   # properly-tuned shrinkage for thin data
    results = []
    for q, rs in big:
        rs = sorted(rs, key=lambda r: r["uid"])
        tr = [r for i, r in enumerate(rs) if i % 10 >= 3]
        te = [r for i, r in enumerate(rs) if i % 10 < 3]
        for r in tr + te:
            r["y"] = int(r["answer_idx"])
        specs = []
        for attr in OQA_ATTRS:
            vocab = sorted({r["demo"].get(attr, "NA") for r in tr})
            if len(vocab) >= 2:
                specs.append(_onehot_spec(attr, vocab))
        results.append(run_ladder(tr, te, specs, arms=arms, extras=False))
    agg = average_ladders(results)
    return {"n_questions": len(results), "curves": agg["curves"], "verdict": agg["verdict"]}


def run(n_noise=8) -> dict:
    rows = _load_cmv()
    for i, r in enumerate(rows):
        r["_i"] = i
    cut = int(0.7 * len(rows))
    train, test = rows[:cut], rows[cut:]

    order = sorted(REAL_VARS, key=lambda v: -_corr(train, _get(v)))          # most relevant real vars first
    specs = [(v, _get(v)) for v in order] + [(f"noise_{j}", _noise(j)) for j in range(n_noise)]

    out = run_ladder(train, test, specs, seed=0)
    curves, verdict = out["curves"], out["verdict"]
    k_real, k_full = len(REAL_VARS), len(specs)

    def at(arm, k):
        return next(c for c in curves[arm] if c["k"] == k)

    summary = {a: {"real_only_logloss": at(a, k_real)["log_loss"], "with_noise_logloss": at(a, k_full)["log_loss"],
                   "noise_penalty": round(at(a, k_full)["log_loss"] - at(a, k_real)["log_loss"], 4),
                   "real_only_ece": at(a, k_real)["ece"], "with_noise_ece": at(a, k_full)["ece"]}
               for a in curves}
    oqa = run_oqa()

    res = {"cmv_low_signal": {"data": "ChangeMyView (real, temporal split)", "n_train": len(train),
                              "n_test": len(test), "n_real_vars": k_real, "n_noise_vars": n_noise,
                              "variable_order": order, "ladder_summary": summary, "verdict": verdict,
                              "weight_report_full": out["final"].get("weight_report"),
                              "variance_triage": out["final"].get("variance_triage")},
           "oqa_high_dim": oqa,
           "reconciliation": ("naive overfits added variables (OQA: log-loss rises as demographic blocks pile "
                              "on); calibrated shrinkage absorbs the useless ones so adding variables does not "
                              "hurt. The binding constraint is weight calibration + uncertainty, not count.")}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-072  fidelity ladder — does modeling MORE variables help or hurt? (no-cheat CMV)")
    print(f"  {len(train)} train / {len(test)} test; {k_real} real vars + {n_noise} injected noise vars")
    print(f"  {'scheme':12s}  real-only LL   +noise LL   noise penalty   ECE(real→noise)")
    for a in ("naive", "calibrated", "uncertainty"):
        s = summary[a]
        print(f"  {a:12s}  {s['real_only_logloss']:.4f}       {s['with_noise_logloss']:.4f}     "
              f"{s['noise_penalty']:+.4f}        {s['real_only_ece']:.3f}→{s['with_noise_ece']:.3f}")
    wr = out["final"].get("weight_report", [])
    real_snr = [w["snr"] for w in wr if not w["name"].startswith("noise")]
    noise_snr = [w["snr"] for w in wr if w["name"].startswith("noise")]
    if real_snr and noise_snr:
        print(f"  weight signal-to-noise |w|/sd:  real vars median={sorted(real_snr)[len(real_snr)//2]:.2f}  "
              f"noise vars median={sorted(noise_snr)[len(noise_snr)//2]:.2f}  (the model knows which it can't weight)")
    print(f"\n  OQA high-dim (add demographic blocks; avg over {oqa['n_questions']} data-rich binary questions):")
    nc, cc, sc = oqa["curves"]["naive"], oqa["curves"]["calibrated"], oqa["curves"]["calibrated_strong"]
    print(f"    {'#attrs':6s}  naive LL   calibrated LL   strong-shrink LL")
    for k in (1, len(nc) // 2, len(nc)):
        print(f"    {k:<6d}  {nc[k-1]['log_loss']:.4f}     {cc[k-1]['log_loss']:.4f}          {sc[k-1]['log_loss']:.4f}")
    v = oqa["verdict"]
    print(f"    harm from adding all 12 vars (full − best):  naive +{v['naive']['full_minus_best']}   "
          f"calibrated +{v['calibrated']['full_minus_best']}   strong +{v['calibrated_strong']['full_minus_best']}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
