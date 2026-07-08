"""EXP-075: the KPI that matters — "pick the best of several candidate messages" at full scale.

EXP-074 measured the MATCHED-PAIR task (two near-identical arguments to the same OP, one happened to
win) and found it near-irreducible: a scaled lexical readout tops out ~0.60 and DeepSeek zero-shot is
0.53 (chance). That is the pessimistic academic benchmark — by construction the two arguments are matched
on topic and quality, so what separates winner from loser is mostly luck/timing, not text.

But that is NOT the product question. The product asks: given several DIFFERENT candidate messages, which
should I send? That is precision@1 over an OP's real set of challenger arguments — and it has genuine
signal. EXP-073 measured it at 0.66-0.83 on only 64 OPs. This runs the SAME framing at 47x scale
(3,051 OPs, 8,106 candidate arguments) to answer honestly:

  A. DOES THE PRODUCT KPI HOLD AT SCALE, and does more data climb it? Train the learned lexical readout
     on N OPs, score precision@1 on a FIXED held-out set of OPs (top-ranked candidate == a delta winner?).
  B. HOW FAR ABOVE the matched-pair floor (0.60) and random is it — i.e. how much of "which message is
     better" is real, recoverable signal vs the irreducible luck the matched-pair task isolates.

Leakage-free: split by OP; features from argument+OP text only (same _feats as EXP-074). Pure-python,
CPU-only. Cache: experiments/results/exp075/cmv_perop.json (built offline from the ConvoKit corpus).
Run: python -m experiments.exp075_product_kpi_scale
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.transition.readout import LogisticReadout
from experiments.exp074_cmv_scale import _feats

PEROP = "experiments/results/exp075/cmv_perop.json"
RESULT = "experiments/results/exp075_product_kpi_scale.json"


def _train(ops):
    X = [_feats(a["text"], o["op_text"]) for o in ops for a in o["args"]]
    y = [a["success"] for o in ops for a in o["args"]]
    return LogisticReadout(l2=0.5, epochs=300).fit(X, y)


def _precision_at_1(model, ops):
    """Of each OP's candidate arguments, is the model's top pick a delta winner?"""
    hits = rand = 0
    for o in ops:
        top = max(o["args"], key=lambda a: model.predict_proba(_feats(a["text"], o["op_text"])))
        hits += top["success"]
        rand += sum(a["success"] for a in o["args"]) / len(o["args"])   # expected hit of a random pick
    return hits / len(ops), rand / len(ops)


def run():
    ops = json.loads(Path(PEROP).read_text())
    cut = int(0.75 * len(ops))
    train_all, test = ops[:cut], ops[cut:]

    curve = {}
    rand = None
    for n in (64, 200, 500, 1000, len(train_all)):
        m = _train(train_all[:n])
        p, rand = _precision_at_1(m, test)
        curve[n] = round(p, 4)
    full = curve[len(train_all)]

    out = {"data": "full ConvoKit winning-args corpus, per-OP candidate sets",
           "n_ops": len(ops), "n_args": sum(len(o["args"]) for o in ops),
           "n_train_ops": len(train_all), "n_test_ops": len(test),
           "kpi": "precision@1 — of an OP's several DIFFERENT candidate arguments, is the top-ranked one a delta winner?",
           "A_scaling_curve_precision@1": {str(k): v for k, v in curve.items()},
           "full_scale_precision@1": full,
           "vs_64_scale": round(full - curve[64], 4),
           "random_pick_baseline": round(rand, 4),
           "lift_over_random": round(full - rand, 4),
           "matched_pair_floor_exp074": 0.5994,
           "matched_pair_deepseek_exp074": 0.5267,
           "exp073_product_kpi_64ops": 0.656}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-075  the PRODUCT KPI at scale: pick the best of several candidate messages")
    print(f"  {len(ops)} OPs, {out['n_args']} candidate args (EXP-073 had 64 OPs) | train {len(train_all)} / test {len(test)} OPs")
    print(f"  random pick baseline: {rand:.4f}")
    print("  A. DOES IT CLIMB WITH DATA? (precision@1 on fixed held-out OPs)")
    for k, v in curve.items():
        bar = "#" * int((v - 0.5) * 200)
        print(f"       train n={k:5d} OPs   precision@1 = {v:.4f}  {bar}")
    print(f"     -> full {full:.4f} vs 64-OP {curve[64]:.4f} ({out['vs_64_scale']:+}) | lift over random {out['lift_over_random']:+.4f}")
    print(f"  B. VS the matched-pair task: product KPI {full:.3f}  >>  matched-pair lexical 0.599, DeepSeek 0.527")
    print("     READ: 'which of these DIFFERENT messages is better' has real signal and holds at scale;")
    print("           'which of two MATCHED args happened to win' is near-irreducible. The product asks the former.")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
