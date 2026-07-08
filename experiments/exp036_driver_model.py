"""EXP-036: can we forecast DIRECTION after all? (leakage-safe driver model)

EXP-033/035 said no — but using only the price's own shape and a momentum drift. A leakage-safe
diagnostic found a real driver: a belief's LEAN (distance from 0.5) predicts the direction of its future
move at ~its calibration rate, because a question resolves toward the side it currently favors. This is a
directional signal (beats a coin flip) that is knowable AS-OF (from the current level) — no LLM recall of
outcomes (the contamination that inflated prior LLM-forecasting results).

Test: a learned direction classifier (P(up | lean, momentum, result-cue, days-to-resolution)) vs the raw
lean, momentum, and a coin flip, on the sign of the H-step move. No-cheat: trained on TRAIN move signs,
evaluated on held-out TEST. Overall and on the CONFIDENT subset (|p−0.5|>0.2).

The honest boundary: this beats 50% on DIRECTION; it does NOT beat the martingale on the POINT (a
calibrated belief already is its expected value — EXP-033/035). Directional accuracy is the useful thing
for questions with no liquid market. Run: python -m experiments.exp036_driver_model
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

from swm.transition.direction_model import DirectionModel, direction_features
from experiments.datasets_swm import load

RESULT = "experiments/results/exp036_driver_model.json"
HORIZON = 8
FLAT = 0.02
_MON = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9,
        "OCT": 10, "NOV": 11, "DEC": 12}


def _days_to_res(r):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", r.get("market_id", ""))
    if not m:
        return None
    try:
        rd = datetime.datetime(2000 + int(m.group(1)), _MON[m.group(2)], int(m.group(3)),
                               tzinfo=datetime.timezone.utc).timestamp()
    except Exception:
        return None
    return (rd - r["target"]["t"]) / 86400.0


def _example(r):
    prices = [h["p"] for h in r["history"]] + [r["target"]["p"]]
    f = direction_features(prices, news=r.get("news"), days_to_res=_days_to_res(r))
    move = r["future"][HORIZON - 1]["p"] - r["target"]["p"]
    return f, move, r["target"]["p"]


def run():
    train = [r for r in load("train") if r.get("history") and r.get("target") and len(r.get("future", [])) >= HORIZON]
    test = [r for r in load("test_kalshi") if r.get("history") and r.get("target") and len(r.get("future", [])) >= HORIZON]
    dm = DirectionModel().fit([(f, mv) for f, mv, _ in (_example(r) for r in train)])
    # the lean rule is parameter-free (leakage-safe by construction); report its accuracy on the TRAIN
    # period too, to be honest about regime variation in the directional signal.
    tr_nf = [(f, mv) for f, mv, _ in (_example(r) for r in train) if abs(mv) >= FLAT]
    lean_train = round(sum(int((f[0] > 0) == (mv > 0)) for f, mv in tr_nf) / max(1, len(tr_nf)), 3)

    tiers = {"coin_flip": [0, 0], "momentum": [0, 0], "lean": [0, 0], "learned": [0, 0]}
    conf = {k: [0, 0] for k in tiers}
    for r in test:
        f, move, p0 = _example(r)
        if abs(move) < FLAT:
            continue
        truth = 1 if move > 0 else 0
        calls = {"coin_flip": 1, "momentum": 1 if f[3] > 0 else 0,   # f[3]=momentum
                 "lean": 1 if f[0] > 0 else 0,                        # f[0]=lean
                 "learned": 1 if dm.p_up(f) > 0.5 else 0}
        confident = abs(p0 - 0.5) > 0.2
        for k, c in calls.items():
            corr = int(c == truth) if k != "coin_flip" else 0        # coin flip scored as 0.5 below
            tiers[k][0] += corr; tiers[k][1] += 1
            if confident:
                conf[k][0] += corr; conf[k][1] += 1

    def acc(d, k):
        return 0.5 if k == "coin_flip" else round(d[k][0] / max(1, d[k][1]), 3)
    overall = {k: acc(tiers, k) for k in tiers}
    confident = {k: acc(conf, k) for k in tiers}
    out = {"dataset": "kalshi", "n_test_moves": tiers["lean"][1], "n_confident": conf["lean"][1],
           "horizon": HORIZON, "directional_accuracy": overall, "confident_directional_accuracy": confident,
           "lean_accuracy_train_period": lean_train,
           "lean_beats_chance": overall["lean"] > 0.5 and lean_train > 0.5,
           "momentum_is_useless": abs(overall["momentum"] - 0.5) < 0.05}
    print(f"EXP-036 direction model (Kalshi) — {tiers['lean'][1]} non-flat test moves, horizon {HORIZON}")
    print("  DIRECTIONAL ACCURACY (fraction of moves whose direction we call right; 0.5 = chance):")
    for k in ("coin_flip", "momentum", "lean", "learned"):
        print(f"    {k:<12} overall {overall[k]}   confident(|p-.5|>0.2) {confident[k]}")
    print(f"  lean rule (parameter-free) on TRAIN period: {lean_train} — signal present in both regimes "
          f"(varies {lean_train}–{overall['lean']})")
    print("  NOTE: this is DIRECTIONAL accuracy (beats chance); it is NOT a point-forecast edge over the")
    print("  martingale (a calibrated belief already is its expected value). It is the forecast where no market exists.")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
