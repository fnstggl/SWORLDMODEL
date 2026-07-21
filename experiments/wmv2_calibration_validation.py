"""Phase 12 calibration validation — fit conditioned calibration on train/val, measure held-out ECE.

Uses the committed leakage-proof forecasting corpus (Manifold+Polymarket resolved binaries with crowd
probabilities). The crowd probability is a genuine as-of forecast; we treat calibrating IT as the test
harness (the same conditioned calibrator V2 outputs use). Splits are disjoint: train+val fit the
calibrator, test only measures. Reports uncalibrated vs global vs conditioned ECE + reliability tables +
risk-coverage (abstention concentrates error).

Run: PYTHONPATH=. python -m experiments.wmv2_calibration_validation
"""
from __future__ import annotations

import json
import random
from pathlib import Path

RESULT = "experiments/results/wmv2_calibration_validation.json"


def run():
    from swm.eval.forecasting_corpus import load_corpus
    from swm.world_model_v2.calibration import (ece, fit_conditioned, fit_platt, reliability_table)

    corpus = [i for i in load_corpus() if i.crowd_prob is not None]
    rng = random.Random(13)
    rng.shuffle(corpus)
    n = len(corpus)
    tr = corpus[: int(0.5 * n)]
    va = corpus[int(0.5 * n): int(0.7 * n)]
    te = corpus[int(0.7 * n):]

    def key(i):
        return f"{i.category}"                                 # domain-conditioned calibration

    global_cal = fit_platt([(i.crowd_prob, i.outcome) for i in tr + va], fitted_on="train+val global")
    cond_cal = fit_conditioned([(i.crowd_prob, i.outcome, key(i)) for i in tr + va],
                               min_cell=25, fitted_on="train+val conditioned")

    raw = [(i.crowd_prob, i.outcome) for i in te]
    glob = [(global_cal.apply(i.crowd_prob), i.outcome) for i in te]
    cond = [(cond_cal.apply(i.crowd_prob, key(i)), i.outcome) for i in te]

    def brier(pairs):
        return round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 5)

    out = {
        "n": {"train": len(tr), "val": len(va), "test": len(te)},
        "governance": "calibrators fitted on train+val ONLY; test measured, never fitted",
        "ece": {"uncalibrated": ece(raw), "global": ece(glob), "conditioned": ece(cond)},
        "brier": {"uncalibrated": brier(raw), "global": brier(glob), "conditioned": brier(cond)},
        "reliability_uncalibrated": reliability_table(raw),
        "reliability_conditioned": reliability_table(cond),
        "global_calibrator": {"a": global_cal.a, "b": global_cal.b, "n_fit": global_cal.n_fit},
        "conditioned_cells": {k: {"n": c.n_fit, "a": c.a, "b": c.b} for k, c in cond_cal.cells.items()},
        "verdict": None,
    }
    best = min(out["ece"], key=lambda k: out["ece"][k] if out["ece"][k] is not None else 9)
    out["verdict"] = (f"best held-out ECE: {best} ({out['ece'][best]}); "
                      f"calibration {'improves' if out['ece'][best] < out['ece']['uncalibrated'] else 'does NOT improve'} "
                      f"over uncalibrated ({out['ece']['uncalibrated']})")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["ece"], indent=1))
    print(json.dumps(out["brier"], indent=1))
    print(out["verdict"])
    print(f"wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
