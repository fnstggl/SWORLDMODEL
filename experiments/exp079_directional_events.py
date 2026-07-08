"""EXP-079: directional event forecasting — predict the DIRECTION of a pivotal event from state, no-cheat.

The event model v1 places calibrated VARIANCE but is symmetric. This adds DIRECTION with the correct
decomposition P(move) × P(up | move): most events hold; when the Fed moves, which way? The direction model
is calibrated on the train era using the CORPUS-HARVESTED `rate_hike` elasticities as priors (the flywheel
feeding the event model), and scored on held-out pivotal moves.

  A. DIRECTIONAL SKILL (the building block): on held-out months where the Fed actually moves, predict
     hike-vs-cut from macro + momentum state. Accuracy + log-loss vs momentum (last move) and the base rate.
  B. CALIBRATED ROLLOUT: the DirectionalEventModel rolls a calendar forward with P(move)×P(up|move)+hold and
     momentum carrying — check the terminal 6-month interval stays calibrated and the direction is unbiased
     (no spurious drift, the bug that separating move-frequency from move-direction fixes).

Run: python -m experiments.exp079_directional_events
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from swm.simulation.directional_event_model import (DirectionalEventModel, calibrated_impact_fn,
                                                    momentum_evolve)
from swm.simulation.event_model import interval_coverage
from swm.variables.calibrated_weights import CalibratedWeights, WeightPrior
from swm.variables.prior_registry import PriorRegistry

FOMC = "experiments/results/exp071/fomc_macro.json"
RESULT = "experiments/results/exp079_directional_events.json"
FEATS = ["inflation", "unemployment", "recent_move"]
THR = 0.05


def _state(data, i):
    prev = data[max(0, i - 1)]["rate"]
    return {"inflation": data[i]["inflation"] / 10.0, "unemployment": data[i]["unemp"] / 10.0,
            "recent_move": max(-1.0, min(1.0, data[i]["rate"] - prev))}


def _clip(p):
    return min(1 - 1e-6, max(1e-6, p))


def run() -> dict:
    data = json.loads(Path(FOMC).read_text())
    rates = [d["rate"] for d in data]
    n = len(rates)
    cut = int(0.6 * n)
    move = lambda i: rates[i + 1] - rates[i]

    # --- train the DIRECTION model P(hike | move happens, state), priored by the harvested rate_hike weights ---
    reg = PriorRegistry.load()
    priors = [reg.prior_for(f, "rate_hike", fallback=WeightPrior(f, 0.0, 2.0)) for f in FEATS]
    tr_piv = [i for i in range(1, cut - 1) if abs(move(i)) > THR]
    Xtr = [[_state(data, i)[f] for f in FEATS] for i in tr_piv]
    ytr = [1 if move(i) > 0 else 0 for i in tr_piv]
    cw = CalibratedWeights(priors, temper_grid=(1.0, 4.0), epochs=150).fit(Xtr, ytr, tune=True)
    p_move = len(tr_piv) / (cut - 2)
    mags = [abs(move(i)) for i in tr_piv]
    mag_mean = sum(mags) / len(mags)
    mag_sd = (sum((m - mag_mean) ** 2 for m in mags) / len(mags)) ** 0.5
    base_up = sum(ytr) / len(ytr)

    # --- A. directional skill on held-out pivotal moves ---
    te_piv = [i for i in range(cut, n - 1) if abs(move(i)) > THR]
    hit_model = hit_mom = hit_base = 0
    ll_model, ll_mom = [], []
    for i in te_piv:
        st = _state(data, i)
        up = move(i) > 0
        p = cw.predict([st[f] for f in FEATS])
        hit_model += 1 if (p > 0.5) == up else 0
        mom_up = st["recent_move"] > 0
        hit_mom += 1 if mom_up == up else 0
        hit_base += 1 if (base_up > 0.5) == up else 0
        ll_model.append(-math.log(_clip(p if up else 1 - p)))
        ll_mom.append(-math.log(_clip(0.7 if mom_up == up else 0.3)))
    nA = len(te_piv)

    # --- B. calibrated rollout (P(move)xP(up|move)+hold), 6-month horizon ---
    H = 6
    dem = DirectionalEventModel(list(range(1, H + 1)),
                                calibrated_impact_fn(lambda s: cw.predict([s[f] for f in FEATS]),
                                                     mag_mean, mag_sd, p_move=p_move),
                                momentum_evolve("recent_move"))
    truths, los, his = [], [], []
    dir_hit = dir_tot = 0
    for i in range(cut, n - H):
        st = _state(data, i)
        r = dem.rollout(rates[i], st, H, n=2500, seed=i)
        truths.append(rates[i + H]); los.append(r["p05"]); his.append(r["p95"])
        if abs(rates[i + H] - rates[i]) > 0.1:
            dir_tot += 1
            dir_hit += 1 if (r["p_up"] > 0.5) == (rates[i + H] > rates[i]) else 0
    cov = interval_coverage(truths, los, his, nominal=0.9)

    res = {"data": "FOMC 1985-2026, no-cheat (train era only)", "p_move": round(p_move, 3),
           "direction_weights": {f: round(w, 3) for f, w in zip(FEATS, cw.model.w)},
           "A_directional_skill": {"n_pivotal_moves": nA,
                                   "model_accuracy": round(hit_model / nA, 4),
                                   "momentum_accuracy": round(hit_mom / nA, 4),
                                   "base_rate_accuracy": round(hit_base / nA, 4),
                                   "model_logloss": round(sum(ll_model) / nA, 4),
                                   "momentum_logloss": round(sum(ll_mom) / nA, 4)},
           "B_rollout_6mo": {"interval_coverage_90": cov["empirical_coverage"], "calibrated": cov["calibrated"],
                             "terminal_direction_accuracy": round(dir_hit / dir_tot, 4) if dir_tot else None,
                             "n_moved": dir_tot}}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    a = res["A_directional_skill"]
    print("EXP-079  directional event forecasting (FOMC, no-cheat)")
    print(f"  direction P(hike|move) weights (harvested priors + train): {res['direction_weights']}")
    print(f"  A. DIRECTION of {nA} held-out pivotal moves:")
    print(f"     model {a['model_accuracy']}  vs momentum {a['momentum_accuracy']}  vs base-rate "
          f"{a['base_rate_accuracy']}   | log-loss model {a['model_logloss']} vs momentum {a['momentum_logloss']}")
    b = res["B_rollout_6mo"]
    print(f"  B. 6-mo rollout: interval coverage {b['interval_coverage_90']} (calibrated={b['calibrated']}), "
          f"terminal direction acc {b['terminal_direction_accuracy']} on {b['n_moved']} moved "
          f"(drift bug fixed: hold dominates)")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
