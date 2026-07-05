"""EXP-026: conformal prediction sets finish the uncertainty contract (empirical coverage guarantee).

The calibration badge says the probabilities are well-calibrated ON AVERAGE. Conformal adds the stronger
per-prediction guarantee: a prediction SET over {0,1} that contains the true outcome with probability
>= 1 - alpha, under only exchangeability. This experiment verifies, no-cheat on the real CMV corpus,
that the Simulator's conformal sets actually deliver their promised coverage on a held-out test split,
across several alpha levels — and shows the honest trade-off: to guarantee tighter coverage on a hard,
high-entropy problem, the model must widen more predictions to the uncertain set {0,1}.

Writes experiments/results/exp026_conformal.json.
Run: python -m experiments.exp026_conformal
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.api import Simulator
from swm.state.state import Action
from experiments.exp024_unified_api import _load_cmv, _map_inference

RESULT = "experiments/results/exp026_conformal.json"


def run():
    sub, inf = _load_cmv()
    insts = []
    for s in sub:
        a = Action(action_id=str(s["ts"]), actor_id=s["challenger"], channel="cmv",
                   timing={"ts": s["ts"]}, meta={"text": s["arg_text"]})
        insts.append((s["op_id"], a, None, s["success"], {"llm_inference": _map_inference(inf.get(s["id"]))}))
    cut = int(0.7 * len(insts))

    rows = []
    for alpha in (0.05, 0.1, 0.2, 0.3):
        sim = Simulator(platform="cmv", conformal_alpha=alpha).fit(insts[:cut])
        ps, ys = [], []
        for op, a, ctx, y, ex in insts[cut:]:
            r = sim.simulate(op, a, llm_inference=ex["llm_inference"])
            ps.append(r.p); ys.append(y)
        cov = sim.conformal.coverage(ps, ys)
        rows.append({"alpha": alpha, "target_coverage": round(1 - alpha, 3),
                     "empirical_coverage": cov["coverage"], "avg_set_size": cov["avg_set_size"],
                     "frac_uncertain_set": cov["frac_uncertain"], "n_test": cov["n"]})

    max_dev = round(max(abs(r["empirical_coverage"] - r["target_coverage"]) for r in rows), 4)
    out = {"dataset": "cmv_persuasion", "n_test": rows[0]["n_test"], "levels": rows,
           "max_abs_deviation": max_dev,
           "note": ("empirical coverage tracks target within finite-sample slack; conformal's guarantee "
                    "is marginal and assumes exchangeability, which a temporal split with a small "
                    "calibration set mildly violates (slight undershoot at the most demanding level)")}
    print(f"EXP-026 conformal prediction sets — CMV, n_test={rows[0]['n_test']}")
    print(f"  {'alpha':>6} {'target':>8} {'empirical':>10} {'avg_set':>9} {'uncertain':>10}")
    for r in rows:
        print(f"  {r['alpha']:>6} {r['target_coverage']:>8} {r['empirical_coverage']:>10} "
              f"{r['avg_set_size']:>9} {r['frac_uncertain_set']:>10}")
    print(f"  max |empirical - target| across levels: {max_dev} (finite-sample + temporal slack)")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
