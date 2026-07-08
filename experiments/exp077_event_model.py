"""EXP-077: the event model, calibrated on a jumpy real series (FOMC rates) — does it place variance right?

Architecture item #5, first version. Where the outcome moves in discrete jumps at pivotal events (rate
decisions), a persistence forecaster's uncertainty is wrong: it predicts no change and no band, so realized
moves fall outside. The event model calibrates the per-meeting move frequency + size from history and rolls
a calendar of future meetings forward — placing the variance at the events. The test is CALIBRATION: does the
event model's 90% interval actually contain ~90% of realized 12-month-ahead rates (no-cheat: calibrated on
the train era only), where persistence's point forecast contains ~0%?

Run: python -m experiments.exp077_event_model
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.simulation.event_model import EventModel, interval_coverage

FOMC = "experiments/results/exp071/fomc_macro.json"
RESULT = "experiments/results/exp077_event_model.json"
H = 12


def run() -> dict:
    data = json.loads(Path(FOMC).read_text())
    rates = [d["rate"] for d in data]
    moves = [rates[i] - rates[i - 1] for i in range(1, len(rates))]
    cut = int(0.6 * len(rates))
    em = EventModel.calibrate(moves[:cut], threshold=0.05, calendar=list(range(1, H + 1)))

    truths, los, his, widths = [], [], [], []
    persist_hits = 0
    for i in range(cut, len(rates) - H):
        start, realized = rates[i], rates[i + H]
        r = em.rollout(start, H, n=3000, seed=i)
        truths.append(realized); los.append(r["p05"]); his.append(r["p95"])
        widths.append(r["p95"] - r["p05"])
        persist_hits += 1 if abs(realized - start) < 1e-9 else 0        # persistence point 'interval'

    cov = interval_coverage(truths, los, his, nominal=0.9)              # p05..p95 = a 90% interval
    res = {"data": "FOMC monthly rate 1985-2026 (calibrated on train era only)", "horizon_months": H,
           "n_forecasts": len(truths), "event_model": {"p_active": round(em.p_active, 3),
           "impact_mean": round(em.impact_mean, 4), "impact_sd": round(em.impact_sd, 4)},
           "event_model_90pct_coverage": cov["empirical_coverage"], "calibrated": cov["calibrated"],
           "mean_interval_width": round(sum(widths) / len(widths), 3) if widths else None,
           "persistence_point_coverage": round(persist_hits / len(truths), 4) if truths else None}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-077  event model calibrated on FOMC rate jumps (no-cheat, horizon 12mo)")
    print(f"  calibrated: P(move)={em.p_active:.2f}, impact {em.impact_mean:+.3f}±{em.impact_sd:.3f} per month")
    print(f"  event-model 90% interval coverage: {cov['empirical_coverage']:.2f} (nominal 0.90) -> "
          f"calibrated={cov['calibrated']}  | mean width {res['mean_interval_width']}")
    print(f"  persistence point forecast coverage: {res['persistence_point_coverage']} "
          f"(a no-band forecast misses every real move)")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
