"""EXP-037: the question->driver->inferred-lean engine — does mapping a question's drivers work?

The front door to "any question": apply the VariableMap architecture to the QUESTION — infer the DRIVERS
acting on its resolution (base-rate first, Fermi-decomposed, balanced), aggregate them into P(outcome) in
log-odds (Tetlock's incremental Bayesian update), and read off the direction (EXP-036). For a question
with no market, this inferred lean IS the forecast.

Validation (no-cheat, leakage-guarded): an 8-agent swarm inferred each question's drivers from the
question + as-of news ONLY — it was NOT given the market price or the future. We then ask:
  (1) MECHANISM: does the driver-inferred P recover the MARKET's probability (correlation)? If yes,
      mapping drivers produces market-consistent probabilities — and the agent couldn't parrot a price it
      never saw.
  (2) DIRECTION vs market: does sign(inferred−0.5) agree with the market's lean?
  (3) DIRECTION vs eventual resolution: does it predict where the belief ends up? (leakage-caveated: the
      agent may partially recall dated outcomes — the honest limit of backtesting LLM forecasting on
      pre-cutoff data; the market-consistency test (1) is the cleaner signal.)

kappa / evidence_shrink are tuned on one half of the questions and evaluated on the other (no leakage of
the aggregation params). Writes JSON. Run: python -m experiments.exp037_question_engine
"""
from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path

from swm.api.question_engine import QuestionEngine, drivers_from_payload

RESULT = "experiments/results/exp037_question_engine.json"


def _load():
    drivers = {}
    for fp in glob.glob("data/qe_drivers_[0-9]*.json") or glob.glob("experiments/results/exp037_qe/qe_drivers.json"):
        for r in json.loads(Path(fp).read_text()):
            if isinstance(r, dict) and "id" in r:
                drivers[r["id"]] = (float(r.get("base_rate", 0.5)), drivers_from_payload(r.get("drivers")))
    tp = Path("data/qe_truth.json")
    truth = json.loads(tp.read_text()) if tp.exists() else json.loads(
        Path("experiments/results/exp037_qe/qe_truth.json").read_text())
    return drivers, truth


def _corr(a, b):
    if len(a) < 3 or statistics.pstdev(a) < 1e-9 or statistics.pstdev(b) < 1e-9:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (statistics.pstdev(a) * statistics.pstdev(b) * len(a))


def _eval(ids, drivers, truth, eng):
    inf, mkt, evt = [], [], []
    da_mkt = [0, 0]; da_evt = [0, 0]
    for qid in ids:
        base, drv = drivers[qid]
        p = eng.aggregate(base, drv)
        m = truth[qid]["market_p"]; e = truth[qid]["eventual_p"]
        inf.append(p); mkt.append(m); evt.append(e)
        if abs(m - 0.5) > 0.05:
            da_mkt[0] += int((p > 0.5) == (m > 0.5)); da_mkt[1] += 1
        if abs(e - 0.5) > 0.05:
            da_evt[0] += int((p > 0.5) == (e > 0.5)); da_evt[1] += 1
    return {"corr_market": round(_corr(inf, mkt), 3), "corr_eventual": round(_corr(inf, evt), 3),
            "mae_market": round(statistics.mean(abs(a - b) for a, b in zip(inf, mkt)), 4),
            "da_vs_market": round(da_mkt[0] / max(1, da_mkt[1]), 3),
            "da_vs_eventual": round(da_evt[0] / max(1, da_evt[1]), 3), "n": len(ids)}


def run():
    drivers, truth = _load()
    ids = [q for q in drivers if q in truth]
    ids.sort()
    half = len(ids) // 2
    tune_ids, test_ids = ids[:half], ids[half:]

    # tune kappa + evidence_shrink on the first half (minimize MAE to the market probability)
    best, best_e = (1.2, 0.7), 1e9
    for kappa in (0.8, 1.2, 1.6, 2.2):
        for shr in (0.5, 0.7, 0.9, 1.0):
            m = _eval(tune_ids, drivers, truth, QuestionEngine(kappa=kappa, evidence_shrink=shr))
            if m["mae_market"] < best_e:
                best_e, best = m["mae_market"], (kappa, shr)
    eng = QuestionEngine(kappa=best[0], evidence_shrink=best[1])
    res = _eval(test_ids, drivers, truth, eng)

    # baseline: base-rate-only (no drivers) — isolates the DRIVERS' contribution
    base_only = _eval(test_ids, {q: (drivers[q][0], []) for q in test_ids}, truth, eng)
    out = {"n_questions": len(ids), "n_test": len(test_ids), "kappa": best[0], "evidence_shrink": best[1],
           "engine": res, "base_rate_only": base_only,
           "drivers_add_corr": round(res["corr_market"] - base_only["corr_market"], 3)}
    print(f"EXP-037 question->driver engine — {len(ids)} questions ({len(test_ids)} held-out for eval), "
          f"kappa={best[0]} shrink={best[1]}")
    print(f"  MECHANISM — driver-inferred P vs MARKET probability (agent never saw the price):")
    print(f"    correlation {res['corr_market']}   MAE {res['mae_market']}   "
          f"(base-rate-only corr {base_only['corr_market']}; drivers add {out['drivers_add_corr']:+.3f})")
    print(f"  DIRECTION — inferred lean vs market lean: {res['da_vs_market']}  "
          f"(base-rate-only {base_only['da_vs_market']})")
    print(f"  DIRECTION — inferred lean vs eventual resolution: {res['da_vs_eventual']} "
          f"(LEAKAGE-CAVEATED — agent may recall dated outcomes)")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
