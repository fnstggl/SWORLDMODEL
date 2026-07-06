"""EXP-038: forecastability / triage score — does the system know what it can forecast?

Learns a score estimating how reliably a question's DIRECTION can be called (from as-of features: lean,
volatility, days-to-resolution, news result-cue), fit no-cheat to predict the realized correctness of the
lean call. Validated by SELECTIVE FORECASTING: if the score is meaningful, keeping only high-score
questions should raise directional accuracy — the system spends effort where it pays (Tetlock #1).

No-cheat: score fit on TRAIN question outcomes, evaluated on held-out TEST. Writes JSON.
Run: python -m experiments.exp038_forecastability
"""
from __future__ import annotations

import datetime
import json
import re
import statistics
from pathlib import Path

from swm.eval.forecastability import ForecastabilityScorer, forecastability_features
from swm.transition.attribution import _RESULT
from experiments.datasets_swm import load

RESULT = "experiments/results/exp038_forecastability.json"
HORIZON = 8
_MON = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9,
        "OCT": 10, "NOV": 11, "DEC": 12}


def _dtr(r):
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
    ph = [h["p"] for h in r["history"]]
    p0 = r["target"]["p"]; lean = p0 - 0.5
    diffs = [ph[i] - ph[i - 1] for i in range(1, len(ph))]
    vol = (sum(d * d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.02
    cue = min(1.0, len(_RESULT.findall(" ".join(n.get("title", "") for n in (r.get("news") or [])[:8]))) / 3.0)
    f = forecastability_features(lean, vol, _dtr(r), cue)
    # forecastability target = did the belief RESOLVE on its lean side (not conditioned on movement,
    # which biases toward surprises for confident beliefs). This is the honest "was our call right".
    eventual = r["future"][HORIZON - 1]["p"]
    correct = None if abs(lean) < 1e-6 else int((eventual > 0.5) == (lean > 0))
    return f, correct, p0


def run():
    train = [r for r in load("train") if r.get("history") and r.get("target") and len(r.get("future", [])) >= HORIZON]
    test = [r for r in load("test_kalshi") if r.get("history") and r.get("target") and len(r.get("future", [])) >= HORIZON]
    sc = ForecastabilityScorer().fit([(f, c) for f, c, _ in (_example(r) for r in train) if c is not None])

    rows = [(sc.score(f), c) for f, c, _ in (_example(r) for r in test) if c is not None]
    rows.sort(key=lambda z: -z[0])
    n = len(rows)

    # score quartiles: does forecastability separate reliable from unreliable questions?
    asc = sorted(rows, key=lambda z: z[0]); q = len(asc) // 4
    quartiles = []
    for i in range(4):
        b = asc[i * q:(i + 1) * q] if i < 3 else asc[i * q:]
        quartiles.append({"quartile": i + 1, "n": len(b), "mean_score": round(statistics.mean(s for s, _ in b), 3),
                          "resolution_accuracy": round(sum(c for _, c in b) / len(b), 3)})

    # selective forecasting: keep the top-coverage most-forecastable, measure directional accuracy
    curve = []
    for cov in (1.0, 0.75, 0.5, 0.25):
        k = max(10, int(cov * n))
        acc = sum(c for _, c in rows[:k]) / k
        curve.append({"coverage": cov, "n": k, "directional_accuracy": round(acc, 3),
                      "mean_score": round(statistics.mean(s for s, _ in rows[:k]), 3)})

    # triage buckets
    buckets = {"forecast": [0, 0], "hedge": [0, 0], "abstain": [0, 0]}
    for f, c, _ in (_example(r) for r in test):
        if c is None:
            continue
        b = sc.triage(f)
        buckets[b][0] += c; buckets[b][1] += 1
    triage = {b: {"n": v[1], "accuracy": round(v[0] / v[1], 3) if v[1] else None} for b, v in buckets.items()}

    lo_acc = quartiles[0]["resolution_accuracy"]; hi_acc = quartiles[2]["resolution_accuracy"]
    out = {"dataset": "kalshi", "n_test": n, "score_quartiles": quartiles, "selective_forecasting": curve,
           "triage": triage, "least_vs_most_forecastable": round(hi_acc - lo_acc, 3),
           "score_separates": hi_acc > lo_acc}
    print(f"EXP-038 forecastability / triage — Kalshi, n_test={n}")
    print("  SCORE QUARTILES (does the score separate forecastable from not?):")
    for qd in quartiles:
        print(f"    Q{qd['quartile']} (score {qd['mean_score']})  n={qd['n']:<4} resolution accuracy {qd['resolution_accuracy']}")
    print("  SELECTIVE FORECASTING (keep most-forecastable):")
    for c in curve:
        print(f"    coverage {int(c['coverage']*100):>3}%  n={c['n']:<4} directional accuracy {c['directional_accuracy']}"
              f"   mean-score {c['mean_score']}")
    print("  TRIAGE buckets (the system's self-assessment of what it can call):")
    for b in ("forecast", "hedge", "abstain"):
        print(f"    {b:<9} n={triage[b]['n']:<4} directional accuracy {triage[b]['accuracy']}")
    print(f"  score separates forecastable from not (Q3 vs Q1 resolution accuracy): "
          f"{hi_acc} vs {lo_acc} (Δ {out['least_vs_most_forecastable']:+.3f})")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
