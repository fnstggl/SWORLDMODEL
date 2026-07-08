"""EXP-087: is the INTERNET enough? — DeepSeek + web as the GENERAL grounding engine.

A general world model can't depend on live feeds — most variables have no dedicated API. The universal
grounder is therefore DeepSeek + web retrieval, and the honest question is: how ACCURATE and how CALIBRATED is
it, across every domain? This grounds a broad set of variables whose true values are KNOWN (stable facts +
recent well-established figures spanning demography, geography, economics, science, sports, politics, biology)
through the `general_router` (NO structured feeds), and measures:

  - ACCURACY: relative error vs the known truth; fraction within 5% / 10%.
  - CALIBRATION: do the grounded CIs cover truth at the nominal rate? Fit the extractor's ci_multiplier on the
    REAL (value, sd, truth) triples — the honest calibration EXP-085 only did against a mock.

Live smoke (needs DEEPSEEK_API_KEY + network; skips gracefully otherwise). Values move with the world; the
accuracy/calibration summary is the deliverable.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp087_general_grounding
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RESULT = "experiments/results/exp087_general_grounding.json"

# (domain, variable, known truth, question-context). Mix of EXACT constants (should ground tight) and ESTIMATES
# (populations, GDP — should ground with an honestly wider CI). Truths are stable/well-established references.
TRUTHS = [
    ("geography", "number of US states", 50, "US civics"),
    ("geography", "number of continents on Earth", 7, "geography"),
    ("geography", "number of member countries in the European Union", 27, "EU"),
    ("science", "boiling point of water at sea level in Celsius", 100, "physics"),
    ("science", "freezing point of water in Celsius", 0, "physics"),
    ("science", "speed of light in kilometers per second", 299792, "physics"),
    ("science", "approximate age of the universe in billions of years", 13.8, "cosmology"),
    ("biology", "number of bones in the adult human body", 206, "anatomy"),
    ("biology", "number of teeth in a typical adult human", 32, "anatomy"),
    ("politics", "number of seats in the US Senate", 100, "US government"),
    ("politics", "number of justices on the US Supreme Court", 9, "US government"),
    ("sports", "number of teams in the NBA", 30, "basketball"),
    ("sports", "number of players on the field per soccer team", 11, "soccer"),
    ("economics", "US federal minimum wage in dollars per hour", 7.25, "US labor"),
    ("demography", "population of the United States (millions)", 335, "US demographics"),
    ("demography", "population of the world (billions)", 8.1, "world population"),
    ("demography", "population of India (billions)", 1.43, "India"),
    ("demography", "population of Japan (millions)", 124, "Japan"),
    ("demography", "US life expectancy at birth in years", 77.5, "US health"),
    ("economics", "US annual GDP in trillions of dollars", 27, "US economy"),
    ("geography", "height of Mount Everest in meters", 8849, "mountains"),
    ("history", "the year Amazon (the company) was founded", 1994, "tech history"),
]
TOL = {"exact": 0.02, "estimate": 0.12}      # relative-error bands used for the accuracy headline


def _cov(triples, mult, z=1.645):
    return sum(1 for v, sd, t in triples if abs(v - t) <= z * sd * mult) / len(triples) if triples else 0.0


def run() -> dict:
    from swm.api.live_grounding import general_router, json_llm
    if json_llm() is None:
        print("EXP-087  general grounding — SKIPPED (no DEEPSEEK_API_KEY / HF_TOKEN configured)")
        return {"skipped": "no LLM backend"}

    router = general_router()                                     # LLM + web only — NO live feeds
    triples, rows = [], []
    for dom, var, truth, q in TRUTHS:
        try:
            gv = router.ground(var, question=q)
        except Exception as e:
            gv = None
            print(f"    ({var}: {str(e)[:50]})")
        if gv is None:
            rows.append({"domain": dom, "variable": var, "truth": truth, "grounded": False})
            continue
        rel = abs(gv.value - truth) / max(abs(truth), 1e-9)
        triples.append((gv.value, gv.sd, float(truth)))
        rows.append({"domain": dom, "variable": var, "truth": truth, "value": round(gv.value, 4),
                     "sd": round(gv.sd, 4), "rel_err": round(rel, 4), "source": gv.source})

    n = len(triples)
    rels = [r["rel_err"] for r in rows if "rel_err" in r]
    rels_sorted = sorted(rels)
    med_rel = rels_sorted[len(rels_sorted) // 2] if rels_sorted else None
    within5 = sum(1 for r in rels if r <= 0.05) / len(rels) if rels else 0.0
    within10 = sum(1 for r in rels if r <= 0.10) / len(rels) if rels else 0.0
    # honest CI calibration on the REAL triples: fit the ci_multiplier to hit 90% coverage
    grid = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    cov_before = round(_cov(triples, 1.0), 3)
    best_m = min(grid, key=lambda m: abs(_cov(triples, m) - 0.9))
    cov_after = round(_cov(triples, best_m), 3)

    res = {"n": n, "grounded": n, "coverage_of_probes": round(n / len(TRUTHS), 3),
           "median_rel_err": round(med_rel, 4) if med_rel is not None else None,
           "within_5pct": round(within5, 3), "within_10pct": round(within10, 3),
           "ci_calibration": {"coverage_before": cov_before, "ci_multiplier": best_m, "coverage_after": cov_after},
           "rows": rows, "note": "live — values move; general engine = DeepSeek + web, no live feeds"}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-087  DeepSeek + web as the GENERAL grounding engine (no live feeds) — accuracy & calibration")
    print(f"  grounded {n}/{len(TRUTHS)} known-truth variables across {len(set(d for d,_,_,_ in TRUTHS))} domains")
    print(f"  ACCURACY: median relative error {res['median_rel_err']}, within-5% {within5*100:.0f}%, "
          f"within-10% {within10*100:.0f}%")
    cc = res["ci_calibration"]
    print(f"  CALIBRATION: CI coverage {cc['coverage_before']} -> {cc['coverage_after']} "
          f"(nominal 0.9) at ci_multiplier {cc['ci_multiplier']}")
    worst = sorted([r for r in rows if "rel_err" in r], key=lambda r: -r["rel_err"])[:4]
    print("  largest errors:")
    for r in worst:
        print(f"    {r['variable'][:44]:44s} grounded {r['value']} vs truth {r['truth']} "
              f"(rel {r['rel_err']})")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
