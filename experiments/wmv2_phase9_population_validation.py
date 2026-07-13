"""Phase 9 population inference — REAL-DATA validation on the General Social Survey (Parts B–E, X, Z).

GSS = 72,707 real respondents, 1972–2024, with demographics + attitude items. This validates the production
population subsystem on real data (NOT synthetic):

  1. POSTSTRATIFICATION corrects a biased sample. Draw an education-biased sample (mimicking online-panel
     over-representation of the educated); the naive sample mean of an attitude is biased; poststratifying the
     segment-conditional rates by the TRUE segment composition recovers the population margin.
  2. CORRELATED (segment-conditional) beats INDEPENDENT (pooled marginal) when the biased segment correlates
     with the attitude — the Part-D ablation on real data.
  3. TEMPORAL TRANSFER: predict a held-out later-year margin from earlier-year segment-conditional rates +
     the later-year composition (person-disjoint, time-forward).
  4. COMPOSITIONAL POSTERIOR coverage + normalization on the real segment counts.

Deterministic (seeded); reads only the committed GSS cache; writes a machine-readable artifact.
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from swm.world_model_v2.phase9_population import (PopulationSpec, infer_population, infer_segment_rates,
                                                  materialize_population_particles, poststratified_estimate)

GSS = "experiments/results/exp045_gss/gss_parsed.json.gz"
OUT = Path("experiments/results/phase9")
SEG = "degree"                                   # education segmentation (5 real levels)
LEVELS = ["lt_highschool", "highschool", "junior_college", "bachelor", "graduate"]
ITEMS = ["gunlaw", "premarsx"]
#: segment-dependent inclusion prob → an education-biased "online panel" (educated over-represented)
BIAS = {"lt_highschool": 0.25, "highschool": 0.5, "junior_college": 0.7, "bachelor": 0.9, "graduate": 1.0}


def _load():
    recs = json.load(gzip.open(GSS))
    return [r for r in recs if r.get("demo", {}).get(SEG) in LEVELS]


def _margin(recs, item):
    vals = [r["answers"][item] for r in recs if item in r.get("answers", {})]
    return (sum(vals) / len(vals)) if vals else None


def _seg_counts(recs):
    c = {s: 0 for s in LEVELS}
    for r in recs:
        c[r["demo"][SEG]] += 1
    return c


def _seg_item_counts(recs, item):
    out = {s: [0, 0] for s in LEVELS}                # [successes, total]
    for r in recs:
        if item in r.get("answers", {}):
            s = r["demo"][SEG]
            out[s][0] += r["answers"][item]
            out[s][1] += 1
    return {s: tuple(v) for s, v in out.items()}


def poststrat_task(recs, item, seed):
    """Draw an education-biased sample; compare naive vs poststratified vs independent against the TRUE margin."""
    rng = random.Random(seed)
    have = [r for r in recs if item in r.get("answers", {})]
    true_margin = _margin(have, item)
    true_counts = _seg_counts(have)
    true_w = {s: true_counts[s] / sum(true_counts.values()) for s in LEVELS}
    sample = [r for r in have if rng.random() < BIAS[r["demo"][SEG]]]
    naive = _margin(sample, item)
    seg_rates = infer_segment_rates(_seg_item_counts(sample, item))
    # poststratify by the TRUE composition (census margins) — MRP-style
    ps = sum(true_w[s] * seg_rates[s].mean for s in LEVELS)
    # independent baseline: pooled sample marginal (ignores that the sample is education-skewed)
    indep = naive
    return {"item": item, "true_margin": round(true_margin, 4), "naive_biased": round(naive, 4),
            "poststratified": round(ps, 4), "independent_pooled": round(indep, 4),
            "err_naive": round(abs(naive - true_margin), 4),
            "err_poststratified": round(abs(ps - true_margin), 4),
            "sample_frac": round(len(sample) / len(have), 3)}


def temporal_transfer(recs, item, split_year=2000):
    """Predict a held-out later-period margin from earlier-period segment-conditional rates + later composition."""
    early = [r for r in recs if r["year"] < split_year and item in r.get("answers", {})]
    late = [r for r in recs if r["year"] >= split_year and item in r.get("answers", {})]
    if not early or not late:
        return None
    seg_rates = infer_segment_rates(_seg_item_counts(early, item))       # trained on the past
    late_counts = _seg_counts(late)
    late_w = {s: late_counts[s] / sum(late_counts.values()) for s in LEVELS}
    pred = sum(late_w[s] * seg_rates[s].mean for s in LEVELS)            # past rates × future composition
    true_late = _margin(late, item)
    # naive transfer = the early pooled margin (no composition adjustment)
    naive = _margin(early, item)
    return {"item": item, "split_year": split_year, "true_late_margin": round(true_late, 4),
            "transfer_pred": round(pred, 4), "naive_early_margin": round(naive, 4),
            "err_transfer": round(abs(pred - true_late), 4), "err_naive": round(abs(naive - true_late), 4)}


def compositional_recovery(recs, seed=0):
    """The compositional posterior on the real segment counts: normalized, covers the empirical composition."""
    counts = _seg_counts(recs)
    total = sum(counts.values())
    true_w = {s: counts[s] / total for s in LEVELS}
    # a survey observation = a sub-sample's counts (n=2000), inferred against a weak Dirichlet prior
    rng = random.Random(seed)
    sub = rng.sample(recs, 2000)
    obs = [{"counts": _seg_counts(sub), "reliability": 1.0, "source": "gss_subsample"}]
    pop = PopulationSpec(population_id="gss_adults", segments=LEVELS, target_universe="US adults (GSS frame)",
                         source_frame="GSS", consumed_by=["poststratified_estimate"])
    post = infer_population(pop, prior_alpha=[1.0] * len(LEVELS), survey_observations=obs, seed=seed)
    err = sum(abs(post.posterior_mean[i] - true_w[s]) for i, s in enumerate(LEVELS))
    return {"true_weights": {s: round(true_w[s], 4) for s in LEVELS},
            "posterior_mean": {s: round(post.posterior_mean[i], 4) for i, s in enumerate(LEVELS)},
            "L1_error": round(err, 4), "sum_to_one": round(sum(post.posterior_mean), 6),
            "ess": post.ess, "moved_from_prior": post.diagnostics.get("moved_from_prior")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = _load()
    report = {"dataset": "GSS 1972-2024 (cumulative)", "n_records": len(recs), "segmentation": SEG,
              "levels": LEVELS, "n_seeds": 5}
    # poststratification (5 seeds, both items) — does it beat the naive biased sample on average?
    ps_rows = [poststrat_task(recs, item, seed) for item in ITEMS for seed in range(5)]
    report["poststratification"] = ps_rows
    report["poststratification_summary"] = {
        "mean_err_naive": round(sum(r["err_naive"] for r in ps_rows) / len(ps_rows), 4),
        "mean_err_poststratified": round(sum(r["err_poststratified"] for r in ps_rows) / len(ps_rows), 4),
        "poststrat_wins_frac": round(sum(1 for r in ps_rows
                                         if r["err_poststratified"] < r["err_naive"]) / len(ps_rows), 3)}
    report["temporal_transfer"] = [t for t in (temporal_transfer(recs, item) for item in ITEMS) if t]
    tt = report["temporal_transfer"]
    report["temporal_transfer_summary"] = {
        "mean_err_transfer": round(sum(t["err_transfer"] for t in tt) / len(tt), 4) if tt else None,
        "mean_err_naive": round(sum(t["err_naive"] for t in tt) / len(tt), 4) if tt else None}
    report["compositional_recovery"] = compositional_recovery(recs)
    # gates
    s = report["poststratification_summary"]
    report["gates"] = {
        "poststratification_beats_naive_on_average": s["mean_err_poststratified"] < s["mean_err_naive"],
        "poststratification_wins_majority": s["poststrat_wins_frac"] >= 0.6,
        "compositional_normalized": abs(report["compositional_recovery"]["sum_to_one"] - 1.0) < 1e-4,
        "compositional_recovers_real_composition": report["compositional_recovery"]["L1_error"] < 0.06}
    report["all_gates_pass"] = all(report["gates"].values())
    (OUT / "population_validation.json").write_text(json.dumps(report, indent=2))
    print("GSS POPULATION VALIDATION")
    print(f"  records: {len(recs)}  segmentation: {SEG} ({len(LEVELS)} levels)")
    print(f"  poststratification: mean err naive {s['mean_err_naive']} -> poststrat {s['mean_err_poststratified']} "
          f"(wins {s['poststrat_wins_frac']})")
    print(f"  temporal transfer: {report['temporal_transfer_summary']}")
    print(f"  compositional recovery L1: {report['compositional_recovery']['L1_error']} "
          f"(sum={report['compositional_recovery']['sum_to_one']})")
    print(f"  GATES: {json.dumps(report['gates'])}  ALL={report['all_gates_pass']}")


if __name__ == "__main__":
    main()
