"""Phase 9 — real-outcome validation with strong baselines + paired bootstrap CIs (final hardening, Part 5).

Two real-outcome tasks, each vs strong simple baselines, with paired bootstrap confidence intervals. We do NOT
claim lift because the output changed — we claim it only when a paired CI excludes zero in the favorable
direction, and we preserve honest nulls.

POPULATION (GSS): predict a held-out attitude margin under an education-biased sample.
  arms: poststratified (structured) · naive (biased sample mean) · homogeneous (uniform segment weights) ·
        prior-only (0.5). Metric = |estimate − true margin|. Paired bootstrap CI of (naive − poststrat).

NETWORK (Enron temporal): predict post-cutoff email edges from pre-cutoff communication.
  arms: exposure posterior (structured) · frequency baseline · prior-only. Metric = Brier. Paired bootstrap
  CI of (baseline − posterior).
"""
from __future__ import annotations

import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from swm.world_model_v2.phase3_posterior import ExposureObservation, infer_edge_posterior_exposure
from swm.world_model_v2.phase9_population import infer_segment_rates

OUT = Path("experiments/results/phase9")
GSS = "experiments/results/exp045_gss/gss_parsed.json.gz"
ENRON = OUT / "enron_comm_edges.json"
LEVELS = ["lt_highschool", "highschool", "junior_college", "bachelor", "graduate"]
BIAS = {"lt_highschool": 0.25, "highschool": 0.5, "junior_college": 0.7, "bachelor": 0.9, "graduate": 1.0}


def _boot_ci(pairs, stat, n_boot=2000, seed=0):
    """Paired bootstrap CI of a statistic over `pairs` (list). Returns (mean, lo, hi)."""
    rng = random.Random(seed)
    vals = []
    m = len(pairs)
    for _ in range(n_boot):
        samp = [pairs[rng.randrange(m)] for _ in range(m)]
        vals.append(stat(samp))
    vals.sort()
    return round(stat(pairs), 5), round(vals[int(0.025 * n_boot)], 5), round(vals[int(0.975 * n_boot)], 5)


def population_task(seed=0):
    recs = [r for r in json.load(gzip.open(GSS)) if r.get("demo", {}).get("degree") in LEVELS]
    rng = random.Random(seed)
    pairs = []                                              # (err_naive, err_ps, err_homog, err_prior)
    for item in ["gunlaw", "premarsx"]:
        have = [r for r in recs if item in r.get("answers", {})]
        true_counts = defaultdict(int)
        for r in have:
            true_counts[r["demo"]["degree"]] += 1
        tw = {s: true_counts[s] / len(have) for s in LEVELS}
        true_margin = sum(r["answers"][item] for r in have) / len(have)
        for sd in range(12):
            rr = random.Random(seed * 100 + sd)
            sample = [r for r in have if rr.random() < BIAS[r["demo"]["degree"]]]
            if len(sample) < 50:
                continue
            naive = sum(r["answers"][item] for r in sample) / len(sample)
            sc = defaultdict(lambda: [0, 0])
            for r in sample:
                sc[r["demo"]["degree"]][0] += r["answers"][item]
                sc[r["demo"]["degree"]][1] += 1
            rates = infer_segment_rates({s: tuple(sc[s]) for s in LEVELS if sc[s][1] > 0})
            ps = sum(tw[s] * rates[s].mean for s in rates)
            homog = sum((1 / len(LEVELS)) * rates[s].mean for s in rates)   # uniform weights
            pairs.append((abs(naive - true_margin), abs(ps - true_margin),
                          abs(homog - true_margin), abs(0.5 - true_margin)))
    mean_naive = sum(p[0] for p in pairs) / len(pairs)
    mean_ps = sum(p[1] for p in pairs) / len(pairs)
    mean_homog = sum(p[2] for p in pairs) / len(pairs)
    mean_prior = sum(p[3] for p in pairs) / len(pairs)
    diff_naive = _boot_ci([(p[0], p[1]) for p in pairs], lambda s: sum(a - b for a, b in s) / len(s))
    diff_homog = _boot_ci([(p[2], p[1]) for p in pairs], lambda s: sum(a - b for a, b in s) / len(s))
    return {"n_evals": len(pairs), "mean_err": {"naive": round(mean_naive, 4), "poststrat": round(mean_ps, 4),
            "homogeneous": round(mean_homog, 4), "prior_only": round(mean_prior, 4)},
            "paired_naive_minus_poststrat": {"mean": diff_naive[0], "ci95": [diff_naive[1], diff_naive[2]]},
            "paired_homogeneous_minus_poststrat": {"mean": diff_homog[0], "ci95": [diff_homog[1], diff_homog[2]]},
            "poststrat_beats_naive_sig": diff_naive[1] > 0, "poststrat_beats_homogeneous_sig": diff_homog[1] > 0}


def network_task(seed=0):
    data = json.loads(ENRON.read_text())
    edges = data["edges"]
    all_ts = sorted(t for e in edges for t in e["ts"])
    cutoff = all_ts[len(all_ts) // 2]
    nodes = data["nodes"]
    cnt, out, fut = defaultdict(int), defaultdict(int), set()
    for e in edges:
        a, b = e["src"], e["dst"]
        for t in e["ts"]:
            if t < cutoff:
                cnt[(a, b)] += 1
                out[a] += 1
            else:
                fut.add((a, b))
    active = [a for a in nodes if out[a] >= 3]
    pairs = []                                              # (brier_posterior, brier_freq, brier_prior)
    for a in active:
        for b in nodes:
            if a == b:
                continue
            y = 1 if (a, b) in fut else 0
            N = min(60, out[a])
            k = min(cnt[(a, b)], N)
            post = infer_edge_posterior_exposure(a, b, "communication",
                                                 [ExposureObservation("repeated_interaction", N, k, 0.9)],
                                                 prior_p=0.05).posterior_p
            freq = min(1.0, k / 5.0)
            pairs.append(((post - y) ** 2, (freq - y) ** 2, (0.05 - y) ** 2))
    mb = lambda i: sum(p[i] for p in pairs) / len(pairs)
    diff_freq = _boot_ci([(p[1], p[0]) for p in pairs], lambda s: sum(a - b for a, b in s) / len(s))
    diff_prior = _boot_ci([(p[2], p[0]) for p in pairs], lambda s: sum(a - b for a, b in s) / len(s))
    return {"n_pairs": len(pairs), "mean_brier": {"posterior": round(mb(0), 4), "frequency": round(mb(1), 4),
            "prior_only": round(mb(2), 4)},
            "paired_frequency_minus_posterior": {"mean": diff_freq[0], "ci95": [diff_freq[1], diff_freq[2]]},
            "paired_prior_minus_posterior": {"mean": diff_prior[0], "ci95": [diff_prior[1], diff_prior[2]]},
            "posterior_beats_frequency_sig": diff_freq[1] > 0,
            "posterior_beats_prior_sig": diff_prior[1] > 0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = population_task()
    net = network_task()
    report = {"population_real_outcome": pop, "network_real_outcome": net,
              "verdict": {
                  "population_lift": ("poststratification significantly beats naive AND homogeneous baselines"
                                      if pop["poststrat_beats_naive_sig"] and pop["poststrat_beats_homogeneous_sig"]
                                      else "population lift not significant on all baselines"),
                  "network_lift": ("edge posterior significantly beats the frequency baseline"
                                   if net["posterior_beats_frequency_sig"] else
                                   "edge posterior does NOT significantly beat the frequency baseline (honest null); "
                                   "it does beat prior-only" if net["posterior_beats_prior_sig"] else
                                   "edge posterior does not significantly beat baselines")}}
    (OUT / "real_outcome_validation.json").write_text(json.dumps(report, indent=2))
    print("REAL-OUTCOME VALIDATION (paired bootstrap CIs)")
    print("  POPULATION:", json.dumps(pop["mean_err"]))
    print("   naive−poststrat:", pop["paired_naive_minus_poststrat"],
          "| homog−poststrat:", pop["paired_homogeneous_minus_poststrat"])
    print("  NETWORK Brier:", json.dumps(net["mean_brier"]))
    print("   freq−posterior:", net["paired_frequency_minus_posterior"],
          "| prior−posterior:", net["paired_prior_minus_posterior"])
    print("  VERDICT:", json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
