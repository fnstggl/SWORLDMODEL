"""Upworthy Research Archive — randomized headline A/B tests (the interventional benchmark).

32,487 randomized experiments (Matias et al. 2021, Scientific Data; OSF jd64p, CC BY 4.0). Each row is a
package/arm — a headline shown to a random slice of visitors — with impressions and clicks. Arms sharing a
`clickability_test_id` are the same experiment. Because the arms were RANDOMIZED, the observed CTR
difference between arms IS the causal effect of the headline (unconfounded) — so "which headline wins" is a
genuine interventional question, and picking one is a real `do(x)`. This is the dataset that lets us score
"what happens if I do X," not just marginal prediction (SIMULATION_AUDIT KPI-A).

Contamination caveat (public since 2021): the archive flags an early randomization issue; we keep only
arms with enough impressions for a reliable CTR and treat the whole set as a MECHANISM benchmark (does the
model pick the causally-better headline), not a leakage-free skill number.

Download once (14 MB; gitignored):
  curl -sSL -o data/upworthy_exploratory.csv https://osf.io/download/3vqmp/
A parsed, test-grouped cache is committed under experiments/results/exp054_upworthy/ for reproducibility.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

CSV = "data/upworthy_exploratory.csv"
CACHE = "experiments/results/exp054_upworthy/upworthy_parsed.json"
MIN_IMPRESSIONS = 1000        # per arm, for a reliable CTR
MIN_ARMS = 2


def parse():
    tests = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                imp = int(row["impressions"]); clk = int(row["clicks"])
            except (ValueError, KeyError):
                continue
            tid = row.get("clickability_test_id"); head = (row.get("headline") or "").strip()
            if not tid or not head or imp < MIN_IMPRESSIONS:
                continue
            tests.setdefault(tid, []).append({"headline": head, "impressions": imp, "clicks": clk,
                                              "ctr": clk / imp})
    # keep tests with >=2 arms that actually differ in CTR (a real choice to get right)
    out = []
    for tid, arms in tests.items():
        if len(arms) < MIN_ARMS:
            continue
        ctrs = [a["ctr"] for a in arms]
        if max(ctrs) - min(ctrs) < 1e-6:
            continue
        out.append({"test_id": tid, "arms": arms})
    return out


def build_cache(subsample=None, seed=0):
    tests = parse()
    if subsample and len(tests) > subsample:
        import random
        rng = random.Random(seed); rng.shuffle(tests); tests = tests[:subsample]
    Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    Path(CACHE).write_text(json.dumps(tests))
    print(f"wrote {len(tests)} A/B tests ({sum(len(t['arms']) for t in tests)} arms) -> {CACHE}")
    return tests


def load():
    if Path(CACHE).exists():
        return json.loads(Path(CACHE).read_text())
    return build_cache()


if __name__ == "__main__":
    build_cache(subsample=6000)
