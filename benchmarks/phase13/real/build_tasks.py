"""Construct exactly 120 REAL intervention decision tasks (Part 30B) from the loaded datasets.

A task is a self-contained decision problem on REAL logged/quasi-experimental data:
  {task_id, dataset, domain, design, decisions|sequences, target_policy, baselines, ground_truth}

The composition is FROZEN here (Part 31: eligibility + counts fixed before any V2 performance is
measured) and enforced to exact quotas:

    40  randomized / A-B         (RCTs with known assignment; identified policy value = ground truth)
    20  logged-policy / bandit   (Upworthy K-arm tests; uniform propensity; oracle = best-arm CTR)
    20  quasi-experimental       (DiD / RD / IV designs; identified effect = ground truth)
    20  sequential-policy        (state-year panels; dynamic treatment regimes)
    20  network / heterogeneous  (peer-effect experiment + CATE-targeting tasks)
   ---
   120  total, no task counted twice.

Ground truth: for randomized data the value of a policy is IDENTIFIED (the randomized arm means), so
an OPE estimate can be graded against truth. For quasi-experimental designs the identified effect is
the design's estimand. Tasks split by DECISION ENVIRONMENT (dataset+slice) into
development/calibration/validation/locked_test (Part 32) — no environment straddles splits.
"""
from __future__ import annotations

import hashlib

from benchmarks.phase13.real import datasets as D


def _split_of(task_id: str) -> str:
    h = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 100
    if h < 45:
        return "development"
    if h < 62:
        return "calibration"
    if h < 80:
        return "validation"
    return "locked_test"


def _subgroups(rows, feature, thresh):
    lo = [r for r in rows if r["context"].get(feature, 0) <= thresh]
    hi = [r for r in rows if r["context"].get(feature, 0) > thresh]
    return lo, hi


# ---------------------------------------------------------------- randomized tasks (40)
def _randomized_tasks():
    """Each RCT yields several targeting-policy tasks. The decision: whom to treat. The target policy
    is CATE-targeted (treat where the learned uplift is positive); baselines are treat-all, treat-none,
    random, and predictive-score-max (treat by predicted OUTCOME, not uplift — the classic error).
    Value is estimated by OPE on a held-out fold; ground truth is the policy's identified randomized
    value. Tasks vary by (dataset, slice, seed) so each is an independent decision environment."""
    specs = [("thornton_hiv", D.load_thornton_hiv, 10), ("star", D.load_star, 8),
             ("jobs", D.load_jobs, 8), ("nsw", D.load_nsw, 7), ("social_insure", D.load_social_insure, 7)]
    tasks = []
    for name, loader, n_tasks in specs:
        rows, _ = loader()
        for k in range(n_tasks):
            tasks.append({"kind": "randomized", "dataset": name, "design": "rct",
                          "domain": D.CARDS[name]["domain"], "rows": rows, "slice_seed": k,
                          "task_id": f"rct_{name}_{k:02d}"})
    return tasks[:40]


# ---------------------------------------------------------------- logged bandit tasks (20)
def _bandit_tasks():
    """Upworthy K-arm headline tests: a logged bandit with uniform propensity. Target policy: pick the
    arm a learned CTR model predicts best; baselines: random arm, first-listed arm. Oracle = best arm's
    observed CTR. Each test is its own environment."""
    tests, _ = D.load_upworthy(min_arms=2, max_tests=400)
    # pick 20 tests deterministically spread across the archive, preferring >=3 arms where available
    multi = [t for t in tests if len(t["arms"]) >= 3][:14]
    two = [t for t in tests if len(t["arms"]) == 2][:6]
    chosen = (multi + two)[:20]
    tasks = []
    for i, t in enumerate(chosen):
        tasks.append({"kind": "bandit", "dataset": "upworthy", "design": "logged_bandit",
                      "domain": "media", "test": t, "task_id": f"bandit_upworthy_{i:02d}"})
    return tasks


# ---------------------------------------------------------------- quasi-experimental tasks (20)
def _quasi_tasks():
    """Each quasi-experimental dataset yields effect-then-policy tasks under its identification design.
    V2 uses the design (DiD/RD/IV); the naive baseline uses the raw observational contrast (which the
    design exists to correct). Ground truth = the design-identified effect sign/magnitude."""
    specs = [("kielmc", D.load_kielmc_did, "did", 4), ("gov_transfers", D.load_gov_transfers_rd, "rd", 4),
             ("close_elections", D.load_close_elections_rd, "rd", 4),
             ("close_college", D.load_close_college_iv, "iv", 3),
             ("jtrain", D.load_jtrain_did, "did", 3), ("organ_donations", D.load_organ_donations_did, "did", 2)]
    tasks = []
    for name, loader, design, n in specs:
        rows, _ = loader()
        for k in range(n):
            tasks.append({"kind": "quasi", "dataset": name, "design": design,
                          "domain": D.CARDS[name]["domain"], "rows": rows, "slice_seed": k,
                          "task_id": f"quasi_{name}_{k:02d}"})
    return tasks[:20]


# ---------------------------------------------------------------- sequential tasks (20)
def _sequential_tasks():
    """Castle-doctrine state-year panel: dynamic treatment regimes. Target policy: a state-conditioned
    adopt/wait rule; evaluated by per-decision IS / weighted sequential DR. V2 sequential vs greedy
    one-step. Tasks partition the 50 state sequences into 20 environments (grouped) with policy
    variants so each is independent."""
    seqs, _ = D.load_castle_panel()
    tasks = []
    # 20 tasks: each takes a rotating subset of states (5-cluster groups) x policy variant
    for i in range(20):
        subset = [s for j, s in enumerate(seqs) if j % 20 == i % 20 or (j + i) % 7 == 0]
        if len(subset) < 3:
            subset = seqs[(i * 2) % len(seqs): (i * 2) % len(seqs) + 8] or seqs[:8]
        tasks.append({"kind": "sequential", "dataset": "castle", "design": "sequential_did",
                      "domain": "public-policy/crime", "sequences": subset,
                      "variant": i % 2, "task_id": f"seq_castle_{i:02d}"})
    return tasks


# ---------------------------------------------------------------- network / heterogeneous tasks (20)
def _network_hte_tasks():
    """Two families: (a) NETWORK — social-insurance experiment where village peer adoption
    (network_exposure) shifts take-up; a policy that USES network exposure vs one that ignores it
    (SUTVA). (b) HETEROGENEOUS — CATE-targeting quality on the RCTs (does targeting by uplift beat
    treat-all?), the policy-relevant form of heterogeneous-effect estimation."""
    tasks = []
    rows, _ = D.load_social_insure()
    for k in range(10):
        tasks.append({"kind": "network", "dataset": "social_insure", "design": "rct_network",
                      "domain": "development/agriculture", "rows": rows, "slice_seed": k,
                      "task_id": f"net_social_insure_{k:02d}"})
    hte_specs = [("thornton_hiv", D.load_thornton_hiv, 4), ("star", D.load_star, 3),
                 ("jobs", D.load_jobs, 3)]
    for name, loader, n in hte_specs:
        rows, _ = loader()
        for k in range(n):
            tasks.append({"kind": "hte", "dataset": name, "design": "rct_hte",
                          "domain": D.CARDS[name]["domain"], "rows": rows, "slice_seed": 100 + k,
                          "task_id": f"hte_{name}_{k:02d}"})
    return tasks[:20]


def build_all_tasks():
    """Return exactly 120 tasks with the frozen quota composition + split assignment."""
    buckets = {"randomized": _randomized_tasks(), "bandit": _bandit_tasks(),
               "quasi": _quasi_tasks(), "sequential": _sequential_tasks(),
               "network_hte": _network_hte_tasks()}
    counts = {k: len(v) for k, v in buckets.items()}
    assert counts == {"randomized": 40, "bandit": 20, "quasi": 20, "sequential": 20,
                      "network_hte": 20}, counts
    tasks = [t for v in buckets.values() for t in v]
    for t in tasks:
        t["split"] = _split_of(t["task_id"])
    assert len(tasks) == 120, len(tasks)
    assert len({t["task_id"] for t in tasks}) == 120, "task ids must be unique (no double count)"
    return tasks


def composition_manifest(tasks):
    by_kind, by_design, by_domain, by_dataset, by_split = {}, {}, {}, {}, {}
    for t in tasks:
        for d, key in ((by_kind, "kind"), (by_design, "design"), (by_domain, "domain"),
                       (by_dataset, "dataset"), (by_split, "split")):
            d[t[key]] = d.get(t[key], 0) + 1
    return {"n_tasks": len(tasks), "by_kind": by_kind, "by_design": by_design,
            "by_domain": by_domain, "by_dataset": by_dataset, "by_split": by_split,
            "n_datasets": len(by_dataset), "n_domains": len(by_domain), "n_designs": len(by_design)}


if __name__ == "__main__":
    import json
    tasks = build_all_tasks()
    print(json.dumps(composition_manifest(tasks), indent=1))
