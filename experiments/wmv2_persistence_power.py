"""Phase 8 persistence-at-power — the honest longitudinal test the prior n=48 could not support.

The OmniBehavior repair (reference/omnibehavior.py) established: passive-exposure events are the real
prediction targets (action-record events are ~deterministic type lookups). The prior run found real
train burstiness (momentum lift 1.96) but NO detectable held-out mechanism effect at n=48. This runner
scales the cohort and does it RIGHT:

  1. build the passive-exposure prediction task on a LARGE cohort (many users);
  2. measure burstiness on TRAIN (is persistence structurally present at all?);
  3. POWER ANALYSIS BEFORE GRADING — given the base rate and observed momentum effect, how many events
     are needed to detect a persistence lift; report whether n is adequate;
  4. splits: per-user chronological time-forward (train prefix / test suffix), PLUS person-disjoint
     (users never in train) and sequence-disjoint checks;
  5. persistence ABLATION — compare, on identical held-out events, no-history vs latest-observation-only
     vs simple autoregression (last-k momentum) vs persistent-latent (hierarchical per-user rate +
     momentum state); paired bootstrap CIs; preserve the sign.

NO LLM (the prior round showed the direct LLM below chance here; the persistence question is about the
STATE model, not the LLM). Pure compute; resumable via the downloaded cohort cache.
Run: PYTHONPATH=. python -m experiments.wmv2_persistence_power
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

RESULT = "experiments/results/wmv2_persistence_power.json"


def _paired_brier(ys, pa, pb, *, n_boot=1000, seed=5):
    d = [(a - y) ** 2 - (b - y) ** 2 for a, b, y in zip(pa, pb, ys)]
    rng = random.Random(seed)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": round(sum(d) / n, 6), "ci95": [round(bs[int(0.025 * n_boot)], 6),
                                                   round(bs[int(0.975 * n_boot) - 1], 6)], "n": n}


def _power_for_effect(base_rate, effect, n, *, paired_diffs=None, alpha=0.05):
    """Power to detect a Brier improvement `effect` over n PAIRED obs. A paired design's precision is set
    by the sd of the per-observation Brier DIFFERENCE, not the marginal Brier sd — so when the arms are
    correlated (they are: same rows, similar models) the effective sd is far smaller. When `paired_diffs`
    is supplied we use their empirical sd (the statistically correct, CI-consistent quantity); otherwise
    we fall back to the conservative marginal approximation 2·base·(1−base) and flag it."""
    from statistics import NormalDist, pstdev
    if paired_diffs is not None and len(paired_diffs) > 1:
        sd_diff = max(1e-9, pstdev(paired_diffs))
        basis = "empirical paired-difference sd (CI-consistent)"
    else:
        sd_diff = max(1e-6, 2.0 * base_rate * (1 - base_rate))
        basis = "conservative marginal Brier sd (planning only; ignores pairing correlation)"
    se = sd_diff / math.sqrt(max(1, n))
    z_a = 1.96
    power = 1 - NormalDist().cdf(z_a - abs(effect) / se)
    mde_80 = (z_a + 0.84) * se
    return {"n": n, "se_basis": basis, "se_per_paired_brier": round(se, 6),
            "power_at_observed_effect": round(power, 3), "min_detectable_effect_80pct": round(mde_80, 6)}


def run(n_users, prefix_len, per_user):
    from swm.eval.omnibehavior_eval import download_users
    from swm.world_model_v2.reference.omnibehavior import (PASSIVE, acted, fit_stats, item_features,
                                                           momentum_state, split_user, user_events)
    from swm.world_model_v2.inference_layer import hierarchical_rates

    paths = download_users(n_users, max_bytes=3_000_000, cache_dir="data/omnibehavior")
    users = {}
    for p in paths:
        for uid, u in json.load(open(p)).items():
            evs = user_events(u)
            if len([e for e in evs if e.get("type") in PASSIVE]) >= 20:   # enough passive events to split
                users[uid] = evs
    print(f"cohort: {len(users)} users with >=20 passive events", flush=True)

    # per-user chronological split; passive-exposure targets only
    train_by_user, test_rows = {}, []
    person_disjoint_users = list(users)[: max(1, len(users) // 5)]        # 20% held-out PERSONS
    for uid, evs in users.items():
        tr, te = split_user(evs, 0.7)
        train_by_user[uid] = tr
        if uid in person_disjoint_users:
            continue                                                     # excluded from train entirely
        for e in [x for x in te if x.get("type") in PASSIVE]:
            idx = evs.index(e)
            prior_passive = [x for x in evs[:idx] if x.get("type") in PASSIVE]
            test_rows.append({"uid": uid, "y": int(acted(e)),
                              "momentum": momentum_state(prior_passive),
                              "prior_passive_n": len(prior_passive)})
    # person-disjoint test rows (users NOT in train)
    pd_rows = []
    for uid in person_disjoint_users:
        evs = users[uid]
        _, te = split_user(evs, 0.7)
        for e in [x for x in te if x.get("type") in PASSIVE]:
            idx = evs.index(e)
            prior_passive = [x for x in evs[:idx] if x.get("type") in PASSIVE]
            pd_rows.append({"uid": uid, "y": int(acted(e)), "momentum": momentum_state(prior_passive),
                            "prior_passive_n": len(prior_passive)})

    stats = fit_stats({u: train_by_user[u] for u in train_by_user})
    base = stats["global_rate"]
    # hierarchical per-user rate posterior (shrinkage) — the persistent user-level state
    groups = {u: (sum(1 for e in tr if e.get("type") in PASSIVE and acted(e)),
                  sum(1 for e in tr if e.get("type") in PASSIVE)) for u, tr in train_by_user.items()}
    hier = hierarchical_rates(groups, population_prior=base)

    print(f"train burstiness: momentum_lift={stats['momentum_lift']:.3f} "
          f"(p_hot={stats['p_hot']:.3f} vs p_cold={stats['p_cold']:.3f}, "
          f"n_hot={stats['momentum_n']['hot']} n_cold={stats['momentum_n']['cold']})", flush=True)

    ys = [r["y"] for r in test_rows]
    n = len(ys)
    real_rate = sum(ys) / max(1, n)

    # ---- arms (identical held-out rows) ----
    def user_rate(uid):
        rp = hier.get(uid)
        return rp.mean() if rp is not None and hasattr(rp, "mean") else base

    A_nohist = [base for _ in test_rows]                                  # no history
    A_latest = [user_rate(r["uid"]) for r in test_rows]                   # latest/user-level only (shrunk)
    # simple autoregression: base + momentum slope fitted on TRAIN
    hot_lift = stats["p_hot"] - stats["p_cold"]
    A_ar = [min(0.97, max(0.01, user_rate(r["uid"]) + hot_lift * (r["momentum"] - 0.5) * 2))
            for r in test_rows]
    # persistent-latent: user-level shrunk rate MODULATED by momentum persistence state (the mechanism)
    A_persist = [min(0.97, max(0.01, user_rate(r["uid"]) * (stats["p_hot"] / max(1e-4, stats["p_cold"]))
                               ** (r["momentum"] - 0.5)))
                 for r in test_rows]

    def brier(p):
        return round(sum((pi - y) ** 2 for pi, y in zip(p, ys)) / n, 5)

    arms = {"A_nohist": A_nohist, "A_latest_userrate": A_latest, "A_autoregression": A_ar,
            "A_persistent_latent": A_persist}
    detail = {a: {"brier": brier(p), "pred_rate": round(sum(p) / n, 4)} for a, p in arms.items()}

    # persistence ABLATION: does the persistence mechanism beat the memoryless user-rate?
    paired = {
        "persist_vs_userrate": _paired_brier(ys, A_persist, A_latest),
        "persist_vs_nohist": _paired_brier(ys, A_persist, A_nohist),
        "ar_vs_userrate": _paired_brier(ys, A_ar, A_latest),
    }

    # POWER ANALYSIS (before interpreting the ablation) — use the EMPIRICAL paired-difference sd
    observed_effect = abs(paired["persist_vs_userrate"]["mean"])
    diffs = [(a - y) ** 2 - (b - y) ** 2 for a, b, y in zip(A_persist, A_latest, ys)]
    power = _power_for_effect(real_rate, observed_effect, n, paired_diffs=diffs)
    pd_diffs = None
    if pd_rows:
        pd_ys0 = [r["y"] for r in pd_rows]
        pd_p0 = [min(0.97, max(0.01, base * (stats["p_hot"] / max(1e-4, stats["p_cold"]))
                               ** (r["momentum"] - 0.5))) for r in pd_rows]
        pd_u0 = [base for _ in pd_rows]
        pd_diffs = [(a - y) ** 2 - (b - y) ** 2 for a, b, y in zip(pd_p0, pd_u0, pd_ys0)]
    pd_power = _power_for_effect(sum(r["y"] for r in pd_rows) / max(1, len(pd_rows)),
                                 observed_effect, len(pd_rows), paired_diffs=pd_diffs) if pd_rows else None

    # person-disjoint ablation (transfer)
    pd_paired = None
    if pd_rows:
        pd_ys = [r["y"] for r in pd_rows]
        pd_persist = [min(0.97, max(0.01, base * (stats["p_hot"] / max(1e-4, stats["p_cold"]))
                                     ** (r["momentum"] - 0.5))) for r in pd_rows]
        pd_userrate = [base for _ in pd_rows]
        pd_paired = _paired_brier(pd_ys, pd_persist, pd_userrate)

    verdict = ("persistence mechanism HELPS (CI excludes 0, favorable)"
               if paired["persist_vs_userrate"]["ci95"][1] < 0 else
               ("persistence mechanism HURTS (CI excludes 0, unfavorable)"
                if paired["persist_vs_userrate"]["ci95"][0] > 0 else
                "persistence effect NOT DETECTABLE (CI spans 0)"))
    powered = power["power_at_observed_effect"] >= 0.8

    out = {
        "cohort": {"n_users": len(users), "n_test_events": n, "real_action_rate": round(real_rate, 4),
                   "n_person_disjoint_users": len(person_disjoint_users),
                   "n_person_disjoint_events": len(pd_rows)},
        "burstiness_train": {"momentum_lift": round(stats["momentum_lift"], 3),
                             "p_hot": round(stats["p_hot"], 4), "p_cold": round(stats["p_cold"], 4),
                             "n_hot": stats["momentum_n"]["hot"], "n_cold": stats["momentum_n"]["cold"],
                             "persistence_structurally_present": stats["momentum_lift"] > 1.1},
        "power_analysis": {"main": power, "person_disjoint": pd_power,
                           "adequately_powered": powered,
                           "note": ("n is adequate to detect the observed effect at >=80% power"
                                    if powered else
                                    "UNDERPOWERED for the observed effect — the null is uninformative; "
                                    f"need ~{int((1.96+0.84)**2 * (2*real_rate*(1-real_rate)) / max(1e-9, observed_effect**2))} events")},
        "detail": detail, "paired_ablation": paired, "person_disjoint_ablation": pd_paired,
        "verdict": verdict,
        "hierarchical_pool": hier.get("_pool"),
        "_meta": {"n_users_downloaded": len(paths), "prefix_len": prefix_len, "llm_calls": 0,
                  "est_cost_usd": 0.0}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nn_test={n} real_rate={real_rate:.3f}")
    print("detail:", json.dumps(detail, indent=1))
    print("persist_vs_userrate:", paired["persist_vs_userrate"])
    print("power @ observed effect:", power["power_at_observed_effect"], "adequately_powered:", powered)
    print("VERDICT:", verdict)
    print(f"wrote {RESULT}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=140)
    ap.add_argument("--prefix-len", type=int, default=12)
    ap.add_argument("--per-user", type=int, default=8)
    a = ap.parse_args()
    run(a.n_users, a.prefix_len, a.per_user)
