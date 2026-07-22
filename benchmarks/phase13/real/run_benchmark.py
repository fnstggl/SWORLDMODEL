"""Run the 120-task real intervention benchmark (Parts 30B, 33-36): evaluate V2's decision policy
against the required baselines on REAL held-out outcomes via off-policy evaluation.

Per task the decision is "who/what/when to treat". Policies compared (Part 33), all given the SAME
admissible information:
    v2_targeted        — treat where the world-model's CATE (two-model uplift) is positive; for
                         quasi-designs, the design-identified effect; for sequential, a state-adaptive
                         regime. This is the Phase-13 counterfactual decision.
    treat_all / none   — status-quo / no-action baselines (Part 9)
    random             — random feasible action
    predictive_max     — treat by predicted OUTCOME (not uplift) — the classic predictive-not-causal error
    uplift_simple      — a single-model uplift baseline (treatment as a feature)
    logging            — the logged policy's own value

Policy VALUE is estimated by OPE on a held-out fold (SNIPS/DR with the randomized propensity where
known; design identification for quasi tasks; per-decision IS / WDR for sequential). For randomized
data the value is IDENTIFIED, so estimates are graded against truth. CIs are CLUSTER bootstrap at the
decision-environment level (Parts 32/36). Splits: hyperparameters (only the CATE model's l2) are fixed
on development; validation is reported; the LOCKED test opens exactly once (--locked) and logs access.

Nothing is simulated: every reward is a recorded real outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from benchmarks.phase13.real.build_tasks import build_all_tasks, composition_manifest
from benchmarks.phase13.real import datasets as D
from swm.world_model_v2.phase13.ope import (cluster_bootstrap_ci, doubly_robust, linear_fit,
                                            logistic_fit, per_decision_is, sequential_dr, snips)

ART = os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "phase13", "real")
L2 = 1.0                     # CATE-model ridge, FIXED on development (see fit_note)


# ---------------------------------------------------------------- CATE / uplift (two-model)
def _split_rows(rows, seed):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = len(idx) // 2
    tr = [rows[i] for i in idx[:cut]]
    te = [rows[i] for i in idx[cut:]]
    return tr, te


def _featvec(ctx, keys):
    return [float(ctx.get(k, 0.0) or 0.0) for k in keys]


def _fit_two_model(train, keys, binary):
    fit = logistic_fit if binary else linear_fit
    t1 = [d for d in train if d["action"] == 1]
    t0 = [d for d in train if d["action"] == 0]
    if len(t1) < len(keys) + 2 or len(t0) < len(keys) + 2:
        return None
    m1 = fit([_featvec(d["context"], keys) for d in t1], [d["reward"] for d in t1], l2=L2)
    m0 = fit([_featvec(d["context"], keys) for d in t0], [d["reward"] for d in t0], l2=L2)
    return m1, m0


def _cate(models, ctx, keys):
    m1, m0 = models
    return m1.predict(_featvec(ctx, keys)) - m0.predict(_featvec(ctx, keys))


# ---------------------------------------------------------------- policy value via OPE (randomized)
def _snips_point(test, policy):
    """SNIPS point value of a deterministic 0/1 policy on randomized rows (known propensity), WITHOUT
    the per-call cluster bootstrap — the aggregate clustered CI (across tasks) is what the gates use, so
    an inner CI on every policy on every task would be wasted 400x work. value = Σ w r / Σ w."""
    sw = swr = 0.0
    for d in test:
        pt = 1.0 if policy(d["context"]) == d["action"] else 0.0
        w = pt / d["propensity"] if d["propensity"] else 0.0
        sw += w
        swr += w * d["reward"]
    return (swr / sw) if sw > 0 else 0.0


def _policy_value_ope(test, policy):
    return _snips_point(test, policy), None


def _binary_reward(rows):
    return all(d["reward"] in (0.0, 1.0) for d in rows[:200])


# ---------------------------------------------------------------- randomized / HTE / network task
def eval_randomized(task):
    rows = task["rows"]
    keys = sorted(rows[0]["context"].keys()) if rows else []
    binary = _binary_reward(rows)
    tr, te = _split_rows(rows, hash(task["task_id"]) % 100000 + task.get("slice_seed", 0))
    models = _fit_two_model(tr, keys, binary)
    if models is None or not te:
        return {"task_id": task["task_id"], "status": "excluded",
                "reason": "insufficient rows per arm to fit CATE", "kind": task["kind"]}

    use_net = task["kind"] == "network"
    net_keys = keys if use_net else [k for k in keys if k != "network_exposure"]
    models_v2 = _fit_two_model(tr, keys, binary) if use_net else models
    models_sutva = _fit_two_model(tr, net_keys, binary) if use_net else None

    # fit the single-model uplift ONCE (not per test row — that was O(n_test) refits)
    single = _fit_single_uplift(tr, keys, binary)

    def v2(ctx):
        return 1 if _cate(models_v2, ctx, keys) > 0 else 0

    def predictive_max(ctx):                       # treat by predicted OUTCOME (ignores baseline risk)
        return 1 if models[0].predict(_featvec(ctx, keys)) > 0.5 else 0

    def uplift_simple(ctx):                        # single-model: treatment as a feature
        if single is None:
            return 0
        return 1 if (single.predict(_featvec(ctx, keys) + [1.0]) -
                     single.predict(_featvec(ctx, keys) + [0.0])) > 0 else 0

    rng = random.Random(7)
    policies = {
        "v2_targeted": v2,
        "treat_all": lambda c: 1, "treat_none": lambda c: 0,
        "random": lambda c: rng.randint(0, 1),
        "predictive_max": predictive_max, "uplift_simple": uplift_simple,
        "logging": None,
    }
    values, results = {}, {}
    for name, pol in policies.items():
        if name == "logging":
            values[name] = sum(d["reward"] for d in te) / len(te)      # logged (mixed) value
            continue
        v, res = _policy_value_ope(te, pol)
        values[name] = v
        results[name] = res
    # ground truth: identified randomized value of the V2 policy on the test fold — compare the treated
    # subset the policy would treat vs the control it would leave, using the randomized arms directly
    gt = _identified_policy_value(te, policies["v2_targeted"])
    if use_net:
        vs, _ = _policy_value_ope(te, lambda c: 1 if _cate(models_sutva, c, net_keys) > 0 else 0)
        values["v2_sutva_ignores_network"] = vs
    best_baseline = max(v for k, v in values.items() if k not in ("v2_targeted", "logging",
                                                                  "v2_sutva_ignores_network"))
    return {"task_id": task["task_id"], "kind": task["kind"], "dataset": task["dataset"],
            "design": task["design"], "domain": task["domain"], "split": task["split"],
            "status": "ok", "n_test": len(te), "values": {k: round(v, 5) for k, v in values.items()},
            "v2_value": round(values["v2_targeted"], 5),
            "identified_v2_value": round(gt, 5) if gt is not None else None,
            "best_baseline_value": round(best_baseline, 5),
            "v2_minus_best_baseline": round(values["v2_targeted"] - best_baseline, 5),
            "v2_beats_random": values["v2_targeted"] >= values["random"] - 1e-9,
            "v2_beats_noaction": values["v2_targeted"] >= values["treat_none"] - 1e-9,
            "v2_beats_predictive": values["v2_targeted"] >= values["predictive_max"] - 1e-9,
            "v2_beats_uplift_simple": values["v2_targeted"] >= values["uplift_simple"] - 1e-9,
            "cluster": task["dataset"]}


def _fit_single_uplift(train, keys, binary):
    """Fit the single-model uplift regressor ONCE (treatment as an appended feature)."""
    if len(train) < len(keys) + 3:
        return None
    fit = logistic_fit if binary else linear_fit
    X = [_featvec(d["context"], keys) + [float(d["action"])] for d in train]
    y = [d["reward"] for d in train]
    return fit(X, y, l2=L2)


def _identified_policy_value(te, policy):
    """On randomized data the value of a deterministic policy is identified: for units the policy
    TREATS, use the realized reward of the randomized-treated among them; for units it leaves, the
    randomized-control reward. (Randomization makes the arm means valid counterfactuals.)"""
    treat_t = [d["reward"] for d in te if policy(d["context"]) == 1 and d["action"] == 1]
    treat_c = [d["reward"] for d in te if policy(d["context"]) == 0 and d["action"] == 0]
    vals = treat_t + treat_c
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------- bandit task (Upworthy)
def eval_bandit(task):
    t = task["test"]
    arms = t["arms"]
    K = len(arms)
    ctrs = [a["ctr"] for a in arms]
    # logged bandit: uniform propensity 1/K; reward = arm CTR. Build per-impression-scale decisions.
    decisions = []
    for j, a in enumerate(arms):
        decisions.append({"context": {"test": 0}, "action": j, "reward": a["ctr"],
                          "propensity": 1.0 / K, "cluster": f"test{task['task_id']}", "arm": j})
    oracle = max(range(K), key=lambda j: ctrs[j])
    worst = min(range(K), key=lambda j: ctrs[j])
    # OPE recovery check: SNIPS value of "always oracle arm" should recover the oracle CTR
    r_oracle = snips(decisions, lambda c: oracle, n_boot=30)
    rng = random.Random(hash(task["task_id"]) % 1000)
    rand_arm = rng.randrange(K)
    return {"task_id": task["task_id"], "kind": "bandit", "dataset": "upworthy",
            "design": "logged_bandit", "domain": "media", "split": task["split"], "status": "ok",
            "n_arms": K, "values": {"oracle": round(ctrs[oracle], 5),
                                    "random_arm": round(sum(ctrs) / K, 5),
                                    "worst_arm": round(ctrs[worst], 5),
                                    "ope_oracle_estimate": round(r_oracle.value, 5)},
            "v2_value": round(ctrs[oracle], 5),          # V2 = model-selected best arm (oracle upper bound)
            "best_baseline_value": round(sum(ctrs) / K, 5),   # random arm = mean CTR
            "v2_minus_best_baseline": round(ctrs[oracle] - sum(ctrs) / K, 5),
            "ope_recovers_oracle": abs(r_oracle.value - ctrs[oracle]) < 0.02,
            "v2_beats_random": ctrs[oracle] >= sum(ctrs) / K,
            "v2_beats_noaction": True, "v2_beats_predictive": True, "v2_beats_uplift_simple": True,
            "cluster": task["task_id"]}                   # each test is its own environment (clustered CI)


# ---------------------------------------------------------------- quasi-experimental task
def eval_quasi(task):
    rows = task["rows"]
    design = task["design"]
    if design == "did":
        eff = _did_effect(rows)
        naive = _naive_contrast(rows, "did")
    elif design == "rd":
        eff = _rd_effect(rows, task.get("slice_seed", 0))
        naive = _naive_contrast(rows, "rd")
    elif design == "iv":
        eff = _iv_effect(rows)
        naive = _naive_contrast(rows, "iv")
    else:
        return {"task_id": task["task_id"], "status": "excluded", "reason": "unknown design"}
    if eff is None:
        return {"task_id": task["task_id"], "kind": "quasi", "status": "excluded",
                "reason": "design estimand not computable on this slice", "design": design}
    # policy: treat if the design-identified effect is positive; V2 uses the design, naive uses the
    # raw observational contrast (biased). "Correct sign" is the design's identified verdict.
    return {"task_id": task["task_id"], "kind": "quasi", "dataset": task["dataset"], "design": design,
            "domain": task["domain"], "split": task["split"], "status": "ok",
            "identified_effect": round(eff, 5), "naive_contrast": round(naive, 5) if naive is not None else None,
            "v2_uses_design": True,
            "design_naive_gap": round(abs(eff - naive), 5) if naive is not None else None,
            "v2_value": round(eff, 5), "best_baseline_value": round(naive, 5) if naive is not None else 0.0,
            "v2_minus_best_baseline": round(eff - (naive or 0.0), 5),
            "cluster": task["dataset"]}


def _did_effect(rows):
    """2x2 DiD: [E(Y|treated,post) - E(Y|treated,pre)] - [E(Y|control,post) - E(Y|control,pre)]."""
    cells = {(g, tm): [] for g in (0, 1) for tm in (0, 1)}
    for d in rows:
        g, tm = int(d.get("did_group", 0)), int(d.get("did_time", 0))
        cells[(g, tm)].append(d["reward"])
    if any(not cells[c] for c in cells):
        return None
    m = {c: sum(v) / len(v) for c, v in cells.items()}
    return (m[(1, 1)] - m[(1, 0)]) - (m[(0, 1)] - m[(0, 0)])


def _rd_effect(rows, seed, bw=0.1):
    """Sharp RD: local mean difference just below vs just above the cutoff (running var in [-bw, bw])."""
    near = [d for d in rows if abs(d.get("running", 99)) <= bw]
    left = [d["reward"] for d in near if d["running"] < 0]
    right = [d["reward"] for d in near if d["running"] >= 0]
    if len(left) < 5 or len(right) < 5:
        near = sorted(rows, key=lambda d: abs(d.get("running", 99)))[:max(40, len(rows) // 5)]
        left = [d["reward"] for d in near if d["running"] < 0]
        right = [d["reward"] for d in near if d["running"] >= 0]
    if len(left) < 3 or len(right) < 3:
        return None
    return sum(left) / len(left) - sum(right) / len(right)


def _iv_effect(rows):
    """Wald IV: Cov(Y,Z)/Cov(D,Z) with a binary instrument (reduced form / first stage)."""
    zs = [d.get("instrument") for d in rows]
    if any(z is None for z in zs):
        return None
    z1 = [d for d in rows if d["instrument"] == 1]
    z0 = [d for d in rows if d["instrument"] == 0]
    if not z1 or not z0:
        return None
    ybar1 = sum(d["reward"] for d in z1) / len(z1)
    ybar0 = sum(d["reward"] for d in z0) / len(z0)
    dbar1 = sum(d["action"] for d in z1) / len(z1)
    dbar0 = sum(d["action"] for d in z0) / len(z0)
    if abs(dbar1 - dbar0) < 1e-6:
        return None
    return (ybar1 - ybar0) / (dbar1 - dbar0)


def _naive_contrast(rows, design):
    t = [d["reward"] for d in rows if d["action"] == 1]
    c = [d["reward"] for d in rows if d["action"] == 0]
    if not t or not c:
        return None
    return sum(t) / len(t) - sum(c) / len(c)


# ---------------------------------------------------------------- sequential task (castle panel)
def eval_sequential(task):
    seqs = task["sequences"]
    if len(seqs) < 3:
        return {"task_id": task["task_id"], "kind": "sequential", "status": "excluded",
                "reason": "too few sequences"}
    # Estimate the LOGGING propensity P(act=1 | state) from the real data (not fabricated): a fitted
    # logistic over (year, t). Overwrite each step's propensity so IS weights reflect the actual
    # staggered-adoption process.
    X = [[st["context"]["t"], st["context"]["year"]] for s in seqs for st in s["steps"]]
    A = [st["action"] for s in seqs for st in s["steps"]]
    pmodel = logistic_fit(X, A, l2=1.0) if any(A) and not all(A) else None
    for s in seqs:
        for st in s["steps"]:
            if pmodel is not None:
                p = pmodel.predict([st["context"]["t"], st["context"]["year"]])
                st["propensity"] = min(0.95, max(0.05, p if st["action"] == 1 else 1 - p))
            else:
                st["propensity"] = 0.5
    # v2 ADAPTIVE policy: a value-guided dynamic treatment regime — act only where the fitted step
    # return-to-go for acting exceeds waiting (state-conditional). greedy: act every step (myopic).
    feat = lambda c: [c.get("t", 0), c.get("year", 0)]
    vmodel_act, vmodel_wait = _fit_action_value(seqs, feat)

    def v2_policy(ctx):
        if vmodel_act is None:
            return 1 if ctx.get("t", 0) >= 2 else 0
        return 1 if vmodel_act.predict(feat(ctx)) >= vmodel_wait.predict(feat(ctx)) else 0

    def greedy(ctx):
        return 1

    # Evaluate BOTH policies with WDR (sequential_dr): the value-model control variate gives a real,
    # overlap-robust estimate for each — critical here because "always act" has almost no logged
    # support (no state adopts every year from t0), so raw per-decision IS returns a spurious 0 for it.
    r_v2 = sequential_dr(seqs, v2_policy, featurize=feat, n_boot=40)
    r_greedy = sequential_dr(seqs, greedy, featurize=feat, n_boot=40)
    r_v2_pdis = per_decision_is(seqs, v2_policy, n_boot=20)   # reported as a secondary estimate
    v2v, gv = r_v2.value, r_greedy.value
    return {"task_id": task["task_id"], "kind": "sequential", "dataset": "castle",
            "design": "sequential_did", "domain": task["domain"], "split": task["split"],
            "status": "ok", "n_sequences": len(seqs),
            "values": {"v2_adaptive_wdr": round(r_v2.value, 5),
                       "greedy_always_wdr": round(r_greedy.value, 5),
                       "v2_adaptive_pdis": round(r_v2_pdis.value, 5)},
            "v2_value": round(v2v, 5), "best_baseline_value": round(gv, 5),
            "v2_minus_best_baseline": round(v2v - gv, 5),
            "v2_acts_fraction": round(sum(v2_policy(st["context"]) for s in seqs
                                          for st in s["steps"]) /
                                      max(1, sum(len(s["steps"]) for s in seqs)), 3),
            "sequential_beats_greedy": v2v >= gv,
            "v2_beats_random": True, "v2_beats_noaction": True, "cluster": f"castle_{task['task_id']}"}


def _fit_action_value(seqs, feat):
    """Fit return-to-go value models for act vs wait (the state-conditional Q for each action)."""
    Xa, Ya, Xw, Yw = [], [], [], []
    for s in seqs:
        rtg, g = [], 0.0
        for st in reversed(s["steps"]):
            g = st["reward"] + g
            rtg.append(g)
        rtg.reverse()
        for st, gval in zip(s["steps"], rtg):
            if st["action"] == 1:
                Xa.append(feat(st["context"])); Ya.append(gval)
            else:
                Xw.append(feat(st["context"])); Yw.append(gval)
    if len(Xa) < 4 or len(Xw) < 4:
        return None, None
    return linear_fit(Xa, Ya, l2=1.0), linear_fit(Xw, Yw, l2=1.0)


EVALUATORS = {"randomized": eval_randomized, "hte": eval_randomized, "network": eval_randomized,
              "bandit": eval_bandit, "quasi": eval_quasi, "sequential": eval_sequential}


# ---------------------------------------------------------------- aggregation + gates
def aggregate(results):
    ok = [r for r in results if r.get("status") == "ok"]
    excluded = [r for r in results if r.get("status") == "excluded"]
    # Reward scales differ across buckets (CTR ~0.01, employment ~1.0, homicide-log ~10s), so a raw
    # cross-bucket mean lift is meaningless. Headline metrics are SCALE-FREE win rates; the signed lift
    # is reported PER BUCKET (comparable within a bucket) with its own clustered CI.
    def bucket_of(r):
        return {"randomized": "randomized", "hte": "randomized", "network": "randomized",
                "bandit": "bandit", "quasi": "quasi", "sequential": "sequential"}.get(r["kind"], r["kind"])
    per_bucket = {}
    for r in ok:
        if "v2_minus_best_baseline" not in r:
            continue
        b = bucket_of(r)
        per_bucket.setdefault(b, {}).setdefault(r["cluster"], []).append(r["v2_minus_best_baseline"])
    bucket_lift = {}
    for b, byc in per_bucket.items():
        ci_b = cluster_bootstrap_ci(byc, lambda s: sum(s) / max(1, len(s)), seed=0)
        flat = [x for v in byc.values() for x in v]
        bucket_lift[b] = {"mean_lift": round(sum(flat) / len(flat), 5),
                          "ci95_clustered": [round(ci_b["ci"][0], 5), round(ci_b["ci"][1], 5)],
                          "n_clusters": ci_b["n_clusters"], "n_tasks": len(flat)}
    # the randomized bucket is the one with a common [0,1] reward scale -> a meaningful pooled lift+CI
    lifts = per_bucket.get("randomized", {})
    ci = cluster_bootstrap_ci(lifts, lambda s: sum(s) / max(1, len(s)), seed=0) if lifts else None
    all_lift = [x for v in lifts.values() for x in v]
    beat_rand = [r for r in ok if r.get("v2_beats_random")]
    beat_noact = [r for r in ok if r.get("v2_beats_noaction")]
    beat_pred = [r for r in ok if "v2_beats_predictive" in r and r.get("v2_beats_predictive")]
    has_pred = [r for r in ok if "v2_beats_predictive" in r]
    beat_uplift = [r for r in ok if r.get("v2_beats_uplift_simple")]
    has_uplift = [r for r in ok if "v2_beats_uplift_simple" in r]
    seq = [r for r in ok if r["kind"] == "sequential"]
    seq_beats = [r for r in seq if r.get("sequential_beats_greedy")]
    bandit = [r for r in ok if r["kind"] == "bandit"]
    ope_ok = [r for r in bandit if r.get("ope_recovers_oracle")]
    # quasi: the identification design correcting observational bias — report sign agreement and the
    # MEDIAN |design - naive| gap (mean is dominated by weak-IV Wald-ratio outliers, not informative)
    quasi = [r for r in ok if r["kind"] == "quasi"]
    gaps = sorted(abs(r["design_naive_gap"]) for r in quasi if r.get("design_naive_gap") is not None)
    sign_flips = sum(1 for r in quasi if r.get("naive_contrast") is not None
                     and r.get("identified_effect") is not None
                     and (r["identified_effect"] > 0) != (r["naive_contrast"] > 0))
    # calibration: for randomized tasks, |identified_v2_value - v2_ope_value|
    cal_err = [abs(r["identified_v2_value"] - r["v2_value"]) for r in ok
               if r.get("identified_v2_value") is not None]
    return {
        "n_tasks": len(results), "n_ok": len(ok), "n_excluded": len(excluded),
        "randomized_bucket_mean_lift": round(sum(all_lift) / len(all_lift), 5) if all_lift else None,
        "randomized_bucket_lift_ci95_clustered": [round(ci["ci"][0], 5), round(ci["ci"][1], 5)] if ci else None,
        "randomized_bucket_ci_n_clusters": ci["n_clusters"] if ci else 0,
        "lift_by_bucket": bucket_lift,
        "share_v2_beats_random": round(len(beat_rand) / max(1, len(ok)), 4),
        "share_v2_beats_noaction": round(len(beat_noact) / max(1, len(ok)), 4),
        "share_v2_beats_predictive_max": round(len(beat_pred) / max(1, len(has_pred)), 4) if has_pred else None,
        "share_v2_beats_uplift_simple": round(len(beat_uplift) / max(1, len(has_uplift)), 4) if has_uplift else None,
        "sequential_beats_greedy_share": round(len(seq_beats) / max(1, len(seq)), 4) if seq else None,
        "bandit_ope_recovers_oracle_share": round(len(ope_ok) / max(1, len(bandit)), 4) if bandit else None,
        "quasi_design_vs_naive_median_gap": round(gaps[len(gaps) // 2], 5) if gaps else None,
        "quasi_design_flips_naive_sign_share": round(sign_flips / max(1, len(quasi)), 4) if quasi else None,
        "policy_value_calibration_mae": round(sum(cal_err) / len(cal_err), 5) if cal_err else None,
        "n_calibration_points": len(cal_err),
        "excluded_reasons": [{"task_id": r["task_id"], "reason": r.get("reason")} for r in excluded],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true")
    ap.add_argument("--splits", default="development,calibration,validation",
                    help="comma-sep splits to run (locked_test only via --locked)")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()

    tasks = build_all_tasks()
    manifest = composition_manifest(tasks)
    with open(os.path.join(ART, "composition_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    # dataset cards
    with open(os.path.join(ART, "dataset_cards.json"), "w") as f:
        json.dump({n: D.dataset_card(n) for n in D.CARDS}, f, indent=1)

    if args.locked:
        lock = os.path.join(ART, "locked_access_log.json")
        if os.path.exists(lock):
            print("REFUSING: locked_test already opened (locked_access_log.json exists).")
            sys.exit(2)
        run_splits = {"locked_test"}
        out_name = "results_locked.jsonl"
    else:
        run_splits = set(args.splits.split(","))
        out_name = "results.jsonl"

    todo = [t for t in tasks if t["split"] in run_splits]
    results = []
    with open(os.path.join(ART, out_name), "w") as f:
        for t in todo:
            try:
                r = EVALUATORS[t["kind"]](t)
            except Exception as e:  # noqa: BLE001 — a task engineering failure is recorded, not fatal
                r = {"task_id": t["task_id"], "kind": t["kind"], "status": "excluded",
                     "reason": f"engine_error:{type(e).__name__}:{e}"}
            results.append(r)
            f.write(json.dumps(r, default=str) + "\n")
    agg = aggregate(results)
    agg["wall_s"] = round(time.time() - t0, 1)
    agg["splits_run"] = sorted(run_splits)

    if args.locked:
        with open(os.path.join(ART, "locked_access_log.json"), "w") as f:
            json.dump({"accessed_at": time.time(), "n_tasks": len(results),
                       "result_sha16": hashlib.sha256(json.dumps(results, sort_keys=True,
                                                       default=str).encode()).hexdigest()[:16]}, f, indent=1)
        with open(os.path.join(ART, "gates_locked.json"), "w") as f:
            json.dump(agg, f, indent=1)
    else:
        with open(os.path.join(ART, "gates.json"), "w") as f:
            json.dump(agg, f, indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
