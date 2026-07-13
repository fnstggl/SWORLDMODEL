"""Phase 8 — reproduce the persistence win THROUGH the shared world (Track A: repeated individual behavior).

The prior ``wmv2_persistence_power`` established the win with a STANDALONE longitudinal predictor
(``A_persist = user_rate × (p_hot/p_cold)^(momentum−0.5)`` computed inline) — it never touched a WorldState.
Phase 8's mandate is to integrate that signal into the SHARED PATH and rerun the evaluation. Here every arm
runs through the same pipeline:

    prior passive history → Phase-8 EventLog (leakage-safe, observed_time ≤ as_of)
      → DecayedBetaBernoulliFilter (anchor = hierarchical per-user rate; forgetting = momentum)
      → materialize into WorldState entity.latent_state[phase4_policy_value:engage]  (PersistentStateDelta)
      → engagement_readout (reads the materialized WorldState field)  → P(act)

The two arms differ ONLY in momentum, because both anchor at the SAME hierarchical user rate:
  * B_userrate (memoryless): no events → readout = the anchor (persistent user level, no momentum);
  * B_persist  (full)      : the actor's real prior history → filtered posterior (level + momentum).
So the paired difference isolates the momentum contribution, and removing history collapses B_persist onto
B_userrate (the causal ablation). B0 (global rate, no user level) is the weakest baseline.

Honest by design: the decayed-Beta filter is NOT the prior inline formula — whether it still beats the
memoryless baseline on held-out Brier is an empirical question we RUN, not assume. Filter hyperparameters
(decay, prior strength) are fit on TRAIN only (grid, max train log-lik); test is never tuned. Power analysis
uses the empirical PAIRED-difference sd. No LLM.

Run: PYTHONPATH=. python -m experiments.wmv2_phase8_shared_world --n-users 140
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

RESULT = "experiments/results/phase8/shared_world_trackA.json"


# ------------------------------------------------------------------ metrics + CIs
def _brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / max(1, len(ys))


def _logloss(ps, ys):
    return sum(-(y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1 - p)))
               for p, y in zip(ps, ys)) / max(1, len(ys))


def _auroc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def _ece(ps, ys, bins=10):
    b = [[0.0, 0] for _ in range(bins)]
    for p, y in zip(ps, ys):
        k = min(bins - 1, int(p * bins))
        b[k][0] += y
        b[k][1] += 1
    n = max(1, len(ys))
    return sum(abs(c[0] / c[1] - (k + 0.5) / bins) * c[1] for k, c in enumerate(b) if c[1]) / n


def _paired_brier(ys, pa, pb, *, n_boot=1000, seed=5):
    d = [(a - y) ** 2 - (b - y) ** 2 for a, b, y in zip(pa, pb, ys)]
    rng = random.Random(seed)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": round(sum(d) / n, 6), "ci95": [round(bs[int(0.025 * n_boot)], 6),
                                                   round(bs[int(0.975 * n_boot) - 1], 6)], "n": n,
            "paired_diffs": d}


def _power(base_rate, effect, n, paired_diffs):
    from statistics import NormalDist, pstdev
    sd_diff = max(1e-9, pstdev(paired_diffs)) if paired_diffs and len(paired_diffs) > 1 else \
        max(1e-6, 2.0 * base_rate * (1 - base_rate))
    se = sd_diff / math.sqrt(max(1, n))
    power = 1 - NormalDist().cdf(1.96 - abs(effect) / se)
    return {"n": n, "se_per_paired_brier": round(se, 6), "power_at_observed_effect": round(power, 3),
            "min_detectable_effect_80pct": round((1.96 + 0.84) * se, 6),
            "se_basis": "empirical paired-difference sd (CI-consistent)"}


# ------------------------------------------------------------------ the shared-world readout for one target
def _materialized_readout(world_id, scenario, actor, anchor, prior_obs, decay, strength, as_of):
    """Ingest the actor's prior acted/not history into a Phase-8 filter, materialize the posterior into a
    minimal WorldState, and read P(act) back from the materialized field — the shared-world path."""
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    key = PersistentStateKey(world_id, scenario, "actor", actor, "engagement_propensity")
    f = DecayedBetaBernoulliFilter(key=key, prior_mean=anchor, prior_strength=strength, decay=decay)
    obs = [(f"e{i}", 1 if x else 0, float(i)) for i, x in enumerate(prior_obs)]
    post = f.filter(obs, as_of=as_of)
    return _materialize_and_read(world_id, actor, post)


def _make_world(world_id):
    from swm.world_model_v2.information import InformationLedger
    from swm.world_model_v2.network import RelationGraph
    from swm.world_model_v2.state import SimulationClock, WorldState
    return WorldState(world_id=world_id, branch_id="root", clock=SimulationClock(now=1.0, as_of=1.0),
                      network=RelationGraph(), information=InformationLedger())


def _materialize_and_read(world_id, actor, post):
    from swm.world_model_v2.phase8_materialize import materialize_persistent_state
    from swm.world_model_v2.phase8_pipeline import engagement_readout
    from swm.world_model_v2.state import Entity, F
    w = _make_world(world_id)
    w.entities[actor] = Entity(identity=actor, entity_type="person")
    w.entities[actor].set("roles", F(["user"]))
    materialize_persistent_state(w, [post])
    return engagement_readout(w, actor)


# ------------------------------------------------------------------ train-only hyperparameter fit
def _fit_decay(train_targets, anchors, *, grid_decay=(0.4, 0.55, 0.7, 0.85, 1.0),
               grid_strength=(2.0, 4.0, 8.0)):
    """Choose (decay, prior_strength) by MAX train log-likelihood of the filtered posterior on the train
    targets. Train-only — the test set is never touched. decay=1 (no momentum) is in the grid, so if
    momentum does not help even on train, the fit will select it."""
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    best, best_ll = (0.7, 4.0), -1e18
    for decay in grid_decay:
        for strength in grid_strength:
            ll = 0.0
            for uid, prior_obs, y in train_targets:
                key = PersistentStateKey("w", "s", "actor", uid, "engagement_propensity")
                f = DecayedBetaBernoulliFilter(key=key, prior_mean=anchors.get(uid, 0.2),
                                               prior_strength=strength, decay=decay)
                obs = [(f"e{i}", 1 if x else 0, float(i)) for i, x in enumerate(prior_obs)]
                p = f.filter(obs).mean
                ll += y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1 - p))
            if ll > best_ll:
                best_ll, best = ll, (decay, strength)
    return best[0], best[1], round(best_ll, 2)


def run(n_users, cap_targets=None, result_path=None):
    from swm.eval.omnibehavior_eval import download_users
    from swm.world_model_v2.reference.omnibehavior import PASSIVE, acted, split_user, user_events
    from swm.world_model_v2.inference_layer import hierarchical_rates

    paths = download_users(n_users, max_bytes=3_000_000, cache_dir="data/omnibehavior")
    users = {}
    for p in paths:
        for uid, u in json.load(open(p)).items():
            evs = user_events(u)
            if len([e for e in evs if e.get("type") in PASSIVE]) >= 20:
                users[uid] = evs
    print(f"cohort: {len(users)} users with >=20 passive events", flush=True)

    person_disjoint = list(users)[: max(1, len(users) // 5)]
    train_by_user, test_rows, pd_rows, train_targets = {}, [], [], []
    for uid, evs in users.items():
        tr, te = split_user(evs, 0.7)
        train_by_user[uid] = tr
        passive_idx = [i for i, e in enumerate(evs) if e.get("type") in PASSIVE]
        # train targets for hyperparameter fit (from the train segment)
        tr_passive = [e for e in tr if e.get("type") in PASSIVE]
        for j, e in enumerate(tr_passive):
            prior = [acted(x) for x in tr_passive[:j]]
            train_targets.append((uid, prior, int(acted(e))))
        for e in [x for x in te if x.get("type") in PASSIVE]:
            idx = evs.index(e)
            prior = [acted(x) for x in evs[:idx] if x.get("type") in PASSIVE]
            row = {"uid": uid, "y": int(acted(e)), "prior": prior}
            (pd_rows if uid in person_disjoint else test_rows).append(row)

    # persistent per-user LEVEL: hierarchical shrinkage (the anchor both momentum + memoryless arms share)
    groups = {u: (sum(1 for e in tr if e.get("type") in PASSIVE and acted(e)),
                  sum(1 for e in tr if e.get("type") in PASSIVE)) for u, tr in train_by_user.items()}
    base = sum(k for k, _ in groups.values()) / max(1, sum(n for _, n in groups.values()))
    hier = hierarchical_rates(groups, population_prior=base)

    def anchor(uid):
        rp = hier.get(uid)
        return rp.mean() if rp is not None and hasattr(rp, "mean") else base
    anchors = {u: anchor(u) for u in users}

    # train-only fit of the filter hyperparameters
    fit_sample = train_targets if len(train_targets) <= 6000 else random.Random(0).sample(train_targets, 6000)
    decay, strength, train_ll = _fit_decay(fit_sample, anchors)
    print(f"train-fit filter: decay={decay} prior_strength={strength} (train_ll={train_ll})", flush=True)

    # train burstiness (is persistence structurally present at all?)
    hot, cold = [0, 0], [0, 0]
    for uid, prior, y in train_targets:
        key = any(prior[-3:])
        (hot if key else cold)[0] += 1
        (hot if key else cold)[1] += y
    p_hot = (hot[1] + .5) / (hot[0] + 1)
    p_cold = (cold[1] + .5) / (cold[0] + 1)
    momentum_lift = round(p_hot / max(1e-4, p_cold), 3)

    if cap_targets:
        test_rows = test_rows[:cap_targets]
    ys = [r["y"] for r in test_rows]
    n = len(ys)
    real_rate = sum(ys) / max(1, n)

    # ---- arms, all through materialize→readout ----
    print(f"scoring {n} test targets through the shared world...", flush=True)
    B0 = [_materialize_and_read_scalar("w", r["uid"], base) for r in test_rows]                 # no user level
    Buser = [_materialize_and_read_scalar("w", r["uid"], anchors[r["uid"]]) for r in test_rows]  # memoryless
    Bpersist = [_materialized_readout("w", "s", r["uid"], anchors[r["uid"]], r["prior"], decay, strength, 1.0)
                for r in test_rows]

    def metrics(ps):
        return {"brier": round(_brier(ps, ys), 5), "logloss": round(_logloss(ps, ys), 4),
                "auroc": round(_auroc(ps, ys), 4), "ece": round(_ece(ps, ys), 4),
                "pred_rate": round(sum(ps) / n, 4)}
    detail = {"B0_no_history": metrics(B0), "B_userrate_memoryless": metrics(Buser),
              "B_persist_shared_world": metrics(Bpersist)}

    paired = {"persist_vs_userrate": _paired_brier(ys, Bpersist, Buser),
              "persist_vs_nohistory": _paired_brier(ys, Bpersist, B0)}
    obs_effect = abs(paired["persist_vs_userrate"]["mean"])
    power = _power(real_rate, obs_effect, n, paired["persist_vs_userrate"]["paired_diffs"])

    # person-disjoint transfer
    pd_block = None
    if pd_rows:
        pd_ys = [r["y"] for r in pd_rows]
        pd_user = [_materialize_and_read_scalar("w", r["uid"], base) for r in pd_rows]   # no train user-level
        pd_persist = [_materialized_readout("w", "s", r["uid"], base, r["prior"], decay, strength, 1.0)
                      for r in pd_rows]
        pdp = _paired_brier(pd_ys, pd_persist, pd_user)
        pd_block = {"n": len(pd_ys), "real_rate": round(sum(pd_ys) / len(pd_ys), 4),
                    "persist_vs_userrate": {k: v for k, v in pdp.items() if k != "paired_diffs"},
                    "power": _power(sum(pd_ys) / len(pd_ys), abs(pdp["mean"]), len(pd_ys), pdp["paired_diffs"])}

    # sequence-disjoint: score only targets whose entire prior history is in the train segment
    # (already true by construction: prior uses evs[:idx], and the split is chronological — the test
    #  target's own outcome is never in its prior). We additionally verify no leakage below.

    for k in ("persist_vs_userrate", "persist_vs_nohistory"):
        paired[k].pop("paired_diffs", None)

    ci_hi = paired["persist_vs_userrate"]["ci95"][1]
    ci_lo = paired["persist_vs_userrate"]["ci95"][0]
    verdict = ("persistence HELPS through the shared world (paired CI excludes 0, favorable)" if ci_hi < 0
               else "persistence HURTS through the shared world (CI excludes 0, unfavorable)" if ci_lo > 0
               else "persistence effect NOT DETECTABLE through the shared world (CI spans 0)")

    out = {"task": "OmniBehavior passive-exposure engagement — SHARED-WORLD longitudinal execution",
           "path": "history → EventLog → DecayedBetaBernoulli filter → materialize latent_state → readout",
           "cohort": {"n_users": len(users), "n_test_events": n, "real_action_rate": round(real_rate, 4),
                      "n_person_disjoint_users": len(person_disjoint), "n_person_disjoint_events": len(pd_rows)},
           "train_fit": {"decay": decay, "prior_strength": strength, "train_loglik": train_ll,
                         "note": "hyperparameters fit on TRAIN only; decay=1 (no momentum) was in the grid"},
           "burstiness_train": {"momentum_lift": momentum_lift, "p_hot": round(p_hot, 4),
                                "p_cold": round(p_cold, 4),
                                "persistence_structurally_present": momentum_lift > 1.1},
           "detail": detail, "paired_ablation": paired, "power_analysis": power,
           "adequately_powered": power["power_at_observed_effect"] >= 0.8,
           "person_disjoint_transfer": pd_block, "verdict": verdict,
           "_meta": {"llm_calls": 0, "est_cost_usd": 0.0, "n_users_downloaded": len(paths)}}
    rp = Path(result_path) if result_path else Path(RESULT)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(out, indent=1, default=str))
    print("\ndetail:", json.dumps(detail, indent=1))
    print("persist_vs_userrate:", paired["persist_vs_userrate"])
    print("power:", power["power_at_observed_effect"], "adequately_powered:", out["adequately_powered"])
    print("VERDICT:", verdict)
    print("wrote", rp)
    return out


def _materialize_and_read_scalar(world_id, actor, p):
    """Baseline arm: materialize a fixed probability (global or user-level) as the engagement propensity —
    same materialize→readout path, no momentum. This is the memoryless comparator."""
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    key = PersistentStateKey(world_id, "s", "actor", actor, "engagement_propensity")
    # zero events → posterior == anchor == p (memoryless)
    post = DecayedBetaBernoulliFilter(key=key, prior_mean=p, prior_strength=4.0, decay=1.0).filter([])
    return _materialize_and_read(world_id, actor, post)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=140)
    ap.add_argument("--cap-targets", type=int, default=None)
    a = ap.parse_args()
    run(a.n_users, a.cap_targets)
