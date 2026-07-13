"""Phase 8 — causal ablations (Part 17), forensic traces (Part 19), checkpoint/performance bench (Part 10).

ABLATIONS: every ablation runs through the SHARED world and reports whether it changes execution (the
materialized field / action distribution) and the held-out Brier on the engagement task. Ablations that a
single-variable engagement task cannot exercise (trust/commitment/institutional/resource) are run as
separate causal-change checks on their own filters/materialization (does removing that history change the
materialized state?) rather than faked — and are labeled as such. A component whose ablation changes nothing
is flagged ORNAMENTAL.

FORENSIC TRACES: paired full-history vs history-removed traces for a stratified sample, showing exactly where
history changed the trajectory (materialized value → action distribution → readout). Machine-readable here;
two human-readable traces are embedded in docs/WMV2_PHASE8_PERSISTENCE.md.

PERFORMANCE: ingestion throughput, replay/filter latency, checkpoint size + save/restore time, replay parity.

Run: PYTHONPATH=. python -m experiments.wmv2_phase8_ablations --n-users 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

OUT = Path("experiments/results/phase8")


# ------------------------------------------------------------------ engagement-task ablation arms
def _load_engagement(n_users, cap):
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
    train_by_user, rows = {}, []
    for uid, evs in users.items():
        tr, te = split_user(evs, 0.7)
        train_by_user[uid] = tr
        for e in [x for x in te if x.get("type") in PASSIVE]:
            idx = evs.index(e)
            prior = [acted(x) for x in evs[:idx] if x.get("type") in PASSIVE]
            rows.append({"uid": uid, "y": int(acted(e)), "prior": prior})
    groups = {u: (sum(1 for e in tr if e.get("type") in PASSIVE and acted(e)),
                  sum(1 for e in tr if e.get("type") in PASSIVE)) for u, tr in train_by_user.items()}
    base = sum(k for k, _ in groups.values()) / max(1, sum(n for _, n in groups.values()))
    hier = hierarchical_rates(groups, population_prior=base)
    raw_rate = {u: (k / n if n else base) for u, (k, n) in groups.items()}   # unshrunk
    if cap:
        rows = rows[:cap]
    return rows, base, hier, raw_rate


def _readout(uid, anchor, prior, decay, strength):
    from experiments.wmv2_phase8_shared_world import _materialize_and_read
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    key = PersistentStateKey("w", "s", "actor", uid, "engagement_propensity")
    obs = [(f"e{i}", 1 if x else 0, float(i)) for i, x in enumerate(prior)]
    post = DecayedBetaBernoulliFilter(key=key, prior_mean=anchor, prior_strength=strength, decay=decay).filter(obs)
    return _materialize_and_read("w", uid, post)


def engagement_ablations(n_users, cap):
    from experiments.wmv2_phase8_shared_world import _brier
    rows, base, hier, raw_rate = _load_engagement(n_users, cap)
    ys = [r["y"] for r in rows]

    def anchor(u):
        rp = hier.get(u)
        return rp.mean() if rp is not None and hasattr(rp, "mean") else base
    # each arm returns predictions on identical held-out rows
    arms = {
        "A21_full": [_readout(r["uid"], anchor(r["uid"]), r["prior"], 0.85, 8.0) for r in rows],
        "A1_no_history": [_readout(r["uid"], anchor(r["uid"]), [], 0.85, 8.0) for r in rows],
        "A2_last_event_only": [_readout(r["uid"], anchor(r["uid"]), r["prior"][-1:], 0.85, 8.0) for r in rows],
        "A5_no_persistent_latent": [base for _ in rows],
        "A4_perfect_memory_no_decay": [_readout(r["uid"], anchor(r["uid"]), r["prior"], 1.0, 8.0) for r in rows],
        "A9_no_person_specific": [_readout("global", base, r["prior"], 0.85, 8.0) for r in rows],
        "A8_no_hierarchical_shrinkage": [_readout(r["uid"], raw_rate.get(r["uid"], base), r["prior"], 0.85, 8.0)
                                         for r in rows],
        "A18_truncated_window_k3": [_readout(r["uid"], anchor(r["uid"]), r["prior"][-3:], 0.85, 8.0)
                                    for r in rows],
    }
    full_b = _brier(arms["A21_full"], ys)
    report = {}
    for name, ps in arms.items():
        b = _brier(ps, ys)
        report[name] = {"brier": round(b, 5), "delta_vs_full": round(b - full_b, 5),
                        "changes_execution": abs(b - full_b) > 1e-6 or name == "A21_full"}
    report["_n"] = len(ys)
    report["_ornamental_arms"] = [k for k, v in report.items() if isinstance(v, dict)
                                  and not v["changes_execution"] and k != "A21_full"]
    return report


# ------------------------------------------------------------------ cross-family causal-change checks
def cross_family_ablations():
    """For families the engagement task does not exercise, verify removing history changes the MATERIALIZED
    world state (the causal gate) on their own filters — honest, not faked as engagement arms."""
    from swm.world_model_v2.phase8_filtering import (AsymmetricTrustFilter, CategoricalStageFilter,
                                                    GaussianStateFilter)
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    out = {}
    # trust: violation history vs none
    tk = PersistentStateKey("w", "s", "dyad", "a|trusts|b", "trust")
    full = AsymmetricTrustFilter(key=tk).filter([("e1", "promise_fulfilled", 1.0),
                                                 ("e2", "promise_violated", 2.0)]).mean
    none = AsymmetricTrustFilter(key=tk).filter([]).mean
    out["trust"] = {"full_history_mean": round(full, 4), "no_history_mean": round(none, 4),
                    "delta": round(full - none, 4), "changes_execution": abs(full - none) > 1e-6}
    # institutional stage: reached via appeal vs directly
    sk = PersistentStateKey("w", "s", "institution", "case1", "institutional_stage")
    via = CategoricalStageFilter(key=sk).filter([("e1", "decision", 1.0), ("e2", "appeal", 2.0),
                                                ("e3", "decision", 3.0)])
    direct = CategoricalStageFilter(key=sk).filter([("e1", "decision", 1.0)])
    out["institutional_stage"] = {"via_appeal_path": via.representation["path"],
                                  "direct_path": direct.representation["path"],
                                  "path_dependent": via.representation["reached_via_appeal"],
                                  "changes_execution": via.representation["path"] != direct.representation["path"]}
    # resource: spend history vs none
    rk = PersistentStateKey("w", "s", "actor", "u", "resource_level")
    spent = GaussianStateFilter(key=rk, prior_mean=10.0).filter([("e1", 7.0, 1.0), ("e2", 4.0, 2.0)]).mean
    unspent = GaussianStateFilter(key=rk, prior_mean=10.0).filter([]).mean
    out["resource_level"] = {"spent_history_mean": round(spent, 3), "no_history_mean": round(unspent, 3),
                             "changes_execution": abs(spent - unspent) > 1e-6}
    return out


# ------------------------------------------------------------------ forensic traces (paired full vs removed)
def forensic_traces(n_users, k=6):
    from experiments.wmv2_phase8_shared_world import _materialize_and_read
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    from swm.world_model_v2.phase8_pipeline import materialize_and_decide
    from swm.world_model_v2.state import Entity, F
    from experiments.wmv2_phase8_shared_world import _make_world
    rows, base, hier, _ = _load_engagement(n_users, cap=None)

    def anchor(u):
        rp = hier.get(u)
        return rp.mean() if rp is not None and hasattr(rp, "mean") else base
    # stratify: pick rows with varied prior lengths + both outcomes
    hot = [r for r in rows if len(r["prior"]) >= 5 and sum(r["prior"][-5:]) >= 3][:k // 2]
    cold = [r for r in rows if len(r["prior"]) >= 5 and sum(r["prior"][-5:]) == 0][:k - len(hot)]
    traces = []
    for r in (hot + cold):
        uid, prior = r["uid"], r["prior"]
        key = PersistentStateKey("w", "s", "actor", uid, "engagement_propensity")
        obs = [(f"e{i}", x, float(i)) for i, x in enumerate(prior)]
        post_full = DecayedBetaBernoulliFilter(key=key, prior_mean=anchor(uid), prior_strength=8.0,
                                               decay=0.85).filter(obs, as_of=1.0)
        post_none = DecayedBetaBernoulliFilter(key=key, prior_mean=anchor(uid), prior_strength=8.0,
                                               decay=0.85).filter([], as_of=1.0)
        # run BOTH through the real ActorView + policy
        traces_row = {"actor": uid, "y_true": r["y"], "n_prior": len(prior),
                      "recent5_acted": sum(prior[-5:]), "anchor_user_rate": round(anchor(uid), 4)}
        for label, post in (("full_history", post_full), ("history_removed", post_none)):
            w = _make_world("w")
            w.entities[uid] = Entity(identity=uid, entity_type="person")
            w.entities[uid].set("roles", F(["user"]))
            ap, deltas, view = materialize_and_decide(w, uid, [post], candidate_actions=["engage", "wait"])
            traces_row[label] = {
                "posterior_mean": round(post.mean, 4),
                "posterior_lineage_tail": post.lineage[-2:],
                "materialized_field": "latent_state[phase4_policy_value:engage]",
                "action_probabilities": {k2: round(v, 4) for k2, v in ap.action_probabilities.items()},
                "n_persistent_deltas": len(deltas), "view_hash": view.view_hash()}
        traces_row["delta_posterior"] = round(traces_row["full_history"]["posterior_mean"]
                                              - traces_row["history_removed"]["posterior_mean"], 4)
        traces_row["history_changed_trajectory"] = (
            traces_row["full_history"]["view_hash"] != traces_row["history_removed"]["view_hash"])
        traces.append(traces_row)
    return traces


# ------------------------------------------------------------------ checkpoint + performance bench
def performance_bench(n_events=5000):
    from swm.world_model_v2.phase8_events import PersistentEvent
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    from swm.world_model_v2.phase8_service import PersistentStore
    store = PersistentStore("w", "s")
    store.register_filter("engagement_propensity", lambda k, obs, ao, sd: DecayedBetaBernoulliFilter(
        key=k, prior_mean=0.2, decay=0.85).filter([(e, 1 if o else 0, t) for e, o, t in obs], as_of=ao))
    t0 = time.time()
    for i in range(n_events):
        store.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                         event_time=float(i), actor_ids=(f"u{i % 50}",), outcome=i % 3 == 0))
    t_ingest = time.time() - t0
    keys = [PersistentStateKey("w", "s", "actor", f"u{i}", "engagement_propensity") for i in range(50)]
    t0 = time.time()
    cp = store.checkpoint(as_of=float(n_events), variable_keys=keys)
    t_replay = time.time() - t0
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/cp.json"
        t0 = time.time()
        store.save_checkpoint(cp, path)
        t_save = time.time() - t0
        size = Path(path).stat().st_size
        t0 = time.time()
        loaded = store.load_checkpoint(path)
        restored = store.restore(loaded)
        t_restore = time.time() - t0
        # replay parity: re-checkpoint from the same log must match
        cp2 = store.checkpoint(as_of=float(n_events), variable_keys=keys)
        parity = PersistentStore.compare(cp, cp2)["identical"]
    return {"n_events": n_events, "n_actors": 50,
            "ingest_throughput_events_per_s": round(n_events / max(1e-6, t_ingest)),
            "replay_latency_s": round(t_replay, 4),
            "replay_per_actor_ms": round(1000 * t_replay / 50, 3),
            "checkpoint_size_bytes": size, "checkpoint_save_s": round(t_save, 4),
            "checkpoint_restore_s": round(t_restore, 4), "n_restored": len(restored),
            "deterministic_replay_parity": parity, "integrity_ok": store.verify(cp)["ok"]}


def main(n_users):
    OUT.mkdir(parents=True, exist_ok=True)
    print("engagement ablations...", flush=True)
    eng = engagement_ablations(n_users, cap=None)
    print("cross-family ablations...", flush=True)
    cross = cross_family_ablations()
    print("forensic traces...", flush=True)
    traces = forensic_traces(n_users)
    print("performance bench...", flush=True)
    perf = performance_bench()
    report = {"engagement_task_ablations": eng, "cross_family_causal_checks": cross, "performance": perf,
              "note": ("engagement ablations report held-out Brier deltas through the shared world; "
                       "cross-family checks verify removing that family's history changes materialized state; "
                       "an ablation that changes nothing is flagged ornamental")}
    (OUT / "ablations.json").write_text(json.dumps(report, indent=1, default=str))
    (OUT / "forensic_traces.json").write_text(json.dumps(traces, indent=1, default=str))
    print("\nENGAGEMENT ABLATIONS (Brier, delta vs full):")
    for k, v in eng.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['brier']} (Δ {v['delta_vs_full']:+})")
    print("ornamental arms:", eng["_ornamental_arms"])
    print("CROSS-FAMILY:", json.dumps({k: v["changes_execution"] for k, v in cross.items()}))
    print("PERF:", json.dumps(perf))
    print("forensic traces:", len(traces), "→", OUT / "forensic_traces.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=40)
    a = ap.parse_args()
    main(a.n_users)
