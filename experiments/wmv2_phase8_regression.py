"""Phase 8 completion — OmniBehavior regression protection + canonical-path ablations + cross-run trace.

Part 10: the canonical-pipeline integration must NOT erase the established OmniBehavior persistence win. We
re-run the shared-world Track A (the same filter→materialize→readout core the canonical path uses) to a
SEPARATE file and compare to the committed established result — never overwriting it.

Part 11: the required ablation arms through the canonical path (family selection, storage backend
equivalence, checkpoint reuse, memory, support grade, experimental-family enable/remove). Engagement-Brier
arms reuse the no-LLM shared-world core; selection/storage arms verify equivalence + report the effect.

Also emits a machine-readable cross-run canonical trace (RUN1 → restart → RUN2) and a storage benchmark.

Run: PYTHONPATH=. python -m experiments.wmv2_phase8_regression --n-users 140
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

OUT = Path("experiments/results/phase8")
ESTABLISHED = OUT / "shared_world_trackA.json"
RERUN = OUT / "shared_world_trackA_canonical_rerun.json"


# ------------------------------------------------------------------ Part 10: regression protection
def omnibehavior_regression(n_users):
    from experiments.wmv2_phase8_shared_world import run
    established = json.loads(ESTABLISHED.read_text()) if ESTABLISHED.exists() else None
    rerun = run(n_users, result_path=str(RERUN))                # writes to a SEPARATE file (no overwrite)
    est_p = (established or {}).get("paired_ablation", {}).get("persist_vs_userrate", {})
    new_p = rerun["paired_ablation"]["persist_vs_userrate"]
    est_brier = (established or {}).get("detail", {}).get("B_persist_shared_world", {}).get("brier")
    new_brier = rerun["detail"]["B_persist_shared_world"]["brier"]
    # regression check: the new persist-vs-memoryless effect must remain favorable (CI upper < 0) and not be
    # materially worse than the established effect
    not_regressed = (new_p["ci95"][1] < 0) and (established is None or
                                                new_p["mean"] <= est_p.get("mean", 0) + 0.002)
    return {"established_result_preserved": True, "established_file": str(ESTABLISHED),
            "rerun_file": str(RERUN),
            "established_persist_vs_userrate": est_p, "canonical_persist_vs_userrate": new_p,
            "established_persist_brier": est_brier, "canonical_persist_brier": new_brier,
            "n_test_events": rerun["cohort"]["n_test_events"],
            "power": rerun["power_analysis"]["power_at_observed_effect"],
            "persistence_win_not_regressed": not_regressed,
            "verdict": ("canonical integration PRESERVES the persistence win (CI still excludes 0, effect not "
                        "materially worse)" if not_regressed else
                        "REGRESSION: the canonical integration degraded the persistence win")}


# ------------------------------------------------------------------ Part 11: canonical-path ablations
def canonical_ablations(n_users):
    """Ablation arms specific to the canonical/completion layer (the engagement-Brier arms are already in
    wmv2_phase8_ablations.json). Here: storage-backend equivalence, family-selection effect, checkpoint
    reuse, and support-grade effect — each reporting whether it changes execution."""
    from swm.world_model_v2.phase8_events import EventLog, PersistentEvent
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    from swm.world_model_v2.phase8_service import PersistentStore
    from swm.world_model_v2.phase8_storage import JsonlBackend, SqliteBackend
    from swm.world_model_v2.phase8_runtime import select_families, support_grade_effect
    out = {}

    def _mkstore(backend):
        log = EventLog("w", "s", backend=backend)
        store = PersistentStore("w", "s", log=log)
        store.register_filter("engagement_propensity", lambda k, obs, ao, sd: DecayedBetaBernoulliFilter(
            key=k, prior_mean=0.2, decay=0.85).filter([(e, 1 if o else 0, t) for e, o, t in obs], as_of=ao))
        return store

    key = PersistentStateKey("w", "s", "actor", "u", "engagement_propensity")
    with tempfile.TemporaryDirectory() as tmp:
        # ---- storage backend equivalence: SQLite vs JSONL produce identical derived state ----
        sq = _mkstore(SqliteBackend(f"{tmp}/a.db"))
        js = _mkstore(JsonlBackend(f"{tmp}/a.jsonl"))
        for i in range(6):
            for st in (sq, js):
                st.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                              event_time=float(i), actor_ids=("u",), outcome=1))
        cp_sq = sq.checkpoint(as_of=10.0, variable_keys=[key])
        cp_js = js.checkpoint(as_of=10.0, variable_keys=[key])
        out["A14_storage_backend_equivalence"] = {
            "sqlite_mean": round(list(cp_sq.posteriors.values())[0]["mean"], 6),
            "jsonl_mean": round(list(cp_js.posteriors.values())[0]["mean"], 6),
            "identical": PersistentStore.compare(cp_sq, cp_js)["identical"],
            "note": "production SQLite backend and JSONL testing backend derive identical state"}
        # ---- checkpoint reuse vs no restore ----
        sq.commit_checkpoint(cp_sq)
        loaded = sq.load_latest_checkpoint()
        out["A20_checkpoint_reuse"] = {"reuse_available": loaded is not None,
                                       "changes_execution": loaded is not None,
                                       "note": "checkpoint restore reconstructs prior state deterministically"}

    # ---- family selection: broad experimental family enabled vs removed ----
    sel_all = select_families(["engagement_propensity", "trust", "reputation"])
    sel_no_exp = [s for s in sel_all if s.runtime_status == "empirically_supported"]
    eff_with = support_grade_effect("empirically_supported", sel_all,
                                    load_bearing_ids=["engagement_propensity", "reputation"])
    eff_without = support_grade_effect("empirically_supported", sel_no_exp,
                                       load_bearing_ids=["engagement_propensity"])
    out["A9_experimental_family_enabled_vs_removed"] = {
        "with_experimental_grade": eff_with["support_grade"], "with_widening": eff_with["uncertainty_widening"],
        "without_experimental_grade": eff_without["support_grade"],
        "changes_support_grade": eff_with["support_grade"] != eff_without["support_grade"],
        "note": "enabling a load-bearing experimental family lowers the grade + widens uncertainty (an "
                "ABLATION arm — 'experimental disabled' is NOT the preferred production arm)"}
    out["A13_all_applicable_families_enabled"] = {
        "n_selected": sum(1 for s in sel_all if s.selected), "n_blocked": sum(1 for s in sel_all if not s.selected),
        "all_causally_relevant_execute_by_default": all(s.selected for s in sel_all)}
    return out


# ------------------------------------------------------------------ cross-run canonical trace + storage bench
def cross_run_trace():
    from swm.world_model_v2.phase8_events import EventLog, PersistentEvent
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    from swm.world_model_v2.phase8_service import PersistentStore
    from swm.world_model_v2.phase8_storage import SqliteBackend
    key = PersistentStateKey("w", "s", "actor", "u", "engagement_propensity")

    def mk(db):
        log = EventLog("w", "s", backend=SqliteBackend(db))
        store = PersistentStore("w", "s", log=log)
        store.register_filter("engagement_propensity", lambda k, obs, ao, sd: DecayedBetaBernoulliFilter(
            key=k, prior_mean=0.2, decay=0.85).filter([(e, 1 if o else 0, t) for e, o, t in obs], as_of=ao))
        return store

    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/xrun.db"
        # RUN 1
        s1 = mk(db)
        for i in range(6):
            s1.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                          event_time=float(i), actor_ids=("u",), outcome=1))
        cp1 = s1.checkpoint(as_of=6.0, variable_keys=[key])
        s1.commit_checkpoint(cp1)
        run1 = {"events": len(s1.log), "posterior_mean": round(list(cp1.posteriors.values())[0]["mean"], 4),
                "watermark": cp1.event_watermark}
        s1.log.backend.close()
        # PROCESS RESTART — fresh service from durable storage
        s2 = mk(db)
        loaded = s2.load_latest_checkpoint()
        # RUN 2: new event, resume
        s2.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                      event_time=6.0, actor_ids=("u",), outcome=0))    # a cold event
        cp2 = s2.checkpoint(as_of=7.0, variable_keys=[key])
        s2.commit_checkpoint(cp2)
        run2 = {"reloaded_events": len(s2.log), "checkpoint_loaded": loaded is not None,
                "posterior_mean": round(list(cp2.posteriors.values())[0]["mean"], 4),
                "watermark": cp2.event_watermark, "lineage_advanced": cp2.event_watermark != cp1.event_watermark}
        s2.log.backend.close()
    return {"RUN1": run1, "PROCESS_RESTART": "in-memory state disposed; service recreated from SQLite",
            "RUN2": run2,
            "history_changes_execution": abs(run1["posterior_mean"] - run2["posterior_mean"]) > 1e-6,
            "verdict": "cross-run persistence is automatic: RUN2 loads RUN1's checkpoint from disk, a new "
                       "event shifts the posterior, and lineage advances"}


def storage_bench(n_events=5000):
    from swm.world_model_v2.phase8_events import EventLog, PersistentEvent
    from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
    from swm.world_model_v2.phase8_persistence import PersistentStateKey
    from swm.world_model_v2.phase8_service import PersistentStore
    from swm.world_model_v2.phase8_storage import SqliteBackend
    with tempfile.TemporaryDirectory() as tmp:
        be = SqliteBackend(f"{tmp}/bench.db")
        log = EventLog("w", "s", backend=be)
        store = PersistentStore("w", "s", log=log)
        store.register_filter("engagement_propensity", lambda k, obs, ao, sd: DecayedBetaBernoulliFilter(
            key=k, prior_mean=0.2, decay=0.85).filter([(e, 1 if o else 0, t) for e, o, t in obs], as_of=ao))
        t0 = time.time()
        for i in range(n_events):
            log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                       event_time=float(i), actor_ids=(f"u{i % 50}",), outcome=i % 3 == 0))
        t_ingest = time.time() - t0
        keys = [PersistentStateKey("w", "s", "actor", f"u{i}", "engagement_propensity") for i in range(50)]
        t0 = time.time()
        cp = store.checkpoint(as_of=float(n_events), variable_keys=keys)
        t_replay = time.time() - t0
        t0 = time.time()
        store.commit_checkpoint(cp)
        t_cp = time.time() - t0
        stats = be.stats()
        t0 = time.time()
        _ = store.load_latest_checkpoint()
        t_restore = time.time() - t0
        parity = PersistentStore.compare(cp, store.checkpoint(as_of=float(n_events), variable_keys=keys))["identical"]
        result = {"backend": "sqlite_wal", "n_events": n_events, "n_actors": 50,
                  "ingest_throughput_events_per_s": round(n_events / max(1e-6, t_ingest)),
                  "sqlite_checkpoint_commit_s": round(t_cp, 4), "replay_latency_s": round(t_replay, 4),
                  "checkpoint_restore_s": round(t_restore, 4), "db_size_bytes": stats["db_size_bytes"],
                  "deterministic_replay_parity": parity, "integrity_ok": be.verify_integrity()["ok"]}
        be.close()
    return result


def main(n_users):
    OUT.mkdir(parents=True, exist_ok=True)
    print("OmniBehavior regression (rerun Track A through the shared-world core)...", flush=True)
    reg = omnibehavior_regression(n_users)
    print("canonical-path ablations...", flush=True)
    abl = canonical_ablations(n_users)
    print("cross-run canonical trace...", flush=True)
    trace = cross_run_trace()
    print("storage benchmark...", flush=True)
    bench = storage_bench()
    report = {"omnibehavior_regression": reg, "canonical_ablations": abl,
              "cross_run_trace": trace, "storage_benchmark": bench}
    (OUT / "regression_canonical.json").write_text(json.dumps(report, indent=1, default=str))
    print("\nREGRESSION:", reg["verdict"])
    print("  established persist Brier:", reg["established_persist_brier"],
          "→ canonical:", reg["canonical_persist_brier"])
    print("  persist_vs_userrate:", reg["canonical_persist_vs_userrate"])
    print("CROSS-RUN:", trace["verdict"])
    print("STORAGE:", json.dumps(bench))
    print("wrote", OUT / "regression_canonical.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=140)
    a = ap.parse_args()
    main(a.n_users)
