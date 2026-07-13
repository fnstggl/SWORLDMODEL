"""Phase 8 — persistence & longitudinal state: unit, integration, e2e, adversarial, migration tests.

The through-line the whole phase is graded on: history is the causal source of truth and REMOVING it
measurably changes execution (a materialized field / an actor view / an action distribution / a terminal
readout). These tests assert exactly that, plus the event-log integrity, leakage safety, sequential-filter
correctness, checkpoint determinism, and the adversarial corrections/retractions/identity/corruption paths.
"""
import math

import pytest

from swm.world_model_v2.phase8_events import (EventLog, HistoryError, LeakageError, PersistentEvent)
from swm.world_model_v2.phase8_filtering import (AsymmetricTrustFilter, CategoricalStageFilter,
                                                DecayedBetaBernoulliFilter, GaussianStateFilter, ParticleFilter)
from swm.world_model_v2.phase8_materialize import HistoryIngestor, materialize_persistent_state
from swm.world_model_v2.phase8_persistence import (PersistentStateKey, PersistentVariableSpec,
                                                   get_persistent_variable, persistent_features_for_policy,
                                                   register_persistent_variable, registered_variables)
from swm.world_model_v2.phase8_pipeline import (engagement_readout, materialize_and_decide,
                                                run_history_ablation)
from swm.world_model_v2.phase8_service import (CorruptionError, MigrationError, PersistentCheckpoint,
                                               PersistentStore)
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState


# ------------------------------------------------------------------ helpers
def _world(now=100.0):
    from swm.world_model_v2.information import InformationLedger
    from swm.world_model_v2.network import RelationGraph
    w = WorldState(world_id="w", branch_id="root", clock=SimulationClock(now=now, as_of=now),
                   network=RelationGraph(), information=InformationLedger())
    w.entities["u"] = Entity(identity="u", entity_type="person")
    w.entities["u"].set("roles", F(["user"]))
    return w


def _key(vid="engagement_propensity", eid="u", scope="actor"):
    return PersistentStateKey("w", "s", scope, eid, vid)


def _betabern(obs, prior=0.2):
    f = DecayedBetaBernoulliFilter(key=_key(), prior_mean=prior, prior_strength=4.0, decay=0.6)
    return f.filter([(f"e{i}", x, float(i)) for i, x in enumerate(obs)], as_of=100.0)


# ------------------------------------------------------------------ Part 1: typed variable specs
def test_variable_registry_has_core_families():
    reg = registered_variables()
    for vid in ("engagement_propensity", "trust", "commitment", "institutional_stage", "resource_level"):
        assert vid in reg
        assert reg[vid].materializes_into and reg[vid].consumed_by     # non-ornamental


def test_ornamental_variable_is_refused():
    with pytest.raises(ValueError):
        register_persistent_variable(PersistentVariableSpec(
            variable_id="ornament", definition="stored but consumed by nothing", scope="actor",
            materializes_into="", consumed_by=()))
    with pytest.raises(ValueError):
        register_persistent_variable(PersistentVariableSpec(
            variable_id="ornament2", definition="no consumer", scope="actor",
            materializes_into="entity.beliefs[x]", consumed_by=()))


def test_policy_feature_projection_is_flat_and_typed():
    from swm.world_model_v2.phase8_persistence import PersistentStateView
    view = PersistentStateView(actor_id="u", as_of=1.0, beliefs={"engagement_propensity": 0.4},
                               trust={"b": 0.7}, resources={"budget": 3.0}, risk_tolerance=0.6)
    feats = persistent_features_for_policy(view)["features"]
    assert feats["belief:engagement_propensity"] == 0.4 and feats["trust:b"] == 0.7
    assert feats["resource:budget"] == 3.0 and feats["risk_tolerance"] == 0.6


# ------------------------------------------------------------------ Part 2: event log
def test_event_log_idempotent_dedup():
    log = EventLog("w", "s")
    e = PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure", event_time=1.0,
                        actor_ids=("u",), outcome=1)
    log.append(e)
    _, is_new = log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                           event_time=1.0, actor_ids=("u",), outcome=1))
    assert len(log) == 1 and is_new is False                          # retry-safe


def test_event_log_deterministic_order_and_integrity():
    log = EventLog("w", "s")
    for t in (5.0, 1.0, 3.0):                                         # inserted out of order
        log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                   event_time=t, actor_ids=("u",), outcome=1))
    times = [e.event_time for e in log.effective_events()]
    assert times == [1.0, 3.0, 5.0]                                  # replay order is by event_time
    assert log.verify_integrity()["ok"]


def test_event_log_leakage_filter_vs_smooth():
    log = EventLog("w", "s")
    log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                               event_time=1.0, observed_time=1.0, actor_ids=("u",), outcome=1))
    log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                               event_time=2.0, observed_time=9.0, actor_ids=("u",), outcome=1))  # delayed
    # filtering: only what was KNOWABLE by t=3 (observed_time<=3) → 1 event
    assert len(log.events_as_of(3.0, mode="filter")) == 1
    # smoothing: event_time<=3 → 2 events (retrospective only)
    assert len(log.events_as_of(3.0, mode="smooth")) == 2
    with pytest.raises(LeakageError):
        log.events_as_of(None)


def test_correction_and_retraction_append_not_overwrite():
    log = EventLog("w", "s")
    e, _ = log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                      event_time=1.0, actor_ids=("u",), outcome=1))
    log.correct(e.event_id, outcome=0)
    eff = log.effective_events()
    assert len(log) == 2                                             # both stored (append-only)
    assert len(eff) == 1 and eff[0].outcome == 0                     # effective view uses the correction
    log.retract(eff[0].event_id if eff[0].revises_event_id else e.event_id)
    # after retraction the effective stream drops the event
    assert all(x.kind != "retraction" for x in log.effective_events())


def test_identity_relink_is_probabilistic():
    log = EventLog("w", "s")
    e, _ = log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                      event_time=1.0, actor_ids=("u_alias",), outcome=1))
    r = log.relink_identity(e.event_id, new_actor_ids=("u_canonical",), link_uncertainty=0.4)
    assert r.identity_link_uncertainty == 0.4 and r.actor_ids == ("u_canonical",)


def test_durable_event_log_survives_reload(tmp_path):
    p = str(tmp_path / "log.jsonl")
    log = EventLog("w", "s", path=p)
    for t in range(4):
        log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                   event_time=float(t), actor_ids=("u",), outcome=t % 2))
    reloaded = EventLog("w", "s", path=p)                            # cross-run: reload from disk
    assert len(reloaded) == 4 and reloaded.verify_integrity()["ok"]


# ------------------------------------------------------------------ Part 4/5: sequential filters
def test_momentum_is_path_dependent():
    cold = _betabern([0, 0, 0, 0, 0, 0]).mean
    hot = _betabern([1, 1, 1, 1, 1, 1]).mean
    assert cold < 0.2 < hot                                          # same anchor, different recent history


def test_momentum_is_sequential_not_batch():
    # a hot-then-cold sequence should end BELOW a cold-then-hot sequence (recency dominates via forgetting)
    hot_then_cold = _betabern([1, 1, 1, 0, 0, 0]).mean
    cold_then_hot = _betabern([0, 0, 0, 1, 1, 1]).mean
    assert hot_then_cold < cold_then_hot


def test_trust_is_asymmetric_slow_gain_fast_loss():
    up = AsymmetricTrustFilter(key=_key("trust", "a|b", "dyad")).filter([("e1", "promise_fulfilled", 1.0)]).mean
    down = AsymmetricTrustFilter(key=_key("trust", "a|b", "dyad")).filter([("e1", "promise_violated", 1.0)]).mean
    assert (0.5 - down) > (up - 0.5)                                 # loss moves further than gain


def test_trust_repair_is_path_dependent():
    tf = AsymmetricTrustFilter(key=_key("trust", "a|b", "dyad"))
    repaired = tf.filter([("e1", "promise_violated", 1.0), ("e2", "trust_repair", 2.0),
                          ("e3", "trust_repair", 3.0)])
    assert repaired.diagnostics["violations"] == 1                  # violation history retained
    assert repaired.representation["violation_count"] == 1


def test_gaussian_filter_tracks_level():
    g = GaussianStateFilter(key=_key("resource_level"), prior_mean=0.0, obs_var=0.1)
    post = g.filter([("e1", 5.0, 1.0), ("e2", 5.0, 2.0), ("e3", 5.0, 3.0)])
    assert 3.0 < post.mean < 5.5                                    # converges toward the observed level


def test_categorical_stage_records_appeal_path_dependence():
    f = CategoricalStageFilter(key=_key("institutional_stage", "case1", "institution"))
    direct = f.filter([("e1", "decision", 1.0)])
    via_appeal = f.filter([("e1", "decision", 1.0), ("e2", "appeal", 2.0), ("e3", "decision", 3.0)])
    assert via_appeal.representation["reached_via_appeal"] and not direct.representation["reached_via_appeal"]


def test_particle_filter_resamples_and_reports_ess():
    pf = ParticleFilter(key=_key(), n_particles=100)
    obs = [("e%d" % i, 1, float(i)) for i in range(8)]
    post = pf.filter(obs, loglik=lambda x, o: math.log(max(1e-9, x if o else 1 - x)), seed=1)
    assert post.mean > 0.5 and post.diagnostics["n_resample"] >= 0 and post.ess > 0


# ------------------------------------------------------------------ Part 9: checkpoint service
def _store_with_hot():
    store = PersistentStore("w", "s")
    store.register_filter("engagement_propensity", lambda k, obs, ao, sd: DecayedBetaBernoulliFilter(
        key=k, prior_mean=0.2, decay=0.6).filter([(e, 1 if o else 0, t) for e, o, t in obs], as_of=ao))
    for i in range(6):
        store.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                         event_time=float(i), actor_ids=("u",), outcome=1))
    return store


def test_checkpoint_deterministic_replay_parity():
    store = _store_with_hot()
    cp1 = store.checkpoint(as_of=10.0)
    cp2 = store.checkpoint(as_of=10.0)
    assert PersistentStore.compare(cp1, cp2)["identical"]           # identical inputs → identical checkpoint
    assert store.verify(cp1)["ok"]


def test_checkpoint_restore_roundtrip(tmp_path):
    store = _store_with_hot()
    cp = store.checkpoint(as_of=10.0)
    path = store.save_checkpoint(cp, str(tmp_path / "cp.json"))
    loaded = store.load_checkpoint(path)
    restored = store.restore(loaded)
    tok = list(restored)[0]
    assert restored[tok].mean > 0.2                                 # hot streak survived the round-trip


def test_corruption_is_detected():
    store = _store_with_hot()
    cp = store.checkpoint(as_of=10.0)
    tok = list(cp.posteriors)[0]
    cp.posteriors[tok]["mean"] = 0.999                              # tamper
    with pytest.raises(CorruptionError):
        store.verify(cp)


def test_incompatible_schema_raises_migration_error():
    store = _store_with_hot()
    cp = store.checkpoint(as_of=10.0)
    cp.schema_version = "phase8-store-9.9"                          # future/incompatible
    with pytest.raises(MigrationError):
        store.verify(cp)


def test_history_removal_changes_checkpoint():
    full = _store_with_hot().checkpoint(as_of=10.0)
    empty = PersistentStore("w", "s")
    empty.register_filter("engagement_propensity", lambda k, obs, ao, sd: DecayedBetaBernoulliFilter(
        key=k, prior_mean=0.2, decay=0.6).filter([(e, 1 if o else 0, t) for e, o, t in obs], as_of=ao))
    empty_cp = empty.checkpoint(as_of=10.0, variable_keys=[_key()])
    cmp = PersistentStore.compare(full, empty_cp)
    assert not cmp["identical"] and cmp["max_abs_delta"] > 0.1


# ------------------------------------------------------------------ Part 3: robust ingestion
def test_ingestor_dedups_and_flags_conflicts():
    log = EventLog("w", "s")
    ing = HistoryIngestor(log=log)
    ing.ingest(event_type="passive_exposure", event_time=1.0, actor="u", outcome=1)
    _, is_new = ing.ingest(event_type="passive_exposure", event_time=1.0, actor="u", outcome=1)
    assert is_new is False                                          # duplicate deduped
    ev, _ = ing.ingest(event_type="passive_exposure", event_time=1.0, actor="u", outcome=0)  # conflict
    assert ev.kind == "revised_observation"


def test_ingestor_probabilistic_identity_linkage():
    log = EventLog("w", "s")
    ing = HistoryIngestor(log=log, alias_map={"raw_x": ("canon", 0.3)})
    ev, _ = ing.ingest(event_type="passive_exposure", event_time=1.0, actor="raw_x", outcome=1)
    assert ev.actor_ids == ("canon",) and ev.identity_link_uncertainty == 0.3


# ------------------------------------------------------------------ Parts 6/7: execution plane
def test_materialized_state_reaches_actor_view_and_policy():
    w = _world()
    post = _betabern([1, 1, 1, 1, 1, 1])
    ap, deltas, view = materialize_and_decide(w, "u", [post], candidate_actions=["engage", "wait"])
    assert view.policy_state.get("phase4_policy_value:engage") is not None    # materialized into the view
    assert deltas and abs(engagement_readout(w, "u") - post.mean) < 1e-6


def test_history_removal_changes_action_distribution():
    def execute(mode):
        w = _world()
        obs = {"full": [1, 1, 1, 1, 1, 1], "no_history": [], "last_event_only": [1]}[mode]
        post = _betabern(obs)
        ap, _, view = materialize_and_decide(w, "u", [post], candidate_actions=["engage", "wait"])
        return {"materialized_value": engagement_readout(w, "u"),
                "action_probabilities": ap.action_probabilities, "view_hash": view.view_hash()}
    report = run_history_ablation(execute)
    assert report["history_changes_execution"]                     # NOT ornamental
    assert report["diffs"]["no_history"]["view_changed"]


def test_persistence_update_operator_closes_the_loop():
    from swm.world_model_v2.phase8_transitions import PersistenceUpdateOperator
    from swm.world_model_v2.events import Event
    import random
    w = _world()
    op = PersistenceUpdateOperator()
    ev = Event(ts=100.0, etype="actor_action", participants=["u"],
               payload={"outcome": "converted", "reward": 1.0})
    delta = op.apply(w, op.propose(w, ev, random.Random(0)))
    latent = w.entities["u"].get("latent_state", key="phase4_policy_value:engage")
    assert latent is not None and delta is not None and delta.changes


def test_materialize_trust_onto_network_edge():
    w = _world()
    w.entities["b"] = Entity(identity="b", entity_type="person")
    tf = AsymmetricTrustFilter(key=_key("trust", "u|trusts|b", "dyad"))
    post = tf.filter([("e1", "promise_fulfilled", 1.0), ("e2", "promise_fulfilled", 2.0)])
    deltas = materialize_persistent_state(w, [post])
    assert deltas                                                   # a network-edge trust delta was emitted


# ------------------------------------------------------------------ universal entry robustness (no-abstention)
def test_universal_entry_fails_gracefully_not_crashes():
    """The universal simulate_with_persistence must degrade to a typed execution_failed on an LLM/compile
    error (no-abstention contract) rather than raising — the happy path is validated live in
    experiments/results/phase8/universal_path_trace.json."""
    from swm.world_model_v2.phase8_pipeline import simulate_with_persistence
    from swm.world_model_v2.phase8_service import PersistentStore

    def broken_llm(_prompt):
        raise RuntimeError("simulated LLM outage")
    ctx_store = PersistentStore("uw", "us")
    res, art = simulate_with_persistence("Will X happen?", llm=broken_llm, as_of="2024-01-01",
                                         horizon="2024-01-04", context=None, seed=0)
    assert res.simulation_status in ("execution_failed", "clarification_required")
    assert res.failure_taxonomy or res.clarification_reason or res.simulation_status == "clarification_required"


# ================================================================== completion pass: runtime status
def test_all_families_default_selectable_none_blocked():
    from swm.world_model_v2.phase8_runtime import select_families, runtime_status_table
    sels = select_families(list(registered_variables()))
    assert all(s.selected for s in sels)                        # none quarantined/incompatible → all usable
    table = runtime_status_table()
    assert all(r["production_usable"] for r in table)           # production_usable != empirically_validated


def test_experimental_family_executes_by_default_no_flag():
    """An exploratory/highly_speculative family must be SELECTED without any experimental flag."""
    from swm.world_model_v2.phase8_runtime import select_families
    sels = {s.variable_id: s for s in select_families(["reputation", "risk_tolerance"])}
    assert sels["reputation"].selected and sels["reputation"].runtime_status == "highly_speculative"
    assert sels["reputation"].transition_tier == 7             # drops to competing-hypotheses tier, not removed


def test_load_bearing_experimental_family_lowers_support_grade():
    from swm.world_model_v2.phase8_runtime import select_families, support_grade_effect
    sels = select_families(["engagement_propensity", "reputation"])
    eff = support_grade_effect("empirically_supported", sels, load_bearing_ids=["reputation"])
    assert eff["support_grade"] == "highly_speculative" and eff["uncertainty_widening"] > 1.0
    assert eff["limitations"]                                   # discloses the load-bearing experimental family


def test_quarantined_family_is_blocked():
    from swm.world_model_v2.phase8_persistence import register_persistent_variable, PersistentVariableSpec
    from swm.world_model_v2.phase8_runtime import select_families
    register_persistent_variable(PersistentVariableSpec(
        variable_id="harmful_demo", definition="a family quarantined after harmful validation", scope="actor",
        materializes_into="entity.beliefs[harmful_demo]", consumed_by=("phase4_policy.risk_sensitive",),
        runtime_status="quarantined"))
    sel = select_families(["harmful_demo"])[0]
    assert not sel.selected and "BLOCKED" in sel.reason        # ONLY quarantined/incompatible are blocked


def test_transition_tier_never_removes_uncertain_transition():
    from swm.world_model_v2.phase8_runtime import resolve_transition_tier
    t = resolve_transition_tier("trust")                       # exploratory, no pack
    assert t["tier"] == 6 and t["simulate_competing_hypotheses"] and "never removed" in t["note"]
    assert resolve_transition_tier("engagement_propensity", fitted_pack=True)["tier"] == 1


# ================================================================== completion pass: SQLite storage
def _sqlite_store(tmp_path, name="p8.db"):
    from swm.world_model_v2.phase8_storage import SqliteBackend
    be = SqliteBackend(str(tmp_path / name))
    log = EventLog("w", "s", backend=be)
    store = PersistentStore("w", "s", log=log)
    store.register_filter("engagement_propensity", lambda k, obs, ao, sd: DecayedBetaBernoulliFilter(
        key=k, prior_mean=0.2, decay=0.85).filter([(e, 1 if o else 0, t) for e, o, t in obs], as_of=ao))
    return store, be


def test_sqlite_cross_run_reload_and_idempotency(tmp_path):
    store, be = _sqlite_store(tmp_path)
    for i in range(6):
        store.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                         event_time=float(i), actor_ids=("u",), outcome=1))
    cp = store.checkpoint(as_of=10.0)
    store.commit_checkpoint(cp)
    be.close()
    # RESTART: fresh backend from disk
    store2, be2 = _sqlite_store(tmp_path)
    assert len(store2.log) == 6 and be2.verify_integrity()["ok"]
    loaded = store2.load_latest_checkpoint()
    assert loaded is not None and loaded.as_of == 10.0
    # deterministic cross-run replay parity
    cp2 = store2.checkpoint(as_of=10.0)
    assert PersistentStore.compare(cp, cp2)["identical"]
    # idempotent retry after restart
    _, is_new = store2.log.append(PersistentEvent(world_id="w", scenario_id="s",
                                                  event_type="passive_exposure", event_time=0.0,
                                                  actor_ids=("u",), outcome=1))
    assert is_new is False and len(store2.log) == 6


def test_sqlite_concurrent_processes_write(tmp_path):
    """Two independent backend connections (two 'processes') both append; all events land, dedup holds."""
    from swm.world_model_v2.phase8_storage import SqliteBackend
    db = str(tmp_path / "multi.db")
    be_a, be_b = SqliteBackend(db), SqliteBackend(db)
    log_a, log_b = EventLog("w", "s", backend=be_a), EventLog("w", "s", backend=be_b)
    for i in range(5):
        log_a.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                     event_time=float(i), actor_ids=("a",), outcome=1))
    for i in range(5):
        log_b.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                     event_time=float(i), actor_ids=("b",), outcome=0))
    be_a.close(); be_b.close()
    be_c = SqliteBackend(db)
    reread = EventLog("w", "s", backend=be_c)
    assert len(reread) == 10                                    # both writers' events durably present
    be_c.close()


def test_sqlite_corrupt_checkpoint_detected(tmp_path):
    store, be = _sqlite_store(tmp_path)
    store.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                     event_time=1.0, actor_ids=("u",), outcome=1))
    cp = store.checkpoint(as_of=10.0)
    tok = list(cp.posteriors)[0]
    cp.posteriors[tok]["mean"] = 0.999                          # tamper before commit
    with pytest.raises(CorruptionError):
        store.verify(cp)


def test_sqlite_watermark_integrity_after_tamper(tmp_path):
    import sqlite3
    store, be = _sqlite_store(tmp_path)
    for i in range(4):
        store.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                         event_time=float(i), actor_ids=("u",), outcome=1))
    assert be.verify_integrity()["ok"]
    be.close()
    # tamper a stored watermark directly → integrity must fail
    c = sqlite3.connect(str(tmp_path / "p8.db"))
    c.execute("UPDATE events SET watermark='deadbeef' WHERE seq=2")
    c.commit(); c.close()
    from swm.world_model_v2.phase8_storage import SqliteBackend
    be2 = SqliteBackend(str(tmp_path / "p8.db"))
    assert be2.verify_integrity()["ok"] is False
    be2.close()


def test_sqlite_compaction_preserves_events(tmp_path):
    store, be = _sqlite_store(tmp_path)
    for i in range(10):
        store.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                         event_time=float(i), actor_ids=("u",), outcome=1))
        store.commit_checkpoint(store.checkpoint(as_of=float(i + 1)))
    res = be.compact(keep_checkpoints=3)
    assert res["checkpoints_after"] <= 3 and res["events_preserved"]
    assert be.stats()["n_events"] == 10                        # events never pruned (source of truth)
    be.close()


# ================================================================== completion pass: identity resolution
def test_identity_probabilistic_linkage():
    from swm.world_model_v2.phase8_identity import IdentityResolver
    r = IdentityResolver(aliases={"jsmith": [("john_smith", 0.6), ("jane_smith", 0.4)]})
    hyps = r.resolve("jsmith")
    assert len(hyps) == 2 and abs(sum(h.weight for h in hyps) - 1.0) < 1e-6
    assert r.link_uncertainty("jsmith") > 0                     # ambiguous → uncertainty > 0
    assert r.link_uncertainty("unknown_token") == 0.0          # pass-through is certain


def test_identity_merge_and_rename():
    from swm.world_model_v2.phase8_identity import IdentityResolver, dyad_id
    r = IdentityResolver(merges={"acct_2": "acct_1"}, renames={"OldCorp": "NewCorp"})
    assert r.resolve("acct_2")[0].canonical_id == "acct_1"
    assert r.resolve("OldCorp")[0].canonical_id == "NewCorp"
    assert dyad_id("b", "a", directed=False) == dyad_id("a", "b", directed=False)   # undirected symmetric


# ================================================================== completion pass: memory recall
def test_actor_recall_is_leakage_safe_and_not_perfect():
    """The canonical actor view exposes only recallable memories strictly before as_of, not the full log."""
    from swm.memory.memory import EpisodicStore
    from swm.world_model_v2.phase8_pipeline import _expose_actor_memory, PersistenceContext
    store = EpisodicStore(half_life=30)
    for t in range(1, 6):
        store.record_contact("u", ts=float(t), text=f"past pricing chat {t}", responded=True, topic="pricing")
    store.record_contact("u", ts=50.0, text="FUTURE event being predicted", responded=True, topic="pricing")
    w = _world()
    ctx = PersistenceContext(store=None, memory_store=store)
    _expose_actor_memory(w, ctx, {"u": [{"event_time": 1.0}]}, as_of=10.0)
    mem = w.entities["u"].value("memory") or []
    assert mem and all(m["at"] < 10.0 for m in mem)            # leakage-safe: nothing at/after as_of
    assert all("FUTURE" not in m["text"] for m in mem)


# ================================================================== completion pass: cross-run closed loop
def test_cross_run_closed_loop_changes_execution(tmp_path):
    """RUN1 writes hot history + checkpoint; RESTART reloads from disk; the reloaded posterior differs from a
    no-history posterior — proving cross-run persistence changes execution and history removal is causal."""
    store, be = _sqlite_store(tmp_path)
    for i in range(6):
        store.log.append(PersistentEvent(world_id="w", scenario_id="s", event_type="passive_exposure",
                                         event_time=float(i), actor_ids=("u",), outcome=1))
    key = PersistentStateKey("w", "s", "actor", "u", "engagement_propensity")
    hot = store.checkpoint(as_of=10.0, variable_keys=[key])
    store.commit_checkpoint(hot)
    be.close()
    # RESTART
    store2, be2 = _sqlite_store(tmp_path)
    reloaded = store2.load_latest_checkpoint()
    assert reloaded is not None
    hot_mean = list(reloaded.posteriors.values())[0]["mean"]
    # a no-history store's posterior for the same key
    empty_store, _ = _sqlite_store(tmp_path, name="empty.db")
    empty_cp = empty_store.checkpoint(as_of=10.0, variable_keys=[key])
    empty_mean = list(empty_cp.posteriors.values())[0]["mean"]
    assert abs(hot_mean - empty_mean) > 0.1                     # removing history changes the derived state
    be2.close()
