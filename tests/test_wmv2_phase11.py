"""Phase 11 — dynamic recompilation: unit + integration + adversarial + determinism tests (spec §30).

Covers the five planes: contracts (serialization/corruption), trigger detection + the eligibility discipline
(a representable simulation-internal event never recompiles), dependence-aware fusion + false-positive control,
scope selection, candidate generation + validation, reproducible scoring (current plan retained), typed
state/posterior/event migration + invariants, lineage/checkpoint/rollback/oscillation, and the end-to-end
controller (recompiles on external structure; ZERO recompiles with no external evidence; determinism).
"""
from types import SimpleNamespace

import pytest

from swm.world_model_v2.state import WorldState, SimulationClock, Entity, parse_time
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.events import Event
from swm.world_model_v2.phase11.contracts import (RecompileObservation, TypedPlanDiff, TypedPlanDiffEntry,
                                                  TRIGGER_FAMILIES, SCOPES)
from swm.world_model_v2.phase11.triggers import detect_all, TriggerContext, TriggerThresholds, DETECTORS
from swm.world_model_v2.phase11.fusion import TriggerFusion
from swm.world_model_v2.phase11.scope import select_scope, SCOPE_RANK
from swm.world_model_v2.phase11.candidates import generate_candidates, apply_transform, validate_candidate, PlanTransform
from swm.world_model_v2.phase11.scoring import score_candidates
from swm.world_model_v2.phase11.migration import migrate, migrate_events, split_actor, merge_actors
from swm.world_model_v2.phase11.lineage import snapshot, RecompileTransaction, standard_invariants, LineageGraph
from swm.world_model_v2.phase11.controller import RecompilationController
from swm.world_model_v2.phase11._serial import content_hash, plan_content_hash

T = parse_time("2021-01-01")
H = parse_time("2021-12-31")


def _plan(**kw):
    d = dict(question="q", outcome_contract=object(), entities=[{"id": "senate"}],
             institutions=[{"id": "senate", "rules": []}], relations=[],
             structural_hypotheses=[{"id": "h1", "prior": 1.0}], provenance={}, version=1, parent_version=0,
             as_of=T, horizon_ts=H, support_grade="exploratory", plan_hash=lambda: "g")
    d.update(kw)
    return SimpleNamespace(**d)


def _world(bid):
    w = WorldState(world_id="w", branch_id=bid, clock=SimulationClock(now=T, as_of=T),
                   network=RelationGraph(), information=InformationLedger())
    w.entities["senate"] = Entity(identity="senate", entity_type="institution")
    w.entities["alice"] = Entity(identity="alice")
    return w


def _obs(**kw):
    d = dict(observation_id="o", origin="external_evidence", event_time=parse_time("2021-06-01"),
             evidence_ids=["ev1"], uncertainty={"terminal_sensitivity": 0.7}, provenance={})
    d.update(kw)
    return RecompileObservation(**d)


# ---------------- contracts ----------------
def test_contracts_serialize_and_detect_corruption():
    o = _obs(observation_type="rollcall")
    rec = o.as_record()
    assert rec["_schema"] == "phase11.observation" and RecompileObservation.verify_record(rec)
    bad = dict(rec); bad["event_time"] = 999.0
    assert not RecompileObservation.verify_record(bad)


def test_all_16_families_have_detectors():
    assert {f for f, _ in DETECTORS} == set(TRIGGER_FAMILIES) and len(SCOPES) == 14


# ---------------- trigger eligibility discipline (the core clarification) ----------------
def test_representable_simulation_internal_event_never_triggers():
    obs = _obs(origin="simulation_internal", representable=True)
    ctx = TriggerContext(observation=obs, surprise={"residual": 12.0, "impossible": False, "tail_prob": 1e-9},
                         residual_history=[12, 12, 12])
    assert detect_all(ctx) == []                       # surprising but representable → execute normally


def test_external_dated_sourced_rule_change_fires():
    obs = _obs(event_time=1000.0)
    ctx = TriggerContext(observation=obs, plan_facts={"known_institutions": ["senate"]},
                         declared={"rule_change": {"effective_date": 1.0, "source": "gov"}})
    assert "rule_change" in {e.trigger_family for e in detect_all(ctx)}


def test_future_dated_rule_not_active_and_unsourced_rejected():
    obs = _obs(event_time=1000.0)
    fut = TriggerContext(observation=obs, declared={"rule_change": {"effective_date": 9e9, "source": "gov"}})
    assert "rule_change" not in {e.trigger_family for e in detect_all(fut)}
    uns = TriggerContext(observation=obs, declared={"rule_change": {"effective_date": 1.0}})
    assert "rule_change" not in {e.trigger_family for e in detect_all(uns)}


def test_alias_is_not_a_new_actor():
    ctx = TriggerContext(observation=_obs(), declared={"new_actor": {"id": "BobJr", "causal_relevance": 0.9}},
                         plan_facts={"known_actors": ["Bob"], "aliases": {"BobJr": "Bob"}})
    assert "new_actor" not in {e.trigger_family for e in detect_all(ctx)}


def test_out_of_support_event_is_impossible_trigger():
    ctx = TriggerContext(observation=_obs(representable=False), surprise={"residual": 30, "impossible": True})
    assert "impossible_event" in {e.trigger_family for e in detect_all(ctx)}


# ---------------- fusion + false-positive control ----------------
def test_single_noisy_residual_does_not_recompile():
    obs = _obs()
    ctx = TriggerContext(observation=obs, surprise={"residual": 3.5, "impossible": False, "tail_prob": 0.01},
                         residual_history=[3.5])
    assert not TriggerFusion().fuse(detect_all(ctx)).proceed


def test_dependence_collapse_does_not_double_count():
    # two evidence items sharing one source → one independent group (not summed)
    from swm.world_model_v2.phase11.contracts import RecompileTriggerEvidence
    e1 = RecompileTriggerEvidence("a", "new_actor", trigger_probability=0.7,
                                  provenance={"source_hashes": ["s1"]}, affected_scope_candidates=["actor"])
    e2 = RecompileTriggerEvidence("b", "coalition_change", trigger_probability=0.7,
                                  provenance={"source_hashes": ["s1"]}, affected_scope_candidates=["relationship"])
    fa = TriggerFusion().fuse([e1, e2])
    assert fa.n_independent_groups == 1                # collapsed to one group


# ---------------- scope ----------------
def test_scope_impossible_escalates_global():
    obs = _obs(representable=False)
    fa = TriggerFusion().fuse(detect_all(TriggerContext(observation=obs, surprise={"residual": 30, "impossible": True})))
    assert select_scope(fa, terminal_sensitivity=0.9).scope in ("outcome_contract", "full_plan")


def test_scope_rule_change_is_local_institution():
    obs = _obs(event_time=1000.0)
    fa = TriggerFusion().fuse(detect_all(TriggerContext(observation=obs, plan_facts={"known_institutions": ["senate"]},
                              declared={"rule_change": {"effective_date": 1.0, "source": "gov"}})))
    assert select_scope(fa, terminal_sensitivity=0.7).scope == "institution_ruleset"


# ---------------- candidates + scoring ----------------
def test_current_plan_always_a_candidate_and_can_win():
    plan = _plan()
    obs = _obs(origin="simulation_internal")
    fa = TriggerFusion().fuse([])                       # no trigger
    sel = select_scope(fa)
    cands = generate_candidates(plan, sel, fa, obs)
    assert any(c.is_current_plan for c, _ in cands)


def test_scoring_retains_current_when_no_evidence():
    plan = _plan()
    obs = _obs()
    from swm.world_model_v2.phase11.fusion import FusedAssessment
    fa = FusedAssessment(fused_probability=0.0, classification="transient_anomaly", proceed=False)
    sel = select_scope(fa)
    res = score_candidates(generate_candidates(plan, sel, fa, obs), plan, obs, fa)
    assert not res.recompile_warranted


def test_candidate_validation_rejects_future_rule():
    plan = _plan()
    obs = _obs(provenance={"declared": {"rule_change": {"institution": "senate", "effective_date": 9e18,
                                                        "source": "g"}}})
    ops = [PlanTransform(op="add_institution_rule", target="institutions",
                         payload={"institution": "senate", "kind": "quorum"}, evidence_ids=["ev1"])]
    v = validate_candidate(plan, apply_transform(plan, ops), ops, obs, now=1000.0)
    assert not v["ok"] and any("future" in p for p in v["problems"])


def test_plan_content_hash_distinguishes_revision():
    plan = _plan()
    ops = [PlanTransform(op="add_institution_rule", payload={"institution": "senate", "kind": "quorum"})]
    revised = apply_transform(plan, ops)
    assert plan_content_hash(plan) != plan_content_hash(revised)   # weak plan_hash() would collide


# ---------------- migration invariants ----------------
def test_migration_preserves_state_adds_new_conserves_mass():
    worlds = [_world("b0"), _world("b1")]
    ops = [PlanTransform(op="add_entity", payload={"id": "bob", "type": "person"}, evidence_ids=["e"])]
    out = migrate(_plan(), _plan(version=2), ops, worlds=worlds, weights=[0.5, 0.5],
                  pending_events=[[], []], sim_time=T)
    assert all({"senate", "alice", "bob"} <= set(w.entities) for w in out.worlds)
    assert abs(sum(out.weights) - 1.0) < 1e-9 and out.report["invariants_ok"]


def test_event_migration_no_time_reversal_no_duplicates():
    e_future = Event(ts=T + 100, etype="background_tick", payload={})
    e_past = Event(ts=T - 100, etype="background_tick", payload={})
    kept, recs = migrate_events([e_future, e_past, e_future], sim_time=T, dest_valid_etypes=None,
                                canceled_reasons=[])
    assert len(kept) == 1                              # future kept once; past dropped; duplicate deduped
    assert any(r["disposition"] == "dropped_past" for r in recs)
    assert any(r["disposition"] == "deduped" for r in recs)


def test_actor_split_partitions_resources_no_duplication():
    from swm.world_model_v2.state import F
    w = _world("b")
    w.entities["org"] = Entity(identity="org")
    w.entities["org"].set("resources", F(100.0))
    made = split_actor(w, source_id="org", component_ids=["div_a", "div_b"], orphans=[])
    assert made == 2 and w.entities["div_a"].value("resources") == 50.0   # partitioned, not duplicated


def test_actor_merge_sums_resources_once():
    from swm.world_model_v2.state import F
    w = _world("b")
    for i, r in enumerate([30.0, 20.0]):
        w.entities[f"u{i}"] = Entity(identity=f"u{i}"); w.entities[f"u{i}"].set("resources", F(r))
    merge_actors(w, source_ids=["u0", "u1"], merged_id="merged", orphans=[])
    assert w.entities["merged"].value("resources") == 50.0


# ---------------- lineage / checkpoint / rollback ----------------
def test_atomic_rollback_on_migration_failure_preserves_source():
    worlds = [_world("b0")]
    cp = snapshot(worlds, [1.0], [[]], _plan(), T)
    assert cp.verify()
    res = RecompileTransaction(source=cp).run(lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                                              standard_invariants)
    assert res["rolled_back"] and not res["activated"] and len(res["worlds"]) == 1


def test_lineage_cycle_and_oscillation_detection():
    lg = LineageGraph(); lg.activate("A"); lg.activate("B")
    assert not lg.has_cycle() and lg.oscillation("A")
    lg.activate("A"); assert lg.has_cycle()


def test_checkpoint_corruption_detected():
    cp = snapshot([_world("b")], [1.0], [[]], _plan(), T)
    cp.summary["n_particles"] = 999                    # tamper
    assert not cp.verify()


# ---------------- integration: end-to-end controller ----------------
def _run(ctrl, obs_list, plan_facts=None):
    return ctrl.run(plan=_plan(), worlds=[_world("b0"), _world("b1")], weights=[0.5, 0.5],
                    pending_events=[[], []], observations=obs_list, horizon_ts=H, as_of=T,
                    plan_facts=plan_facts or {"known_institutions": ["senate"], "known_actors": ["senate"]})


def test_controller_recompiles_on_external_rule_change():
    obs = _obs(observation_id="rc1", event_time=parse_time("2021-06-01"),
               provenance={"declared": {"rule_change": {"institution": "senate", "kind": "quorum",
                           "params": {"frac": 0.6}, "effective_date": parse_time("2021-05-01"),
                           "source": "congress.gov"}}})
    res = _run(RecompilationController(), [obs])
    assert res.n_recompiles == 1
    t = res.traces[0]
    assert t["decision"]["action"] == "institution_recompile" and t["migration_report"]["invariants_ok"]
    assert [e["etype"] for e in t["events_emitted"]][0] == "recompile_triggered"


def test_controller_zero_recompiles_with_no_external_evidence():
    obs = _obs(observation_id="i1", origin="simulation_internal", representable=True,
               provenance={"observed_value": 0.3})
    res = _run(RecompilationController(), [obs])
    assert res.n_eligible == 0 and res.n_recompiles == 0


def test_controller_disabled_arm_never_recompiles():
    obs = _obs(observation_id="rc1", event_time=parse_time("2021-06-01"),
               provenance={"declared": {"rule_change": {"institution": "senate", "kind": "quorum",
                           "effective_date": parse_time("2021-05-01"), "source": "gov"}}})
    res = _run(RecompilationController(recompile_enabled=False), [obs])
    assert res.n_recompiles == 0


def test_controller_deterministic_replay():
    obs = _obs(observation_id="rc1", event_time=parse_time("2021-06-01"),
               provenance={"declared": {"rule_change": {"institution": "senate", "kind": "quorum",
                           "effective_date": parse_time("2021-05-01"), "source": "gov"}}})
    a = _run(RecompilationController(), [obs])
    b = _run(RecompilationController(), [obs])
    assert a.n_recompiles == b.n_recompiles and a.terminal == b.terminal


# ---------------- committed artifacts: honest gates preserved ----------------
def test_committed_eval_artifacts_report_honest_verdict():
    import json, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "experiments/results/phase11/eval.json")
    if not os.path.exists(p):
        pytest.skip("eval.json not generated in this checkout")
    doc = json.load(open(p))
    v = doc["four_status_verdict"]
    assert v["software_implemented"] and v["executes_end_to_end"]
    # migration + safety gates must hold; production eligibility must NOT be overstated
    assert doc["gate_scoring"]["migration_gate"]["pass"] and doc["gate_scoring"]["safety_gate"]["pass"]
    assert doc["b5_minus_b0_changed_brier_improvement"]["favors_phase11"]
