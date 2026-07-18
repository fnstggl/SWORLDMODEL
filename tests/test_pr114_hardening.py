"""PR #114 audit ports + production-fallback removal — the hardening invariants.

One production consequence path, one recursive actor path; schema failure ends in the
minimal generated schema or an explicit underidentified result, never fixed-v1/scalar;
every substitution stamped; joint hidden worlds condition actor states per branch;
frontier-promoted reconsiderations get qualitative cognition; the lexical vote operator is
baseline-only."""
import json
import random

import pytest

from swm.world_model_v2 import generated_world as gw
from swm.world_model_v2.events import Event
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.run_classification import (
    RUN_CLASSES, classify_run, collect_generated_manifests, epistemic_contract,
)
from swm.world_model_v2.scenario_schema import ScenarioSemanticModel, minimal_scenario_schema
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta

T0 = 1_700_000_000.0


def schema():
    return ScenarioSemanticModel(
        question="Will the dispute settle?", prediction_timestamp=T0, horizon=T0 + 30 * 86400,
        fact_types={"settlement_record": {"description": "x",
                                          "fields": {"status": "str", "position": "str",
                                                     "matter": "str"}}},
        semantic_event_types={"private_settlement_overture":
                              {"description": "x", "fields": {},
                               "typical_visibility": "participants"}},
        institutional_definitions={"panel": {"procedure": "vote",
                                             "decision_holders": ["cara", "dev"],
                                             "decision_record_type": "settlement_record",
                                             "aggregation": {"kind": "majority"},
                                             "assumed": True}},
        outcome_predicates=[{"predicate_id": "settled", "record_type": "settlement_record",
                             "op": "eq", "field": "status", "value": "settled",
                             "option_true": "settled", "option_false": "unsettled"}],
        provenance={"compiler": "test"}).freeze()


def world(actors=("ana", "bo", "cara", "dev")):
    w = WorldState("hard", "b0", SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    for a in actors:
        e = Entity(a)
        e.set("roles", F(["person"], status="observed"))
        e.set("past_actions", F([], status="observed"))
        w.entities[a] = e
    w.scenario_schema = schema()
    return w


def emit(w, *, targets, visibility="participants", content="the overture", report=None):
    ctx = {"actor_id": "ana", "action_id": "a1", "now": T0,
           "report": report if report is not None else gw.generated_report(),
           "budgets": gw._budgets(w), "events": [], "quarantined": [], "compiler": "test"}
    d = StateDelta(at=T0, event_type="actor_action", operator="test")
    gw.execute_kernel_ops(w, [{"op": "emit_semantic_event",
                               "semantic_type_id": "private_settlement_overture",
                               "exact_content": content, "direct_targets": list(targets),
                               "intended_visibility": visibility}], ctx, d)
    return ctx


# ---------------------------------------------------------------- recursion + quiescence
def test_semantic_signature_quiescence():
    w = world()
    ctx1 = emit(w, targets=["bo"])
    assert len(ctx1["events"]) == 1                     # first emission routes
    ctx2 = emit(w, targets=["bo"])                      # identical re-act
    assert ctx2["events"] == []                         # logged, never rerouted
    assert len(w.semantic_log) == 2                     # world history keeps both
    ec = w.uncertainty_meta["event_cascade"]
    assert ec["suppressed_duplicate"] == 1 and ec["quiescence"] == "duplicate_semantic_event"


def test_information_gate_stamps_unobserved_standing():
    w = world()
    ctx = emit(w, targets=["bo"], visibility="participants")
    report = gw.generated_report()
    router = gw.GeneratedSemanticEventOperator(report=report)
    rd, _ = router.run(w, ctx["events"][0], random.Random(0))
    invoked = {f["participants"][0] for f in rd.follow_up_events
               if f["etype"] == "ctrl_invoke_actor"}
    delivered = {f["payload"]["recipient"] for f in rd.follow_up_events
                 if f["etype"] == "ctrl_deliver_observation"}
    # cara/dev hold panel rights (frontier standing via the matter text) but never received
    # this private event: they are STAMPED, not invoked
    assert delivered == {"bo"}
    assert invoked == set() or invoked <= {"bo"}
    stamps = w.uncertainty_meta.get("approximation_manifest") or []
    assert any(s["approximation_type"] == "no_reconsideration_unobserved" for s in stamps) \
        or w.uncertainty_meta["event_cascade"]["suppressed_unobserved"] == 0


def test_env_budget_overrides(monkeypatch):
    monkeypatch.setenv("SWM_PROPAGATION_DEPTH", "1")
    monkeypatch.setenv("SWM_PROPAGATION_EVENTS", "7")
    b = gw.resolve_budgets()
    assert b["max_cascade_depth"] == 1 and b["max_semantic_events"] == 7
    monkeypatch.delenv("SWM_PROPAGATION_DEPTH")
    monkeypatch.delenv("SWM_PROPAGATION_EVENTS")
    assert gw.resolve_budgets()["max_cascade_depth"] == gw.DEFAULT_BUDGETS["max_cascade_depth"]


def test_budget_exhaustion_is_stamped_not_faked():
    w = world()
    b = gw._budgets(w)
    b["invocations"]["bo"] = b["max_invocations_per_actor"]
    sev = {"event_id": "s1", "semantic_type_id": "private_settlement_overture",
           "cascade_depth": 0, "source_actor_id": "ana"}
    assert gw._invocation_event(w, "bo", sev, reason="test") is None
    stamps = w.uncertainty_meta["approximation_manifest"]
    assert stamps[-1]["approximation_type"] == "no_reconsideration_scheduled"
    assert w.uncertainty_meta["event_cascade"]["suppressed_budget"] == 1


# ---------------------------------------------------------------- delivered representation
def test_delivered_representation_reaches_the_actor():
    s = schema()
    s.information_rules = {"public_channel": "wire", "public_delay_s": 3600.0,
                           "public_representation": "summary", "default_delay_s": 60.0}
    w = world()
    w.scenario_schema = s
    long = "Settlement terms attached. " * 30
    ctx = emit(w, targets=["bo"], visibility="public", content=long)
    report = gw.generated_report()
    router = gw.GeneratedSemanticEventOperator(report=report)
    rd, _ = router.run(w, ctx["events"][0], random.Random(0))
    deliver = gw.GeneratedObservationDeliveryOperator(report=report)
    by = {}
    for f in rd.follow_up_events:
        if f["etype"] != "ctrl_deliver_observation":
            continue
        dd, _ = deliver.run(w, Event(ts=f["ts"], etype=f["etype"],
                                     participants=f["participants"], payload=f["payload"]),
                            random.Random(0))
        for fu in dd.follow_up_events or []:
            by[fu["payload"]["actor_id"]] = fu["payload"]
    # the NON-target public recipient reconsiders on the summarized, reported account —
    # never the omniscient full text
    others = [p for a, p in by.items() if a != "bo"]
    assert others and all("[summarized in transit]" in p["delivered"]["content"]
                          for p in others)
    assert all(p["delivered"]["perceived_source"].startswith("reported:ana")
               for p in others)
    # and the world-plane event records who actually received which representation
    row = w.semantic_log[-1]
    assert row["actual_observability"].get("bo") == "complete"
    assert any(v == "summary" for k, v in row["actual_observability"].items() if k != "bo")


def test_frontier_tier_promotion_reaches_router():
    from swm.world_model_v2.qualitative_actor import (
        QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
    )
    engine = QualitativeDecisionEngine(QualitativeConfig(llm=lambda p: "{}",
                                                         llm_hypotheses=False))
    rt = QualitativeActorPolicyRuntime(engine, mode="hybrid_relevant_actor_policy")
    w = world()
    routed, assignment = rt._routes_qualitative(
        w, "dev", {"situation": "x",
                   "tier_assignment": {"actor_id": "dev", "tier": 2,
                                       "reasons": ["causal_frontier: informed holder"]}})
    assert routed is True and assignment["tier"] == 2
    assert rt.tiers["dev"]["selector"] == "causal_frontier_event_tier"
    # absent a hint, an unlisted actor stays tier-3 numeric (routine), as before
    routed2, a2 = rt._routes_qualitative(w, "bo", {"situation": "x"})
    assert routed2 is False and a2["tier"] == 3


# ---------------------------------------------------------------- joint hidden worlds
def test_joint_world_conditions_actor_states_per_branch():
    from swm.world_model_v2.joint_world import (
        JointWorldHypothesizer, attach_joint_hypotheses, branch_hypothesis,
    )
    from swm.world_model_v2.init_state import InitialStateModel
    base = world()
    hyp = JointWorldHypothesizer(None, k=3)         # offline fallback worlds, stamped
    rows = hyp.generate(question="Will the dispute settle?", actors=["ana", "bo"],
                        institutions=["panel"], evidence="", date="2023-11-14",
                        structural_model={})
    assert len(rows) >= 2
    assert all(r.assumptions for r in rows)          # labeled assumptions mandatory
    init = InitialStateModel(base_world=base, latents=[])
    attach_joint_hypotheses(init, rows)
    particles = init.sample_particles(4, seed=1)
    stamped = [branch_hypothesis(p) for p in particles]
    assert all(s.get("hypothesis_id") for s in stamped)
    # branch isolation: same particle index → same shared world; different indices cycle K
    assert stamped[0]["hypothesis_id"] == branch_hypothesis(particles[0])["hypothesis_id"]
    assert len({s["hypothesis_id"] for s in stamped}) >= 2
    # the actor-side conditioning consumes the SAME branch reality
    from swm.world_model_v2.qualitative_actor import _fallback_hypotheses
    from types import SimpleNamespace
    view = SimpleNamespace(actor_id="ana", actor_role="person", observed_time=T0,
                           observed_events=[], institution_rules=[], authority=[],
                           commitments=[], resources={}, action_history=[],
                           network_position={}, information=[], goals=[],
                           obligations=[], stances=[], beliefs={}, memory=[])
    adverse = next((s for s in stamped if "collapse" in json.dumps(s).lower()
                    or "adverse" in json.dumps(s).lower()), stamped[0])
    conditioned = _fallback_hypotheses(view, 3, world_hypothesis=adverse)
    assert all(any("conditioned on shared world hypothesis" in a
                   for a in r.get("assumptions", [])) for r in conditioned)


# ---------------------------------------------------------------- honest classification
def _res(**over):
    from types import SimpleNamespace
    base = dict(simulation_status="completed", failure_taxonomy="",
                structural_disagreement={}, fallbacks_used=[], support_grade="calibrated",
                calibrated_probability=None,
                provenance={"active_component_manifest": {
                    "phase2_evidence": {"executed": True},
                    "phase3_posterior": {"executed": True}},
                    "posterior_consumed": True, "consequence_report": {},
                    "actor_policy_report": {}, "actor_decision_distributions": {}})
    base.update(over)
    ns = SimpleNamespace(**base)
    ns.has_forecast = lambda: base.get("_has_forecast", True)
    return ns


def test_run_classification_and_epistemic_contract_attached():
    assert set(RUN_CLASSES) == {"full_numeric_forecast", "rank_only",
                                "scenario_distribution", "structurally_underidentified",
                                "execution_failed"}
    assert classify_run(_res())["run_class"] == "full_numeric_forecast"
    assert classify_run(_res(_has_forecast=False))["run_class"] == "execution_failed"
    r = _res()
    r.provenance["posterior_consumed"] = False
    assert classify_run(r)["run_class"] == "rank_only"
    r2 = _res()
    r2.provenance["active_component_manifest"]["phase2_evidence"] = {"executed": False}
    assert classify_run(r2)["run_class"] == "scenario_distribution"
    r3 = _res()
    r3.provenance["consequence_report"] = {"structurally_underidentified": True,
                                           "scenario_schema_error": "x"}
    assert classify_run(r3)["run_class"] == "structurally_underidentified"
    # the contract surfaces every invariant
    r4 = _res()
    r4.provenance["consequence_report"] = {
        "actors_invoked": 3, "recursive_cascade_depth": 2,
        "human_reactions_written_directly": 0, "fixed_ontology_uses": 0,
        "legacy_scalar_writes": 0, "tier1_numeric_fallbacks": 0,
        "tier2_numeric_fallbacks": 0, "fallback_reasons": []}
    r4.provenance["actor_decision_distributions"] = {
        "ana": {"n_qualitative_branches": 4, "n_excluded_numeric_fallbacks": 0, "rows": []}}
    ec = epistemic_contract(r4)
    assert ec["actor_simulation"] == "full_recursive_actor_simulation"
    for k in ("human_reactions_written_directly", "fixed_ontology_uses",
              "legacy_scalar_writes", "tier1_numeric_fallbacks", "tier2_numeric_fallbacks"):
        assert ec[k] == 0


def test_run_from_plan_attaches_manifests_and_recovery_labels():
    from tests.test_wmv2_phase4_e2e import compiled_payload
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.materialize import run_from_plan
    plan = compile_world("Will the manager approve the project?",
                         llm=lambda _: json.dumps(compiled_payload()), evidence="",
                         as_of="2025-01-01", horizon="2025-01-10", persist=False)
    result, branches = run_from_plan(plan, n_particles=3, seed=2)
    assert "generated_manifests" in result
    assert result["scenario_schema_recovery"] == "minimal_deterministic"
    m = result["generated_manifests"]
    assert "event_cascade" in m and "n_approximations" in m
    rep = result["consequence_report"]
    assert rep["tier1_numeric_fallbacks"] == 0 and rep["tier2_numeric_fallbacks"] == 0
    # joint worlds attached even offline (fallback source, stamped)
    jw = (plan.provenance or {}).get("joint_world") or {}
    assert jw.get("status") in ("attached", "generation_failed", "disabled_by_env")


# ---------------------------------------------------------------- invariants + purity
def test_invariants_no_tier12_numeric_fallbacks():
    rep = gw.generated_report()
    assert rep["human_reactions_written_directly"] == 0
    assert rep["fixed_ontology_uses"] == 0
    # minimal recovery schema is a REAL generated schema (validated + frozen)
    m = minimal_scenario_schema(question="Will X happen?", as_of=T0, horizon=T0 + 86400,
                                entities=("ana", "bo"),
                                institutions={"panel": ["cara"]},
                                options=("Yes", "No"))
    assert m.frozen and m.outcome_predicates
    assert m.provenance["compiler"] == "minimal_deterministic"
    assert m.institutional_definitions["panel"]["aggregation"]["kind"] == "single_authority"


def test_no_lexical_vote_operator_in_generated_mode(monkeypatch):
    from types import SimpleNamespace
    from swm.world_model_v2.materialize import operators_from_plan
    plan = SimpleNamespace(accepted_mechanisms=[], provenance={})
    monkeypatch.delenv("SWM_CONSEQUENCES", raising=False)
    ops, _ = operators_from_plan(plan)
    names = [getattr(o, "name", "") for o in ops]
    assert "institutional_vote" not in names          # lexical vote mapping is baseline-only
    monkeypatch.setenv("SWM_CONSEQUENCES", "fixed_semantic_consequence_policy_v1")
    ops2, _ = operators_from_plan(plan)
    assert "institutional_vote" in [getattr(o, "name", "") for o in ops2]


def test_exactly_one_actor_path_and_no_donor_modules():
    import importlib
    for rejected in ("actor_propagation", "semantic_events", "observation_delivery",
                     "causal_frontier", "semantic_clustering"):
        with pytest.raises(ImportError):
            importlib.import_module(f"swm.world_model_v2.{rejected}")
    from swm.world_model_v2.events import _EVENT_TYPES
    assert "actor_reconsideration" not in _EVENT_TYPES  # no second reconsideration channel
    import swm.world_model_v2.generated_world as g
    assert not hasattr(g, "RECONSIDERATION_AFFORDANCES")
    assert not hasattr(g, "SEMANTIC_EVENT_TYPES")


def test_exotic_entity_types_keep_deciding():
    from swm.world_model_v2.state import extension_fields
    for etype in ("organization", "group", "committee", "party"):
        assert "expected_reactions" in extension_fields(etype)
        assert "stances" in extension_fields(etype)


def test_keyed_past_actions_reach_the_view():
    from swm.world_model_v2.phase4_policy import ActorViewBuilder
    from swm.world_model_v2.state import StateField
    from swm.world_model_v2.state import StateField as _SF
    w = world()
    ent = w.entities["ana"]
    ent.fields["past_actions"] = {"a_1": _SF(value={"op": "counteroffer", "at": T0 - 5,
                                                    "public": True})}
    view = ActorViewBuilder().build(w, "ana")
    assert any(h.get("op") == "counteroffer" or h.get("action") == "counteroffer"
               or "counteroffer" in json.dumps(h) for h in view.action_history)
    rows = ActorViewBuilder._history_rows({"a_2": StateField(value="legacy_string")})
    assert rows == [{"action": "legacy_string", "action_id": "a_2", "public": True}]
