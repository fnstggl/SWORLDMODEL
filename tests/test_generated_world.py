"""Generated actor-mediated world — the 34 required invariants, offline and deterministic.

The engine may hardcode integrity, scheduling, access, feasibility, and aggregation ONLY.
Scenario semantics (types, events, processes, predicates) are generated per question;
control-plane tasks never masquerade as world events; every human response comes from that
actor's own persistent simulation; budgets terminate recursion loudly; the fixed-v1 and
legacy scalar modes stay runnable as explicit baselines."""
import copy
import json

import pytest

from swm.world_model_v2 import generated_world as gw
from swm.world_model_v2 import semantic_consequences as sc
from swm.world_model_v2.events import Event, EventQueue, _EVENT_TYPES
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import ActorPolicyRuntime, ProductionActorPolicyOperator
from swm.world_model_v2.qualitative_actor import (
    QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
)
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.scenario_schema import (
    UNMODELED_EVENT_TYPE, ScenarioSemanticModel, extend_schema, validate_scenario_schema,
)
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta

T0 = 1_700_000_000.0


def lab_schema():
    """A replication-controversy scenario — nothing about it exists in repository code."""
    return ScenarioSemanticModel(
        question="Will the replication controversy end in a retraction?",
        prediction_timestamp=T0, horizon=T0 + 30 * 86400,
        entity_types={"research_lab": {"description": "lab", "fields": {"name": "str"}}},
        fact_types={"replication_attempt": {"description": "attempt",
                                            "fields": {"finding": "str", "status": "str",
                                                       "matter": "str", "position": "str"}},
                    "retraction_record": {"description": "retraction",
                                          "fields": {"paper": "str", "position": "str",
                                                     "matter": "str"}}},
        semantic_event_types={
            "replication_failure_disclosed": {"description": "x",
                                              "fields": {"finding": "str"},
                                              "typical_visibility": "participants"},
            "public_defense_statement": {"description": "x", "fields": {},
                                         "typical_visibility": "public"}},
        relation_types={"scientific_rival_of": {"description": "rivalry", "directed": False}},
        resource_definitions={"grant_funds": {"unit": "usd", "conserved": True}},
        actor_roles={"dr_okafor": {"role": "original author", "why_consequential": "her paper",
                                   "affordances": ["issue public defense"]}},
        outcome_predicates=[{"predicate_id": "retracted", "record_type": "retraction_record",
                             "op": "exists", "option_true": "retraction",
                             "option_false": "no_retraction"}],
        provenance={"compiler": "test"}).freeze()


def ownership_schema():
    """An ENTIRELY different domain (sports ownership dispute) on the SAME kernel."""
    return ScenarioSemanticModel(
        question="Will the club's ownership dispute force a sale?",
        prediction_timestamp=T0, horizon=T0 + 90 * 86400,
        entity_types={"football_club": {"description": "club", "fields": {"name": "str"}}},
        fact_types={"ownership_stake": {"description": "stake",
                                        "fields": {"holder": "str", "percent": "float"}},
                    "forced_sale_order": {"description": "order",
                                          "fields": {"issued_by": "str"}}},
        semantic_event_types={"minority_owner_files_oppression_claim":
                              {"description": "x", "fields": {"claim": "str"},
                               "typical_visibility": "public"}},
        outcome_predicates=[{"predicate_id": "sale", "record_type": "forced_sale_order",
                             "op": "exists", "option_true": "sale_forced",
                             "option_false": "no_sale"}],
        provenance={"compiler": "test"}).freeze()


def world(schema=None, actors=("dr_okafor", "dr_lin")):
    w = WorldState("gen", "b0", SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    for name in actors:
        e = Entity(name)
        e.set("roles", F(["person"], status="observed"))
        e.set("resources", F(50_000.0, status="observed"), key="grant_funds")
        e.set("past_actions", F([], status="observed"))
        w.entities[name] = e
    if len(actors) >= 2:
        w.network.add(actors[0], "communicates_with", actors[1])
    w.scenario_schema = schema if schema is not None else lab_schema()
    return w


def ctx_for(w, actor="dr_lin", action="a1", report=None):
    return {"actor_id": actor, "action_id": action, "now": w.clock.now,
            "report": report if report is not None else gw.generated_report(),
            "budgets": gw._budgets(w), "events": [], "quarantined": [], "compiler": "test"}


def run_ops(w, ops, **kw):
    ctx = ctx_for(w, **kw)
    d = StateDelta(at=w.clock.now, event_type="actor_action", operator="test")
    events = gw.execute_kernel_ops(w, ops, ctx, d)
    return ctx, d, events


# ------------------------------------------------- 1-3: no global catalogs in production mode
def test_1_2_3_no_global_type_event_or_stage_catalogs():
    assert not hasattr(gw, "OBJECT_TYPES")
    assert not hasattr(gw, "PROCESS_STAGES")
    assert not hasattr(gw, "ACTION_PATHWAY_EFFECTS")
    # the kernel op list is storage mechanics, not scenario semantics
    assert set(gw.KERNEL) == set(gw.KERNEL_OPS)
    # record/event acceptance is decided ONLY by the branch schema
    w = world()
    ctx, d, _ = run_ops(w, [{"op": "create_or_update_record", "record_type": "product",
                             "fields": {}}])
    assert ctx["quarantined"] and "not in this scenario" in ctx["quarantined"][0]["reason"]
    ctx2, _, _ = run_ops(w, [{"op": "emit_semantic_event",
                              "semantic_type_id": "message_delivered", "exact_content": "x"}])
    assert ctx2["quarantined"]                      # the old global name means nothing here


# ------------------------------------------------- 4-6: schema-driven kernel, no registration
def test_4_same_kernel_accepts_two_entirely_different_schemas():
    w1, w2 = world(lab_schema()), world(ownership_schema(), actors=("owner_a", "owner_b"))
    run_ops(w1, [{"op": "create_or_update_record", "record_type": "replication_attempt",
                  "fields": {"finding": "effect X", "status": "failed"}}])
    run_ops(w2, [{"op": "create_or_update_record", "record_type": "ownership_stake",
                  "fields": {"holder": "owner_a", "percent": 33.0}}], actor="owner_a")
    assert any(o.object_type == "replication_attempt" for o in w1.objects.values())
    assert any(o.object_type == "ownership_stake" for o in w2.objects.values())


def test_5_6_scenario_types_and_events_need_no_repository_registration():
    w = world(ownership_schema(), actors=("owner_a", "owner_b"))
    ctx, d, events = run_ops(w, [
        {"op": "emit_semantic_event",
         "semantic_type_id": "minority_owner_files_oppression_claim",
         "exact_content": "Claim filed in commercial court.",
         "structured_fields": {"claim": "oppression"}}], actor="owner_b")
    assert not ctx["quarantined"]
    assert w.semantic_log[0]["semantic_type_id"] == "minority_owner_files_oppression_claim"
    # NOT in the global event registry — the envelope rode a ctrl_* control task instead
    assert "minority_owner_files_oppression_claim" not in _EVENT_TYPES
    assert [e.etype for e in events] == ["ctrl_semantic_event"]


# ------------------------------------------------- 7-8: runtime extension, branch isolation
def test_7_8_schema_extends_during_rollout_versioned_and_branch_isolated():
    w = world()
    wb = w.clone(branch_id="b1")
    ctx, d, _ = run_ops(wb, [
        {"op": "declare_schema_definition", "reason": "journal enters",
         "definitions": {"fact_types": {"journal_expression_of_concern":
                                        {"description": "EoC", "fields": {"paper": "str"}}}}},
        {"op": "create_or_update_record", "record_type": "journal_expression_of_concern",
         "fields": {"paper": "effect X"}}])
    assert not ctx["quarantined"]
    assert wb.scenario_schema.version == "2" and wb.scenario_schema.ancestry
    assert w.scenario_schema.version == "1"
    assert "journal_expression_of_concern" not in w.scenario_schema.fact_types
    # extensions can never redefine existing semantics
    ok, why = extend_schema(wb.scenario_schema,
                            {"fact_types": {"retraction_record": {"fields": {}}}},
                            reason="rewrite attempt")
    assert not ok and "immutable" in why[0]


# ------------------------------------------------- 9-10: control plane ≠ world plane
def test_9_10_internal_tasks_never_appear_as_world_semantics():
    w = world()
    _, _, events = run_ops(w, [
        {"op": "emit_semantic_event", "semantic_type_id": "replication_failure_disclosed",
         "exact_content": "n=400, null result", "direct_targets": ["dr_okafor"]}])
    report = gw.generated_report()
    router = gw.GeneratedSemanticEventOperator(report=report)
    deliver = gw.GeneratedObservationDeliveryOperator(report=report)
    rd, _ = router.run(w, events[0], None)
    follow = rd.follow_up_events
    assert all(f["etype"].startswith("ctrl_") for f in follow)
    for f in follow:
        if f["etype"] == "ctrl_deliver_observation":
            dd, _ = deliver.run(w, Event(ts=f["ts"], etype=f["etype"],
                                         participants=f["participants"],
                                         payload=f["payload"]), None)
    # the world-plane log records ONLY scenario-typed happenings
    assert all(not str(x.get("semantic_type_id", "")).startswith("ctrl")
               for x in w.semantic_log)
    assert all(x.get("semantic_type_id") in w.scenario_schema.semantic_event_types
               for x in w.semantic_log)


class ScriptedBackend:
    """Answers decision prompts (per-actor behavior) and generated compile prompts."""

    def __init__(self, choices=None, compile_ops=None):
        #: actor -> (action, target) or a LIST of them consumed in order (then wait)
        self.choices = choices or {}
        self.compile_ops = compile_ops or {}
        self.prompts = []
        self._used = {}

    def _next_choice(self, actor):
        c = self.choices.get(actor, ("wait", ""))
        if isinstance(c, tuple):
            return c
        i = self._used.get(actor, 0)
        self._used[actor] = i + 1
        return c[i] if i < len(c) else ("wait", "")

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if "CONSEQUENCE COMPILER" in prompt:
            raise AssertionError("fixed-v1 consequence compiler ran in generated mode")
        if "DIRECT-EFFECT COMPILER" in prompt:
            actor = prompt.split("ACTOR:")[1].split("\n")[0].strip()
            return json.dumps(self.compile_ops.get(actor, []))
        actor = prompt.split("You ARE ")[1].split(",")[0].split(".")[0].strip() \
            if "You ARE " in prompt else "?"
        chosen, target = self._next_choice(actor)
        act_or_wait = "wait" if chosen == "wait" else "act"
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "situation_interpretation": {"what_changed": "x", "why_it_matters": "y",
                                         "perceived_opportunities": "",
                                         "perceived_threats": ""},
            "actor_state_update": {"current_private_beliefs": [], "beliefs_about_others": {},
                                   "personal_condition": "steady", "important_memories": []},
            "anticipated_reactions": [{"actor_or_group": target or "them",
                                       "expected_reaction": "they will fold immediately",
                                       "reasoning_summary": "s",
                                       "uncertainty_description": "low"}],
            "decision": {"act_or_wait": act_or_wait, "chosen_action": chosen,
                         "target": target, "timing": "immediate",
                         "observability": "private", "intended_effect": "advance"},
            "novel_action_proposal": {"present": chosen not in ("wait",),
                                      "description": chosen.replace("_", " "),
                                      "required_authority": "none",
                                      "required_resources": "none",
                                      "proposed_mechanisms": "communication"},
            "alternatives_considered": [], "decision_summary": f"I {chosen}"})


def generated_runtime(backend):
    engine = QualitativeDecisionEngine(QualitativeConfig(llm=backend, llm_hypotheses=False,
                                                         n_hypotheses=2))
    return QualitativeActorPolicyRuntime(engine, mode="persistent_qualitative_llm_policy",
                                         consequence_mode="generated_actor_mediated_world")


def run_cascade(w, backend, *, kickoff_actor, kickoff_situation, horizon_days=20, seed=5):
    rt = generated_runtime(backend)
    report = rt.consequence_report
    ops = [ProductionActorPolicyOperator(runtime=rt),
           gw.GeneratedSemanticEventOperator(report=report),
           gw.GeneratedObservationDeliveryOperator(report=report),
           gw.GeneratedActorInvocationOperator(rt, report=report)]
    q = EventQueue(horizon_ts=T0 + horizon_days * 86400)
    q.schedule(Event(ts=T0 + 60, etype="decision_opportunity", participants=[kickoff_actor],
                     payload={"situation": kickoff_situation}))
    branch = RolloutEngine(operators=ops).run_branch(w, q, seed=seed)
    return rt, report, branch


# ------------------------------------------------- 11-16: actor-mediated causality
def test_11_12_16_reconsideration_wait_and_own_policy_response():
    backend = ScriptedBackend(
        choices={"dr_lin": [("disclose_replication_failure", "dr_okafor")],   # then waits
                 "dr_okafor": [("issue_public_defense", "")]},
        compile_ops={"dr_lin": [
            {"op": "emit_semantic_event",
             "semantic_type_id": "replication_failure_disclosed",
             "exact_content": "Private heads-up: our replication found nothing.",
             "direct_targets": ["dr_okafor"],
             "structured_fields": {"finding": "effect X"}}],
            "dr_okafor": [
            {"op": "emit_semantic_event", "semantic_type_id": "public_defense_statement",
             "exact_content": "The replication used the wrong reagent lot.",
             "intended_visibility": "public"}]})
    w = world()
    rt, report, branch = run_cascade(w, backend, kickoff_actor="dr_lin",
                                     kickoff_situation="your replication failed; decide")
    log = [x["semantic_type_id"] for x in w.semantic_log]
    assert "replication_failure_disclosed" in log
    # 11: receiving the disclosure triggered okafor's reconsideration via the control plane
    assert report["actors_reconsidered"] >= 1 and report["observations_delivered"] >= 1
    # 16: okafor's response came from HER policy (public defense), NOT from lin's
    # expectation ("they will fold immediately" — stored as lin's subjective state only)
    assert "public_defense_statement" in log
    lin_expect = w.entity("dr_lin").value("expected_reactions") or {}
    assert "fold immediately" in json.dumps(lin_expect)      # subjective, actor-local
    assert not any("fold immediately" in str(x.get("exact_content", ""))
                   + json.dumps(x.get("structured_fields", {}))
                   for x in w.semantic_log)                  # never became world truth
    # 12: dr_lin, re-invoked on the defense, decided nothing was warranted — first-class
    assert report["actors_declined_to_act"] >= 1
    assert report["human_reactions_written_directly"] == 0
    assert report["fixed_ontology_uses"] == 0
    assert report["actual_mode"] == "generated_actor_mediated_world"


def test_13_14_novel_action_and_no_required_menu():
    backend = ScriptedBackend(
        choices={"dr_okafor": ("commission_independent_replication_consortium", "")},
        compile_ops={"dr_okafor": [
            {"op": "create_or_update_record", "record_type": "replication_attempt",
             "fields": {"finding": "effect X", "status": "consortium_commissioned"}}]})
    w = world()
    rt = generated_runtime(backend)
    report = rt.consequence_report
    invoke = gw.GeneratedActorInvocationOperator(rt, report=report)
    ev = Event(ts=T0 + 10, etype="ctrl_invoke_actor", participants=["dr_okafor"],
               payload={"actor_id": "dr_okafor",
                        "triggering_semantic_event": {"event_id": "sev_x",
                                                      "semantic_type_id":
                                                          "replication_failure_disclosed",
                                                      "exact_content": "null result"},
                        "reason_actor_may_be_causally_relevant": "her paper",
                        "cascade_depth": 1})
    import random
    d, vr = invoke.run(w, ev, random.Random(0))
    # 13: the action is NOVEL — absent from the schema's affordance examples
    assert report["actor_actions_executed"] == 1
    assert any(o.attributes.get("status") == "consortium_commissioned"
               for o in w.objects.values())
    # 14: no generic candidate set was required — the decision prompt carried no
    # acknowledge/ignore menu, only the schema's affordance EXAMPLES (if any)
    decision_prompts = [p for p in backend.prompts if "You ARE dr_okafor" in p]
    assert decision_prompts and "acknowledge" not in decision_prompts[0]
    assert "ignore" not in decision_prompts[0].split("OPTIONS")[-1][:200]


def test_15_compiler_cannot_write_human_reactions():
    w = world()
    ctx, d, _ = run_ops(w, [
        {"op": "create_or_update_record", "record_type": "replication_attempt",
         "fields": {"finding": "x", "status": "set_belief_of_dr_okafor"},
         "belief_of": "dr_okafor"}])
    assert ctx["quarantined"]
    assert "mind/choice" in ctx["quarantined"][0]["reason"]
    assert ctx["report"]["human_reactions_written_directly"] == 0
    assert ctx["report"].get("human_reactions_attempted_directly", 0) == 1


# ------------------------------------------------- 17-19: visibility and representations
def test_17_18_19_visibility_reach_and_representations():
    schema = lab_schema()
    schema.information_rules = {"public_channel": "science_press",
                                "public_delay_s": 7200.0,
                                "public_representation": "summary",
                                "default_delay_s": 60.0}
    w = world(schema, actors=("dr_okafor", "dr_lin", "dr_padilla"))
    long_text = "The replication used the wrong reagent lot. " * 12
    _, _, events = run_ops(w, [
        {"op": "emit_semantic_event", "semantic_type_id": "public_defense_statement",
         "exact_content": long_text, "intended_visibility": "public",
         "direct_targets": ["dr_lin"]}], actor="dr_okafor")
    report = gw.generated_report()
    router = gw.GeneratedSemanticEventOperator(report=report)
    rd, _ = router.run(w, events[0], None)
    deliveries = [f for f in rd.follow_up_events if f["etype"] == "ctrl_deliver_observation"]
    # 17: public information reaches several actors (both non-sources)
    assert {d["payload"]["recipient"] for d in deliveries} == {"dr_lin", "dr_padilla"}
    # 19: the DIRECT target gets the complete text now; the public gets the summarized
    # press representation later
    by = {d["payload"]["recipient"]: d for d in deliveries}
    assert by["dr_lin"]["payload"]["representation"] == "complete"
    assert by["dr_padilla"]["payload"]["representation"] == "summary"
    assert by["dr_padilla"]["ts"] > by["dr_lin"]["ts"]
    deliver = gw.GeneratedObservationDeliveryOperator(report=report)
    for f in deliveries:
        deliver.run(w, Event(ts=f["ts"], etype=f["etype"], participants=f["participants"],
                             payload=f["payload"]), None)
    pad = w.information.visible_to("dr_padilla", at=T0 + 10_000)
    assert any("[summarized in transit]" in item.content for item, _ in pad)
    # 18: PRIVATE information does not reach unrelated actors
    w2 = world(lab_schema(), actors=("dr_okafor", "dr_lin", "dr_padilla"))
    _, _, ev2 = run_ops(w2, [
        {"op": "emit_semantic_event", "semantic_type_id": "replication_failure_disclosed",
         "exact_content": "private null result", "direct_targets": ["dr_okafor"],
         "intended_visibility": "participants"}])
    rd2, _ = gw.GeneratedSemanticEventOperator(report=report).run(w2, ev2[0], None)
    recips = {f["payload"]["recipient"] for f in rd2.follow_up_events
              if f["etype"] == "ctrl_deliver_observation"}
    assert recips == {"dr_okafor"} and "dr_padilla" not in recips


# ------------------------------------------------- 20: frontier promotion
def test_20_frontier_discovery_promotes_institutional_holder():
    schema = lab_schema()
    schema.institutional_definitions["ethics_panel"] = {
        "procedure": "panel", "decision_holders": ["dr_padilla"],
        "decision_record_type": "retraction_record",
        "aggregation": {"kind": "single_authority"}, "assumed": True}
    w = world(schema, actors=("dr_okafor", "dr_lin", "dr_padilla"))
    sev = {"event_id": "s1", "semantic_type_id": "replication_failure_disclosed",
           "exact_content": "the ethics_panel must see this", "direct_targets": ["dr_okafor"],
           "source_actor_id": "dr_lin", "intended_visibility": "participants",
           "cascade_depth": 0}
    frontier = gw.discover_causal_frontier(w, sev)
    ids = {a for a, _r in frontier}
    assert "dr_padilla" in ids                      # promoted: never a target, holds the right
    reasons = dict(frontier)
    assert "decision holder" in reasons["dr_padilla"]


# ------------------------------------------------- 21-23: canonical queue, quiescence, budgets
def test_21_22_23_canonical_queue_quiescence_and_budgets():
    backend = ScriptedBackend(
        choices={"dr_lin": ("disclose_replication_failure", "dr_okafor"),
                 "dr_okafor": ("issue_public_defense", "")},
        compile_ops={"dr_lin": [
            {"op": "emit_semantic_event",
             "semantic_type_id": "replication_failure_disclosed",
             "exact_content": "null", "direct_targets": ["dr_okafor"],
             "structured_fields": {"finding": "x"}}],
            "dr_okafor": [
            {"op": "emit_semantic_event", "semantic_type_id": "public_defense_statement",
             "exact_content": "defense", "intended_visibility": "public"}]})
    w = world()
    rt, report, branch = run_cascade(w, backend, kickoff_actor="dr_lin",
                                     kickoff_situation="decide")
    # 21: the cascade ran through the canonical RolloutEngine queue — control-plane deltas
    # appear in the branch log with their operator names
    ops_seen = {d.operator for d in branch.log}
    assert {"generated_semantic_event_router", "generated_observation_delivery",
            "generated_actor_invocation"} <= ops_seen
    # 22: quiescence — the same-event/same-actor dedup means the finite log terminates well
    # under the event cap, and repeated identical observations do not re-invoke
    assert len(branch.log) < 100
    keys = w.uncertainty_meta.get("pending_reconsiderations", [])
    assert len(keys) == len(set(keys))
    # 23: per-actor invocation budgets bound recursion; exhaustion is stamped when hit
    budgets = gw._budgets(w)
    assert all(v <= budgets["max_invocations_per_actor"]
               for v in budgets["invocations"].values())


# ------------------------------------------------- 24-25: institutions aggregate, never choose
def test_24_25_institutional_arithmetic_counts_real_choices_only():
    schema = lab_schema()
    schema.institutional_definitions["ethics_panel"] = {
        "procedure": "panel vote", "decision_holders": ["dr_okafor", "dr_lin", "dr_padilla"],
        "decision_record_type": "replication_attempt",
        "aggregation": {"kind": "quorum_majority", "threshold": 2}, "assumed": True}
    w = world(schema, actors=("dr_okafor", "dr_lin", "dr_padilla"))
    # nobody has decided: aggregation reports nothing passed and writes NO positions
    res0 = gw.run_institutional_aggregation(w, "ethics_panel", matter_record_id="m1")
    assert res0["cast"] == 0 and res0["passed"] is False
    assert not any(o.attributes.get("position") for o in w.objects.values())
    for who, pos in (("dr_lin", "yes"), ("dr_padilla", "yes")):
        run_ops(w, [{"op": "create_or_update_record", "record_type": "replication_attempt",
                     "fields": {"finding": "x", "position": pos, "matter": "m1",
                                "status": "voted"}}], actor=who)
    res = gw.run_institutional_aggregation(w, "ethics_panel", matter_record_id="m1")
    assert res["passed"] is True and res["cast"] == 2 and res["yes"] == 2
    assert res["decisions"] == {"dr_lin": "yes", "dr_padilla": "yes"}


# ------------------------------------------------- 26-27: predicates over dynamic records
def test_26_27_predicates_reference_dynamic_records_and_cannot_be_set_directly():
    w = world()
    readout = gw.make_generated_predicate_readout(w.scenario_schema)
    assert readout(w) == "no_retraction"
    run_ops(w, [{"op": "create_or_update_record", "record_type": "retraction_record",
                 "fields": {"paper": "effect X"}}], actor="dr_okafor")
    assert readout(w) == "retraction"               # 26: resolves from generated records
    # 27: no op can mint a probability/forecast field toward the predicate
    ctx, _, _ = run_ops(w, [{"op": "create_or_update_record",
                             "record_type": "retraction_record",
                             "fields": {"paper": "y", "probability_of_retraction": 0.9}}])
    assert ctx["quarantined"] and "forbidden" in ctx["quarantined"][0]["reason"]


# ------------------------------------------------- 28-29: baseline writers dead in production
def test_28_29_no_scalar_or_fixed_v1_writers_in_generated_mode():
    backend = ScriptedBackend(
        choices={"dr_lin": ("disclose_replication_failure", "dr_okafor")},
        compile_ops={"dr_lin": [
            {"op": "emit_semantic_event",
             "semantic_type_id": "replication_failure_disclosed",
             "exact_content": "null", "direct_targets": ["dr_okafor"],
             "structured_fields": {"finding": "x"}}]})
    w = world()
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    register_quantity_type("pathway_progress", units="process_state")
    w.quantities["pathway_progress:cooperative_agreement"] = Quantity(
        "pathway_progress:cooperative_agreement", "pathway_progress", value=0.5)
    rt, report, branch = run_cascade(w, backend, kickoff_actor="dr_lin",
                                     kickoff_situation="decide")
    assert w.quantities["pathway_progress:cooperative_agreement"].value == 0.5   # untouched
    assert report["legacy_scalar_writes"] == 0
    # 29: the fixed-v1 compiler never ran (ScriptedBackend raises on its prompt) and no
    # fixed-catalog objects appeared
    assert not any(o.object_type in sc.OBJECT_TYPES for o in w.objects.values()) \
        or all(o.object_type in w.scenario_schema.record_types()
               for o in w.objects.values())
    with pytest.raises(RuntimeError):
        ActorPolicyRuntime._apply_pathway_effects(
            w, None, StateDelta(at=T0, event_type="x", operator="t"),
            consequence_mode="generated_actor_mediated_world")


# ------------------------------------------------- 30-32: counterfactuals, pairing, replay
def test_30_31_32_matched_counterfactuals_pairing_and_replay():
    def build(backend):
        w = world()
        return run_cascade(w, backend, kickoff_actor="dr_lin",
                           kickoff_situation="decide", seed=11), w

    def backend_factory():
        return ScriptedBackend(
            choices={"dr_lin": ("disclose_replication_failure", "dr_okafor"),
                     "dr_okafor": ("issue_public_defense", "")},
            compile_ops={"dr_lin": [
                {"op": "emit_semantic_event",
                 "semantic_type_id": "replication_failure_disclosed",
                 "exact_content": "null", "direct_targets": ["dr_okafor"],
                 "structured_fields": {"finding": "x"}}],
                "dr_okafor": [
                {"op": "emit_semantic_event",
                 "semantic_type_id": "public_defense_statement",
                 "exact_content": "defense", "intended_visibility": "public"}]})

    (_, rep1, b1), w1 = build(backend_factory())
    (_, rep2, b2), w2 = build(backend_factory())
    # 31/32: common randomness + provenance ⇒ deterministic replay: identical worlds, seeds,
    # and scripted providers reproduce the SAME world-plane history
    strip = lambda log: [(x["semantic_type_id"], x["source_actor_id"], x["exact_content"])  # noqa: E731
                         for x in log]
    assert strip(w1.semantic_log) == strip(w2.semantic_log)
    # 30: a Phase-13-style matched clone carries the SAME generated runtime semantics —
    # the cloned branch keeps the schema and diverges only through its own actor decisions
    w3 = world()
    clone = w3.clone(branch_id="b0:intervention")
    assert clone.scenario_schema is not w3.scenario_schema
    assert clone.scenario_schema.schema_id == w3.scenario_schema.schema_id
    run_ops(clone, [{"op": "create_or_update_record", "record_type": "retraction_record",
                     "fields": {"paper": "x"}}], actor="dr_okafor")
    assert clone.objects and not w3.objects


# ------------------------------------------------- 33-34: loud gaps, runnable baselines
def test_33_unsupported_semantics_are_surfaced():
    w = world()
    ctx, d, _ = run_ops(w, [
        {"op": "create_or_update_record", "record_type": "replication_attempt",
         "fields": {"undeclared_field": "x", "finding": "effect X"}},
        {"op": "run_physics_simulation", "model": "invented"}])
    # unknown kernel op → quarantined; undeclared field on a DECLARED type → dropped LOUDLY
    # while the op's declared semantics still apply
    assert len(ctx["quarantined"]) == 1
    assert ctx["report"]["unsupported_semantics"] == 1
    assert ctx["report"]["undeclared_fields_dropped"] == 1
    assert any(r.startswith("fields_dropped:") for r in d.reason_codes)
    rec = next(o for o in w.objects.values() if o.object_type == "replication_attempt")
    assert rec.attributes == {"finding": "effect X"}
    assert any("kernel_ops" in r for r in d.reason_codes)
    # the compiler's total-fallback path preserves the exact action as a SCAFFOLDING event,
    # stamped as a fallback, never as modeled semantics
    comp = gw.GeneratedActionCompiler(llm=None)
    from swm.world_model_v2.phase4_policy import ActionTarget, TypedAction
    a = TypedAction(action_id="a", actor_id="dr_lin", actor_role="r",
                    action_family="generic", action_name="interpretive_dance",
                    target=ActionTarget("person", "dr_okafor"),
                    mechanisms_triggered=["semantic_consequences"])
    rep = gw.generated_report()
    ops, meta = comp.compile(w, a, qualitative={"decision_summary": "I dance"}, report=rep)
    assert meta["compiler"] == "deterministic_fallback"
    assert ops[0]["semantic_type_id"] == UNMODELED_EVENT_TYPE
    assert any(fr.get("kind") == "action_semantics_unmodeled"
               for fr in rep["fallback_reasons"])


def test_34_fixed_v1_and_legacy_baselines_remain_runnable():
    assert ActorPolicyRuntime(
        consequence_mode="fixed_semantic_consequence_policy_v1").consequence_mode \
        == "fixed_semantic_consequence_policy_v1"
    assert ActorPolicyRuntime(
        consequence_mode="legacy_scalar_pathway_consequences").consequence_mode \
        == "legacy_scalar_pathway_consequences"
    # and the fixed-v1 catalog still exists FOR THE BASELINE, outside production semantics
    assert sc.OBJECT_TYPES and sc.PROCESS_STAGES and sc.PRIMITIVES
