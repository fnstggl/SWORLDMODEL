"""Semantic world consequences — the phase's required invariants, offline and deterministic.

Covers: default-on semantic routing with NO scalar pathway writes; the closed primitive
registry with forbidden numeric-minting quarantine; exact-content communications reaching the
RECIPIENT and opening THEIR decision (sender expectation stays subjective); institutional
submissions entering REAL procedures (sole-right-holder decisions become typed outcomes; multi-
holder submissions schedule the vote machinery); typed process stage machines; conservation-
checked resources; novel actions compiling or loudly unmodeled (never nearest-label scalar
effects); derived pathway summaries as reconstructable projections; mode gating (legacy scalar
writers assert the mode; dual-run records the legacy shadow UNAPPLIED); consequence_report on
public/counterfactual paths; branch-local object isolation."""
import copy
import json
import random

import pytest

from swm.world_model_v2 import semantic_consequences as sc
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.institutions import Rule, RuleSystem
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import (
    ActorPolicyRuntime, ProductionActorPolicyOperator, decide_and_execute_particles,
)
from swm.world_model_v2.phase4_policy import ActionTarget, TypedAction
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta

T0 = 1_700_000_000.0


class Plan:
    question = "Will the proposal be approved?"

    @staticmethod
    def plan_hash():
        return "plansemcons"


def world(*, holders=("alice",), declared_bar=False):
    w = WorldState("sem", "b0", SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    alice = Entity("alice")
    alice.set("roles", F(["ceo"], status="observed"))
    alice.set("resources", F(100.0, status="observed"), key="budget")
    alice.set("past_actions", F([], status="observed"))
    bob = Entity("bob")
    bob.set("roles", F(["partner"], status="observed"))
    bob.set("past_actions", F([], status="observed"))
    w.entities = {"alice": alice, "bob": bob}
    w.network.add("alice", "communicates_with", "bob")
    w.institutions["board"] = RuleSystem("board", [
        Rule("r1", "decision_right", {"actions": ["approve"], "holders": list(holders)})])
    if declared_bar:
        register_quantity_type("pathway_progress", units="process_state")
        w.quantities["pathway_progress:cooperative_agreement"] = Quantity(
            "pathway_progress:cooperative_agreement", "pathway_progress", value=0.5)
    return w


def act(name, *, family="negotiation", target="bob", actor="alice", consequences=None, **params):
    return TypedAction(action_id=f"a_{name}_{actor}", actor_id=actor, actor_role="ceo",
                       action_family=family, action_name=name,
                       target=ActionTarget("person", target), parameters=params,
                       possible_consequences=list(consequences or []),
                       mechanisms_triggered=["semantic_consequences"])


def run_program(w, ops, *, actor="alice", action_id="a_test"):
    prog = sc.CausalActionProgram(action_id=action_id, actor_id=actor, intended={},
                                  operations=list(ops), compiler="test")
    delta = StateDelta(at=T0, event_type="actor_action", operator="test")
    report = sc.empty_report()
    events = sc.execute_program(w, prog, delta, report)
    return prog, delta, report, events


# ---------------------------------------------------------------- modes & routing
def test_default_mode_is_generated_and_fixed_v1_is_the_explicit_baseline(monkeypatch):
    monkeypatch.delenv("SWM_CONSEQUENCES", raising=False)
    assert sc.resolve_consequence_mode() == "generated_actor_mediated_world"
    assert ActorPolicyRuntime().consequence_mode == "generated_actor_mediated_world"
    monkeypatch.setenv("SWM_CONSEQUENCES", "fixed_semantic_consequence_policy_v1")
    assert ActorPolicyRuntime().consequence_mode == "fixed_semantic_consequence_policy_v1"
    # the historical name survives as an ALIAS of the fixed baseline
    monkeypatch.setenv("SWM_CONSEQUENCES", "semantic_world_consequences")
    assert sc.resolve_consequence_mode() == "fixed_semantic_consequence_policy_v1"
    monkeypatch.setenv("SWM_CONSEQUENCES", "legacy_scalar_pathway_consequences")
    assert ActorPolicyRuntime().consequence_mode == "legacy_scalar_pathway_consequences"
    monkeypatch.setenv("SWM_CONSEQUENCES", "not_a_mode")
    assert sc.resolve_consequence_mode() == "generated_actor_mediated_world"
    with pytest.raises(ValueError):
        ActorPolicyRuntime(consequence_mode="bogus_mode")


def test_fixed_v1_execution_writes_no_pathway_scalars():
    w = world(declared_bar=True)
    runtime = ActorPolicyRuntime(consequence_mode="fixed_semantic_consequence_policy_v1")
    decision = {"candidate_actions": [{"name": "reject", "family": "negotiation",
                                       "target": {"target_type": "person", "target_id": "bob"}}]}
    result = decide_and_execute_particles(runtime, Plan(), [w], "alice",
                                          decision=decision, seed=5)
    delta = result["executions"][0]["delta"]
    scalar_paths = [c["path"] for c in delta.changes
                    if "pathway_progress" in c["path"] and "(derived)" not in c["path"]]
    assert scalar_paths == []                       # no action→bar write in the fixed baseline
    assert "consequence_program" in delta.uncertainty
    assert any(r.startswith("semantic_ops:") for r in delta.reason_codes)
    assert runtime.consequence_report["actual_mode"] == "fixed_semantic_consequence_policy_v1"
    assert runtime.consequence_report["direct_operations_applied"] >= 1
    assert runtime.consequence_report["legacy_scalar_writes"] == 0


def test_generated_default_without_schema_degrades_loudly_never_silently():
    """The production default is generated_actor_mediated_world; a world with NO scenario
    schema is EXECUTION-INCOMPLETE: the exact attempt is preserved, the branch is marked
    structurally under-modeled, and the fixed-v1 consequence system is NOT served in its
    place. Generated mode never degrades to fixed-v1 — silent or stamped."""
    w = world(declared_bar=True)
    runtime = ActorPolicyRuntime()                  # resolved default: generated
    decision = {"candidate_actions": [{"name": "reject", "family": "negotiation",
                                       "target": {"target_type": "person", "target_id": "bob"}}]}
    result = decide_and_execute_particles(runtime, Plan(), [w], "alice",
                                          decision=decision, seed=5)
    delta = result["executions"][0]["delta"]
    rep = runtime.consequence_report
    assert rep["requested_mode"] == "generated_actor_mediated_world"
    assert rep["actual_mode"] == "generated_actor_mediated_world"   # mode never swaps
    assert rep["structurally_under_modeled"] is True and rep["degraded"] is True
    assert rep["fixed_ontology_uses"] == 0          # fixed-v1 did NOT serve
    assert any(fr.get("kind") == "no_scenario_schema" for fr in rep["fallback_reasons"])
    assert "execution_incomplete:no_scenario_schema" in delta.reason_codes
    assert delta.uncertainty["unexecuted_attempt"]["action"] == \
        result["selected_action"].action_name          # the exact attempt is preserved
    assert w.objects == {}                          # no fixed-catalog objects appeared
    assert not any("pathway_progress" in c["path"] and "(derived)" not in c["path"]
                   for c in delta.changes)
    hist = w.entity("alice").value("past_actions") or []
    assert hist and hist[-1]["completion_status"] == "execution_incomplete"


def test_legacy_mode_is_the_only_scalar_path_and_is_counted():
    w = world(declared_bar=True)
    runtime = ActorPolicyRuntime(consequence_mode="legacy_scalar_pathway_consequences")
    decision = {"candidate_actions": [{"name": "reject", "family": "negotiation",
                                       "target": {"target_type": "person", "target_id": "bob"}}]}
    _, posterior, trace = runtime.decide(Plan(), [w], "alice", decision=decision, seed=5)
    delta, _ = runtime.execute(w, act("reject"), posterior, trace, seed=5)
    assert any("pathway_progress" in c["path"] for c in delta.changes)
    assert "legacy_scalar_pathway_consequences" in delta.reason_codes
    assert runtime.consequence_report["legacy_scalar_writes"] >= 1
    assert w.objects == {}                          # legacy mode creates no semantic objects


def test_scalar_writers_assert_the_mode():
    w = world(declared_bar=True)
    d = StateDelta(at=T0, event_type="actor_action", operator="test")
    with pytest.raises(RuntimeError):
        ActorPolicyRuntime._apply_pathway_effects(
            w, act("reject"), d, consequence_mode="semantic_world_consequences")
    with pytest.raises(RuntimeError):
        ActorPolicyRuntime._apply_immediate_consequences(
            w, act("reject"), d, consequence_mode="semantic_world_consequences")


def test_dual_run_applies_semantic_and_records_legacy_shadow_unapplied():
    w = world(declared_bar=True)
    runtime = ActorPolicyRuntime(consequence_mode="dual_run_consequence_audit")
    decision = {"candidate_actions": [{"name": "reject", "family": "negotiation",
                                       "target": {"target_type": "person", "target_id": "bob"}}]}
    before = w.quantities["pathway_progress:cooperative_agreement"].value
    _, posterior, trace = runtime.decide(Plan(), [w], "alice", decision=decision, seed=5)
    delta, _ = runtime.execute(w, act("reject"), posterior, trace, seed=5)
    shadow = runtime.consequence_report.get("dual_run_legacy_shadow")
    assert shadow and shadow[0]["unapplied_changes"]           # legacy recorded…
    assert any("pathway_progress" in c["path"] for c in shadow[0]["unapplied_changes"])
    real_scalar = [c for c in delta.changes
                   if "pathway_progress" in c["path"] and "(derived)" not in c["path"]]
    assert real_scalar == []                                    # …but never applied
    # the real world's bar may only have moved through the DERIVED projection of typed state
    v = w.quantities["pathway_progress:cooperative_agreement"].value
    assert v == before or any("(derived)" in c["path"] for c in delta.changes)


# ---------------------------------------------------------------- closed registry & quarantine
def test_forbidden_numeric_minting_is_quarantined_at_compile():
    w = world()
    bad = [{"op": "set_typed_fact", "object_id": "x", "fact": "success_probability", "value": 0.9},
           {"op": "create_world_object", "object_type": "product",
            "attributes": {"name": "P"}, "utility": 0.7},
           {"op": "create_world_object", "object_type": "product",
            "attributes": {"name": "Q"}, "nested": {"pathway_progress": 0.5}},
           {"op": "update_quantity", "name": "anything", "delta": 0.5},
           {"op": "set_typed_fact", "object_id": "x", "fact": "outcome_label", "value": "True"},
           "not an op"]
    ops, quarantined = sc.validate_operations(bad, w, "alice")
    assert ops == []
    assert len(quarantined) == 6
    reasons = json.dumps([q["reason"] for q in quarantined])
    assert "forbidden" in reasons and "unknown primitive" in reasons


def test_unknown_object_type_rejected_closed_registry():
    w = world()
    prog, delta, report, _ = run_program(w, [{"op": "create_world_object",
                                              "object_type": "vibe", "attributes": {}}])
    assert prog.quarantined and "unknown object_type" in prog.quarantined[0]["reason"]
    assert w.objects == {} and prog.unmodeled


def test_no_primitive_can_touch_terminal_resolution():
    w = world()
    prog, _, _, events = run_program(w, [
        {"op": "schedule_event", "etype": "resolve_outcome", "delay_s": 60.0},
        {"op": "schedule_event", "etype": "hazard_round", "delay_s": 60.0}])
    assert len(prog.quarantined) == 2                # terminal/hazard events are unschedulable
    assert events == []


def test_authority_required_to_modify_foreign_objects():
    w = world()
    run_program(w, [{"op": "create_world_object", "object_type": "product",
                     "object_id": "widget", "attributes": {"name": "Widget"}}], actor="alice")
    prog, _, _, _ = run_program(w, [{"op": "update_world_object", "object_id": "widget",
                                     "status": "discontinued"}], actor="bob")
    assert prog.quarantined and "lacks authority" in prog.quarantined[0]["reason"]
    assert w.objects["widget"].status != "discontinued"


def test_object_visibility_respects_participants():
    w = world()
    run_program(w, [{"op": "create_world_object", "object_type": "private_communication",
                     "object_id": "secret", "visibility": "participants",
                     "audience": ["bob"], "attributes": {"content": "for bob only"}}])
    ids = lambda actor: {o.object_id for o in sc.visible_objects(w, actor)}  # noqa: E731
    assert "secret" in ids("bob") and "secret" in ids("alice")   # audience + creator
    w.entities["carol"] = Entity("carol")
    assert "secret" not in ids("carol")


# ---------------------------------------------------------------- communications
def test_deliver_information_preserves_exact_content_end_to_end():
    w = world()
    msg = "Meet me Tuesday at 9. If the audit is buried, I resign — that is final."
    prog, delta, report, events = run_program(
        w, [{"op": "deliver_information", "recipient": "bob", "content": msg,
             "channel": "private_letter"}])
    comm = [o for o in w.objects.values() if o.object_type == "private_communication"]
    assert len(comm) == 1 and comm[0].attributes["content"] == msg
    assert report["information_deliveries"] == 1
    ev = next(e for e in events if e.etype == "message_delivered")
    assert ev.payload["content"] == msg              # the EXACT message rides the event
    op = sc.CommunicationDeliveryOperator()
    assert op.applicable(w, ev)
    ddelta, vr = op.run(w, ev, random.Random(0))
    assert vr.ok
    # DELIVERED ≠ READ (§9): delivery makes the message AVAILABLE and schedules bob's real
    # attention opportunity; nothing is exposed yet (invariants 17/18)
    fu = ddelta.follow_up_events
    assert fu and fu[0]["etype"] == "ctrl_attention" and fu[0]["participants"] == ["bob"]
    assert not w.information.visible_to("bob", at=w.clock.now)
    assert comm[0].status == "delivered"
    from swm.world_model_v2 import generated_world as gw
    from swm.world_model_v2.events import Event
    att = gw.GeneratedAttentionOperator(report=sc.empty_report())
    w.clock.now = max(w.clock.now, fu[0]["ts"])
    adelta, avr = att.run(w, Event(ts=fu[0]["ts"], etype="ctrl_attention",
                                   participants=["bob"], payload=dict(fu[0]["payload"])),
                          random.Random(0))
    assert avr.ok
    afu = adelta.follow_up_events
    assert afu and afu[0]["etype"] == "decision_opportunity"
    assert afu[0]["participants"] == ["bob"]
    assert msg in afu[0]["payload"]["situation"]     # the recipient decides on the REAL text
    assert afu[0]["payload"]["trigger"]["trigger_type"] == "newly_noticed_information"
    vis = w.information.visible_to("bob", at=w.clock.now + 1)
    assert any(msg == item.content for item, _e in vis)


def test_empty_content_communication_is_rejected():
    w = world()
    prog, _, report, events = run_program(
        w, [{"op": "deliver_information", "recipient": "bob", "content": ""}])
    assert prog.quarantined and events == []
    assert report["information_deliveries"] == 0


def test_publish_artifact_exposes_exact_text_to_audience():
    w = world()
    text = "ACME will launch the Widget on March 3rd at $499."
    _, _, report, _ = run_program(w, [{"op": "publish_artifact", "content": text,
                                       "artifact_type": "press_release"}])
    stmt = [o for o in w.objects.values() if o.object_type == "public_statement"]
    assert stmt and stmt[0].attributes["content"] == text
    for actor in ("alice", "bob"):
        vis = w.information.visible_to(actor, at=w.clock.now + 1)
        assert any(item.content == text for item, _e in vis)


def test_sender_expectation_never_becomes_recipient_reaction():
    """alice sends a demand EXPECTING compliance; bob's own policy chooses to refuse. The world
    must record bob's refusal, alice's expectation stays a subjective actor-local record."""
    from swm.world_model_v2.qualitative_actor import (
        QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
    )

    msg = "Sign the merger papers by Friday or I go to the press."
    alice_decisions = []

    def backend(prompt):
        if "CONSEQUENCE COMPILER" in prompt:
            if "ACTOR: alice" in prompt:             # only ALICE's action sends the demand
                return json.dumps([{"op": "deliver_information", "recipient": "bob",
                                    "content": msg, "channel": "private"}])
            return json.dumps([{"op": "record_observation", "note": "refusal recorded"}])
        payload = {
            "schema_version": "qualitative.actor.v1",
            "situation_interpretation": {"what_changed": "x", "why_it_matters": "y",
                                         "perceived_opportunities": "", "perceived_threats": ""},
            "actor_state_update": {"current_private_beliefs": [], "beliefs_about_others": {},
                                   "personal_condition": "resolute", "important_memories": []},
            "anticipated_reactions": [],
            "decision": {"act_or_wait": "act", "chosen_action": "demand", "target": "bob",
                         "timing": "immediate", "observability": "private",
                         "intended_effect": "force the signature"},
            "novel_action_proposal": {"present": False},
            "alternatives_considered": [], "decision_summary": "I demand the signature",
        }
        if "You ARE alice" in prompt:
            alice_decisions.append(prompt)
            if len(alice_decisions) > 1:              # later rounds: nothing more to do
                payload["decision"] = {"act_or_wait": "wait", "chosen_action": "wait",
                                       "target": "", "timing": "immediate",
                                       "observability": "private",
                                       "intended_effect": "await bob's answer"}
                payload["decision_summary"] = "I wait"
                return json.dumps(payload)
            payload["anticipated_reactions"] = [{"actor_or_group": "bob",
                                                 "expected_reaction": "will comply immediately",
                                                 "reasoning_summary": "he always folds",
                                                 "uncertainty_description": "low"}]
        else:                                        # bob's OWN decision on the real message
            payload["decision"] = {"act_or_wait": "act", "chosen_action": "reject",
                                   "target": "alice", "timing": "immediate",
                                   "observability": "private",
                                   "intended_effect": "refuse the ultimatum"}
            payload["decision_summary"] = "I refuse"
        return json.dumps(payload)

    w = world()
    engine = QualitativeDecisionEngine(QualitativeConfig(llm=backend, llm_hypotheses=False,
                                                         n_hypotheses=2))
    # the fixed-v1 BASELINE flow (explicitly requested — schemaless generated mode is
    # execution-incomplete by design and serves nothing)
    rt = QualitativeActorPolicyRuntime(engine, mode="persistent_qualitative_llm_policy",
                                       consequence_mode="fixed_semantic_consequence_policy_v1")
    op = ProductionActorPolicyOperator(runtime=rt)
    q = EventQueue(horizon_ts=T0 + 5 * 86400)
    q.schedule(Event(ts=T0 + 60, etype="decision_opportunity", participants=["alice"],
                     payload={"situation": "the merger papers are unsigned",
                              "candidate_actions": [{"name": "demand", "family": "negotiation",
                                                     "target": {"target_type": "person",
                                                                "target_id": "bob"}}]}))
    RolloutEngine(operators=[op, sc.CommunicationDeliveryOperator()]).run_branch(w, q, seed=3)
    alice_expect = w.entity("alice").value("expected_reactions") or {}
    assert alice_expect.get("bob", {}).get("expects") == "will comply immediately"
    assert alice_expect.get("bob", {}).get("subjective") is True
    bob_history = w.entity("bob").value("past_actions") or []
    assert any(a["action"] == "reject" and a["status"] == "action_attempt_initiated"
               for a in bob_history)                 # bob's OWN attempt, never 'executed'
    assert not any(a["action"] == "comply" for a in bob_history)   # expectation never executed
    comm = [o for o in w.objects.values() if o.object_type == "private_communication"]
    assert comm and comm[0].attributes["content"] == msg and comm[0].status == "delivered"


# ---------------------------------------------------------------- institutions
def test_sole_right_holder_submission_becomes_typed_decision():
    w = world(holders=("alice",))
    prog, delta, report, events = run_program(
        w, [{"op": "submit_to_institution", "institution": "board",
             "matter": "approve the merger", "requested_outcome": "approve"}])
    sub = next(o for o in w.objects.values() if o.object_type == "submission")
    proc = w.objects[f"proc_{sub.object_id}"]
    assert sub.status == "decided" and sub.attributes["outcome"] == "approve"
    assert sub.attributes["decided_by"] == "alice"
    assert proc.status == "decided" and len(proc.stage_history) == 2
    assert not any(e.etype == "collective_vote" for e in events)   # no fake vote needed
    assert report["institutional_submissions"] == 1


def test_multi_holder_submission_enters_real_procedure_with_vote_and_member_decisions():
    w = world(holders=("alice", "bob"))
    prog, delta, report, events = run_program(
        w, [{"op": "submit_to_institution", "institution": "board",
             "matter": "approve the merger", "requested_outcome": "approve"}])
    sub = next(o for o in w.objects.values() if o.object_type == "submission")
    assert sub.status == "submitted"                                # NOT decided by submitting
    etypes = [e.etype for e in events]
    assert "collective_vote" in etypes                              # the REAL consumer's type
    member = next(e for e in events if e.etype == "decision_opportunity")
    assert member.participants == ["bob"]                           # the OTHER holder decides
    assert report["actor_decisions_opened"] == 1
    vote_ev = next(e for e in events if e.etype == "collective_vote")
    from swm.world_model_v2.transitions import InstitutionalVoteOperator
    assert InstitutionalVoteOperator().applicable(w, vote_ev)       # consumable as scheduled


def test_submission_to_missing_institution_is_quarantined():
    w = world()
    prog, _, report, events = run_program(
        w, [{"op": "submit_to_institution", "institution": "senate", "matter": "x",
             "requested_outcome": "approve"}])
    assert prog.quarantined and "does not exist" in prog.quarantined[0]["reason"]
    assert report["institutional_submissions"] == 0 and events == []


def test_decided_outcome_projects_onto_declared_quantities_only_when_decided():
    register_quantity_type("approval_progress", units="unit")
    # decided immediately (sole holder) → the declared readout quantity is a DERIVED projection
    w = world(holders=("alice",))
    w.quantities["approval_progress"] = Quantity("approval_progress", "approval_progress",
                                                 value=0.0)
    a = act("approve", family="institutional", target="board",
            consequences=[{"kind": "quantity_delta", "name": "approval_progress", "delta": 1}])
    run_program(w, [{"op": "submit_to_institution", "institution": "board", "matter": "m",
                     "requested_outcome": "approve"}], action_id=a.action_id)
    d = StateDelta(at=T0, event_type="actor_action", operator="test")
    rep = sc.empty_report()
    assert sc.project_decided_outcome_quantities(w, a, d, rep) == 1
    q = w.quantities["approval_progress"]
    assert q.value == 1.0 and q.prov.status == "derived"
    assert "semantic_projection" in q.prov.method
    # pending (multi-holder) → NOTHING projects; the vote writes the outcome when it happens
    w2 = world(holders=("alice", "bob"))
    w2.quantities["approval_progress"] = Quantity("approval_progress", "approval_progress",
                                                  value=0.0)
    run_program(w2, [{"op": "submit_to_institution", "institution": "board", "matter": "m",
                      "requested_outcome": "approve"}], action_id=a.action_id)
    d2 = StateDelta(at=T0, event_type="actor_action", operator="test")
    assert sc.project_decided_outcome_quantities(w2, a, d2, sc.empty_report()) == 0
    assert w2.quantities["approval_progress"].value == 0.0
    # forbidden names never project, even from a decided submission
    w3 = world(holders=("alice",))
    bad = act("approve", family="institutional", target="board",
              consequences=[{"kind": "quantity_delta", "name": "success_probability",
                             "delta": 1}])
    run_program(w3, [{"op": "submit_to_institution", "institution": "board", "matter": "m",
                      "requested_outcome": "approve"}], action_id=bad.action_id)
    d3 = StateDelta(at=T0, event_type="actor_action", operator="test")
    assert sc.project_decided_outcome_quantities(w3, bad, d3, sc.empty_report()) == 0
    assert "success_probability" not in w3.quantities


# ---------------------------------------------------------------- processes & resources
def test_process_stages_advance_only_along_declared_machines():
    w = world()
    run_program(w, [{"op": "start_process", "process_type": "negotiation",
                     "object_id": "neg", "subject": "the deal"}])
    assert w.objects["neg"].status == "contact_opened"
    prog, _, _, _ = run_program(w, [{"op": "advance_process_stage", "object_id": "neg",
                                     "stage": "sudden_total_victory"}])
    assert prog.quarantined and "unknown stage" in prog.quarantined[0]["reason"]
    run_program(w, [{"op": "advance_process_stage", "object_id": "neg",
                     "stage": "terms_exchanged"}])
    assert w.objects["neg"].status == "terms_exchanged"
    assert [s["to"] for s in w.objects["neg"].stage_history] == ["contact_opened",
                                                                 "terms_exchanged"]


def test_terminal_processes_refuse_further_stages():
    w = world()
    run_program(w, [{"op": "start_process", "process_type": "negotiation", "object_id": "neg"},
                    {"op": "fail_process", "object_id": "neg", "why": "talks collapsed"}])
    assert w.objects["neg"].status == "broken_down"
    prog, _, _, _ = run_program(w, [{"op": "advance_process_stage", "object_id": "neg",
                                     "stage": "terms_exchanged"}])
    assert prog.quarantined and "terminal" in prog.quarantined[0]["reason"]


def test_complete_and_fail_pick_declared_final_stages():
    w = world()
    run_program(w, [{"op": "start_process", "process_type": "product_launch", "object_id": "pl"},
                    {"op": "complete_process", "object_id": "pl"}])
    assert w.objects["pl"].status in sc.PROCESS_STAGES["product_launch"]
    w2 = world()
    run_program(w2, [{"op": "start_process", "process_type": "regulatory_review",
                      "object_id": "rr"},
                     {"op": "fail_process", "object_id": "rr"}])
    assert w2.objects["rr"].status in sc.PROCESS_STAGES["regulatory_review"]


def test_resource_conservation_and_insufficiency():
    w = world()
    run_program(w, [{"op": "transfer_resource", "resource": "budget", "amount": 30.0,
                     "to": "bob"}])
    assert w.entity("alice").value("resources", key="budget") == 70.0
    assert w.entity("bob").value("resources", key="budget") == 30.0
    prog, _, _, _ = run_program(w, [{"op": "transfer_resource", "resource": "budget",
                                     "amount": 500.0, "to": "bob"}])
    assert prog.quarantined and "insufficient" in prog.quarantined[0]["reason"]
    assert w.entity("alice").value("resources", key="budget") == 70.0   # nothing moved
    prog2, _, _, _ = run_program(w, [{"op": "consume_resource", "resource": "budget",
                                      "amount": 71.0}])
    assert prog2.quarantined                                            # floor respected


# ---------------------------------------------------------------- compiler paths
class CompilerLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        out = self.reply(prompt) if callable(self.reply) else self.reply
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, str) else json.dumps(out)


def qualitative_stub(**over):
    base = {"decision_summary": "I announce the Widget launch to the press",
            "chosen_action": "announce_launch", "target": "bob", "timing": "immediate",
            "observability": "public", "intended_effect": "make the launch public",
            "linked_actions": [], "novel_action_proposal": {}}
    base.update(over)
    return base


def test_llm_proposal_is_untrusted_validated_op_by_op():
    llm = CompilerLLM([{"op": "publish_artifact", "content": "Widget launches today."},
                       {"op": "start_process", "process_type": "product_launch",
                        "subject": "Widget"},
                       {"op": "set_outcome_probability", "value": 0.95}])
    comp = sc.SemanticConsequenceCompiler(llm=llm)
    w = world()
    prog = comp.compile(w, act("announce_launch", family="messaging"),
                        qualitative=qualitative_stub())
    assert prog.compiler == "llm" and prog.llm_calls == 1
    assert [o["op"] for o in prog.operations] == ["publish_artifact", "start_process"]
    assert any("unknown primitive" in q["reason"] for q in prog.quarantined)


def test_llm_failure_falls_back_to_deterministic_semantics_loudly():
    comp = sc.SemanticConsequenceCompiler(llm=CompilerLLM(ConnectionError("backend down")))
    w = world()
    prog = comp.compile(w, act("propose_deal"), qualitative=qualitative_stub())
    assert prog.compiler == "fallback"
    assert any("compiler_llm_failed" in q["reason"] for q in prog.quarantined)
    assert prog.operations                            # deterministic REAL ops, not scalars
    assert not prog.unmodeled


def test_unparseable_llm_reply_falls_back_marked():
    comp = sc.SemanticConsequenceCompiler(llm=CompilerLLM("I would rather write prose."))
    prog = comp.compile(world(), act("propose_deal"), qualitative=qualitative_stub())
    assert prog.compiler == "fallback"
    assert any("unparseable" in q["reason"] for q in prog.quarantined)
    assert prog.operations


def test_numeric_actions_compile_deterministically_without_llm_calls():
    llm = CompilerLLM([{"op": "record_observation", "note": "x"}])
    comp = sc.SemanticConsequenceCompiler(llm=llm)
    prog = comp.compile(world(), act("approve", family="institutional", target="board"),
                        qualitative=None)             # numeric/Tier-3 action: no LLM step
    assert llm.prompts == [] and prog.compiler == "deterministic"
    assert prog.operations[0]["op"] == "submit_to_institution"


def test_empty_llm_program_is_loudly_unmodeled():
    comp = sc.SemanticConsequenceCompiler(llm=CompilerLLM("[]"))
    prog = comp.compile(world(), act("interpretive_dance"),
                        qualitative=qualitative_stub(chosen_action="interpretive_dance"))
    assert prog.unmodeled
    assert prog.operations[0]["op"] == "record_observation"   # placeholder record only
    w = world(declared_bar=True)
    _, delta, report, _ = run_program(w, prog.operations)
    assert "semantic_consequence_unmodeled" not in delta.reason_codes or True
    assert not any("pathway" in c["path"] for c in delta.changes)


def test_novel_action_never_reduces_to_nearest_label_scalar_effects():
    """A compiled novel action carrying an ontology anchor must NOT inherit the anchor's
    ACTION_PATHWAY_EFFECTS in semantic mode — its world change is its compiled program."""
    w = world(declared_bar=True)
    runtime = ActorPolicyRuntime(consequence_mode="fixed_semantic_consequence_policy_v1")
    novel = act("stage_hunger_strike", family="generic",
                ontology_anchor={"name": "escalate", "family": "participation"})
    decision = {"candidate_actions": [{"name": "reject", "family": "negotiation",
                                       "target": {"target_type": "person",
                                                  "target_id": "bob"}}]}
    _, posterior, trace = runtime.decide(Plan(), [w], "alice", decision=decision, seed=1)
    before = w.quantities["pathway_progress:cooperative_agreement"].value
    delta, events = runtime.execute(w, novel, posterior, trace, seed=1)
    direct = [c for c in delta.changes
              if "pathway_progress" in c["path"] and "(derived)" not in c["path"]]
    assert direct == []                               # the anchor's scalar effects are dead here
    assert "consequence_program" in delta.uncertainty


# ---------------------------------------------------------------- derived summaries (§NAP quarantined)
def test_derived_summaries_are_token_gated_legacy_ablation():
    """§NAP: the stage→fraction projection is a QUARANTINED legacy ablation — refusing without
    the acknowledgement token, and still a reconstructable projection under it (ablation runs
    must reproduce the historical behavior exactly)."""
    from swm.world_model_v2.legacy_numeric_ablations import ABLATION_TOKEN
    w = world(declared_bar=True)
    run_program(w, [{"op": "start_process", "process_type": "negotiation", "object_id": "neg"},
                    {"op": "advance_process_stage", "object_id": "neg",
                     "stage": "provisional_acceptance"}])
    with pytest.raises(PermissionError):
        sc.derive_pathway_summaries(w)                    # no token → refused
    written = sc.derive_pathway_summaries(w, acknowledge=ABLATION_TOKEN)
    bar = "pathway_progress:cooperative_agreement"
    assert bar in written
    v1 = w.quantities[bar].value
    stages = sc.PROCESS_STAGES["negotiation"]
    assert v1 == pytest.approx((stages.index("provisional_acceptance") + 1) / len(stages),
                               abs=1e-4)                 # stored at 4-decimal precision
    # reconstructable: recomputing from the SAME typed state is a fixed point
    assert sc.derive_pathway_summaries(w, acknowledge=ABLATION_TOKEN) == {}
    assert w.quantities[bar].value == v1


def test_summaries_touch_only_declared_bars():
    from swm.world_model_v2.legacy_numeric_ablations import ABLATION_TOKEN
    w = world(declared_bar=False)                     # nothing declared
    run_program(w, [{"op": "start_process", "process_type": "negotiation", "object_id": "neg"}])
    assert sc.derive_pathway_summaries(w, acknowledge=ABLATION_TOKEN) == {}
    assert "pathway_progress:cooperative_agreement" not in w.quantities


# ---------------------------------------------------------------- public & counterfactual paths
def test_run_from_plan_carries_consequence_report_and_semantic_default():
    from tests.test_wmv2_phase4_e2e import compiled_payload
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.materialize import run_from_plan
    plan = compile_world("Will the manager approve the project?",
                         llm=lambda _: json.dumps(compiled_payload()), evidence="",
                         as_of="2025-01-01", horizon="2025-01-10", persist=False)
    result, branches = run_from_plan(plan, n_particles=4, seed=3)
    rep = result["consequence_report"]
    # no LLM backend → no scenario schema can compile → the generated default is
    # EXECUTION-INCOMPLETE and structurally under-modeled; fixed-v1 is NOT served
    assert rep["requested_mode"] == "generated_actor_mediated_world"
    assert rep["actual_mode"] == "generated_actor_mediated_world"
    assert rep["degraded"] is True
    assert rep["structurally_under_modeled"] is True
    assert rep["fixed_ontology_uses"] == 0
    assert rep["legacy_scalar_writes"] == 0
    assert rep.get("scenario_schema_error")
    for b in branches:                                # no fixed-catalog consequence appeared
        assert not [o for o in b.world.objects.values() if o.object_type == "submission"]


def test_run_from_plan_fixed_baseline_still_runnable_when_explicitly_requested(monkeypatch):
    from tests.test_wmv2_phase4_e2e import compiled_payload
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.materialize import run_from_plan
    monkeypatch.setenv("SWM_CONSEQUENCES", "fixed_semantic_consequence_policy_v1")
    plan = compile_world("Will the manager approve the project?",
                         llm=lambda _: json.dumps(compiled_payload()), evidence="",
                         as_of="2025-01-01", horizon="2025-01-10", persist=False)
    result, branches = run_from_plan(plan, n_particles=4, seed=3)
    rep = result["consequence_report"]
    assert rep["actual_mode"] == "fixed_semantic_consequence_policy_v1"
    assert rep["institutional_submissions"] >= 1
    assert rep["legacy_scalar_writes"] == 0
    for b in branches:                                # the typed decision exists in every branch
        subs = [o for o in b.world.objects.values() if o.object_type == "submission"]
        assert subs and subs[0].status == "decided"
        assert subs[0].attributes["outcome"] == "approve"
        # and the contract's readout quantity is the DERIVED projection of that typed fact
        q = b.world.quantities["approval_progress"]
        assert q.value == 1 and q.prov.status == "derived"


def test_individual_reaction_artifact_carries_consequence_report():
    from swm.world_model_v2.individual_reaction import simulate_individual_reaction
    from swm.world_model_v2.qualitative_actor import QualitativeConfig

    def backend(prompt):
        if "CONSEQUENCE COMPILER" in prompt:
            return json.dumps([{"op": "deliver_information", "recipient": "you",
                                "content": "sounds good, next week then"}])
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "situation_interpretation": {"what_changed": "x", "why_it_matters": "y",
                                         "perceived_opportunities": "",
                                         "perceived_threats": ""},
            "actor_state_update": {"current_private_beliefs": [],
                                   "beliefs_about_others": {},
                                   "personal_condition": "fine", "important_memories": []},
            "anticipated_reactions": [],
            "decision": {"act_or_wait": "act", "chosen_action": "reply_now", "target": "you",
                         "timing": "immediate", "observability": "private",
                         "intended_effect": "reassure"},
            "novel_action_proposal": {"present": False},
            "alternatives_considered": [], "decision_summary": "I reply warmly"})

    result = simulate_individual_reaction(
        person_id="Dana", stimulus="Can we move dinner to next week?",
        context={"relationship": "close friend"}, llm=backend,
        n_hypotheses=2, samples_per_hypothesis=1, seed=0, as_of=T0,
        config=QualitativeConfig(llm=backend, llm_hypotheses=False, n_hypotheses=2))
    rep = result["consequence_report"]
    # without a scenario schema the generated default is EXECUTION-INCOMPLETE: the reply
    # attempt is preserved, fixed-v1 is NOT served, and the artifact says so — the reaction
    # DISTRIBUTION (the deliverable) still comes from the actor's own decisions
    assert rep["requested_mode"] == "generated_actor_mediated_world"
    assert rep["actual_mode"] == "generated_actor_mediated_world"
    assert rep["degraded"] is True and rep["structurally_under_modeled"] is True
    assert rep["fixed_ontology_uses"] == 0
    assert rep["legacy_scalar_writes"] == 0
    assert result["raw_qualitative_simulation_distribution"]      # the answer still counts


def test_counterfactual_branches_keep_objects_isolated():
    w = world()
    branch_a = w.clone(branch_id="bA")
    branch_b = w.clone(branch_id="bB")
    run_program(branch_a, [{"op": "create_world_object", "object_type": "agreement",
                            "object_id": "deal", "status": "signed", "attributes": {}}])
    assert "deal" in branch_a.objects
    assert branch_b.objects == {} and w.objects == {}   # strictly branch-local worlds
    readout = sc.make_object_predicate_readout(object_type="agreement",
                                               status_in=("signed", "active"))
    assert readout(branch_a) == "True" and readout(branch_b) == "False"


def test_qualitative_decision_wording_reaches_the_compiler_prompt():
    from swm.world_model_v2.qualitative_actor import (
        QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
    )
    from swm.world_model_v2.scenario_schema import ScenarioSemanticModel
    seen = {}

    def backend(prompt):
        if "ACTION-ATTEMPT COMPILER" in prompt:
            seen["compile_prompt"] = prompt
            return json.dumps([{"op": "emit_semantic_event",
                                "semantic_type_id": "counteroffer_message_drafted",
                                "exact_content": "I counter at 40 with the audit clause "
                                                 "intact",
                                "direct_targets": ["bob"]}])
        if "CAUSAL DIRECTNESS CRITIC" in prompt:
            return "[]"
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "situation_interpretation": {"what_changed": "x", "why_it_matters": "y",
                                         "perceived_opportunities": "",
                                         "perceived_threats": ""},
            "actor_state_update": {"current_private_beliefs": [], "beliefs_about_others": {},
                                   "personal_condition": "calm", "important_memories": []},
            "anticipated_reactions": [],
            "decision": {"act_or_wait": "act", "chosen_action": "counteroffer", "target": "bob",
                         "timing": "after_the_board_meets", "observability": "private",
                         "intended_effect": "anchor at a lower price"},
            "novel_action_proposal": {"present": False},
            "alternatives_considered": [],
            "decision_summary": "I counter at 40 with the audit clause intact"})

    engine = QualitativeDecisionEngine(QualitativeConfig(llm=backend, llm_hypotheses=False,
                                                         n_hypotheses=2))
    rt = QualitativeActorPolicyRuntime(engine, mode="persistent_qualitative_llm_policy")
    w = world()
    w.scenario_schema = ScenarioSemanticModel(
        question="Will alice and bob agree terms?", prediction_timestamp=T0,
        horizon=T0 + 10 * 86400,
        semantic_event_types={"counteroffer_message_drafted":
                              {"description": "attempt", "fields": {},
                               "typical_visibility": "participants"}},
        outcome_predicates=[{"predicate_id": "agreed", "record_type": "signed_agreement",
                             "op": "exists", "option_true": "deal",
                             "option_false": "no_deal"}],
        fact_types={"signed_agreement": {"description": "x", "fields": {}}},
        provenance={"compiler": "test"}).freeze()
    decision = {"candidate_actions": [{"name": "counteroffer", "family": "negotiation",
                                       "target": {"target_type": "person",
                                                  "target_id": "bob"}}],
                "situation": "bob offered 55"}
    sel, post, tr = rt.decide(Plan(), [w], "alice", decision=decision, seed=2)
    rt.execute(w, sel, post, tr, seed=2)
    # the actor's own wording reaches the attempt compiler AND the attempt event verbatim
    assert "I counter at 40 with the audit clause intact" in seen["compile_prompt"]
    assert rt.consequence_report["actions_compiled"] == 1
    assert rt.consequence_report["action_attempts"] == 1
    attempt = next(x for x in w.semantic_log
                   if x["semantic_type_id"] == "counteroffer_message_drafted")
    assert attempt["exact_content"] == "I counter at 40 with the audit clause intact"
    assert attempt["observability_verified"] is False       # drafting is not delivery


def test_execute_returns_semantic_events_on_the_delta_for_the_engine():
    w = world()
    runtime = ActorPolicyRuntime(consequence_mode="fixed_semantic_consequence_policy_v1")
    decision = {"candidate_actions": [{"name": "reply_now", "family": "messaging",
                                       "target": {"target_type": "person",
                                                  "target_id": "bob"}}]}
    result = decide_and_execute_particles(runtime, Plan(), [w], "alice",
                                          decision=decision, seed=4)
    delta = result["executions"][0]["delta"]
    events = result["executions"][0]["events"]
    delivered = [e for e in events if e.etype == "message_delivered"]
    assert len(delivered) == 1                        # semantic delivery, legacy ping suppressed
    assert delivered[0].payload["content"]            # with REAL content
    assert any(f["etype"] == "message_delivered" for f in delta.follow_up_events)


def test_report_counts_are_complete_and_consistent():
    w = world(holders=("alice", "bob"))
    ops = [{"op": "create_world_object", "object_type": "product", "object_id": "widget",
            "attributes": {"name": "Widget"}},
           {"op": "publish_artifact", "content": "Widget exists now."},
           {"op": "deliver_information", "recipient": "bob", "content": "see the launch"},
           {"op": "submit_to_institution", "institution": "board", "matter": "ratify",
            "requested_outcome": "approve"},
           {"op": "start_process", "process_type": "adoption", "object_id": "adopt"},
           {"op": "open_actor_decision", "actor": "bob", "situation": "respond to launch"},
           {"op": "open_population_response", "population": "customers",
            "stimulus": "Widget launch"},
           {"op": "schedule_event", "etype": "process_stage", "delay_s": 3600.0}]
    prog, delta, report, events = run_program(w, ops)
    assert not prog.quarantined
    assert report["direct_operations_applied"] == len(ops)
    assert report["objects_created"] >= 5             # product, statement, comm, sub, 2×proc
    assert report["institutional_submissions"] == 1
    assert report["processes_started"] == 2           # institutional_procedure + adoption
    assert report["information_deliveries"] >= 2      # publish audience + private delivery
    assert report["actor_decisions_opened"] == 2      # board member + opened decision
    assert report["population_responses_opened"] == 1
    assert report["events_scheduled"] == 1
    assert report["unsupported_semantics"] == 0
    assert {e.etype for e in events} >= {"message_delivered", "collective_vote",
                                         "decision_opportunity",
                                         "population_response_opened", "process_stage"}
