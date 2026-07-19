"""Causal truth boundary — the §19 focused architecture/regression tests.

The governing rule under test: an action may directly create only facts that remain
guaranteed after assuming the actor successfully performed every step under that actor's
unilateral mechanical control. Anything that could still fail because of a channel, platform,
system, institution, another actor, a physical constraint, an administrative/legal process,
acceptance, delivery, visibility, processing, settlement, or later execution must occur
through an explicit scenario mechanism — or stay honestly unresolved.

All scenario semantics here (a grant-application scenario, a carrier-pigeon registry) are
test-generated: nothing in production code names these domains."""
import json

import pytest

from swm.world_model_v2 import causal_boundary as cb
from swm.world_model_v2 import generated_world as gw
from swm.world_model_v2 import semantic_consequences as sc
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import ActorPolicyRuntime, ProductionActorPolicyOperator
from swm.world_model_v2.phase4_policy import ActionTarget, TypedAction
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.scenario_schema import ScenarioSemanticModel
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta

T0 = 1_700_000_000.0


def grant_schema():
    """A grant-application scenario with its OWN mechanisms: a departmental email channel,
    a university postal intake, and a registrar acceptance process. Generated-per-scenario
    semantics — none of this exists in repository code."""
    return ScenarioSemanticModel(
        question="Will Prof. Ada's consortium application be accepted?",
        prediction_timestamp=T0, horizon=T0 + 60 * 86400,
        entity_types={"university_department": {"description": "dept",
                                                "fields": {"name": "str"}}},
        fact_types={
            "application_form": {"description": "ada's own draft application",
                                 "fields": {"title": "str", "status": "str"}},
            "own_signature_record": {"description": "one party's own signature",
                                     "fields": {"document": "str"},
                                     "controlled_by": "prof_ada"},
            "intake_log_entry": {"description": "the registrar's intake log — the SYSTEM "
                                                "writes this, never the applicant",
                                 "fields": {"application": "str", "matter": "str"}},
            "bilateral_mou": {"description": "a recognized two-party memorandum",
                              "fields": {"parties": "list"},
                              "controlled_by": "joint_signature_process"},
            "confirmed_meeting": {"description": "a meeting both sides confirmed",
                                  "fields": {"with_whom": "str"},
                                  "controlled_by": "prof_boyd"},
            "acceptance_decision": {"description": "the registrar's acceptance decision",
                                    "fields": {"application": "str"}}},
        relation_types={
            "consortium_partner_of": {"description": "recognized partnership",
                                      "directed": False},
            "follows_work_of": {"description": "one-sided attention",
                                "directed": True, "unilateral": True}},
        semantic_event_types={
            "colleague_note_drafted": {"description": "ATTEMPT: an outgoing note",
                                       "fields": {"about": "str"},
                                       "typical_visibility": "participants"},
            "colleague_note_delivered": {"description": "channel outcome",
                                         "fields": {"about": "str"},
                                         "typical_visibility": "participants"},
            "application_package_submitted": {"description": "ATTEMPT: package handed to "
                                                             "the post",
                                              "fields": {"title": "str"},
                                              "typical_visibility": "participants"},
            "application_intake_confirmed": {"description": "intake outcome",
                                             "fields": {"title": "str"},
                                             "typical_visibility": "participants"},
            "meeting_invitation_issued": {"description": "ATTEMPT: an invitation",
                                          "fields": {"when": "str"},
                                          "typical_visibility": "participants"},
            "results_announcement_posted": {"description": "ATTEMPT: press release",
                                            "fields": {},
                                            "typical_visibility": "public"},
            "results_announcement_published": {"description": "wire outcome", "fields": {},
                                               "typical_visibility": "public"}},
        resource_definitions={"grant_funds": {"unit": "usd", "conserved": True,
                                              "settlement_mechanism":
                                                  "university_finance_settlement"}},
        actor_roles={"prof_ada": {"role": "applicant", "why_consequential": "her application",
                                  "affordances": ["submit application"]}},
        mechanism_definitions={
            "dept_email_channel": {
                "description": "the departmental mail system carrying colleague notes",
                "triggering_event_types": ["colleague_note_drafted"],
                "accepted_inputs": {"about": "str"},
                "controlling_actor_or_system": "dept_mail_infrastructure",
                "state_machine": {"queued": ["delivered", "bounced"]},
                "initial_state": "queued",
                "success_states": ["delivered"], "failure_states": ["bounced"],
                "unresolved_states": [],
                "transition_rules": [
                    {"from": "queued", "to": "bounced",
                     "when": {"entity": "prof_boyd", "field": "latent_state",
                              "key": "mailbox_state", "equals": "over_quota"}},
                    {"from": "queued", "to": "delivered",
                     "when": {"entity": "prof_boyd", "field": "latent_state",
                              "key": "mailbox_state", "equals": "normal"}}],
                "possible_output_event_types": {
                    "on_success": ["colleague_note_delivered"], "on_failure": []},
                "observation_rules": {"recipients": "direct_targets",
                                      "representation": "complete"},
                "timing_rules": {"delay_s": 60.0},
                "assumptions": ["standard smtp relay"],
                "uncertainty_source": "the recipient's mailbox state (branch-hidden)"},
            "university_postal_intake": {
                "description": "campus post + registrar intake logging",
                "triggering_event_types": ["application_package_submitted"],
                "accepted_inputs": {"title": "str"},
                "controlling_actor_or_system": "registrar_office",
                "state_machine": {"in_transit": ["logged"]},
                "initial_state": "in_transit",
                "success_states": ["logged"], "failure_states": ["lost_in_post"],
                "unresolved_states": [],
                "possible_output_event_types": {
                    "on_success": ["application_intake_confirmed"], "on_failure": []},
                "possible_record_updates": ["intake_log_entry"],
                "observation_rules": {"recipients": "initiator",
                                      "representation": "complete"},
                "timing_rules": {"delay_s": 3600.0},
                "assumptions": ["campus post operates"],
                "uncertainty_source": "physical transit"},
            "campus_press_wire": {
                "description": "the campus press office wire",
                "triggering_event_types": ["results_announcement_posted"],
                "accepted_inputs": {},
                "controlling_actor_or_system": "campus_press_office",
                "state_machine": {"drafted_at_desk": ["published"]},
                "initial_state": "drafted_at_desk",
                "success_states": ["published"], "failure_states": ["spiked"],
                "unresolved_states": [],
                "possible_output_event_types": {
                    "on_success": ["results_announcement_published"], "on_failure": []},
                "observation_rules": {"recipients": "direct_targets",
                                      "availability": "public",
                                      "representation": "complete"},
                "timing_rules": {"delay_s": 600.0},
                "assumptions": ["press office reviews same day"],
                "uncertainty_source": "editorial pickup"},
            "university_finance_settlement": {
                "description": "the university finance office settling fund transfers",
                "triggering_event_types": [],
                "accepted_inputs": {"resource": "str", "amount": "float", "to": "str"},
                "controlling_actor_or_system": "university_finance_office",
                "state_machine": {"authorization_requested": ["settled"]},
                "initial_state": "authorization_requested",
                "success_states": ["settled"], "failure_states": ["declined"],
                "unresolved_states": [],
                "possible_output_event_types": {"on_success": [], "on_failure": []},
                "observation_rules": {"recipients": "initiator"},
                "timing_rules": {"delay_s": 3600.0},
                "executor_binding": "conserved_resource_settlement",
                "assumptions": ["finance office processes daily"],
                "uncertainty_source": "authorization checks"}},
        outcome_predicates=[{"predicate_id": "accepted",
                             "record_type": "acceptance_decision", "op": "exists",
                             "option_true": "accepted", "option_false": "not_accepted"}],
        provenance={"compiler": "test"}).freeze()


def grant_world(schema=None, actors=("prof_ada", "prof_boyd"), mailbox="normal"):
    w = WorldState("cb", "b0", SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    for name in actors:
        e = Entity(name)
        e.set("roles", F(["person"], status="observed"))
        e.set("resources", F(10_000.0, status="observed"), key="grant_funds")
        e.set("past_actions", F([], status="observed"))
        w.entities[name] = e
    if "prof_boyd" in w.entities:
        # branch-hidden state the email mechanism's declared rules read
        w.entities["prof_boyd"].set("latent_state", F(mailbox, status="sampled"),
                                    key="mailbox_state")
    w.scenario_schema = schema if schema is not None else grant_schema()
    return w


def ctx_for(w, actor="prof_ada", action="a1", report=None):
    return {"actor_id": actor, "action_id": action, "now": w.clock.now,
            "report": report if report is not None else gw.generated_report(),
            "budgets": gw._budgets(w), "events": [], "quarantined": [],
            "compiler": "test", "plane": "direct_action"}


def run_ops(w, ops, **kw):
    ctx = ctx_for(w, **kw)
    d = StateDelta(at=w.clock.now, event_type="actor_action", operator="test")
    events = gw.execute_kernel_ops(w, ops, ctx, d)
    return ctx, d, events


def step_all_mechanisms(w, report, *, llm=None, rounds=4):
    """Drive pending instances to their next transitions + route verified outputs."""
    op = cb.MechanismRuntimeOperator(report=report, llm=llm)
    router = gw.GeneratedSemanticEventOperator(report=report)
    deliver = gw.GeneratedObservationDeliveryOperator(report=report)
    invocations = []
    for _ in range(rounds):
        pending = [i for i in w.mechanism_instances.values() if i.status == "pending"]
        if not pending:
            break
        for inst in pending:
            ts = max(w.clock.now, inst.pending_transition_at)
            if ts > w.clock.now:
                w.clock.advance_to(ts)
            d, _ = op.run(w, Event(ts=ts, etype="ctrl_mechanism_step",
                                   payload={"instance_id": inst.instance_id}), None)
            for fu in d.follow_up_events:
                if fu["etype"] != "ctrl_semantic_event":
                    continue
                rd, _ = router.run(w, Event(ts=fu["ts"], etype=fu["etype"],
                                            participants=list(fu.get("participants") or []),
                                            payload=dict(fu.get("payload") or {})), None)
                for f2 in rd.follow_up_events:
                    if f2["etype"] == "ctrl_deliver_observation":
                        if f2["ts"] > w.clock.now:
                            w.clock.advance_to(f2["ts"])
                        dd, _ = deliver.run(
                            w, Event(ts=f2["ts"], etype=f2["etype"],
                                     participants=list(f2.get("participants") or []),
                                     payload=dict(f2.get("payload") or {})), None)
                        invocations.extend(x for x in dd.follow_up_events
                                           if x["etype"] == "ctrl_invoke_actor")
                    elif f2["etype"] == "ctrl_invoke_actor":
                        invocations.append(f2)
    return invocations


PIGEON_NOTE = "Meet me at the observatory at dusk — bring the ledger. — A. ✈️"


# ================================================================= 1. sending ≠ delivery
def test_1_sent_message_creates_outgoing_attempt_but_not_delivery():
    w = grant_world()
    report = gw.generated_report()
    ctx, d, events = run_ops(w, [
        {"op": "emit_semantic_event", "semantic_type_id": "colleague_note_drafted",
         "exact_content": PIGEON_NOTE, "direct_targets": ["prof_boyd"],
         "structured_fields": {"about": "the ledger"}}], report=report)
    attempt = w.semantic_log[0]
    assert attempt["causal_layer"] == "actor_controlled"
    assert attempt["observability_verified"] is False
    assert attempt["actual_recipients"] == []          # intent ≠ receipt
    assert report["intended_deliveries"] == 1 and report["actual_deliveries"] == 0
    # the channel mechanism took the attempt; delivery has NOT happened
    assert any(i.mechanism_id == "dept_email_channel"
               for i in w.mechanism_instances.values())
    assert all(i.status == "pending" for i in w.mechanism_instances.values())
    assert not any(x["semantic_type_id"] == "colleague_note_delivered"
                   for x in w.semantic_log)
    assert w.information.visible_to("prof_boyd") == []


# ================================================================= 2. no invocation pre-delivery
def test_2_recipient_not_invoked_before_actual_delivery():
    w = grant_world()
    report = gw.generated_report()
    run_ops(w, [{"op": "emit_semantic_event", "semantic_type_id": "colleague_note_drafted",
                 "exact_content": PIGEON_NOTE, "direct_targets": ["prof_boyd"]}],
            report=report)
    attempt = w.semantic_log[0]
    router = gw.GeneratedSemanticEventOperator(report=report)
    rd, _ = router.run(w, Event(ts=w.clock.now, etype="ctrl_semantic_event",
                                payload={"semantic_event": attempt}), None)
    assert not any(f["etype"] in ("ctrl_deliver_observation", "ctrl_invoke_actor")
                   for f in rd.follow_up_events)       # nobody moves on an undelivered note
    invocations = step_all_mechanisms(w, report)       # the channel actually delivers
    assert any(inv["payload"]["actor_id"] == "prof_boyd" for inv in invocations)
    assert report["actual_deliveries"] == 1


# ================================================================= 3. failed delivery
def test_3_failed_delivery_produces_no_recipient_observation():
    w = grant_world(mailbox="over_quota")              # THIS branch's hidden state
    report = gw.generated_report()
    run_ops(w, [{"op": "emit_semantic_event", "semantic_type_id": "colleague_note_drafted",
                 "exact_content": PIGEON_NOTE, "direct_targets": ["prof_boyd"]}],
            report=report)
    invocations = step_all_mechanisms(w, report)
    inst = next(iter(w.mechanism_instances.values()))
    assert inst.status == "failed" and inst.state == "bounced"
    assert report["mechanism_failures"] == 1 and report["actual_deliveries"] == 0
    assert w.information.visible_to("prof_boyd") == []
    assert invocations == []                           # no observation, no reaction
    assert not any(x["semantic_type_id"] == "colleague_note_delivered"
                   for x in w.semantic_log)
    # the attempt record carries the failure honestly
    hist_status = cb._completion_status_for_action(w, "a1", None)[0]
    assert hist_status == "mechanism_failed"


# ================================================================= 4. publication ≠ awareness
def test_4_publication_attempt_does_not_directly_create_awareness():
    w = grant_world(actors=("prof_ada", "prof_boyd", "prof_chen"))
    report = gw.generated_report()
    run_ops(w, [{"op": "emit_semantic_event",
                 "semantic_type_id": "results_announcement_posted",
                 "exact_content": "Consortium results are in.",
                 "intended_visibility": "public"}], report=report)
    assert report["intended_publications"] == 1 and report["actual_publications"] == 0
    for other in ("prof_boyd", "prof_chen"):
        assert w.information.visible_to(other) == []   # announced ≠ available ≠ aware
    step_all_mechanisms(w, report)                     # the press wire publishes
    assert report["actual_publications"] == 1
    seen = {a for a in ("prof_boyd", "prof_chen") if w.information.visible_to(a)}
    assert seen == {"prof_boyd", "prof_chen"}          # awareness ONLY after actual publication


# ================================================================= 5. submission ≠ acceptance
def test_5_submission_does_not_directly_create_institutional_acceptance():
    w = grant_world()
    report = gw.generated_report()
    # the applicant cannot write the registrar's intake log directly
    ctx, _, _ = run_ops(w, [{"op": "create_or_update_record",
                             "record_type": "intake_log_entry",
                             "fields": {"application": "consortium"}}], report=report)
    assert ctx["quarantined"] and "controlled by" in ctx["quarantined"][0]["reason"]
    assert report["directness_claims_rejected"] >= 1
    # the attempt enters the intake mechanism; the log entry appears only on its success
    run_ops(w, [{"op": "emit_semantic_event",
                 "semantic_type_id": "application_package_submitted",
                 "exact_content": "the consortium application",
                 "structured_fields": {"title": "consortium"}}], report=report)
    assert not any(o.object_type == "intake_log_entry" for o in w.objects.values())
    step_all_mechanisms(w, report)
    entry = [o for o in w.objects.values() if o.object_type == "intake_log_entry"]
    assert entry and entry[0].created_by == "registrar_office"   # the SYSTEM wrote it
    # and intake is still not ACCEPTANCE: the outcome predicate stays unsatisfied
    readout = gw.make_generated_predicate_readout(w.scenario_schema)
    assert readout(w) == "not_accepted"


# ================================================================= 6. invitation ≠ meeting
def test_6_proposed_meeting_does_not_directly_create_completed_meeting():
    w = grant_world()
    report = gw.generated_report()
    ctx, _, _ = run_ops(w, [
        {"op": "emit_semantic_event", "semantic_type_id": "meeting_invitation_issued",
         "exact_content": "Could we meet Thursday at 3?", "direct_targets": ["prof_boyd"],
         "structured_fields": {"when": "thursday_1500"}},
        {"op": "create_or_update_record", "record_type": "confirmed_meeting",
         "fields": {"with_whom": "prof_boyd"}}], report=report)
    # the invitation attempt stands; the CONFIRMED meeting is boyd's act, not ada's claim
    assert any(x["semantic_type_id"] == "meeting_invitation_issued"
               for x in w.semantic_log)
    assert ctx["quarantined"]
    assert "controlled by 'prof_boyd'" in ctx["quarantined"][0]["reason"]
    assert not any(o.object_type == "confirmed_meeting" for o in w.objects.values())


# ================================================================= 7. one signature ≠ agreement
def test_7_unilateral_signature_does_not_create_bilateral_agreement():
    w = grant_world()
    report = gw.generated_report()
    ctx, _, _ = run_ops(w, [
        {"op": "create_or_update_record", "record_type": "own_signature_record",
         "fields": {"document": "mou_draft"}},
        {"op": "create_or_update_record", "record_type": "bilateral_mou",
         "fields": {"parties": ["prof_ada", "prof_boyd"]}},
        {"op": "create_or_remove_relation", "relation": "consortium_partner_of",
         "src": "prof_ada", "dst": "prof_boyd"},
        {"op": "create_or_remove_relation", "relation": "follows_work_of",
         "src": "prof_ada", "dst": "prof_boyd"}], report=report)
    # her own signature: Layer A, allowed
    assert any(o.object_type == "own_signature_record" for o in w.objects.values())
    # the bilateral memorandum and the recognized partnership: rejected
    reasons = json.dumps([q["reason"] for q in ctx["quarantined"]])
    assert "joint_signature_process" in reasons
    assert "not declared unilateral" in reasons
    assert not any(o.object_type == "bilateral_mou" for o in w.objects.values())
    # the schema-declared UNILATERAL relation from herself: allowed
    edges = [(e.src, e.rel, e.dst) for e in w.network.edges]
    assert ("prof_ada", "follows_work_of", "prof_boyd") in edges
    assert not any(r == "consortium_partner_of" for _s, r, _d in edges)


# ================================================================= 8. scheduling ≠ occurrence
def test_8_scheduled_future_action_does_not_occur_at_scheduling_time():
    w = grant_world()
    report = gw.generated_report()
    ctx, d, events = run_ops(w, [
        {"op": "schedule_semantic_event", "semantic_type_id": "colleague_note_drafted",
         "exact_content": PIGEON_NOTE, "direct_targets": ["prof_boyd"],
         "delay_s": 7200.0}], report=report)
    assert w.semantic_log == []                        # nothing HAPPENED yet
    assert report["scheduled_attempts"] == 1
    sched = [e for e in events if e.etype == "ctrl_scheduled_attempt"]
    assert len(sched) == 1 and sched[0].ts == T0 + 7200.0
    # at fire time the attempt executes through the SAME boundary (mechanism handoff then)
    w.clock.advance_to(sched[0].ts)
    op = cb.ScheduledAttemptOperator(report=report)
    d2, vr = op.run(w, sched[0], None)
    assert vr.ok
    assert [x["semantic_type_id"] for x in w.semantic_log] == ["colleague_note_drafted"]
    assert w.semantic_log[0]["occurred_at"] == T0 + 7200.0
    assert any(i.mechanism_id == "dept_email_channel"
               for i in w.mechanism_instances.values())
    # scheduling a mechanism OUTPUT as a future fact is rejected outright
    ctx2, _, _ = run_ops(w, [{"op": "schedule_semantic_event",
                              "semantic_type_id": "colleague_note_delivered",
                              "exact_content": "x", "delay_s": 60.0}], report=report)
    assert ctx2["quarantined"] and "mechanism" in ctx2["quarantined"][0]["reason"]


# ================================================================= 9. novel mechanism, no handler
def test_9_novel_scenario_mechanism_executes_without_hardcoded_domain_handler():
    schema = ScenarioSemanticModel(
        question="Will the message reach the mountain observatory?",
        prediction_timestamp=T0, horizon=T0 + 7 * 86400,
        fact_types={"pigeon_flight_log": {"description": "registry log",
                                          "fields": {"leg": "str"}}},
        semantic_event_types={
            "pigeon_dispatched": {"description": "ATTEMPT", "fields": {},
                                  "typical_visibility": "participants"},
            "pigeon_arrival_witnessed": {"description": "registry outcome", "fields": {},
                                         "typical_visibility": "participants"}},
        mechanism_definitions={
            "notarized_carrier_pigeon_registry": {
                "description": "a notarized registry relaying messages by carrier pigeon",
                "triggering_event_types": ["pigeon_dispatched"],
                "accepted_inputs": {},
                "controlling_actor_or_system": "mountain_registry_notary",
                "state_machine": {"aloft": ["over_the_pass"],
                                  "over_the_pass": ["arrived"]},
                "initial_state": "aloft",
                "intermediate_states": ["over_the_pass"],
                "success_states": ["arrived"], "failure_states": ["lost_to_weather"],
                "unresolved_states": [],
                "possible_output_event_types": {
                    "on_success": ["pigeon_arrival_witnessed"], "on_failure": []},
                "possible_record_updates": ["pigeon_flight_log"],
                "observation_rules": {"recipients": "direct_targets"},
                "timing_rules": {"delay_s": 300.0},
                "assumptions": ["clear weather on this branch"],
                "uncertainty_source": "mountain weather"}},
        outcome_predicates=[{"predicate_id": "reached", "record_type": "pigeon_flight_log",
                             "op": "exists", "option_true": "reached",
                             "option_false": "not_reached"}],
        provenance={"compiler": "test"}).freeze()
    w = grant_world(schema=schema, actors=("sender_sam", "watcher_wen"))
    report = gw.generated_report()
    run_ops(w, [{"op": "emit_semantic_event", "semantic_type_id": "pigeon_dispatched",
                 "exact_content": PIGEON_NOTE, "direct_targets": ["watcher_wen"]}],
            actor="sender_sam", report=report)
    step_all_mechanisms(w, report)                     # two deterministic legs, no handler
    inst = next(iter(w.mechanism_instances.values()))
    assert inst.status == "succeeded" and inst.state == "arrived"
    assert [t["to"] for t in inst.transitions] == ["over_the_pass", "arrived"]
    assert any(x["semantic_type_id"] == "pigeon_arrival_witnessed"
               for x in w.semantic_log)
    delivered = next(x for x in w.semantic_log
                     if x["semantic_type_id"] == "pigeon_arrival_witnessed")
    assert delivered["exact_content"] == PIGEON_NOTE   # content preserved through the flight
    assert report["mechanism_successes"] == 1


# ================================================================= 10. unresolved stays unresolved
def test_10_unresolved_mechanism_remains_unresolved_never_succeeds():
    schema = grant_schema()
    # a mechanism whose branching cannot be decided: two candidates, no executable rule,
    # and no adjudication backend in this run
    schema.mechanism_definitions["dept_email_channel"] = dict(
        schema.mechanism_definitions["dept_email_channel"],
        transition_rules=[])
    w = grant_world(schema=schema)
    report = gw.generated_report()
    run_ops(w, [{"op": "emit_semantic_event", "semantic_type_id": "colleague_note_drafted",
                 "exact_content": PIGEON_NOTE, "direct_targets": ["prof_boyd"]}],
            report=report)
    step_all_mechanisms(w, report, llm=None)
    inst = next(iter(w.mechanism_instances.values()))
    assert inst.status == "unresolved"                 # NOT succeeded, NOT assumed
    assert report["mechanism_unresolved"] == 1 and report["mechanism_successes"] == 0
    assert w.information.visible_to("prof_boyd") == []
    assert cb._completion_status_for_action(w, "a1", None)[0] == "mechanism_unresolved"


# ================================================================= 11. no fixed-v1 degradation
def test_11_generated_mode_does_not_degrade_to_fixed_v1_when_schema_missing():
    w = WorldState("nofb", "b0", SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    for name in ("alice", "bob"):
        e = Entity(name)
        e.set("roles", F(["person"], status="observed"))
        e.set("past_actions", F([], status="observed"))
        w.entities[name] = e
    runtime = ActorPolicyRuntime()                     # resolved default: generated
    assert runtime.consequence_mode == "generated_actor_mediated_world"
    action = TypedAction(action_id="t1", actor_id="alice", actor_role="person",
                         action_family="messaging", action_name="reply_now",
                         target=ActionTarget("actor", "bob"),
                         parameters={"content": PIGEON_NOTE},
                         mechanisms_triggered=["message_delivery"])
    _sel, posterior, trace = runtime.decide(
        None, [w], "alice", seed=1,
        decision={"candidate_actions": [{"name": "reply_now", "family": "messaging",
                                         "target": {"target_type": "person",
                                                    "target_id": "bob"}}]})
    delta, events = runtime.execute(w, action, posterior, trace, seed=1)
    rep = runtime.consequence_report
    assert rep["actual_mode"] == "generated_actor_mediated_world"   # never swapped
    assert rep["structurally_under_modeled"] is True
    assert rep["fixed_ontology_uses"] == 0 and rep["numeric_fallbacks"] == 0
    assert "execution_incomplete:no_scenario_schema" in delta.reason_codes
    assert delta.uncertainty["unexecuted_attempt"]["content"] == PIGEON_NOTE
    assert w.objects == {}                             # no fixed-catalog object appeared
    assert not any(e.etype == "message_delivered" for e in events)
    hist = w.entity("alice").value("past_actions")[-1]
    assert hist["completion_status"] == "execution_incomplete"


# ================================================================= 12. phase 13, same path
def test_12_phase13_decision_actions_use_the_same_causal_mechanism_path():
    from swm.world_model_v2.phase13.interventions import DecisionActionOperator, to_intervention
    from swm.world_model_v2.phase13.ontology import ActionSchema
    w = grant_world()
    report_seed = gw.generated_report()

    def compile_backend(prompt):
        if "ACTION-ATTEMPT COMPILER" in prompt:
            return json.dumps([{"op": "emit_semantic_event",
                                "semantic_type_id": "colleague_note_drafted",
                                "exact_content": PIGEON_NOTE,
                                "direct_targets": ["prof_boyd"]}])
        if "CAUSAL DIRECTNESS CRITIC" in prompt:
            return "[]"
        raise AssertionError(f"unexpected prompt: {prompt[:60]}")

    op = DecisionActionOperator(llm=compile_backend)
    action = ActionSchema(action_id="p13a", actor="prof_ada", operation="communicate",
                          object="prof_boyd", recipients=["prof_boyd"],
                          observability="private", content={"text": PIGEON_NOTE})
    q = EventQueue(horizon_ts=T0 + 7 * 86400)
    to_intervention(action).apply(w, q)
    ev = q.next_event()
    assert ev.etype == "decision_action"
    delta, vr = op.run(w, ev, None)
    assert vr.ok and "phase13_causal_boundary" in delta.reason_codes
    # SAME path: an attempt + a mechanism instance — never message_delivered, never a
    # direct information exposure, never an immediate external effect
    assert not any(f["etype"] == "message_delivered" for f in delta.follow_up_events)
    assert any(x["semantic_type_id"] == "colleague_note_drafted" for x in w.semantic_log)
    assert any(i.mechanism_id == "dept_email_channel"
               for i in w.mechanism_instances.values())
    assert w.information.visible_to("prof_boyd") == []
    assert op.report["action_attempts"] == 1
    assert op.report["intended_deliveries"] == 1 and op.report["actual_deliveries"] == 0
    # the report seeded above stays untouched — the operator's own report carries the truth
    assert report_seed["action_attempts"] == 0


# ================================================================= 13. PR #115 content survives
def test_13_message_content_survives_byte_exact_through_attempt_and_mechanism():
    winner_text = ("Subject: the ledger\n\nDear Prof. Boyd — I'll be brief. " + PIGEON_NOTE
                   + "\n\n— Ada (sent 19:02, “no edits”)")
    w = grant_world()
    report = gw.generated_report()
    run_ops(w, [{"op": "emit_semantic_event", "semantic_type_id": "colleague_note_drafted",
                 "exact_content": winner_text, "direct_targets": ["prof_boyd"]}],
            report=report)
    attempt = w.semantic_log[0]
    assert attempt["exact_content"] == winner_text     # byte-exact on the attempt
    inst = next(iter(w.mechanism_instances.values()))
    assert inst.inputs["exact_content"] == winner_text  # byte-exact into the mechanism
    step_all_mechanisms(w, report)
    delivered = next(x for x in w.semantic_log
                     if x["semantic_type_id"] == "colleague_note_delivered")
    assert delivered["exact_content"] == winner_text   # byte-exact out of the mechanism
    items = w.information.visible_to("prof_boyd")
    assert items and items[0][0].content == winner_text  # byte-exact into the recipient's view


# ================================================================= 14. observability drives invocation
def test_14_actual_observability_not_intended_visibility_controls_invocation():
    w = grant_world(actors=("prof_ada", "prof_boyd", "prof_chen"))
    report = gw.generated_report()
    # intended PUBLIC visibility on the attempt — still reaches nobody by itself
    run_ops(w, [{"op": "emit_semantic_event", "semantic_type_id": "colleague_note_drafted",
                 "exact_content": "please advise", "direct_targets": ["prof_boyd"],
                 "intended_visibility": "public"}], report=report)
    attempt = w.semantic_log[0]
    router = gw.GeneratedSemanticEventOperator(report=report)
    rd, _ = router.run(w, Event(ts=w.clock.now, etype="ctrl_semantic_event",
                                payload={"semantic_event": attempt}), None)
    assert rd.follow_up_events == []                   # intended publicity invoked NOBODY
    invocations = step_all_mechanisms(w, report)
    invoked = {inv["payload"]["actor_id"] for inv in invocations}
    assert invoked == {"prof_boyd"}                    # only the ACTUAL recipient
    # the channel verified participants-only observability: no public availability appeared
    delivered = next(x for x in w.semantic_log
                     if x["semantic_type_id"] == "colleague_note_delivered")
    assert delivered["availability"] != "public"
    assert w.information.visible_to("prof_chen") == []


# ================================================================= 15. default-on public route
def test_15_ordinary_run_route_uses_boundary_compiler_and_mechanism_runtime_by_default():
    # (a) the resolved default mode IS the generated causal-boundary architecture
    assert sc.resolve_consequence_mode() == "generated_actor_mediated_world"
    # (b) the plan→operators funnel used by simulate_world (run_with_persistence AND
    #     run_from_plan) wires the mechanism runtime + scheduled-attempt runtime by default
    from swm.world_model_v2.materialize import operators_from_plan

    class _P:
        accepted_mechanisms = [{"mech_id": "m1", "operator": "production_actor_policy"}]

    def fake_llm(prompt):
        if "ACTION-ATTEMPT COMPILER" in prompt:
            return "[]"
        if "CAUSAL DIRECTNESS CRITIC" in prompt:
            return "[]"
        return json.dumps({"decision": {"act_or_wait": "wait", "chosen_action": "wait"}})

    ops, _rej = operators_from_plan(_P(), llm=fake_llm)
    names = [getattr(op, "name", "") for op in ops]
    assert "scenario_mechanism_runtime" in names
    assert "scheduled_attempt_runtime" in names
    assert "generated_semantic_event_router" in names
    assert "generated_observation_delivery" in names
    # (c) an end-to-end micro-rollout through the CANONICAL RolloutEngine queue on the
    #     default runtime: decision → attempt → mechanism → delivery → recipient invocation
    boyd_prompts = []

    def backend(prompt):
        if "ACTION-ATTEMPT COMPILER" in prompt:
            actor = prompt.split("ACTOR:")[1].split("\n")[0].strip()
            if actor == "prof_ada":
                return json.dumps([{"op": "emit_semantic_event",
                                    "semantic_type_id": "colleague_note_drafted",
                                    "exact_content": PIGEON_NOTE,
                                    "direct_targets": ["prof_boyd"]}])
            return "[]"
        if "CAUSAL DIRECTNESS CRITIC" in prompt:
            return "[]"
        if "You ARE prof_boyd" in prompt:
            boyd_prompts.append(prompt)
        who = "prof_ada" if "You ARE prof_ada" in prompt else "prof_boyd"
        chosen = "send_note" if who == "prof_ada" and not w.semantic_log else "wait"
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "decision": {"act_or_wait": "act" if chosen != "wait" else "wait",
                         "chosen_action": chosen, "target": "prof_boyd",
                         "timing": "immediate", "observability": "private",
                         "intended_effect": "reach boyd"},
            "situation_interpretation": {"what_changed": "x", "why_it_matters": "y"},
            "actor_state_update": {}, "anticipated_reactions": [],
            "novel_action_proposal": {"present": chosen == "send_note"},
            "alternatives_considered": [], "decision_summary": f"I {chosen}"})

    from swm.world_model_v2.qualitative_actor import (
        QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
    )
    w = grant_world()
    rt = QualitativeActorPolicyRuntime(
        QualitativeDecisionEngine(QualitativeConfig(llm=backend, llm_hypotheses=False,
                                                    n_hypotheses=2)),
        mode="persistent_qualitative_llm_policy",
        consequence_mode="generated_actor_mediated_world")
    report = rt.consequence_report
    ops = [ProductionActorPolicyOperator(runtime=rt),
           gw.GeneratedSemanticEventOperator(report=report),
           gw.GeneratedObservationDeliveryOperator(report=report),
           gw.GeneratedActorInvocationOperator(rt, report=report),
           cb.MechanismRuntimeOperator(report=report),
           cb.ScheduledAttemptOperator(report=report)]
    q = EventQueue(horizon_ts=T0 + 10 * 86400)
    q.schedule(Event(ts=T0 + 60, etype="decision_opportunity", participants=["prof_ada"],
                     payload={"situation": "you need boyd's counsel"}))
    RolloutEngine(operators=ops).run_branch(w, q, seed=7)
    log = [x["semantic_type_id"] for x in w.semantic_log]
    assert log == ["colleague_note_drafted", "colleague_note_delivered"]
    assert report["action_attempts"] >= 1
    assert report["mechanisms_invoked"] == 1 and report["mechanism_successes"] == 1
    assert report["intended_deliveries"] == 1 and report["actual_deliveries"] == 1
    assert boyd_prompts and PIGEON_NOTE[:30] in boyd_prompts[0]   # boyd saw the REAL content
    # §18 pure-run contract
    assert report["human_reactions_written_directly"] == 0
    assert report["external_successes_written_directly"] == 0
    assert report["fixed_ontology_uses"] == 0
    assert report["numeric_fallbacks"] == 0
    assert report["causal_action_reports"]
    car = report["causal_action_reports"][0]
    assert car["completion_status"] in ("mechanism_pending", "mechanism_succeeded")
    assert car["exact_content"]


# ================================================================= settlement (§11 transfers)
def test_transfer_with_settlement_mechanism_is_escrowed_not_credited():
    w = grant_world()
    report = gw.generated_report()
    ctx, _, _ = run_ops(w, [{"op": "transfer_conserved_quantity", "resource": "grant_funds",
                             "amount": 500.0, "to": "prof_boyd"}], report=report)
    assert ctx["quarantined"] and "settle" in ctx["quarantined"][0]["reason"]
    assert w.entities["prof_boyd"].value("resources", key="grant_funds") == 10_000.0
    # the honest route: invoke the settlement mechanism — escrow now, credit on settlement
    ctx2, d2, _ = run_ops(w, [{"op": "invoke_scenario_mechanism",
                               "mechanism_id": "university_finance_settlement",
                               "exact_payload": {"resource": "grant_funds", "amount": 500.0,
                                                 "to": "prof_boyd"}}], report=report)
    assert not ctx2["quarantined"]
    assert w.entities["prof_ada"].value("resources", key="grant_funds") == 9_500.0
    assert w.entities["prof_boyd"].value("resources", key="grant_funds") == 10_000.0
    step_all_mechanisms(w, report)
    assert w.entities["prof_boyd"].value("resources", key="grant_funds") == 10_500.0


# ================================================================= mechanism outputs are gated
def test_action_cannot_emit_a_mechanism_output_event():
    w = grant_world()
    report = gw.generated_report()
    ctx, _, _ = run_ops(w, [{"op": "emit_semantic_event",
                             "semantic_type_id": "colleague_note_delivered",
                             "exact_content": "totally delivered, trust me",
                             "direct_targets": ["prof_boyd"]}], report=report)
    assert ctx["quarantined"]
    assert "produced by mechanism" in ctx["quarantined"][0]["reason"]
    assert report["directness_claims_rejected"] >= 1
    assert w.semantic_log == []


# ================================================================= directness validator converts
def test_directness_validator_converts_output_claims_into_mechanism_invocations():
    w = grant_world()
    report = gw.generated_report()
    compiler = cb.CausalActionCompiler(
        llm=lambda p: json.dumps([{"op": "emit_semantic_event",
                                   "semantic_type_id": "colleague_note_delivered",
                                   "exact_content": PIGEON_NOTE,
                                   "direct_targets": ["prof_boyd"]}]))
    action = TypedAction(action_id="c1", actor_id="prof_ada", actor_role="applicant",
                         action_family="messaging", action_name="send_note",
                         target=ActionTarget("actor", "prof_boyd"),
                         mechanisms_triggered=["semantic_consequences"])
    program = compiler.compile(w, action, qualitative={"decision_summary": "I email boyd"},
                               report=report)
    cb.DirectnessValidator(llm=None).validate(w, program, report=report)
    assert program.rejected_claims
    assert program.rejected_claims[0]["test"] == "external_acceptance"
    assert any(op.get("mechanism_id") == "dept_email_channel"
               for op in program.mechanism_invocations)
    assert report["directness_claims_rejected"] >= 1
    # executing the converted program starts the channel, not the delivery
    ctx = ctx_for(w, report=report)
    d = StateDelta(at=w.clock.now, event_type="actor_action", operator="test")
    gw.execute_kernel_ops(w, program.kernel_ops(), ctx, d)
    assert any(i.mechanism_id == "dept_email_channel"
               for i in w.mechanism_instances.values())
    assert not any(x["semantic_type_id"] == "colleague_note_delivered"
                   for x in w.semantic_log)


# ================================================================= mechanism generation traced
def test_mechanism_compiler_uses_separate_proposal_and_critic_calls_with_cache():
    calls = []

    def llm(prompt):
        calls.append(prompt[:40])
        if "CAUSAL MECHANISM COMPILER" in prompt:
            return json.dumps({"mechanism_definitions": {
                "village_bulletin_board": {
                    "description": "the physical board where notices become visible",
                    "triggering_event_types": ["notice_pinned"],
                    "controlling_actor_or_system": "village_hall",
                    "state_machine": {"pinned_face_down": ["visible"]},
                    "initial_state": "pinned_face_down",
                    "success_states": ["visible"], "failure_states": ["blown_away"],
                    "possible_output_event_types": {"on_success": ["notice_readable"]},
                    "observation_rules": {"recipients": "direct_targets",
                                          "availability": "public"},
                    "timing_rules": {"delay_s": 60},
                    "assumptions": ["board exists"],
                    "uncertainty_source": "weather"}},
                "new_semantic_event_types": {
                    "notice_readable": {"description": "board outcome", "fields": {},
                                        "typical_visibility": "public"}}})
        if "CAUSAL-BOUNDARY CRITIC" in prompt:
            return json.dumps({"missing_mechanisms": [], "verdict": "usable"})
        raise AssertionError("unexpected prompt")

    model = ScenarioSemanticModel(
        question="Will the villagers see the notice?", prediction_timestamp=T0,
        horizon=T0 + 86400,
        semantic_event_types={"notice_pinned": {"description": "ATTEMPT", "fields": {},
                                                "typical_visibility": "public"}},
        fact_types={"village_meeting_record": {"description": "x", "fields": {}}},
        outcome_predicates=[{"predicate_id": "seen", "record_type":
                             "village_meeting_record", "op": "exists",
                             "option_true": "seen", "option_false": "unseen"}],
        provenance={"compiler": "test"})
    pristine = model.as_dict()                             # identical semantic inputs
    trace = cb.MechanismCompiler(llm).attach(model)
    assert model.mechanism_definitions.get("village_bulletin_board")
    assert "notice_readable" in model.semantic_event_types
    roles = [c["role"] for c in trace["calls"]]
    assert roles[:2] == ["proposal", "boundary_critic"]     # two SEPARATE calls
    assert trace["call_count"] == 2 and trace["accepted"] == 1
    # content-addressed cache: identical semantic inputs → a reused actual LLM result
    model2 = ScenarioSemanticModel.from_dict(pristine)
    n_before = len(calls)
    trace2 = cb.MechanismCompiler(llm).attach(model2)
    assert trace2["cache_hit"] is True and len(calls) == n_before
    assert model2.mechanism_definitions.get("village_bulletin_board")


def test_no_llm_backend_leaves_structural_gap_never_invented_semantics():
    model = ScenarioSemanticModel(
        question="q", prediction_timestamp=T0, horizon=T0 + 86400,
        fact_types={"r": {"description": "x", "fields": {}}},
        outcome_predicates=[{"predicate_id": "p", "record_type": "r", "op": "exists",
                             "option_true": "t", "option_false": "f"}],
        provenance={"compiler": "test"})
    cb.MechanismCompiler(None).attach(model)
    assert model.mechanism_definitions == {}               # nothing invented
    assert model.provenance["mechanism_model_error"] == "no_llm_backend"
    assert any("structurally under-modeled" in u for u in model.unresolved_mechanisms)
