"""Phase 4 action, observability, feasibility, policy, and execution acceptance tests."""
import copy
import random
from types import SimpleNamespace

import pytest

from swm.world_model_v2.events import Event
from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.institutions import Rule, RuleSystem
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import (
    ActorPolicyRuntime, ProductionActorPolicyOperator, apply_adaptation,
    decide_and_execute_particles,
)
from swm.world_model_v2.phase4_policy import (
    ACTION_ONTOLOGY, ActionSpaceBuilder, ActionTarget, ActorPolicyModel, ActorViewBuilder,
    FeasibilityEngine, SCHEMA_VERSION, TemperatureCalibrator, TypedAction,
    phase6_policy_registry_records,
)
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

T0 = 1_700_000_000.0


class Plan:
    question = "Will the organization approve the proposal?"

    @staticmethod
    def plan_hash():
        return "planphase4"


def world():
    w = WorldState("p4", "b0", SimulationClock(T0, T0), network=RelationGraph(),
                   information=InformationLedger())
    alice = Entity("alice")
    alice.set("roles", F(["manager"], status="observed", sources=["org"]));
    alice.set("goals", F(["complete_project"], status="observed", sources=["brief"]));
    alice.set("preferences", F({"quality": 0.8}, status="inferred"))
    alice.set("beliefs", F(0.7, status="inferred"), key="proposal_succeeds")
    alice.set("resources", F(10.0, status="observed"), key="budget")
    alice.set("authority", F(["approve"], status="observed"))
    alice.set("commitments", F([], status="observed"))
    alice.set("attention", F(0.8, status="inferred"))
    alice.set("private_information", F("reservation=3", status="observed"))
    alice.set("latent_state", F("omniscient_truth", status="sampled"), key="hidden_truth")
    alice.set("past_actions", F([], status="observed"))
    bob = Entity("bob")
    bob.set("roles", F(["analyst"], status="observed"))
    bob.set("private_information", F("private_belief", status="observed"))
    bob.set("beliefs", F(0.1, status="inferred"), key="private_belief")
    bob.set("resources", F(3.0, status="observed"), key="budget")
    w.entities = {"alice": alice, "bob": bob}
    w.network.add("alice", "communicates_with", "bob", visibility="private")
    w.information.publish(InformationItem("public1", "proposal filed", source="registry", created_at=T0 - 5))
    w.information.publish(InformationItem("private1", "bob opposes", kind="private", source="bob", created_at=T0 - 3))
    w.information.publish(InformationItem("future1", "proposal resolved", source="future", created_at=T0 + 50))
    w.information.expose("alice", "public1", T0 - 4)
    w.information.expose("bob", "private1", T0 - 2)
    w.information.expose("alice", "future1", T0 + 50)
    w.institutions["board"] = RuleSystem("board", [
        Rule("approve_right", "decision_right", {"actions": ["approve"], "holders": ["alice"]}),
        Rule("budget", "budget", {"actions": ["purchase"], "available": 5.0}),
    ])
    return w


def action(name="approve", **kw):
    defaults = dict(
        action_id=f"a:{name}", actor_id="alice", actor_role="manager",
        action_family="institutional" if name in ACTION_ONTOLOGY["institutional"] else "generic",
        action_name=name, target=ActionTarget("institution", "board"),
        mechanisms_triggered=["institution_processing", "reaction_scheduling"],
    )
    defaults.update(kw)
    return TypedAction(**defaults)


def test_typed_action_round_trip_and_extension_requires_mechanism():
    a = action(parameters={"matter": "m1"}, support_status="fitted")
    assert TypedAction.from_dict(a.as_dict()) == a
    with pytest.raises(ValueError):
        TypedAction("x", "alice", "manager", "generic", "novel_action")
    novel = TypedAction("x", "alice", "manager", "generic", "novel_action",
                        mechanisms_triggered=["record_action"])
    assert novel.action_name == "novel_action"


def test_typed_action_legacy_migration_is_semantic_only():
    migrated = TypedAction.from_dict({
        "actor": "alice", "role": "manager", "family": "institutional", "name": "approve",
        "target": "board", "mechanisms_triggered": ["institution_processing"],
    })
    assert migrated.semantic_version == SCHEMA_VERSION
    assert migrated.actor_id == "alice" and migrated.target.target_id == "board"
    assert migrated.action_id.startswith("action:")
    with pytest.raises(ValueError, match="behavioral numeric fields"):
        TypedAction.from_dict({**migrated.as_dict(), "probability": 0.99})


def test_policy_families_export_through_phase6_governed_records():
    records = phase6_policy_registry_records()
    assert len(records) == 22 and all(r.status == "implemented" for r in records)
    assert all(r.executable() and r.packs and r.promotion_blockers("production_eligible") for r in records)


def test_actor_view_excludes_private_future_hidden_and_other_actor_state():
    v = ActorViewBuilder().build(world(), "alice")
    assert v.observed_evidence_ids == ["public1"]
    assert "private1" not in v.observed_evidence_ids and "future1" not in v.observed_evidence_ids
    rendered = str(v.as_dict())
    assert "reservation=3" not in rendered
    assert "omniscient_truth" not in rendered
    assert "private_belief" not in rendered
    assert "private_information" in v.hidden_fields_excluded


def test_phase2_visibility_bridge_populates_only_permitted_actor_ledger():
    from swm.world_model_v2.evidence_materialize import EvidenceObservationOperator
    w = world(); op = EvidenceObservationOperator()
    ev = SimpleNamespace(etype="observe_evidence", payload={
        "claim_id": "claim:sealed", "subject": "proposal", "value": "has defect",
        "source": "audit", "visibility": "private_actor", "actors": ["alice"],
    })
    delta, vr = op.run(w, ev, random.Random(1))
    assert vr.ok and any(c["path"].startswith("information.exposures") for c in delta.changes)
    assert "claim:sealed" in ActorViewBuilder().build(w, "alice").observed_evidence_ids
    assert "claim:sealed" not in ActorViewBuilder().build(w, "bob").observed_evidence_ids


@pytest.mark.parametrize("payload", [
    {"ts": T0 + 1, "visibility": "public", "resolution_outcome": True},
    {"ts": T0 - 1, "visibility": "private", "participants": ["bob"], "content": "secret"},
    {"ts": T0 - 1, "visibility": "public", "posterior_truth": 0.99},
])
def test_adversarial_event_leakage_is_blocked(payload):
    v = ActorViewBuilder().build(world(), "alice", observed_events=[payload])
    assert "secret" not in str(v.as_dict())
    assert "resolution_outcome" not in str(v.observed_events)
    assert "posterior_truth" not in str(v.observed_events)


def test_action_space_uses_structured_candidates_not_question_keywords():
    w = world(); view = ActorViewBuilder().build(w, "alice")
    decision = {"candidate_actions": [
        {"name": "approve", "family": "institutional", "target": {"target_type": "institution", "target_id": "board"},
         "mechanisms_triggered": ["institution_processing"]},
        {"name": "defer", "family": "institutional", "target": {"target_type": "institution", "target_id": "board"},
         "mechanisms_triggered": ["institution_processing"]},
    ]}
    a = ActionSpaceBuilder().build(Plan(), w, view, decision=decision)
    class OtherPlan(Plan):
        question = "Completely unrelated words that share no domain keywords"
    b = ActionSpaceBuilder().build(OtherPlan(), w, view, decision=decision)
    assert [x.action_name for x in a] == [x.action_name for x in b]


def test_action_space_cold_start_never_abstains_system_forecast():
    w = world(); w.institutions = {}; view = ActorViewBuilder().build(w, "alice")
    actions = ActionSpaceBuilder().build(Plan(), w, view, decision={})
    assert {a.action_name for a in actions} == {"wait", "abstain"}
    assert all(a.support_status == "tier_7_broad_prior" for a in actions)


def test_feasibility_distinguishes_perceived_and_actual_and_masks_known_impossible():
    w = world(); v = ActorViewBuilder().build(w, "alice"); eng = FeasibilityEngine()
    impossible = action("approve", authority_requirements=["veto"])
    d = eng.classify(impossible, v, w)
    assert d.perceived_status == "outside_authority" and not d.perceived_feasible
    costly = action("approve", resource_requirements={"budget": 11})
    d2 = eng.classify(costly, v, w)
    assert d2.perceived_status == "unaffordable"
    mistaken = action("approve", target=ActionTarget("institution", "missing"))
    d3 = eng.classify(mistaken, v, w)
    assert d3.perceived_feasible and d3.actual_status == "physically_impossible"


def test_policy_zero_mass_on_masked_action_and_represents_family_uncertainty():
    w = world(); v = ActorViewBuilder().build(w, "alice"); eng = FeasibilityEngine()
    good = action("approve")
    bad = action("veto", authority_requirements=["veto"])
    decisions = [[eng.classify(a, v, w) for a in (good, bad)]]
    p = ActorPolicyModel().decide([v], [good, bad], decisions, seed=3)
    assert set(p.action_probabilities) == {good.action_id}
    assert p.action_probabilities[good.action_id] == 1.0
    assert len(p.policy_family_posterior.weights) > 1
    assert not p.provenance["llm_probability_minting"]


def test_world_particles_change_policy_probabilities():
    w1 = world(); w2 = copy.deepcopy(w1); w2.branch_id = "b1"
    w1.entities["alice"].set("past_actions", F([{"action": "approve"}] * 6))
    w2.entities["alice"].set("past_actions", F([{"action": "defer"}] * 6))
    runtime = ActorPolicyRuntime(ActorPolicyModel({
        **ActorPolicyModel._broad_pack(),
        "policy_family_weights": {"habit": 1.0}, "habit_strength": 1.0,
    }))
    decision = {"candidate_actions": [
        {"name": "approve", "family": "institutional", "target": {"target_type": "institution", "target_id": "board"},
         "mechanisms_triggered": ["institution_processing"]},
        {"name": "defer", "family": "institutional", "target": {"target_type": "institution", "target_id": "board"},
         "mechanisms_triggered": ["institution_processing"]},
    ]}
    _, p1, _ = runtime.decide(Plan(), [w1], "alice", decision=decision, seed=1)
    _, pm, _ = runtime.decide(Plan(), [w1, w2], "alice", decision=decision, seed=1)
    assert p1.action_probabilities != pm.action_probabilities


def test_temperature_calibration_normalizes():
    c = TemperatureCalibrator(2.0, fitted_on="calibration_only")
    out = c.apply({"a": 0.9, "b": 0.1})
    assert abs(sum(out.values()) - 1) < 1e-12 and 0.5 < out["a"] < 0.9


def test_deterministic_replay_and_trace_corruption_detection():
    w = world(); runtime = ActorPolicyRuntime()
    decision = {"candidate_actions": [{"name": "approve", "family": "institutional",
                                        "target": {"target_type": "institution", "target_id": "board"},
                                        "mechanisms_triggered": ["institution_processing"]}]}
    a1, p1, t1 = runtime.decide(Plan(), [w], "alice", decision=decision, seed=17)
    a2, p2, t2 = runtime.decide(Plan(), [copy.deepcopy(w)], "alice", decision=decision, seed=17)
    assert a1.action_id == a2.action_id and p1.action_probabilities == p2.action_probabilities
    assert t1.trace_id == t2.trace_id and t1.verify()
    t1.sampled_action_id = "corrupt"
    assert not t1.verify()


def test_execution_consumes_resources_creates_commitment_events_and_delta():
    # the fixed-v1 BASELINE (quantity projection of a decided submission is its machinery);
    # schemaless generated mode is execution-incomplete by design and projects nothing
    w = world(); runtime = ActorPolicyRuntime(
        consequence_mode="fixed_semantic_consequence_policy_v1")
    decision = {"candidate_actions": [{
        "name": "approve", "family": "institutional",
        "target": {"target_type": "institution", "target_id": "board"},
        "resource_requirements": {"budget": 2}, "resource_costs": {"budget": 2},
        "commitments_created": [{"id": "deliver", "binding": True}],
        "possible_consequences": [{"kind": "quantity_delta", "name": "approval_progress", "delta": 1}],
        "mechanisms_triggered": ["institution_processing", "reaction_scheduling"],
    }]}
    result = decide_and_execute_particles(runtime, Plan(), [w], "alice", decision=decision, seed=5)
    delta = result["executions"][0]["delta"]
    assert delta.event_type == "actor_action" and delta.changes
    assert w.entity("alice").value("resources", key="budget") == 8.0
    assert w.entity("alice").value("commitments")[0]["id"] == "deliver"
    assert w.quantities["approval_progress"].value == 1.0
    assert {e.etype for e in result["executions"][0]["events"]} >= {"actor_action", "institution_submission"}
    assert delta.as_dict()["follow_up_events"]


def test_attempted_but_actually_invalid_action_is_explicit_delta():
    w = world(); runtime = ActorPolicyRuntime()
    decision = {"candidate_actions": [{
        "name": "approve", "family": "institutional",
        "target": {"target_type": "institution", "target_id": "missing"},
        "mechanisms_triggered": ["institution_processing"],
    }]}
    selected, posterior, trace = runtime.decide(Plan(), [w], "alice", decision=decision, seed=2)
    delta, events = runtime.execute(w, selected, posterior, trace, seed=2)
    assert delta.event_type == "action_blocked"
    assert "attempted_but_blocked" in delta.reason_codes
    assert events[0].etype == "action_blocked"


def test_reaction_chain_uses_same_policy_operator():
    class CapturingPolicy(ActorPolicyModel):
        seen = None

        def decide(self, views, actions, feasibility, *, seed=0):
            self.seen = views[0].observed_events
            return super().decide(views, actions, feasibility, seed=seed)

    w = world(); model = CapturingPolicy(); op = ProductionActorPolicyOperator(model)
    ev = Event(T0, etype="actor_reaction", participants=["bob", "alice"],
               payload={"candidate_actions": ["acknowledge", "ignore"]})
    delta, vr = op.run(w, ev, random.Random(4))
    assert vr.ok and delta.event_type == "actor_action"
    assert w.entity("bob").value("current_action")["action_name"] in ("acknowledge", "ignore")
    assert any(row.get("etype") == "actor_reaction" for row in model.seen)


def test_adaptation_and_history_change_later_probabilities():
    w = world()
    apply_adaptation(w, actor_id="alice", action_name="approve", reward=1.0,
                     outcome="success", source_event_id="out1", learning_rate=1.0)
    pack = {**ActorPolicyModel._broad_pack(),
            "policy_family_weights": {"reinforcement_learning": 1.0}}
    runtime = ActorPolicyRuntime(ActorPolicyModel(pack))
    decision = {"candidate_actions": [
        {"name": "approve", "family": "institutional", "target": {"target_type": "institution", "target_id": "board"},
         "mechanisms_triggered": ["institution_processing"]},
        {"name": "defer", "family": "institutional", "target": {"target_type": "institution", "target_id": "board"},
         "mechanisms_triggered": ["institution_processing"]},
    ]}
    _, learned, _ = runtime.decide(Plan(), [w], "alice", decision=decision, seed=1)
    clean = world()
    _, forgotten, _ = runtime.decide(Plan(), [clean], "alice", decision=decision, seed=1)
    approve_id = next(k for k in learned.action_probabilities if k in forgotten.action_probabilities and
                      learned.action_probabilities[k] != forgotten.action_probabilities[k])
    assert learned.action_probabilities[approve_id] > forgotten.action_probabilities[approve_id]
