"""Phase 4 natural-language compiler to shared-world end-to-end tests."""
import json

from swm.world_model_v2.compiler import compile_world
from swm.world_model_v2.materialize import run_from_plan


def compiled_payload():
    return {
        "coherent": True,
        "outcome": {"family": "binary", "options": ["True", "False"],
                    "resolution_rule": "approval progress reaches one", "readout_var": "approval_progress"},
        "entities": [
            {"id": "manager", "type": "person", "fields": {
                "roles": ["manager"], "goals": ["approve_project"], "authority": ["approve"],
                "resources": {"budget": 10}, "commitments": [], "beliefs": {"success": 0.7},
                "past_actions": [], "attention": 0.8}},
            {"id": "analyst", "type": "person", "fields": {"roles": ["analyst"], "past_actions": []}},
        ],
        "institutions": [{"id": "board", "rules": [
            {"kind": "decision_right", "params": {"actions": ["approve"], "holders": ["manager"]}},
        ]}],
        "relations": [{"src": "manager", "rel": "communicates_with", "dst": "analyst"}],
        "quantities": [{"name": "approval_progress", "qtype": "approval_progress", "value": 0}],
        "actor_decisions": [{
            "actor": "manager", "role": "manager", "at": "2025-01-02",
            "candidate_actions": [{
                "name": "approve", "family": "institutional",
                "target": {"target_type": "institution", "target_id": "board"},
                "resource_requirements": {"budget": 1}, "resource_costs": {"budget": 1},
                "possible_consequences": [{"kind": "quantity_delta", "name": "approval_progress", "delta": 1}],
                "mechanisms_triggered": ["institution_processing", "reaction_scheduling"],
                "inclusion_reason": "manager decision causally controls approval",
            }],
        }],
        "scheduled_events": [], "hazards": [],
        "mechanisms": ["agent_decision"],
        "required_causal_processes": ["manager_decision"],
        "structural_hypotheses": [
            {"id": "deliberative", "describe": "manager follows evidence", "prior": 0.6, "lean": "weak_yes"},
            {"id": "status_quo", "describe": "manager avoids change", "prior": 0.4, "lean": "weak_no"},
        ],
        "domain": "organizational", "population_kind": "named_actors", "time_scale": "days",
        "available_data": ["organization chart", "decision rights"],
        "rationale": "typed organizational approval with one decision and one reaction",
    }


def test_compiler_routes_legacy_agent_proposal_to_production_policy_and_e2e_delta():
    payload = compiled_payload()
    plan = compile_world(
        "Will the manager approve the project?", llm=lambda _: json.dumps(payload),
        evidence="", as_of="2025-01-01", horizon="2025-01-10", persist=False,
    )
    operators = [m["operator"] for m in plan.accepted_mechanisms]
    assert "production_actor_policy" in operators and "agent_decision" not in operators
    assert plan.actor_decisions and any(e["etype"] == "decision_opportunity" for e in plan.scheduled_events)
    result, branches = run_from_plan(plan, n_particles=8, seed=7)
    action_deltas = [d for branch in branches for d in branch.log if d.operator == "production_actor_policy"]
    assert action_deltas and all(d.changes for d in action_deltas)
    assert all(any(c["path"] == "quantities[approval_progress]" for c in d.changes) for d in action_deltas)
    assert result["readout"] == "terminal_states"


def test_selected_action_changes_terminal_state_without_direct_probability_assignment():
    plan = compile_world(
        "Will the manager approve the project?", llm=lambda _: json.dumps(compiled_payload()),
        evidence="", as_of="2025-01-01", horizon="2025-01-10", persist=False,
    )
    result, branches = run_from_plan(plan, n_particles=4, seed=3)
    assert all(b.world.quantities["approval_progress"].value == 1 for b in branches)
    assert not any("probab" in c["path"].lower() for b in branches for d in b.log for c in d.changes)


def test_compiler_never_accepts_llm_probability_fields():
    payload = compiled_payload()
    payload["actor_decisions"][0]["candidate_actions"][0]["probability"] = 0.999
    plan = compile_world(
        "Will the manager approve the project?", llm=lambda _: json.dumps(payload), evidence="",
        as_of="2025-01-01", horizon="2025-01-10", persist=False,
    )
    decision = plan.actor_decisions[0]
    # The semantic compiler record may preserve unknown metadata, but the typed action
    # builder has no probability field and numeric policy recomputes its distribution.
    from swm.world_model_v2.materialize import build_world
    from swm.world_model_v2.phase4_policy import ActionSpaceBuilder, ActorViewBuilder
    w = build_world(plan)
    v = ActorViewBuilder().build(w, "manager")
    actions = ActionSpaceBuilder().build(plan, w, v, decision=decision)
    assert all("probability" not in a.as_dict() for a in actions)

