"""Causal-boundary smoke run — the REAL public runtime, end to end, offline.

One lightweight pass through ``swm.facade.forecast(architecture="world_model_v2")`` (the
ordinary caller's route: simulate_world → run_with_persistence → operators_from_plan) with a
deterministic scripted backend standing in for the LLM. Proves DEFAULT-ON routing of the
causal truth boundary: scenario mechanisms generated (proposal + independent boundary critic),
actor actions compiled as ATTEMPTS, external effects only through mechanism instances, and the
§18 pure-run counters at zero. No accuracy claim is made or implied — this is wiring proof.

Run:  PYTHONPATH=. python experiments/causal_boundary_smoke.py
"""
import json
import sys

QUESTION = "Will the manager approve the analyst's expansion project?"

COMPILED = {
    "coherent": True,
    "outcome": {"family": "binary", "options": ["True", "False"],
                "resolution_rule": "the board records an approval decision",
                "readout_var": "approval_progress"},
    "entities": [
        {"id": "manager", "type": "person",
         "fields": {"roles": ["manager"], "goals": ["evaluate_project"],
                    "authority": ["approve"], "resources": {"budget": 10},
                    "past_actions": []}},
        {"id": "analyst", "type": "person",
         "fields": {"roles": ["analyst"], "past_actions": []}},
    ],
    "institutions": [{"id": "board", "rules": [
        {"kind": "decision_right", "params": {"actions": ["approve"],
                                              "holders": ["manager"]}}]}],
    "relations": [{"src": "manager", "rel": "communicates_with", "dst": "analyst"}],
    "quantities": [{"name": "approval_progress", "qtype": "approval_progress", "value": 0}],
    "actor_decisions": [{
        "actor": "analyst", "role": "analyst", "at": "2025-01-02",
        "candidate_actions": [{
            "name": "send_briefing", "family": "messaging",
            "target": {"target_type": "person", "target_id": "manager"},
            "mechanisms_triggered": ["message_delivery"],
            "inclusion_reason": "the analyst must brief the manager"}]}],
    "scheduled_events": [], "hazards": [], "mechanisms": ["agent_decision"],
    "required_causal_processes": ["analyst_briefing"],
    "structural_hypotheses": [],
    "domain": "organizational", "population_kind": "named_actors", "time_scale": "days",
    "available_data": ["org chart"], "rationale": "one briefing, one decision",
}

SCHEMA = {
    "entity_types": {"project_dossier": {"description": "the dossier",
                                         "fields": {"title": "str"}}},
    "fact_types": {
        "briefing_memo": {"description": "the analyst's own memo",
                          "fields": {"summary": "str"}},
        "approval_decision_record": {"description": "the board-recorded decision",
                                     "fields": {"matter": "str", "position": "str"}}},
    "relation_types": {},
    "semantic_event_types": {
        "briefing_memo_sent": {"description": "ATTEMPT: memo handed to internal mail",
                               "fields": {"summary": "str"},
                               "typical_visibility": "participants"},
        "briefing_memo_delivered": {"description": "internal-mail outcome",
                                    "fields": {"summary": "str"},
                                    "typical_visibility": "participants"}},
    "process_definitions": {},
    "institutional_definitions": {
        "board": {"procedure": "manager decides", "decision_holders": ["manager"],
                  "decision_record_type": "approval_decision_record",
                  "aggregation": {"kind": "single_authority"}, "assumed": True}},
    "physical_constraints": {}, "resource_definitions": {},
    "information_rules": {"default_delay_s": 60.0},
    "actor_roles": {"manager": {"role": "decision maker",
                                "why_consequential": "holds the approval right",
                                "affordances": ["record approval decision"]},
                    "analyst": {"role": "author", "why_consequential": "writes the briefing",
                                "affordances": ["send briefing memo"]}},
    "outcome_predicates": [{"predicate_id": "approved",
                            "record_type": "approval_decision_record",
                            "field": "position", "op": "eq", "value": "approve",
                            "option_true": "True", "option_false": "False"}],
    "unresolved_mechanisms": [], "assumptions": ["offline smoke scenario"],
}

MECHANISMS = {
    "mechanism_definitions": {
        "internal_mail_run": {
            "description": "the office internal-mail run carrying memos between desks",
            "triggering_event_types": ["briefing_memo_sent"],
            "accepted_inputs": {"summary": "str"},
            "controlling_actor_or_system": "office_mail_room",
            "state_machine": {"in_the_tray": ["delivered_to_desk"]},
            "initial_state": "in_the_tray",
            "success_states": ["delivered_to_desk"], "failure_states": ["misfiled"],
            "unresolved_states": [],
            "possible_output_event_types": {"on_success": ["briefing_memo_delivered"],
                                            "on_failure": []},
            "observation_rules": {"recipients": "direct_targets",
                                  "representation": "complete"},
            "timing_rules": {"delay_s": 1800.0},
            "assumptions": ["the mail run happens twice daily"],
            "uncertainty_source": "mail-room routing"}},
    "new_semantic_event_types": {},
}


def scripted_llm(prompt: str) -> str:
    p = prompt
    if "WORLD-SLICE COMPILER" in p:
        return json.dumps(COMPILED)
    if "SCENARIO SEMANTICS COMPILER" in p:
        return json.dumps(SCHEMA)
    if "adversarial critic of a GENERATED world-semantics model" in p:
        return json.dumps({"missing_decisive_elements": [], "relabeled_progress_bars": [],
                           "hidden_answer_risks": [],
                           "direct_human_reaction_encodings": [],
                           "public_posture_vs_private_reality_conflations": [],
                           "outcome_predicate_matches_question": True,
                           "missing_nonhuman_mechanisms": [], "verdict": "usable"})
    if "CAUSAL MECHANISM COMPILER" in p:
        return json.dumps(MECHANISMS)
    if "CAUSAL-BOUNDARY CRITIC" in p and "RUNTIME schema extension" not in p:
        return json.dumps({"missing_mechanisms": [], "results_masquerading_as_triggers": [],
                           "assumed_success_paths": [], "human_reaction_encodings": [],
                           "repairs_required": [], "verdict": "usable"})
    if "RUNTIME schema extension" in p:
        return json.dumps({"verdict": "reject", "reason": "no runtime extensions in smoke"})
    if "CAUSAL DIRECTNESS CRITIC" in p:
        return "[]"
    if "You adjudicate ONE next transition" in p:
        return json.dumps({"next_state": "delivered_to_desk", "why": "routine run"})
    if "ACTION-ATTEMPT COMPILER" in p:
        actor = p.split("ACTOR:")[1].split("\n")[0].strip()
        if actor == "analyst":
            return json.dumps([
                {"op": "emit_semantic_event", "semantic_type_id": "briefing_memo_sent",
                 "exact_content": "Expansion briefing: projections attached, verbatim.",
                 "direct_targets": ["manager"],
                 "structured_fields": {"summary": "expansion case"}}])
        if actor == "manager":
            return json.dumps([
                {"op": "create_or_update_record",
                 "record_type": "approval_decision_record",
                 "fields": {"matter": "expansion", "position": "approve"},
                 "status": "recorded"}])
        return "[]"
    if "CONSEQUENCE COMPILER" in p:
        raise AssertionError("fixed-v1 consequence compiler ran in a generated smoke run")
    if "You ARE analyst" in p:
        chosen = "send_briefing" if "RECENT ACTIONS: (none yet)" in p else "wait"
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "decision": {"act_or_wait": "act" if chosen != "wait" else "wait",
                         "chosen_action": chosen, "target": "manager",
                         "timing": "immediate", "observability": "private",
                         "intended_effect": "brief the manager"},
            "situation_interpretation": {"what_changed": "deadline",
                                         "why_it_matters": "decision pending"},
            "actor_state_update": {}, "anticipated_reactions": [],
            "novel_action_proposal": {"present": False}, "alternatives_considered": [],
            "decision_summary": f"I {chosen}"})
    if "You ARE manager" in p:
        acted = "record_approval_decision" in p.split("RECENT ACTIONS:")[-1][:200] \
            if "RECENT ACTIONS:" in p else False
        chosen = "wait" if acted else "record_approval_decision"
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "decision": {"act_or_wait": "act" if chosen != "wait" else "wait",
                         "chosen_action": chosen, "target": "board",
                         "timing": "immediate", "observability": "private",
                         "intended_effect": "record my decision"},
            "situation_interpretation": {"what_changed": "briefing arrived",
                                         "why_it_matters": "my call"},
            "actor_state_update": {}, "anticipated_reactions": [],
            "novel_action_proposal": {"present": True,
                                      "description": "record the approval decision",
                                      "required_authority": "approve",
                                      "required_resources": "none",
                                      "proposed_mechanisms": "board record"},
            "alternatives_considered": [], "decision_summary": f"I {chosen}"})
    if "constructing ALTERNATIVE HYPOTHESES" in p:
        return "[]"                                # template fallback set serves, labeled
    return "{}"                                    # every guarded auxiliary phase records omission


def main():
    from swm.facade import forecast
    res = forecast(QUESTION, architecture="world_model_v2", llm=scripted_llm,
                   as_of="2025-01-01", horizon="2025-01-15",
                   execution_policy={"drop_phases": ["phase2_evidence",
                                                     "phase11_recompilation", "fidelity",
                                                     "event_time", "activation_synthesis"]},
                   seed=11)
    prov = res.get("provenance") or {}
    rep = dict(prov.get("consequence_report") or {})
    rep.pop("causal_action_reports", None)
    out = {
        "question": QUESTION,
        "simulation_status": res.get("simulation_status"),
        "raw_distribution": res.get("raw_distribution"),
        "requested_mode": rep.get("requested_mode"),
        "actual_mode": rep.get("actual_mode"),
        "consequence_report": rep,
        "causal_action_reports_sample":
            (prov.get("consequence_report") or {}).get("causal_action_reports", [])[:3],
        "actor_policy_mode": (prov.get("actor_policy_report") or {}).get(
            "actual_actor_policy_mode"),
        "audit": {
            "old_consequence_path_used": bool(rep.get("fixed_ontology_uses", 0)),
            "external_success_written_directly":
                int(rep.get("external_successes_written_directly", 0) or 0),
            "human_reactions_written_directly":
                int(rep.get("human_reactions_written_directly", 0) or 0),
            "numeric_fallbacks": int(rep.get("numeric_fallbacks", 0) or 0),
            "action_attempts": int(rep.get("action_attempts", 0) or 0),
            "mechanisms_invoked": int(rep.get("mechanisms_invoked", 0) or 0),
            "mechanism_successes": int(rep.get("mechanism_successes", 0) or 0),
            "intended_deliveries": int(rep.get("intended_deliveries", 0) or 0),
            "actual_deliveries": int(rep.get("actual_deliveries", 0) or 0),
        },
    }
    print(json.dumps(out, indent=2, default=str))
    ok = (out["requested_mode"] == "generated_actor_mediated_world"
          and out["actual_mode"] == "generated_actor_mediated_world"
          and out["audit"]["old_consequence_path_used"] is False
          and out["audit"]["external_success_written_directly"] == 0
          and out["audit"]["human_reactions_written_directly"] == 0
          and out["audit"]["action_attempts"] >= 1
          and out["audit"]["mechanisms_invoked"] >= 1)
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
