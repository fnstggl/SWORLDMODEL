"""Activation synthesis + phase consumers + replay-lab sealing tests."""
from __future__ import annotations
import json
import os

import pytest

from swm.world_model_v2.compiler import WorldExecutionPlan
from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.state import parse_time
from swm.world_model_v2.activation_synthesis import phase_requirements, synthesize_activation
from swm.world_model_v2.materialize import run_from_plan

AS_OF = parse_time("2025-01-01")
HZ = parse_time("2025-02-01")


def _plan(question="Will the assembly confirm the nominee?", processes=(), institutions=None,
          populations=None, relations=None, entities=None, posterior=None):
    def read(w):
        q = w.quantities.get("outcome")
        return q.value if q else None
    c = OutcomeContract(family="binary", options=["confirmed", "not_confirmed"], resolution_rule="r",
                        readout=read, readout_var="outcome", horizon_ts=HZ).validate()
    p = WorldExecutionPlan(question=question, outcome_contract=c, as_of=AS_OF, horizon_ts=HZ)
    p.quantities = [{"name": "outcome", "qtype": "outcome", "value": None}]
    p.entities = entities if entities is not None else []
    p.institutions = institutions or []
    p.populations = populations or []
    p.relations = relations or []
    p.mechanism_choices = [{"process": pr} for pr in processes]
    p.scheduled_events = [{"etype": "resolve_outcome", "ts": HZ - 1.0, "participants": [],
                           "payload": {"outcome_var": "outcome", "family": "binary",
                                       "options": ["confirmed", "not_confirmed"], "lean": "neutral"}}]
    p.accepted_mechanisms = [{"mech_id": "generic_outcome_prior", "operator": "generic_outcome_prior",
                              "causal_role": "safety net"}]
    if posterior:
        p.posterior_rate_particles = posterior
    return p


SENATE = [{"id": "senate", "sensitivity": 0.9,
           "rules": [{"kind": "quorum", "params": {"quorum": 51, "total": 100}}]}]
POPS = [{"id": "electorate", "segments": [{"id": "base", "weight": 0.6, "differs_on": ["enthusiasm"]},
                                          {"id": "swing", "weight": 0.4, "differs_on": ["attention"]}]}]
RELS = [{"src": "a", "rel": "influences", "dst": "b"}, {"src": "b", "rel": "trusts", "dst": "c"}]


def _deltas_by_op(branches):
    ops = {}
    for b in branches:
        for d in b.log:
            ops[d.operator] = ops.get(d.operator, 0) + 1
    return ops


# ---------------------------------------------------------------- relevance gate
def test_gate_requires_process_AND_declared_structure():
    p = _plan(question="Will it rain in the city tomorrow?", processes=["weather_system_progression"],
              institutions=SENATE, populations=POPS, relations=RELS)
    req = phase_requirements(p)
    assert not req["phase10_institutions"]["required"]     # declared but no institutional process → context
    assert not req["phase9_populations"]["required"]
    assert not req["phase9_networks"]["required"]


def test_gate_fires_on_institutional_process_with_declared_institution():
    p = _plan(processes=["evaluate_quorum_and_threshold_vote"], institutions=SENATE)
    assert phase_requirements(p)["phase10_institutions"]["required"]


def test_gate_reads_question_wording_when_process_list_empty():
    p = _plan(question="Will the FOMC vote to cut the rate?", processes=[], institutions=SENATE)
    assert phase_requirements(p)["phase10_institutions"]["required"]


# ---------------------------------------------------------------- P10 execution + causal effect
def test_p10_institutional_decision_executes_and_changes_terminal():
    p = _plan(processes=["evaluate_quorum_and_threshold_vote"], institutions=SENATE,
              posterior=[(0.75, 1.0)])
    synthesize_activation(p)
    res, branches = run_from_plan(p, llm=None, seed=3)
    ops = _deltas_by_op(branches)
    assert ops.get("institutional_decision", 0) > 0
    p_full = res["distribution"].get("confirmed", 0.0)
    # matched ablation: same plan, P10 forced off
    p2 = _plan(processes=["evaluate_quorum_and_threshold_vote"], institutions=SENATE,
               posterior=[(0.75, 1.0)])
    req = phase_requirements(p2)
    req["phase10_institutions"] = {"required": False, "why": "ablation"}
    synthesize_activation(p2, req)
    res2, _ = run_from_plan(p2, llm=None, seed=3)
    assert abs(p_full - res2["distribution"].get("confirmed", 0.0)) > 0.02   # rule sharpening is real


def test_p10_threshold_rule_transforms_posterior_not_invents_rate():
    # supermajority (67/100) with propensity 0.55 must pass far less often than simple majority (51/100)
    def once(needed):
        inst = [{"id": "x", "sensitivity": 0.9,
                 "rules": [{"kind": "quorum", "params": {"quorum": needed, "total": 100}}]}]
        p = _plan(processes=["ratification_vote"], institutions=inst, posterior=[(0.55, 1.0)])
        synthesize_activation(p)
        res, _ = run_from_plan(p, llm=None, seed=5)
        return res["distribution"].get("confirmed", 0.0)
    assert once(51) > once(67) + 0.2


# ---------------------------------------------------------------- P9 consumers
def test_p9_population_aggregation_executes_and_modulates():
    p = _plan(question="Will turnout exceed the threshold?", processes=["turnout_participation"],
              populations=POPS)
    rep = synthesize_activation(p)
    assert any(a["phase"] == "phase9_populations" for a in rep["actions"])
    res, branches = run_from_plan(p, llm=None, seed=3)
    assert _deltas_by_op(branches).get("population_aggregation", 0) > 0
    rev = [e for e in p.scheduled_events if e["etype"] == "resolve_outcome"][0]
    assert rev["payload"]["rate_modulation"]                     # consumer wired into the terminal


def test_p9_network_diffusion_executes():
    p = _plan(question="Will the rumor spread through the network?", processes=["contagion_spread"],
              relations=RELS)
    synthesize_activation(p)
    _, branches = run_from_plan(p, llm=None, seed=3)
    assert _deltas_by_op(branches).get("network_diffusion", 0) > 0


def test_modulation_total_weight_capped():
    from swm.world_model_v2.activation_synthesis import _add_modulation
    p = _plan(processes=[])
    for i in range(6):
        _add_modulation(p, f"v{i}", 0.2)
    rev = [e for e in p.scheduled_events if e["etype"] == "resolve_outcome"][0]
    assert sum(m["weight"] for m in rev["payload"]["rate_modulation"]) <= 0.45 + 1e-9


# ---------------------------------------------------------------- P7 + P4
def test_p7_state_step_chain_executes():
    p = _plan(question="Will adoption pass the tipping point?", processes=["threshold_tipping_cascade"])
    synthesize_activation(p)
    _, branches = run_from_plan(p, llm=None, seed=3)
    assert _deltas_by_op(branches).get("nonlinear_state_step", 0) > 0


def test_p4_gated_off_removes_ornamental_decisions():
    p = _plan(question="Will the temperature exceed 40C?", processes=["heatwave_progression"],
              entities=[{"id": "x", "type": "person", "fields": {}}])
    p.scheduled_events.append({"etype": "decision_opportunity", "ts": HZ - 5.0, "participants": ["x"],
                               "payload": {"actions": [{"type": "act"}, {"type": "wait"}]}})
    rep = synthesize_activation(p)
    assert not any(e["etype"] == "decision_opportunity" for e in p.scheduled_events)
    assert any(a["action"] == "gated_off_ornamental_decisions" for a in rep["actions"])


def test_p4_undeclared_actor_decision_dropped_not_crashing():
    p = _plan(processes=["strategic_decision_negotiation"],
              entities=[{"id": "x", "type": "person", "fields": {}, "sensitivity": 0.8}])
    p.scheduled_events.append({"etype": "decision_opportunity", "ts": HZ - 5.0,
                               "participants": ["ghost_actor"], "payload": {}})
    synthesize_activation(p)
    run_from_plan(p, llm=None, seed=3)                          # must not raise
    assert any(o.get("kind") == "decision_event_undeclared_actor" for o in p.omissions)


def test_action_polarity_lexicon():
    from swm.world_model_v2.phase_consumers import action_polarity
    assert action_polarity("vote_yes") == 1
    assert action_polarity("reject_deal") == -1
    assert action_polarity("wait") == 0


# ---------------------------------------------------------------- synthesis honesty
def test_synthesis_never_invents_structure():
    """P10 required by process wording but NO institution declared → nothing synthesized."""
    p = _plan(question="Will the committee vote pass?", processes=["committee_vote"], institutions=[])
    rep = synthesize_activation(p)
    assert not any(e["etype"] == "institutional_decision" for e in p.scheduled_events)
    assert not any(a["phase"] == "phase10_institutions" for a in rep["actions"])


def test_unified_runtime_wires_synthesis():
    import inspect
    from swm.world_model_v2 import unified_runtime as U
    src = inspect.getsource(U)
    assert "synthesize_activation" in src and "phase_requirements" in src


# ---------------------------------------------------------------- replay lab sealing
def test_sealed_resolutions_refuse_forecaster_access(tmp_path, monkeypatch):
    from swm.replay import vault
    monkeypatch.delenv("REPLAY_SCORER", raising=False)
    if not vault.SEALED.exists():
        pytest.skip("vault not built")
    with pytest.raises(PermissionError):
        vault.sealed_resolutions()


def test_public_events_carry_no_outcome():
    from swm.replay import vault
    if not vault.EVENTS.exists():
        pytest.skip("vault not built")
    raw = json.loads(vault.EVENTS.read_text())
    assert "outcome" not in json.dumps(raw["events"]).lower().replace("outcome_contract", "")


def test_freeze_hash_detects_tampering():
    from swm.replay.vault import freeze_hash
    row = {"event_id": "e", "p_yes": 0.4}
    h = freeze_hash(row)
    row["p_yes"] = 0.9
    assert freeze_hash(row) != h


def test_blinding_preserves_structure_hides_names():
    from swm.replay.blinding import apply_mapping, blind_question
    mapping = {"Donald Trump": "Candidate A", "Trump": "Candidate A", "Kamala Harris": "Candidate B"}
    q = blind_question("Will Donald Trump defeat Kamala Harris in the 2024 election?", mapping)
    assert "Trump" not in q and "Harris" not in q and "2024" not in q
    assert "Candidate A" in q and "Candidate B" in q
    assert apply_mapping("Trumpet sound", {"Trump": "X"}) == "Trumpet sound"   # whole-word only
