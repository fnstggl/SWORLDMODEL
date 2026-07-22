"""Global-default routing: every public action/policy entry point uses the scenario-generated
layer on generated worlds; fixed-v1 is reachable only by explicit request; deterministic
mechanical candidates skip LLM planning but never skip the canonical contract/feasibility/
queue/result path.
"""
import copy

import pytest

from swm.world_model_v2.phase13 import api as p13
from swm.world_model_v2.phase13.contracts import DecisionProblem
from swm.world_model_v2.phase13.scenario_actions.candidates import ConcreteAction, PlanStep
from tests.scenario_fixtures import build_context, council_schema
from tests.test_scenario_action_layer import MAKER, OFFICER, filing_candidate, officer_grants


def problem(**kw):
    kw.setdefault("decision_id", "route1")
    kw.setdefault("decision_maker", MAKER)
    kw.setdefault("authority", ["petitioner"])
    kw.setdefault("horizon", "2023-12-31T00:00:00Z")
    return DecisionProblem(**kw)


def ctx(**kw):
    kw.setdefault("script", {OFFICER: officer_grants})
    kw.setdefault("n_particles", 2)
    return build_context(council_schema(), [MAKER, OFFICER], **kw)


def _forbid_legacy(monkeypatch):
    """Any touch of the legacy catalog machinery raises — silent fallback is impossible."""
    import swm.world_model_v2.phase13.ontology as onto
    import swm.world_model_v2.phase13.search as legacy_search

    def boom(*a, **k):
        raise AssertionError("legacy fixed-v1 machinery reached on a generated world")

    monkeypatch.setattr(onto, "operation_spec", boom)
    monkeypatch.setattr(onto, "operation_registered", boom)
    monkeypatch.setattr(legacy_search, "select_and_run", boom)


# ------------------------------------------------------------------ per-entry-point defaults
def test_recommend_action_defaults_to_generated(monkeypatch):
    _forbid_legacy(monkeypatch)
    wc, rep, _ = ctx()
    res = p13.recommend_action(problem(), wc, goal_text="obtain the variance", seed=1)
    assert "scenario_report" in res.provenance


def test_evaluate_actions_defaults_to_generated(monkeypatch):
    _forbid_legacy(monkeypatch)
    wc, rep, _ = ctx()
    res = p13.evaluate_actions(problem(), [filing_candidate()], wc,
                               goal_text="obtain the variance", seed=1)
    assert "scenario_report" in res.provenance
    assert res.recommended == "file_petition"
    assert rep["scenario_events_emitted"] >= 1          # canonical runtime actually ran


def test_optimize_policy_defaults_to_generated_with_contingent_plan(monkeypatch):
    _forbid_legacy(monkeypatch)
    wc, rep, _ = ctx()
    plan = filing_candidate("contingent")
    from swm.world_model_v2.phase13.scenario_actions.candidates import ConditionSpec
    follow = PlanStep(step_id="contingent_s2",
                      intent="publicly thank the panel after the grant issues",
                      conditions=[ConditionSpec(kind="record", record_type="variance_grant",
                                                op="exists", description="grant exists")],
                      after_steps=["contingent_s1"], visibility="public")
    follow.compiled_ops = [{"op": "emit_semantic_event",
                            "semantic_type_id": "petition_filed_notice",
                            "exact_content": "thank you", "intended_visibility": "public"}]
    follow.compile_meta = {"compiler": "test_precompiled"}
    plan.steps[0].step_id = "contingent_s1"
    plan.steps.append(follow)
    res = p13.optimize_policy(problem(), [plan], wc, seed=2)
    sr = res.provenance["scenario_report"]
    assert "contingent" in sr["evaluations"]
    diag = sr["trajectory_summaries"]["contingent"]
    # the contingent step FIRED after its observed condition held (sequential + contingent)
    assert diag["step_stats"].get("contingent_s2", {}).get("completed", 0) >= 1


def test_value_of_information_defaults_to_generated(monkeypatch):
    _forbid_legacy(monkeypatch)
    wc, rep, _ = ctx()
    out = p13.value_of_information(problem(), [], wc, goal_text="obtain the variance", seed=1)
    assert out["value_of_information"].get("route") == "scenario_generated"


# ------------------------------------------------------------------ quarantine + loud failure
def test_legacy_reachable_only_by_explicit_request():
    wc, _, _ = ctx()
    with pytest.raises(ValueError, match="unknown mode"):
        p13.recommend_action(problem(), wc, mode="fixed")     # typo'd modes refused loudly
    with pytest.raises(ValueError, match="unknown mode"):
        p13.optimize_policy(problem(), [], wc, mode="v1")


def test_missing_scenario_schema_fails_loudly():
    from swm.world_model_v2.information import InformationLedger
    from swm.world_model_v2.network import RelationGraph
    from swm.world_model_v2.state import SimulationClock, WorldState
    from swm.world_model_v2.events import EventQueue

    class BareInitial:
        schema = None
        def sample_particles(self, n, seed=0):
            return [WorldState("bare", f"p{i}", SimulationClock(0.0, 0.0),
                               network=RelationGraph(), information=InformationLedger())
                    for i in range(n)]

    wc = {"initial": BareInitial(), "queue_builder": lambda w: EventQueue(horizon_ts=1e12),
          "operators": [], "contract": None, "n_particles": 1}
    from swm.world_model_v2.phase13.scenario_actions.api import evaluate_actions_generated
    with pytest.raises(RuntimeError, match="under-modeled"):
        evaluate_actions_generated(problem(), [filing_candidate()], wc)


# ------------------------------------------------------------------ deterministic mechanical path
def test_precompiled_mechanical_candidate_runs_with_zero_llm_calls(monkeypatch):
    """§3: deterministic mechanical operations skip LLM planning but still enter through the
    same contract, feasibility, queue, and result contract."""
    _forbid_legacy(monkeypatch)
    called = {"n": 0}

    def llm_that_must_not_be_needed(prompt):
        called["n"] += 1
        return "{}"

    wc, rep, _ = ctx()
    cand = filing_candidate()                       # fully precompiled kernel ops
    res = p13.evaluate_actions(problem(), [cand], wc, goal_text="obtain the variance",
                               seed=3)
    sr = res.provenance["scenario_report"]
    assert res.recommended == "file_petition"
    assert sr["feasibility"]["file_petition"]["feasible_everywhere"]     # feasibility ran
    assert sr["compiled_effects"]["file_petition"][0]["ops"]             # same result contract
    assert res.cost["llm_calls"] == 0                                    # zero LLM planning
    assert called["n"] == 0


def test_all_entry_points_share_one_result_contract():
    wc, _, _ = ctx()
    r1 = p13.recommend_action(problem(), wc, goal_text="obtain the variance", seed=1)
    r2 = p13.evaluate_actions(problem(), [filing_candidate()], wc,
                              goal_text="obtain the variance", seed=1)
    r3 = p13.optimize_policy(problem(), [filing_candidate("polplan")], wc, seed=1)
    for r in (r1, r2, r3):
        sr = r.provenance["scenario_report"]
        for key in ("goal_contract", "action_language_summary", "evaluations",
                    "simulation_coverage", "candidate_ancestry"):
            assert key in sr
