"""No-abstention production contract (Part A) — unit + integration + adversarial + legacy-migration tests.

Every coherent question SIMULATES; epistemic weakness lowers the support grade, never refuses. Only genuine
incoherence → clarification_required; only technical failure → execution_failed. All scripted (no network).
"""
import json

import pytest

from swm.world_model_v2.pipeline import simulate
from swm.world_model_v2.result import (RECOMMENDATION_STATUSES, SIMULATION_STATUSES, SUPPORT_GRADES,
                                        ClarificationRequired, CompilerExecutionError, SimulationResult,
                                        migrate_legacy_result)

AS_OF, HORIZON = "2023-05-01", "2023-06-01"


def _llm(decomp):
    return lambda p: json.dumps(decomp)


# ---------------------------------------------------------------- the result contract itself
def test_result_axes_are_independent_and_validated():
    r = SimulationResult(question="q", simulation_status="completed_with_degradation",
                         support_grade="exploratory", recommendation_status="limited",
                         raw_probability=0.4, raw_distribution={"yes": 0.4, "no": 0.6})
    assert r.has_forecast() and r.as_dict()["abstain"] is False       # a forecast that ran is NOT an abstention
    assert r.simulation_status in SIMULATION_STATUSES
    assert r.support_grade in SUPPORT_GRADES
    assert r.recommendation_status in RECOMMENDATION_STATUSES
    with pytest.raises(ValueError):                                    # completed result needs a valid grade
        SimulationResult(question="q", simulation_status="completed", support_grade="not_a_grade")


def test_only_clarification_maps_to_legacy_abstain():
    clar = SimulationResult(question="q", simulation_status="clarification_required",
                            clarification_reason="no coherent outcome")
    assert clar.as_dict()["abstain"] is True                          # ONLY genuine incoherence
    fail = SimulationResult(question="q", simulation_status="execution_failed",
                            failure_taxonomy="runtime_exception")
    assert fail.as_dict()["abstain"] is False                         # a technical failure is NOT an abstention


# ---------------------------------------------------------------- integration: coherent Q always forecasts
def test_simulate_produces_forecast_for_coherent_question():
    decomp = {"outcome": {"family": "binary", "options": ["approve", "reject"],
                          "resolution_rule": "approved", "readout_var": "decision"},
              "outcome_lean": "weak_yes", "entities": [{"id": "vp", "type": "person", "fields": {}}],
              "required_causal_processes": ["approval_decision"], "rationale": "vp decides"}
    res = simulate("Will the VP approve the hire?", llm=_llm(decomp), evidence="",
                   as_of=AS_OF, horizon=HORIZON, seed=1)
    assert res.simulation_status in ("completed", "completed_with_degradation")
    assert res.has_forecast() and res.raw_distribution and res.raw_probability is not None
    assert res.support_grade in SUPPORT_GRADES
    assert any(fb.get("tier") for fb in res.fallbacks_used)            # a fallback tier is always identified


def test_weak_question_forecasts_at_low_grade_not_refusal():
    """A question with no mechanisms and no readout still forecasts — degraded, not refused."""
    res = simulate("Will it happen?", llm=_llm({"outcome": {"family": "binary"}}), evidence="",
                   as_of=AS_OF, horizon=HORIZON, seed=1)
    assert res.has_forecast()
    assert res.simulation_status == "completed_with_degradation"
    assert res.support_grade in ("exploratory", "highly_speculative")
    assert res.limitations                                            # weakness is surfaced, not hidden


# ---------------------------------------------------------------- adversarial: incoherence vs technical failure
def test_incoherent_question_clarifies_rarely():
    res = simulate("purple", llm=_llm({"coherent": False, "why": "no simulable outcome",
                                       "interpretations": []}),
                   evidence="", as_of=AS_OF, horizon=HORIZON, seed=1)
    assert res.simulation_status == "clarification_required"
    assert not res.has_forecast()                                     # genuinely incoherent → no forecast (rare)


def test_empty_llm_output_is_execution_failed_not_abstention():
    res = simulate("Will X happen?", llm=lambda p: "", evidence="", as_of=AS_OF, horizon=HORIZON, seed=1)
    assert res.simulation_status == "execution_failed"
    assert res.failure_taxonomy == "parser_failure_after_retries"    # engineering failure, taxonomy'd
    assert res.as_dict()["abstain"] is False                         # NOT an epistemic abstention


def test_llm_probability_minting_is_ignored():
    """Adversarial: the LLM tries to inject a terminal probability; the forecast must come from the prior."""
    decomp = {"outcome": {"family": "binary", "options": ["yes", "no"], "resolution_rule": "r",
                          "readout_var": "out"}, "outcome_lean": "neutral",
              "p": {"yes": 0.99}, "probability": 0.99,               # injected — must be ignored
              "entities": [{"id": "a", "type": "person", "fields": {}}],
              "required_causal_processes": ["decide"], "rationale": "x"}
    res = simulate("Will yes?", llm=_llm(decomp), evidence="", as_of=AS_OF, horizon=HORIZON, seed=5)
    assert res.raw_probability != 0.99                               # the injected number never becomes the forecast


# ---------------------------------------------------------------- legacy artifact migration (read, don't edit)
def test_migrate_legacy_coherence_abstention_becomes_clarification():
    old = {"question": "q", "abstain": True, "abstain_reason": "question is ambiguous; cannot interpret"}
    m = migrate_legacy_result(old)
    assert m["simulation_status"] == "clarification_required" and m["_migrated"] is True
    assert m["abstain"] is True                                       # original field preserved, not edited


def test_migrate_legacy_mechanism_abstention_becomes_execution_failed():
    old = {"question": "q", "abstain": True, "abstain_reason": "no executable mechanism for this slice"}
    m = migrate_legacy_result(old)
    assert m["simulation_status"] == "execution_failed"
    assert m["failure_taxonomy"] == "missing_required_operator"
    assert m["provenance"]["migrated_from_legacy_abstention"] == "no executable mechanism for this slice"


def test_migrate_legacy_nonabstention_is_passthrough():
    old = {"question": "q", "abstain": False, "p": 0.6}
    m = migrate_legacy_result(old)
    assert m["simulation_status"] == "completed" and m["_migrated"] is False


# ---------------------------------------------------------------- salvage: truncated decomposition still forecasts
def test_truncated_decomposition_is_salvaged():
    from swm.world_model_v2.compiler import _salvage_json
    trunc = ('{"outcome": {"family": "binary", "options": ["win","lose"], "resolution_rule": "r", '
             '"readout_var": "res"}, "outcome_lean": "weak_yes", "entities": [{"id": "a", "type": "per')
    o = _salvage_json(trunc)
    assert o.get("outcome", {}).get("options") == ["win", "lose"]     # prefix (outcome contract) recovered
    assert o.get("outcome_lean") == "weak_yes"
