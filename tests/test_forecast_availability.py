"""The forecast-availability contract (user directive): grounding quality and forecast
availability are SEPARATE. Statuses describe the run; they never erase an available
probability. Proofs required:

  1. under_modeled can still return a probability;
  2. unresolved can still return a probability;
  3. weak grounding changes labels and uncertainty, not availability;
  4. no generic 0.5 fallback exists anywhere in the recovery;
  5. truly malformed/no-source questions may still return no probability;
  6. (live harness) all five lean BTF-3 runs produce scoreable probabilities — exp110;
  +  regression: status labels NEVER control whether a valid probability is returned."""
from __future__ import annotations

from types import SimpleNamespace

from swm.world_model_v2.forecast_recovery import (ForecastRecovery, GROUNDING_GRADES,
                                                  PROBABILITY_SOURCES, recover_forecast)
from swm.world_model_v2.result import SimulationResult


# ------------------------------------------------------------------ layered recovery
def test_partial_rollouts_blend_and_disclose_never_renormalize():
    rec = recover_forecast(distribution={"yes": 0.30, "no": 0.10, "unresolved_mechanism": 0.60},
                           options=["yes", "no"], unresolved_mass=0.60,
                           posterior_mean=0.85, posterior_n_eff=3)
    assert rec.probability_source == "partial_rollouts"
    assert rec.probability_conditional_on_resolved == 0.75          # 0.30/0.40, weights kept
    # final = resolved_share*conditional + unresolved_share*posterior = 0.4*0.75 + 0.6*0.85
    assert rec.probability == 0.81
    assert rec.unresolved_mass == 0.6                               # disclosed, not hidden
    assert rec.uncertainty_interval == (0.3, 0.9)                   # worst/best case swings
    assert rec.weight_sensitive is True                             # interval crosses 0.5
    assert rec.grounding_grade == "partially_grounded"


def test_completed_rollouts_keep_simulated_frequencies():
    rec = recover_forecast(distribution={"yes": 0.7, "no": 0.3}, options=["yes", "no"],
                           unresolved_mass=0.0, posterior_mean=0.5, posterior_n_eff=5)
    assert rec.probability == 0.7 and rec.probability_source == "completed_rollouts"
    assert rec.grounding_grade == "grounded" and rec.weight_sensitive is False


def test_prior_layers_serve_when_no_mass_resolved():
    ev = recover_forecast(distribution={}, options=["yes", "no"], unresolved_mass=1.0,
                          posterior_mean=0.85, posterior_n_eff=2)
    assert ev.probability == 0.85 and ev.probability_source == "evidence_conditioned_prior"
    assert ev.grounding_grade == "exploratory" and ev.weight_sensitive is True
    gr = recover_forecast(distribution={}, options=["yes", "no"], unresolved_mass=1.0,
                          prior_mean=0.62, prior_source_class="recurrence")
    assert gr.probability == 0.62 and gr.probability_source == "grounded_reference_prior"
    ex = recover_forecast(distribution={}, options=["yes", "no"], unresolved_mass=1.0,
                          prior_mean=0.3, prior_source_class="lean")
    assert ex.probability == 0.3 and ex.probability_source == "exploratory_model_estimate"
    assert ex.grounding_grade == "ungrounded"


def test_no_generic_half_fallback_exists():
    """With absolutely no defensible source, the recovery returns None — it cannot invent a
    neutral 0.5. AST-verified: no 0.5 literal is ever ASSIGNED or RETURNED in the recovery
    module (0.5 in comparisons — the weight-sensitivity checks — is allowed)."""
    assert recover_forecast(distribution={}, options=["yes", "no"], unresolved_mass=1.0) is None
    import ast
    import inspect
    from swm.world_model_v2 import forecast_recovery as FR
    tree = ast.parse(inspect.getsource(FR))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Return, ast.keyword)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Compare):
                    break
            else:
                for sub in ast.walk(node):
                    assert not (isinstance(sub, ast.Constant) and sub.value == 0.5), \
                        f"literal 0.5 assigned/returned at line {sub.lineno}"


def test_weak_grounding_changes_labels_not_availability():
    strong = recover_forecast(distribution={"yes": 0.7, "no": 0.3}, options=["yes", "no"],
                              unresolved_mass=0.0, posterior_mean=0.6, posterior_n_eff=4)
    weak = recover_forecast(distribution={}, options=["yes", "no"], unresolved_mass=1.0,
                            prior_mean=0.7, prior_source_class="lean")
    assert strong.probability is not None and weak.probability is not None
    assert strong.grounding_grade == "grounded" and weak.grounding_grade == "ungrounded"
    assert strong.confidence != weak.confidence
    assert weak.uncertainty_interval == (0.0, 1.0)                  # width, not absence


# ------------------------------------------------------------------ result contract
def _mk(status, p=None, dist=None, rr=None, **kw):
    kw.setdefault("support_grade", "exploratory")
    if status in ("unresolved", "partially_resolved"):
        kw.setdefault("resolution_report", rr or {"unresolved_share": 1.0,
                                                  "missing_mechanisms": [{"mechanism": "m"}]})
    if status == "under_modeled":
        kw.setdefault("under_modeled_subtypes", ["under_modeled_nonhuman_mechanism"])
    return SimulationResult(question="q?", simulation_status=status, raw_probability=p,
                            raw_distribution=dist or {}, **kw)


def test_under_modeled_and_unresolved_can_carry_a_forecast():
    um = _mk("under_modeled", p=0.62, grounding_grade="exploratory",
             probability_source="evidence_conditioned_prior")
    assert um.has_forecast() and um.raw_probability == 0.62
    un = _mk("unresolved", p=0.41, grounding_grade="exploratory",
             probability_source="grounded_reference_prior")
    assert un.has_forecast() and un.raw_probability == 0.41


def test_status_labels_never_control_probability_availability():
    """THE regression: for every status, has_forecast() is decided by probability/distribution
    presence alone. A weak forecast is a labeled forecast, never a missing one."""
    for status in ("completed", "completed_with_degradation", "under_modeled", "unresolved",
                   "partially_resolved", "truncated", "temporally_truncated"):
        with_p = _mk(status, p=0.3)
        without = _mk(status, p=None)
        assert with_p.has_forecast() is True, f"{status} suppressed an available probability"
        assert without.has_forecast() is False, f"{status} invented a forecast from nothing"


def test_malformed_questions_still_return_no_probability():
    cl = SimulationResult(question="?", simulation_status="clarification_required",
                          clarification_reason="ambiguous")
    assert not cl.has_forecast() and cl.raw_probability is None
    ef = SimulationResult(question="?", simulation_status="execution_failed",
                          failure_taxonomy="runtime_exception")
    assert not ef.has_forecast()


def test_result_carries_the_separated_fields():
    r = _mk("unresolved", p=0.44, grounding_grade="exploratory",
            probability_source="grounded_reference_prior", confidence="very_low",
            unresolved_mass=1.0, uncertainty_interval=[0.0, 1.0], weight_sensitive=True)
    assert r.grounding_grade in GROUNDING_GRADES
    assert r.probability_source in PROBABILITY_SOURCES
    assert r.unresolved_mass == 1.0 and r.weight_sensitive is True
    d = r.as_dict() if hasattr(r, "as_dict") else r.__dict__
    for k in ("probability_source", "grounding_grade", "confidence", "unresolved_mass",
              "weight_sensitive"):
        assert k in d


# ------------------------------------------------------------------ guard integration
def test_guard_serves_prior_on_unresolved_execution_without_changing_status(monkeypatch):
    monkeypatch.delenv("SWM_ALLOW_GENERIC_PRIOR", raising=False)   # live-run configuration
    import swm.world_model_v2.unified_runtime as U
    res = _mk("execution_failed", p=None)
    res.failure_taxonomy = "runtime_exception"
    U._apply_result_guards(res, posterior=None,
                           prior_spec=SimpleNamespace(mean=0.34, source_class="recurrence"))
    assert res.simulation_status == "unresolved"
    assert res.raw_probability == 0.34
    assert res.probability_source == "grounded_reference_prior"
    assert res.has_forecast()
    assert any("labeled" in l for l in res.limitations)


def test_all_five_lean_btf3_runs_recovered_scoreable_probabilities():
    """Requirement 6 (live harness): the committed EXP-110 recovery over the five lean BTF-3
    checkpoints produced a scoreable, labeled probability for every question — from stored
    simulation artifacts (weighted distributions + evidence-updated posteriors), never a rerun,
    never a neutral default."""
    import json
    from pathlib import Path
    p = Path("experiments/results/exp110_recovered_forecasts.json")
    assert p.exists(), "run experiments/exp110_recover_lean_forecasts first"
    rows = json.loads(p.read_text())["lean"]
    assert len(rows) == 5
    for r in rows:
        assert r.get("recovered_probability") is not None, f"{r.get('qid')} unscoreable"
        assert r.get("probability_source"), f"{r.get('qid')} missing source"
        assert r.get("grounding_grade") in ("grounded", "partially_grounded", "exploratory",
                                            "ungrounded")
        assert r["recovered_probability"] != 0.5 or r.get("probability_source") != "", \
            "a bare neutral default is not a source"
