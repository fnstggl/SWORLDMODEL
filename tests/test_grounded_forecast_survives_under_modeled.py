"""Pins the p=None fix: forecasting mode must NEVER return a silent None when a GROUNDED outside-view
forecast exists. The malformed Knesset run had a usable ~5% grounded prior but the ensemble aggregator
returned p=None/under_modeled because it built its final result without a no-silent-None guard and never
consulted the prior. The ensemble-level guard now serves the grounded reference-class estimate as the
headline while KEEPING the epistemically-weak status as a warning — and still refuses to headline a
GENERIC (no-reference-class) prior (§NAP preserved).

These pin the two guard helpers directly (fast, no LLM); the live Knesset rerun is the end-to-end check."""
from types import SimpleNamespace

from swm.world_model_v2.phase3_priors import (grounded_estimate_prior, generic_lean_prior,
                                              is_grounded_prior)
from swm.world_model_v2.structural_runtime import (_ensemble_grounded_forecast,
                                                   _serve_grounded_outside_view)
from swm.world_model_v2.result import SimulationResult


def _plan(options=("True", "False")):
    return SimpleNamespace(outcome_contract=SimpleNamespace(options=list(options)))


def _promoted(*model_ids):
    return [SimpleNamespace(model_id=m, executable_plan=_plan()) for m in model_ids]


def _knesset_prior():
    # the exact grounded 5% outside-view prior the malformed run discarded (source_class
    # "llm_estimated_reference", reference class named) -> is_grounded_prior True
    return grounded_estimate_prior("Knesset dissolutions", 0.05, transport_risk="moderate",
                                   n_examples=20, is_recurrence=False, evidence_quality="model_memory")


def _under_modeled_none():
    return SimulationResult(
        question="Will the Knesset be dissolved by June 30?", simulation_status="under_modeled",
        support_grade="highly_speculative", raw_distribution={}, raw_probability=None,
        under_modeled_subtypes=["under_modeled_nonhuman_mechanism"],
        under_modeled_components=[{"component": "dissolution vote", "kind": "nonhuman_mechanism",
                                   "why": "no validated mechanism", "sensitivity": "decisive"}],
        resolution_report={"missing_mechanisms": [{"mechanism": "dissolution vote"}]})


def _unresolved_none():
    return SimulationResult(
        question="Will the Knesset be dissolved by June 30?", simulation_status="unresolved",
        support_grade="highly_speculative", raw_distribution={}, raw_probability=None,
        resolution_report={"unresolved_share": 1.0,
                           "missing_mechanisms": [{"mechanism": "dissolution vote"}]})


# ---- helper 1: collecting the grounded forecast across promoted models --------------------------

def test_grounded_prior_is_collected_as_forecast():
    assert is_grounded_prior(_knesset_prior())
    runs = {"m1": {"prior_spec": _knesset_prior(), "posterior": None}}
    g = _ensemble_grounded_forecast(runs, _promoted("m1"))
    assert g is not None
    assert abs(g["mean"] - _knesset_prior().mean) < 1e-6
    assert g["source"] == "grounded_prior_mean"
    assert "Knesset dissolutions" in g["reference_classes"]


def test_generic_only_prior_is_not_grounded_nap_preserved():
    # a generic weakly-informative prior (no reference class) is NOT a headline forecast (§NAP)
    gen = generic_lean_prior("neutral")
    assert not is_grounded_prior(gen)
    runs = {"m1": {"prior_spec": gen, "posterior": None}}
    assert _ensemble_grounded_forecast(runs, _promoted("m1")) is None


def test_evidence_updated_posterior_preferred_over_prior():
    post = SimpleNamespace(n_effective_observations=4, outcome_rate_mean=0.18)
    runs = {"m1": {"prior_spec": _knesset_prior(), "posterior": post}}
    g = _ensemble_grounded_forecast(runs, _promoted("m1"))
    assert g["source"] == "evidence_updated_posterior_mean"
    assert abs(g["mean"] - 0.18) < 1e-6


def test_grounded_means_averaged_across_models():
    p_low = grounded_estimate_prior("class A", 0.10, n_examples=10, evidence_quality="model_memory")
    p_high = grounded_estimate_prior("class B", 0.30, n_examples=10, evidence_quality="model_memory")
    runs = {"m1": {"prior_spec": p_low, "posterior": None},
            "m2": {"prior_spec": p_high, "posterior": None}}
    g = _ensemble_grounded_forecast(runs, _promoted("m1", "m2"))
    assert g["n_models"] == 2
    assert abs(g["mean"] - (p_low.mean + p_high.mean) / 2) < 1e-6


# ---- helper 2: serving the grounded forecast while keeping the weak status ----------------------

def test_serve_populates_forecast_keeps_under_modeled():
    res = _under_modeled_none()
    g = _ensemble_grounded_forecast({"m1": {"prior_spec": _knesset_prior(), "posterior": None}},
                                    _promoted("m1"))
    out = _serve_grounded_outside_view(res, g, "under_modeled", _plan())
    assert out.raw_probability is not None and out.raw_probability < 0.25   # ~5%
    assert out.raw_distribution                                            # non-empty binary dist
    assert out.has_forecast() is True                                     # THE invariant
    assert out.simulation_status == "under_modeled"                       # weak status KEPT as warning
    assert out.provenance["grounded_outside_view_fallback"]["used"] is True
    assert any("UNDER-MODELED FORECAST" in l for l in out.limitations)     # loud warning leads
    # the served probability round-trips through the binary projection as P(YES)
    assert out.raw_distribution.get("True") == out.raw_probability


def test_serve_retags_unresolved_to_under_modeled_with_named_gap():
    res = _unresolved_none()
    assert res.has_forecast() is False                                    # unresolved carries no forecast
    g = _ensemble_grounded_forecast({"m1": {"prior_spec": _knesset_prior(), "posterior": None}},
                                    _promoted("m1"))
    out = _serve_grounded_outside_view(res, g, "unresolved", _plan())
    assert out.simulation_status == "under_modeled"                       # retagged (not a refusal)
    assert out.under_modeled_components                                   # gap is NAMED (contract)
    assert out.has_forecast() is True
    assert out.raw_probability is not None
    assert out.provenance["grounded_outside_view_fallback"]["original_status"] == "unresolved"


def test_forecasting_never_none_when_grounded_end_to_end():
    """The exact malformed-run condition: a None-headline under_modeled result + a grounded prior on the
    promoted model. The guard's two steps together must yield a real forecast — never a silent None."""
    for factory, status in ((_under_modeled_none, "under_modeled"), (_unresolved_none, "unresolved")):
        res = factory()
        assert res.raw_probability is None
        runs = {"m1": {"prior_spec": _knesset_prior(), "posterior": None}}
        g = _ensemble_grounded_forecast(runs, _promoted("m1"))
        assert g is not None, status
        res = _serve_grounded_outside_view(res, g, status, _plan())
        assert res.raw_probability is not None, status
        assert res.has_forecast() is True, status


def test_generic_only_stays_none_no_manufactured_number():
    """§NAP counterpart: with only a generic prior the guard finds no grounded forecast, so the result
    stays a refusal — a generic ~0.5 number is never manufactured into a headline."""
    runs = {"m1": {"prior_spec": generic_lean_prior("neutral"), "posterior": None}}
    assert _ensemble_grounded_forecast(runs, _promoted("m1")) is None


# ---- top-level forecasting-mode floor: even a total structural failure never returns a silent None ----

def _stub_llm(payload):
    import json
    return lambda *a, **k: json.dumps(payload)


def _execution_failed():
    # the exact rerun failure: the ensemble collapsed to zero models before any prior was built
    return SimulationResult(
        question="Will the Knesset dissolve the committee bill by July 1?",
        simulation_status="execution_failed", failure_taxonomy="no_executable_structural_candidate",
        raw_distribution={}, raw_probability=None,
        limitations=["no executable structural candidate remained after generation, critics and repair"])


def test_floor_serves_grounded_forecast_on_execution_failed():
    from swm.world_model_v2.unified_runtime import _forecasting_mode_floor
    res = _execution_failed()
    assert res.has_forecast() is False
    grounded_payload = {"reference_class": "Knesset committee bill first readings", "is_recurrence": False,
                        "stage": "mere_proposal_or_speculation", "base_rate": 0.05,
                        "status_quo": "the bill stalls in committee", "n_examples": 20,
                        "transport_risk": "moderate", "evidence_quality": "model_memory"}
    out = _forecasting_mode_floor(res, res.question, as_of="2026-05-14", horizon="2026-07-01",
                                  llm=_stub_llm(grounded_payload))
    assert out.raw_probability is not None and out.raw_probability < 0.25    # ~5%
    assert out.has_forecast() is True                                       # THE invariant
    assert out.simulation_status == "under_modeled"                        # retagged, forecast-carrying
    gof = out.provenance["grounded_outside_view_fallback"]
    assert gof["used"] is True and gof["floor"] == "top_level_forecasting_mode"
    assert gof["original_status"] == "execution_failed"                    # failure is NAMED, not hidden
    assert gof["original_failure_taxonomy"] == "no_executable_structural_candidate"
    assert any("UNDER-MODELED FORECAST" in l for l in out.limitations)
    assert out.under_modeled_components                                     # gap named (contract)


def test_floor_refuses_when_only_generic_prior_available_nap():
    # the LLM proposes NO reference class -> generic prior -> §NAP: no number manufactured, stays failed
    from swm.world_model_v2.unified_runtime import _forecasting_mode_floor
    res = _execution_failed()
    generic_payload = {"reference_class": "", "is_recurrence": False, "stage": "unknown",
                       "base_rate": 0.5, "status_quo": "", "n_examples": 0,
                       "transport_risk": "severe", "evidence_quality": "model_memory"}
    out = _forecasting_mode_floor(res, res.question, as_of="2026-05-14", horizon="2026-07-01",
                                  llm=_stub_llm(generic_payload))
    assert out.raw_probability is None                                      # §NAP preserved
    assert out.simulation_status == "execution_failed"                     # untouched


def test_floor_leaves_a_real_forecast_untouched():
    from swm.world_model_v2.unified_runtime import _forecasting_mode_floor
    good = SimulationResult(question="Q", simulation_status="completed", support_grade="exploratory",
                            raw_distribution={"True": 0.7, "False": 0.3}, raw_probability=0.7)
    out = _forecasting_mode_floor(good, "Q", as_of="2026-05-14", horizon="2026-07-01",
                                  llm=_stub_llm({"reference_class": "x", "base_rate": 0.1, "stage": "unknown",
                                                 "is_recurrence": False, "n_examples": 5,
                                                 "transport_risk": "high", "evidence_quality": "model_memory"}))
    assert out.raw_probability == 0.7 and out.simulation_status == "completed"   # never overwrites a forecast


def test_floor_never_floors_clarification_required():
    from swm.world_model_v2.unified_runtime import _forecasting_mode_floor
    clar = SimulationResult(question="Q", simulation_status="clarification_required",
                            clarification_reason="ambiguous", raw_distribution={}, raw_probability=None)
    out = _forecasting_mode_floor(clar, "Q", as_of="2026-05-14", horizon="2026-07-01",
                                  llm=_stub_llm({"reference_class": "x", "base_rate": 0.1, "stage": "unknown",
                                                 "is_recurrence": False, "n_examples": 5,
                                                 "transport_risk": "high", "evidence_quality": "model_memory"}))
    assert out.simulation_status == "clarification_required" and out.raw_probability is None
