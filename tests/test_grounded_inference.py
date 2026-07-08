"""Unit tests for the grounded-inference engine (the three pillars) — pure functions, no network."""
from swm.variables.grounded_inference import (ensemble_infer, reference_prior, fit_calibration,
                                              apply_calibration, shrink, grounded_estimate)


def test_ensemble_spread_reflects_agreement():
    agree = ensemble_infer(lambda p: "0.5", "prompt", float, k=3)
    assert agree[0] == 0.5 and agree[1] == 0.0            # identical samples -> zero spread (tight bars)


def test_ensemble_handles_unparseable():
    out = ensemble_infer(lambda p: "not a number", "prompt", lambda s: None, k=3)
    assert out is None


def test_reference_prior_fallback():
    br = {"200": (0.45, 0.2)}
    assert reference_prior(br, "200", (0.0, 0.4)) == (0.45, 0.2)
    assert reference_prior(br, "999", (0.0, 0.4)) == (0.0, 0.4)   # unknown class -> fallback


def test_calibration_recovers_linear_map():
    raw = [0.0, 1.0, 2.0, 3.0]
    truth = [1.0, 3.0, 5.0, 7.0]                          # truth = 2*raw + 1
    a, b, rmse = fit_calibration(raw, truth)
    assert abs(a - 2.0) < 1e-6 and abs(b - 1.0) < 1e-6 and rmse < 1e-6
    assert abs(apply_calibration((a, b), 4.0) - 9.0) < 1e-6


def test_shrink_pulls_uncertain_estimates_to_prior():
    # a very uncertain estimate collapses toward the base rate; a confident one barely moves
    loose, _ = shrink(1.0, est_sd=10.0, prior_mean=0.0, prior_sd=0.2)
    tight, _ = shrink(1.0, est_sd=0.01, prior_mean=0.0, prior_sd=0.2)
    assert abs(loose - 0.0) < abs(tight - 0.0)
    assert tight > 0.9                                    # confident estimate keeps its deviation


def test_grounded_estimate_uses_base_rate_when_no_llm():
    est = grounded_estimate(llm_mean=None, llm_spread=None, cal=None, class_prior=(0.3, 0.2))
    assert est.value == 0.3 and est.provenance == "base_rate_only"


def test_grounded_estimate_calibrates_then_shrinks():
    # LLM says 0.8 on a compressed scale; calibration (truth=1.2*raw) expands it, then it shrinks toward 0.4
    est = grounded_estimate(llm_mean=0.8, llm_spread=0.05, cal=(1.2, 0.0, 0.1), class_prior=(0.4, 0.3))
    assert 0.4 < est.value < 0.96                         # between the base rate and the calibrated estimate
    assert est.sd > 0
