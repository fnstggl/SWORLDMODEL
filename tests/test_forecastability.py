"""Tests for the forecastability / triage score (Tetlock #1 — say what you can forecast)."""
from swm.eval.forecastability import (FEATURE_NAMES, ForecastabilityScorer, forecastability_features)


def test_features_shape_and_bounds():
    f = forecastability_features(lean=0.3, volatility=0.05, days_to_res=30, result_cue=0.5)
    assert len(f) == len(FEATURE_NAMES)
    assert all(0.0 <= x <= 1.0 for x in f)


def test_features_monotone_in_lean_and_saturate():
    lo = forecastability_features(0.05, 0.05)[0]
    hi = forecastability_features(0.45, 0.05)[0]
    assert hi > lo                                         # bigger lean -> higher abs_lean feature
    assert forecastability_features(0.0, 5.0)[1] == 1.0    # volatility saturates at 1


def test_days_to_res_none_defaults_high():
    # unknown horizon should not be treated as imminent-resolution (which would over-boost the score)
    f = forecastability_features(0.2, 0.05, days_to_res=None)
    assert f[2] == 1.0


def test_scorer_learns_that_confident_calls_are_reliable():
    # confident-lean examples are usually correct; near-0.5 examples are coin flips
    examples = []
    for i in range(60):
        examples.append((forecastability_features(0.4, 0.03), 1))       # confident + right
        examples.append((forecastability_features(0.02, 0.09), i % 2))  # unsure + 50/50
    sc = ForecastabilityScorer().fit(examples)
    assert sc.score(forecastability_features(0.4, 0.03)) > sc.score(forecastability_features(0.02, 0.09))


def test_triage_buckets_high_low():
    examples = []
    for i in range(60):
        examples.append((forecastability_features(0.45, 0.02), 1))
        examples.append((forecastability_features(0.01, 0.1), i % 2))
    sc = ForecastabilityScorer().fit(examples)
    assert sc.triage(forecastability_features(0.45, 0.02)) == "forecast"
    assert sc.triage(forecastability_features(0.01, 0.1)) in ("hedge", "abstain")


def test_unfit_scorer_is_neutral():
    sc = ForecastabilityScorer()
    assert sc.score(forecastability_features(0.3, 0.05)) == 0.5
