"""Tests for inner-crowd aggregation + GDELT index aggregation (deterministic, no network)."""
from swm.api.inner_crowd import aggregate
from swm.retrieval.gdelt_index import _summ


def test_aggregate_logodds_and_extremize():
    # a panel leaning YES: log-odds mean is > 0.5; extremizing (>1) sharpens it further
    panel = [0.6, 0.7, 0.55, 0.65]
    a = aggregate(panel)
    assert 0.5 < a < 0.75
    assert aggregate(panel, extremize=2.0) > a                # extremization sharpens toward the shared signal
    assert aggregate([0.5, 0.5, 0.5]) == 0.5                  # a tied panel stays at 0.5
    assert aggregate([None, 0.8, None]) == 0.8                # ignores missing forecasters
    assert aggregate([]) is None


def test_aggregate_median_is_robust_to_outlier():
    panel = [0.55, 0.6, 0.58, 0.99]                           # one wild optimist
    assert aggregate(panel, method="median") < aggregate(panel)   # median ignores the outlier, mean is pulled up


def test_gdelt_summary_normalizes_rates():
    b = {"n": 100, "tone_sum": -150.0, "gold_sum": 40.0, "protest": 5, "assault": 3, "fight": 2,
         "mass_violence": 1, "consult": 10, "cooperate_diplo": 8}
    s = _summ(b)
    assert s["tone"] == -1.5 and s["goldstein"] == 0.4
    assert s["protest_rate"] == 0.05 and s["violence_rate"] == 0.06 and s["diplomacy_rate"] == 0.18
