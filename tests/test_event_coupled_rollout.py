"""Tests for the event-coupled population rollout helpers (EXP-046)."""
from experiments.exp046_event_coupled_rollout import _slope


def test_slope_positive_trend():
    assert _slope([(2000, 0.2), (2002, 0.3), (2004, 0.4)]) > 0


def test_slope_flat_is_zero():
    assert abs(_slope([(2000, 0.5), (2002, 0.5), (2004, 0.5)])) < 1e-9


def test_slope_magnitude():
    # +0.1 per 2 years -> slope 0.05/yr
    assert abs(_slope([(2000, 0.0), (2002, 0.1), (2004, 0.2)]) - 0.05) < 1e-6


def test_slope_single_point_safe():
    assert _slope([(2000, 0.5)]) == 0.0
    assert _slope([]) == 0.0
