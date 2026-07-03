"""Tests for the proper scoring rules (audit E.2). The evaluator must be trustworthy first."""
import math

import pytest

from swm.eval.metrics import (
    base_rate,
    brier_score,
    crps_ensemble,
    expected_calibration_error,
    log_loss,
    uplift_at_k,
)


def test_log_loss_perfect_vs_wrong():
    assert log_loss([1, 0], [1.0, 0.0]) < 1e-6
    # confidently wrong is heavily penalized
    assert log_loss([1], [0.001]) > log_loss([1], [0.4])


def test_brier_bounds():
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_ece_perfectly_calibrated_is_low():
    # 100 items, half at p=0.0 all-negative, half at p=1.0 all-positive -> ECE ~ 0
    y = [0] * 50 + [1] * 50
    p = [0.0] * 50 + [1.0] * 50
    assert expected_calibration_error(y, p, n_bins=10) < 1e-6


def test_ece_detects_overconfidence():
    # model says 0.9 for everyone but only 50% actually happen -> ECE ~ 0.4
    y = [1] * 50 + [0] * 50
    p = [0.9] * 100
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.4, abs=0.05)


def test_uplift_positive_when_ranking_is_informative():
    # score correlates with outcome -> top-k rate exceeds base rate
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    score = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert uplift_at_k(y, score, k=0.3) > 0
    assert base_rate(y) == pytest.approx(0.3)


def test_uplift_zero_when_ranking_is_random_ish():
    # constant score -> top-k is just the first items; with alternating labels ~ base rate
    y = [1, 0] * 5
    score = [0.5] * 10
    assert abs(uplift_at_k(y, score, k=0.2)) <= 0.5


def test_crps_smaller_when_samples_center_on_obs():
    near = crps_ensemble(0.0, [-0.1, 0.0, 0.1])
    far = crps_ensemble(0.0, [4.9, 5.0, 5.1])
    assert near < far
    assert math.isfinite(near)
