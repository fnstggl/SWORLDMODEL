"""Tests for the live post-mortem loop (Tetlock #8/#10 — leakage-free skill + do-no-harm recalibration)."""
import random

from swm.eval.postmortem import PostMortemLog


def _stream(n, transform, seed=0):
    rng = random.Random(seed)
    log = PostMortemLog()
    for i in range(n):
        p_true = rng.random()
        outcome = int(rng.random() < p_true)
        log.log(f"f-{i}", transform(p_true), made_at=i, resolves_at=i + 1)
        log.resolve(f"f-{i}", outcome)
    return log


def test_skill_needs_minimum_resolved():
    log = PostMortemLog()
    log.log("a", 0.6, made_at=0, resolves_at=1)
    log.resolve("a", 1)
    assert log.skill()["n"] == 1
    assert "note" in log.skill()                            # too few to score


def test_skill_is_leakage_free_by_construction():
    log = PostMortemLog()
    # a forecast whose made_at is NOT before its resolves_at must be excluded from skill
    log.log("leak", 0.9, made_at=5, resolves_at=5)          # made == resolves -> not leakage-free
    log.resolve("leak", 1)
    for i in range(12):
        log.log(f"ok-{i}", 0.6, made_at=i, resolves_at=i + 10)
        log.resolve(f"ok-{i}", 1)
    s = log.skill()
    assert s["leakage_free"] is True
    assert s["n"] == 12                                     # the leaky forecast is dropped


def test_skill_reports_proper_metrics():
    log = _stream(100, lambda p: p)                         # a well-calibrated forecaster
    s = log.skill()
    assert s["n"] == 100
    assert 0.0 <= s["brier"] <= 0.25
    assert 0.0 <= s["ece"] <= 1.0
    assert "log_loss" in s and "directional_accuracy" in s


def test_recalibration_fixes_underconfidence():
    # a forecaster that hedges toward 0.5; the loop should recover calibration on held-out data
    under = lambda p: 0.5 + 0.6 * (p - 0.5)
    log = _stream(400, under)
    cut = 200
    raw = log.skill(before=cut)                             # first-half calibration (as-of)
    log.fit_recalibration(cut)
    assert log._platt is not None                           # enough history -> deploys
    # evaluate on the second half (resolved at/after cut), raw vs recalibrated
    late_raw = log.skill(recalibrated=False)
    late_cal = log.skill(recalibrated=True)
    assert late_cal["ece"] < late_raw["ece"]               # recalibration improved calibration


def test_recalibration_does_no_harm_on_small_history():
    log = _stream(24, lambda p: p)                          # too little history for a trustworthy guard
    log.fit_recalibration(12)
    assert log._platt is None                               # abstains rather than deploy a noisy map
    assert log.recalibrate(0.7) == 0.7                     # identity


def test_log_forecast_accepts_engine_outputs():
    from dataclasses import dataclass

    @dataclass
    class _QF:                                             # mimics QuestionForecast
        p_outcome: float
        direction: int = 1
        confidence: float = 0.6

    @dataclass
    class _Pred:                                           # mimics Simulator's Prediction
        p: float
        confidence: float = 0.5
        regime: str = "inference_driven"

    log = PostMortemLog()
    log.log_forecast("qf", _QF(0.7), made_at=0, resolves_at=1)
    log.log_forecast("pr", _Pred(0.3), made_at=0, resolves_at=1)
    log.log_forecast("raw", 0.55, made_at=0, resolves_at=1)
    assert log.forecasts["qf"]["p"] == 0.7
    assert log.forecasts["qf"]["meta"]["confidence"] == 0.6   # carries engine metadata through
    assert log.forecasts["pr"]["p"] == 0.3
    assert log.forecasts["pr"]["meta"]["regime"] == "inference_driven"
    assert log.forecasts["raw"]["p"] == 0.55


def test_recalibration_does_no_harm_when_already_calibrated():
    log = _stream(400, lambda p: p)                         # already calibrated -> nothing to fix
    log.fit_recalibration(200)
    # either it abstains, or any deployed map does not worsen calibration on held-out
    raw = log.skill(recalibrated=False)
    cal = log.skill(recalibrated=True)
    assert cal["ece"] <= raw["ece"] + 0.03
