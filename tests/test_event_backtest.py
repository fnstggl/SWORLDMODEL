"""Tests for the no-cheat event backtest: leakage guard, skill vs baselines."""
import pytest

from swm.eval.event_backtest import Question, assert_asof, backtest


def test_asof_guard_rejects_leakage():
    good = [Question("a", 1.0, {"base_rate": 0.5}, asof="2020-01", resolved="2020-06")]
    assert assert_asof(good)
    bad = [Question("b", 1.0, {"base_rate": 0.5}, asof="2020-06", resolved="2020-01")]   # evidence AFTER outcome
    with pytest.raises(ValueError):
        assert_asof(bad)


def test_skill_positive_when_model_beats_baseline():
    # binary events; persistence baseline is a coin flip, the model is confidently right
    qs = [Question(f"e{i}", float(i % 2), {"persistence": 0.5}) for i in range(20)]
    def perfect(q):
        return 0.98 if q.outcome == 1 else 0.02
    out = backtest(qs, perfect, check_asof=False)
    assert out["metric"] == "log_loss"
    assert out["skill_vs"]["persistence"] > 0.5           # large positive skill over the coin-flip baseline
    assert out["beats_all_baselines"] is True
    assert out["winrate_vs"]["persistence"] == 1.0


def test_continuous_share_uses_mae_and_reports_no_skill_when_tied():
    # a continuous share; the model just echoes persistence -> zero skill
    qs = [Question(f"s{i}", 0.5 + 0.01 * i, {"persistence": 0.5 + 0.01 * i}) for i in range(10)]
    out = backtest(qs, lambda q: q.baselines["persistence"], check_asof=False)
    assert out["metric"] == "mae"
    assert abs(out["skill_vs"]["persistence"]) < 1e-6     # tied with persistence => ~zero skill
