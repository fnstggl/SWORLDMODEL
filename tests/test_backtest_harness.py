"""Deterministic tests for the backtest infrastructure (pure logic — no network, no LLM)."""
import math

from swm.api.model_spec import ModelSpec, SpecVar
from swm.eval.backtest_harness import _apply_toggles, _p_from_forecast, score
from swm.eval.forecasting_corpus import _cat


def test_category_tagging():
    assert _cat("Will Trump win the 2024 presidential election?") == "election"
    assert _cat("Will Bitcoin cross $100k?") == "crypto"
    assert _cat("Will the US enter a recession?") == "economy"
    assert _cat("Will my friend text me back?") == "other"


def test_p_from_forecast_prefers_event_then_target_then_mean():
    assert _p_from_forecast({"p_event": 0.7, "mean": 0.3}) == 0.7
    assert _p_from_forecast({"p_target": 0.4}) == 0.4
    assert _p_from_forecast({"mean": 0.55}) == 0.55
    assert _p_from_forecast({"mean": 42.0}) is None            # out of [0,1] -> not a probability


def _readout(vals_weights):
    vs = [SpecVar(name=n, value=v, est_sd=0.0, weight=w, weight_sd=0.0, center=0.5, lo=0.0, hi=1.0)
          for n, v, w in vals_weights]
    return ModelSpec(mechanism="calibrated_readout", variables=vs,
                     outcome={"event": {"op": ">", "value": 0.5}}, extra={"intercept": 0.0})


def test_apply_toggles_ungrounds_and_limits_vars():
    spec = _readout([("a", 0.9, 3.0), ("b", 0.8, 2.0), ("c", 0.5, 0.01)])
    ung = _apply_toggles(_readout([("a", 0.9, 3.0), ("b", 0.8, 2.0)]), ground=False)
    assert all(v.value == v.center for v in ung.variables)     # neutralized to center (no state info)
    top1 = _apply_toggles(_readout([("a", 0.9, 3.0), ("b", 0.8, 2.0), ("c", 0.5, 0.01)]), max_vars=1)
    zeroed = [v for v in top1.variables if v.weight == 0.0]
    assert len(zeroed) == 2                                     # only the top-leverage var keeps its weight


def test_score_skill_vs_crowd_and_base():
    # model perfectly calibrated, crowd at coin-flip -> model should show positive skill vs crowd
    rows = [{"outcome": 1, "p_model": 0.9, "p_crowd": 0.5, "p_direct": 0.8, "category": "x", "cutoff_clean": True},
            {"outcome": 0, "p_model": 0.1, "p_crowd": 0.5, "p_direct": 0.2, "category": "x", "cutoff_clean": True},
            {"outcome": 1, "p_model": 0.8, "p_crowd": 0.5, "p_direct": 0.7, "category": "y", "cutoff_clean": False},
            {"outcome": 0, "p_model": 0.2, "p_crowd": 0.5, "p_direct": 0.3, "category": "y", "cutoff_clean": False}]
    res = score(rows)
    assert res["overall"]["skill_vs_crowd"] > 0                 # a sharp, correct model beats the coin-flip crowd
    assert res["clean"]["n"] == 2 and res["by_category"]["x"]["n"] == 2
    assert res["overall"]["direct_skill_vs_crowd"] is not None  # leakage meter present when p_direct given


def test_temperature_scaling_fixes_overconfidence():
    from experiments.exp090_ablation import _temperature, _sig, _logit
    # overconfident predictions (0.98 when true rate is ~0.6) -> fitted temperature should be < 1 (shrink)
    ps = [0.98, 0.02, 0.98, 0.02, 0.98, 0.98, 0.02, 0.98]
    ys = [1, 0, 0, 0, 1, 1, 1, 1]                              # 0.98s are only ~70% right -> overconfident
    lam = _temperature(ps, ys)
    assert lam < 1.0
    assert _sig(lam * _logit(0.98)) < 0.98                     # calibrated prob is pulled in from the extreme
