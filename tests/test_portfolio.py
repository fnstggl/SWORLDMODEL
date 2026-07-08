"""Tests for the calibrated_readout mechanism (weight-uncertainty integration) and the portfolio harness."""
from swm.api.compiler import CompiledModel
from swm.api.model_spec import parse_spec
from swm.eval.event_backtest import Question
from swm.eval.portfolio import Domain, run_portfolio


def test_calibrated_readout_integrates_weight_uncertainty():
    # same variable value; a wider weight CI must widen the outcome interval (unknown weight -> wider forecast)
    def spec(weight_sd):
        return parse_spec({"mechanism": "calibrated_readout", "extra": {"intercept": 0.0},
                           "variables": [{"name": "v", "value": 0.9, "weight": 1.5, "weight_sd": weight_sd}],
                           "outcome": {"event": {"op": ">", "value": 0.5}}})
    tight = CompiledModel(spec(0.05)).run(n=4000)
    loose = CompiledModel(spec(2.0)).run(n=4000)
    w = lambda o: o["interval_80"][1] - o["interval_80"][0]
    assert w(loose) > w(tight)


def test_portfolio_maps_fidelity_and_baselines():
    # a synthetic domain: low fidelity = the base-rate; high fidelity = a near-perfect forecaster
    outcomes = [float(i % 2) for i in range(30)]
    def build(fid):
        qs = [Question(f"q{i}", o, {"persistence": 0.5, "base_rate": 0.5}) for i, o in enumerate(outcomes)]
        if fid == "few":
            return qs, (lambda q: 0.5)
        return qs, (lambda q: 0.95 if q.outcome == 1 else 0.05)
    port = run_portfolio([Domain("toy", build, ("few", "full"))], check_asof=False)
    m = port["map"]["toy"]
    assert m["fidelity_helps"] is True and m["beats_all_baselines"] is True
    assert m["high_fidelity_skill"] > m["low_fidelity_skill"]
