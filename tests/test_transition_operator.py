"""Tests for the calibrated transition operator — recover KNOWN dynamics from synthetic trajectories, and
confirm the honest null: a random walk collapses back to persistence (no hallucinated drift)."""
import random

from swm.simulation.transition_operator import (TransitionOperator, persistence_rollout,
                                                quadratic_self_basis)


def _mse(pred, truth):
    return sum((p - t) ** 2 for p, t in zip(pred, truth)) / len(truth)


def test_recovers_mean_reversion_and_beats_persistence_at_horizon():
    # AR(1): x_{t+1} = 0.7 x_t + eps  =>  Δx = -0.3 x  (mean-reverting to 0)
    rng = random.Random(0)
    series = []
    x = 0.0
    for _ in range(1200):
        x = 0.7 * x + rng.gauss(0, 1)
        series.append({"x": x})
    tr, te = series[:900], series[900:]
    op = TransitionOperator(names=["x"]).fit([tr])
    a_diag = op.coupling_report()["coupling"]["x"]["x"]
    assert -0.55 < a_diag < -0.12                         # recovers the negative (mean-reverting) coefficient

    # multi-step: from each test origin, forecast x 8 steps ahead; the operator (reverts to 0) should beat
    # persistence (holds the current, possibly-extreme value) in aggregate MSE.
    H = 8
    op_err = pers_err = 0.0
    k = 0
    for i in range(len(te) - H):
        start = {"x": te[i]["x"]}
        truth = te[i + H]["x"]
        op_pred = op.rollout(start, H, n=400, seed=i)["x"]["mean"]
        pers_pred = persistence_rollout(start, ["x"], H)["x"]["mean"]
        op_err += (op_pred - truth) ** 2
        pers_err += (pers_pred - truth) ** 2
        k += 1
    assert op_err / k < pers_err / k                      # calibrated dynamics beat persistence at horizon


def test_random_walk_collapses_to_persistence():
    # x_{t+1} = x_t + eps  =>  no learnable drift; the EB temper must shrink A toward 0 (persistence null)
    rng = random.Random(1)
    series = []
    x = 0.0
    for _ in range(600):
        x += rng.gauss(0, 1)
        series.append({"x": x})
    op = TransitionOperator(names=["x"]).fit([series])
    a_diag = op.coupling_report()["coupling"]["x"]["x"]
    assert abs(a_diag) < 0.05                             # essentially persistence — no hallucinated dynamics


def test_recovers_cross_variable_coupling_sign():
    # x1 is an AR(1); x2 is pushed UP by the level of x1:  Δx2 = +0.5 x1 + eps
    rng = random.Random(2)
    series = []
    x1 = x2 = 0.0
    for _ in range(1500):
        x1 = 0.8 * x1 + rng.gauss(0, 1)
        x2 = x2 + 0.5 * x1 + rng.gauss(0, 0.3)
        series.append({"x1": x1, "x2": x2})
    op = TransitionOperator(names=["x1", "x2"]).fit([series])
    coup = op.coupling_report()["coupling"]
    assert coup["x2"]["x1"] > 0.25                        # Δx2 responds POSITIVELY to x1 (recovered coupling)
    assert abs(coup["x1"]["x2"]) < 0.2                    # x2 does NOT drive x1 (no spurious reverse coupling)


def test_rollout_interval_is_calibrated():
    # AR(1) with known innovation sd; the rolled-out 90% interval should cover truth near 90%
    rng = random.Random(3)
    series = []
    x = 0.0
    for _ in range(1500):
        x = 0.6 * x + rng.gauss(0, 1.0)
        series.append({"x": x})
    tr, te = series[:1000], series[1000:]
    op = TransitionOperator(names=["x"]).fit([tr])
    H = 5
    covered = tot = 0
    for i in range(len(te) - H):
        r = op.rollout({"x": te[i]["x"]}, H, n=600, seed=i)["x"]
        truth = te[i + H]["x"]
        covered += 1 if r["p05"] <= truth <= r["p95"] else 0
        tot += 1
    assert 0.80 <= covered / tot <= 0.99                  # honest predictive interval (nominal 90%)


def test_quadratic_basis_learns_saturating_growth():
    # logistic diffusion Δx = 0.3 x (1 - x); a quadratic self-basis should learn upward drift in mid-curve
    rng = random.Random(4)
    series = []
    x = 0.02
    for _ in range(400):
        x = min(0.999, max(0.001, x + 0.3 * x * (1 - x) + rng.gauss(0, 0.005)))
        series.append({"x": x})
    op = TransitionOperator(names=["x"], basis=quadratic_self_basis, los=[0.0], his=[1.0]).fit([series])
    # from the steep middle of the S-curve, the operator must predict a RISE (persistence would stay flat)
    fwd = op.rollout({"x": 0.3}, 6, n=500, seed=0)["x"]["mean"]
    assert fwd > 0.3 + 0.02                               # learned the climb, not a flat random walk


def test_fit_is_deterministic():
    rng = random.Random(5)
    series = [{"x": rng.gauss(0, 1)} for _ in range(200)]
    a = TransitionOperator(names=["x"]).fit([series]).coef
    b = TransitionOperator(names=["x"]).fit([series]).coef
    assert a == b
