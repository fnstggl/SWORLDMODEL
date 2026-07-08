"""Tests for the full calibration engine: priors-with-CIs, empirical-Bayes temper, integrate-out, triage, active learning."""
import random

from swm.variables.calibrated_weights import (CalibratedWeights, effect_size_prior, empirical_bayes_temper,
                                              llm_elasticity_prior, uninformative_prior)


def _signal_noise(n, seed):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        s = rng.gauss(0, 1)
        noise = [rng.gauss(0, 1) for _ in range(4)]
        p = 1.0 / (1.0 + 2.718 ** (-1.6 * s))
        X.append([s] + noise)
        y.append(1 if rng.random() < p else 0)
    return X, y


def test_priors_carry_confidence_intervals():
    tight = effect_size_prior("x", 0.8, ci95=0.1)
    loose = llm_elasticity_prior("y", 0.8, ci95=2.0)
    assert tight.precision() > loose.precision()          # a tighter CI => a stronger (higher-precision) prior
    assert tight.source == "literature" and loose.source == "llm"


def test_calibrated_recovers_signal_and_shrinks_noise():
    X, y = _signal_noise(600, 0)
    priors = [uninformative_prior("signal")] + [uninformative_prior(f"noise{j}") for j in range(4)]
    cw = CalibratedWeights(priors).fit(X, y, tune=True)
    rep = {r["name"]: r for r in cw.weight_report()}
    assert rep["signal"]["weight"] > 0 and rep["signal"]["snr"] > 1.0        # signal weight pinned down
    assert rep["signal"]["snr"] > max(rep[f"noise{j}"]["snr"] for j in range(4))  # better known than noise
    assert cw.temper in cw.temper_grid                    # empirical-Bayes chose a shrinkage level


def test_predict_dist_integrates_weight_uncertainty():
    Xs, ys = _signal_noise(40, 1)
    Xl, yl = _signal_noise(1500, 1)
    priors = [uninformative_prior("s")] + [uninformative_prior(f"n{j}") for j in range(4)]
    cs = CalibratedWeights(priors).fit(Xs, ys, tune=False)
    cl = CalibratedWeights(priors).fit(Xl, yl, tune=False)
    x = [1.5, 0, 0, 0, 0]
    assert cs.predict_dist(x)["sd"] > cl.predict_dist(x)["sd"]   # less data -> wider (unknown weight widens)


def test_active_learning_points_at_high_leverage_uncertain_weight():
    X, y = _signal_noise(120, 2)
    priors = [uninformative_prior("signal")] + [uninformative_prior(f"noise{j}") for j in range(4)]
    cw = CalibratedWeights(priors).fit(X, y, tune=False)
    targets = cw.active_learning_targets(X)
    # the signal has the leverage; with modest data its weight is still worth sharpening -> ranks high
    assert targets[0][0] == "signal"


def test_empirical_bayes_returns_a_grid_value():
    X, y = _signal_noise(300, 3)
    t = empirical_bayes_temper(X, y, [0.0] * 5, [1.0] * 5, grid=(0.5, 1.0, 4.0), seed=0)
    assert t in (0.5, 1.0, 4.0)
