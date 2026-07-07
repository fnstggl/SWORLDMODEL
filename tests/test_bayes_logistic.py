"""Tests for the Bayesian weight layer: posterior weight uncertainty, integrate-out, variance triage."""
import random

from swm.variables.bayes_logistic import BayesianLogistic, variance_contribution


def _signal_noise_data(n=600, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x0 = rng.gauss(0, 1)                 # the real driver
        xn = rng.gauss(0, 1)                 # pure noise
        p = 1.0 / (1.0 + 2.718 ** (-1.5 * x0))
        X.append([x0, xn])
        y.append(1 if rng.random() < p else 0)
    return X, y


def test_weight_uncertainty_separates_signal_from_noise():
    X, y = _signal_noise_data()
    m = BayesianLogistic(l2=1.0).fit(X, y)
    rep = m.weight_report(["signal", "noise"])
    by = {r["name"]: r for r in rep}
    assert by["signal"]["weight"] > 0                       # recovers the positive driver
    assert by["signal"]["snr"] > by["noise"]["snr"]         # the model KNOWS the signal weight better
    assert all(r["sd"] > 0 for r in rep)                    # every weight carries a posterior SD


def test_predict_dist_widens_when_weights_are_uncertain():
    # tiny data -> loose posterior -> wide predictive spread; more data -> tighter
    Xs, ys = _signal_noise_data(n=40, seed=1)
    Xl, yl = _signal_noise_data(n=2000, seed=1)
    ms = BayesianLogistic(l2=1.0).fit(Xs, ys)
    ml = BayesianLogistic(l2=1.0).fit(Xl, yl)
    x = [1.5, 0.0]
    assert ms.predict_dist(x)["sd"] > ml.predict_dist(x)["sd"]   # unknown weights -> wider prediction


def test_variance_contribution_ranks_high_leverage_first():
    # feature 0 has weight+variance; feature 1 is nearly constant -> contributes ~nothing regardless of weight
    X = [[random.Random(i).gauss(0, 1), 0.5] for i in range(200)]
    tri = variance_contribution(X, [1.0, 5.0])
    assert tri[0][0] == 0                                    # the varying feature dominates outcome variance
