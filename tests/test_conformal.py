"""Tests for split-conformal prediction sets — the finite-sample coverage guarantee."""
import random

from swm.uncertainty.conformal import ConformalBinary


def test_coverage_guarantee_holds_on_calibrated_model():
    """With a well-calibrated probability model, empirical coverage should meet the 1 - alpha target."""
    rng = random.Random(0)
    # a calibrated generator: p is the true probability, y ~ Bernoulli(p)
    cal_p = [rng.random() for _ in range(2000)]
    cal_y = [int(rng.random() < p) for p in cal_p]
    test_p = [rng.random() for _ in range(2000)]
    test_y = [int(rng.random() < p) for p in test_p]
    for alpha in (0.1, 0.2):
        cf = ConformalBinary(alpha=alpha).fit(cal_p, cal_y)
        cov = cf.coverage(test_p, test_y)
        # split-conformal guarantees marginal coverage >= 1 - alpha up to finite-sample slack
        assert cov["coverage"] >= (1 - alpha) - 0.04
        assert 1 <= cov["avg_set_size"] <= 2


def test_confident_model_yields_singletons():
    """A sharp, correct model should produce mostly singleton sets (small avg size)."""
    rng = random.Random(1)
    # near-deterministic: p close to 0 or 1 and correct
    cal_p, cal_y = [], []
    for _ in range(1000):
        y = rng.randint(0, 1)
        cal_p.append(0.97 if y else 0.03); cal_y.append(y)
    cf = ConformalBinary(alpha=0.1).fit(cal_p, cal_y)
    assert cf.predict_set(0.97) == [1]
    assert cf.predict_set(0.03) == [0]
    cov = cf.coverage(cal_p, cal_y)
    assert cov["avg_set_size"] < 1.2               # confident model -> tight sets


def test_uncertain_prediction_returns_both_labels():
    cf = ConformalBinary(alpha=0.1)
    cf.q = 0.6                                      # threshold that admits both labels near 0.5
    assert set(cf.predict_set(0.5)) == {0, 1}
