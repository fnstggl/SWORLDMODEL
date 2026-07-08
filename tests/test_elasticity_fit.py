"""Elasticity-fitting harness: recover weights, earn a grade, and carry it into the optimizer.

Validated on synthetic data with a KNOWN ground-truth elasticity model (the estimator-validation stance
the repo uses when real outcome logs are private).
"""
import statistics

from swm.decision.elasticity_fit import (fit_elasticities, grade_fit, synthetic_reply_dataset, _predict)
from swm.decision.message_pipeline import RecipientState, optimize_message
from swm.eval.metrics import log_loss

SKEPTIC = {"status_orientation": 0.85, "skepticism": 0.9, "status": 0.9, "openness_to_outreach": 0.9,
           "attention_availability": 0.4, "platform_response_norm": 0.3, "relationship_strength": 0.0}


def _corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return cov / den if den else 0.0


def test_fit_recovers_ground_truth_and_grades_well():
    data, truth = synthetic_reply_dataset(2000, seed=3)
    fit = grade_fit(data, split=0.7, temporal=True)
    assert fit.grade["grade"] in ("A", "B")               # calibrated on held-out synthetic data
    assert fit.grade["ece"] < 0.10
    names = list(truth)
    corr = _corr([fit.weights[n][0] for n in names], [truth[n] for n in names])
    assert corr > 0.8                                     # recovers the ground-truth elasticities


def test_prior_helps_on_thin_data():
    data, _ = synthetic_reply_dataset(2200, seed=5)
    thin, test = data[:70], data[1800:2100]
    y = [o for *_, o in test]
    ll_prior = log_loss(y, [_predict(fit_elasticities(thin, use_prior=True), r, s, b) for r, s, b, _ in test])
    ll_ridge = log_loss(y, [_predict(fit_elasticities(thin, use_prior=False), r, s, b) for r, s, b, _ in test])
    assert ll_prior <= ll_ridge + 0.03                    # world-knowledge prior doesn't hurt (usually helps)


def test_fit_is_deterministic():
    data, _ = synthetic_reply_dataset(800, seed=7)
    assert grade_fit(data, split=0.7).grade == grade_fit(data, split=0.7).grade


def test_too_little_data_is_graded_F_not_crash():
    data, _ = synthetic_reply_dataset(20, seed=1)
    fit = grade_fit(data)
    assert fit.grade["grade"] == "F" and "error" in fit.grade


def test_fitted_scorer_carries_grade_into_optimizer():
    data, _ = synthetic_reply_dataset(2000, seed=2)
    fit = grade_fit(data, split=0.7)
    rs = RecipientState(vars=SKEPTIC, base_mean=0.2, base_n_effective=6.0, label="Skeptic")
    # priors -> unvalidated
    plain = optimize_message(rs, n_mc=600, seed=0).summary()
    assert plain["calibration_grade"] == "unvalidated"
    # fitted -> real grade
    graded = optimize_message(rs, fit=fit, n_mc=600, seed=0).summary()
    assert graded["calibration_grade"] in ("A", "B", "C")
    assert "CALIBRATED" in graded["honesty"]


def test_scorer_for_produces_a_graded_scorer():
    data, _ = synthetic_reply_dataset(1200, seed=4)
    fit = grade_fit(data, split=0.7)
    sc = fit.scorer_for(SKEPTIC, 0.2)
    assert sc.weights is not None and sc.grade["grade"] in ("A", "B", "C")
