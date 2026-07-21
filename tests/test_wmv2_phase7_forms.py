"""Phase 7 — structural forms, safety, pooling, structural uncertainty, applicability, composition."""
import math
import pytest

from swm.world_model_v2.nonlinear import forms, safety, pooling
from swm.world_model_v2.nonlinear import structural_uncertainty as su
from swm.world_model_v2.nonlinear import applicability as ap
from swm.world_model_v2.nonlinear import composition as comp


def test_registry_has_core_forms():
    have = set(forms.list_forms())
    required = {"linear", "logistic", "cloglog_hazard", "hill", "threshold_hard", "threshold_smooth",
                "fatigue", "habituation", "hysteresis", "change_point", "finite_mixture", "regime_model",
                "survival_hazard", "self_exciting", "finite_population", "gam", "logistic_growth"}
    assert required <= have, required - have


def test_every_form_evaluates_and_is_finite():
    inp = {"x": 1.5, "features": {"a": 1.0, "cum_count": 2.0, "tenure": 3.0}, "n_exposures": 2.0,
           "time_since_last": 5.0, "window_days": 1.0, "active": 0, "cum_count": 2.0,
           "infected": 3.0, "susceptible": 5.0, "kernel_sum": 1.0, "regime": "0",
           "regime_posterior": {"0": 1.0}}
    dummy = {"form_id": "linear", "params": {"weights": {}, "intercept": 0.1}, "weight": 1.0}
    params = {
        "linear": {"weights": {"a": 1.0}, "intercept": 0.0},
        "logistic": {"weights": {"a": 1.0}, "intercept": 0.0},
        "cloglog_hazard": {"weights": {"a": 0.1}, "intercept": -2.0},
        "piecewise_linear": {"knots": [0, 1, 2], "values": [0, 1, 0]},
        "monotonic_spline": {"x": [0, 1, 2], "y": [0, 0.5, 1]},
        "cubic_spline": {"x": [0, 1, 2], "segments": [{"a": 0, "b": 1, "c": 0, "d": 0},
                                                      {"a": 1, "b": 1, "c": 0, "d": 0}]},
        "threshold_hard": {"tau": 1.0, "low": 0, "high": 1}, "threshold_smooth": {"tau": 1.0, "s": 0.5},
        "hill": {"theta": 1.0, "n": 2.0, "k": 1.0}, "michaelis_menten": {"Vmax": 1.0, "K": 1.0},
        "logistic_saturation": {"L": 1.0, "k": 1.0, "x0": 1.0}, "exp_saturation": {"L": 1.0, "k": 1.0},
        "inverted_u": {"peak": 1.0, "spread": 1.0, "height": 1.0}, "u_shaped": {"a": 1.0, "m": 1.0, "c": 0.0},
        "fatigue": {"A": 1.0, "gamma": 0.5}, "habituation": {"A": 1.0, "lam": 1.0},
        "refractory": {"A": 1.0, "rho": 2.0}, "hysteresis": {"tau_up": 2.0, "tau_down": 1.0},
        "change_point": {"cp": 1.0, "a0": 0, "b0": 1, "a1": 0, "b1": 2},
        "finite_mixture": {"components": [dict(dummy)]},
        "regime_model": {"regimes": {"0": {"form_id": "linear", "params": {"weights": {}, "intercept": 1.0}}}},
        "hmm_regime": {"regimes": {"0": {"form_id": "linear", "params": {"weights": {}, "intercept": 1.0}}}},
        "mixture_of_experts": {"gate": {"e0": {"weights": {}, "intercept": 0.0}},
                               "experts": {"e0": {"form_id": "linear", "params": {"weights": {},
                                                                                  "intercept": 1.0}}}},
        "survival_hazard": {"log_lambda": -1.0, "weights": {"a": 0.1}},
        "recurrent_event_hazard": {"lam0": 0.1, "beta_count": 0.1, "beta_recency": 0.1},
        "self_exciting": {"mu": 0.1, "alpha": 0.5, "omega": 0.01},
        "self_inhibiting": {"mu": 0.1, "beta": 0.5, "omega": 0.01},
        "finite_population": {"beta": 0.1, "N": 10.0},
        "nonlinear_state_space": {"drift": {"form_id": "linear", "params": {"weights": {}, "intercept": 0.1}},
                                  "dt": 1.0},
        "gam": {"linear": {"a": 1.0}, "smooth": {"tenure": {"knots": [1, 2], "coefs": [0.1, 0.2, 0.3]}},
                "intercept": 0.0},
        "logistic_growth": {"r": 0.5, "L": 1.0}, "linear_growth": {"g": 0.1, "c": 0.0},
    }
    for fid in forms.list_forms():
        f = forms.get_form(fid)
        y = f.eval(params[fid], inp)
        assert isinstance(y, float) and math.isfinite(y), fid


def test_output_domains_clamped():
    # a probability form cannot leave [0,1]
    assert 0.0 <= forms.get_form("logistic").eval({"weights": {"a": 1e6}, "intercept": 0}, {"features": {"a": 1}}) <= 1.0
    # a rate form is non-negative
    assert forms.get_form("hill").eval({"theta": -5, "n": 2, "k": 1}, {"x": 2}) >= 0.0


def test_hill_monotone_increasing_and_saturates():
    f = forms.get_form("hill"); p = {"theta": 1.0, "n": 2.0, "k": 4.0}
    ys = [f.eval(p, {"x": x}) for x in (0, 1, 4, 16, 100)]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))  # increasing
    assert ys[-1] > 0.95                                               # saturates toward theta


def test_fatigue_decreases_with_exposure():
    f = forms.get_form("fatigue"); p = {"A": 1.0, "gamma": 0.6}
    ys = [f.eval(p, {"n_exposures": n}) for n in range(6)]
    assert all(ys[i] > ys[i + 1] for i in range(len(ys) - 1))


def test_hysteresis_is_path_dependent():
    f = forms.get_form("hysteresis"); p = {"tau_up": 0.7, "tau_down": 0.3}
    # same x=0.5, different prior state → different output (the defining property)
    assert f.eval(p, {"x": 0.5, "active": 0}) == 0.0
    assert f.eval(p, {"x": 0.5, "active": 1}) == 1.0


def test_zero_over_zero_is_guarded_to_finite():
    # michaelis_menten guards a 0/0 to a finite 0.0 rather than emitting nan (internal safety)
    assert forms.get_form("michaelis_menten").eval({"Vmax": 1.0, "K": 0.0}, {"x": 0.0}) == 0.0


def test_nonfinite_output_raises():
    # a genuine overflow (x**n → inf, inf/inf → nan) is caught by the eval-level non-finite guard
    with pytest.raises(forms.FormError):
        forms.get_form("hill").eval({"theta": 1.0, "n": 1000.0, "k": 2.0}, {"x": 10.0})


def test_safety_prob_and_rate_guards_record():
    r = safety.GuardReport()
    assert safety.safe_prob(1.5, r) == 1.0 and r.clamped
    r2 = safety.GuardReport()
    assert safety.safe_rate(-3.0, r2) == 0.0 and r2.clamped
    with pytest.raises(safety.StabilityError):
        safety.safe_prob(float("nan"))


def test_branching_ratio_guard():
    with pytest.raises(safety.StabilityError):
        safety.check_branching(1.0)          # explosive
    safety.check_branching(0.9)              # ok


def test_partial_pooling_shrinks_sparse_groups():
    # a data-rich group keeps its signal; a 1-obs group shrinks to the pool
    stats = {"big": {"mean": 0.9, "n": 1000}, "tiny": {"mean": 0.1, "n": 1}}
    pooled = pooling.pool_gaussian(stats)
    assert pooled.groups["tiny"]["shrinkage"] > pooled.groups["big"]["shrinkage"]
    assert abs(pooled.estimate("tiny") - pooled.grand_mean) < abs(0.1 - pooled.grand_mean)


def test_structural_uncertainty_keeps_ambiguous_forms():
    fp = su.FormPosterior("m", [su.FormCandidate("linear", {}, 0.10, 200),
                                su.FormCandidate("hill", {}, 0.101, 200)]).normalize()
    assert fp.is_ambiguous()                 # two near-tied forms → do not collapse
    assert fp.selected().form_id == "linear"  # but the marginally-better one is 'selected'


def test_applicability_keeps_linear_without_support():
    box = ap.SupportBox(ranges={"x": [0, 10]}, n_train=100)
    res = ap.evaluate_applicability(phenomenon_supported=True, history_available=False,
                                    context_available=False, inputs={"x": 5}, support=box)
    assert res.verdict == "keep_linear" and res.keep_linear_fallback


def test_applicability_widens_on_strong_extrapolation():
    box = ap.SupportBox(ranges={"x": [0, 10]}, n_train=100)
    res = ap.evaluate_applicability(phenomenon_supported=True, history_available=True,
                                    context_available=True, inputs={"x": 40}, support=box)
    assert res.extrapolation == "strong_extrapolation"
    assert res.uncertainty_widening > 1.0 and res.support_grade_delta < 0


def test_composition_detects_duplicate_saturation():
    slots = [comp.MechanismSlot("m1", "hill", "quantities[adopt]"),
             comp.MechanismSlot("m2", "michaelis_menten", "quantities[adopt]")]
    rep = comp.detect_conflicts(slots)
    assert any(c["type"] == "duplicate_saturation" for c in rep["conflicts"])
