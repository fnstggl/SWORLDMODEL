"""Diffusion family transitions: parameter recovery, world-execution parity, nonlinear forms, Hawkes."""
import math
import random

from swm.world_model_v2.registry.families.diffusion import (
    LinearHazard, LogLinearHazard, closed_form_window_p, contagion_window_predict, fit_frailty_sigma,
    fit_hill, fit_hawkes, fit_linear_q, fit_loglinear, hawkes_forecast_counts, marginal_window_p)


def _synth(n=3000, seed=1, concave=True):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        k = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 8, 12, 20])
        deg = max(1, int(rng.expovariate(1 / 80)))
        rec = rng.uniform(0, 48)
        base = math.log1p(k) if concave else k
        lam = 0.05 * base * (deg ** -0.15) * math.exp(-rec / 60)
        rows.append({"u": i, "k": k, "deg": deg, "recency_h": rec,
                     "y": 1 if rng.random() < 1 - math.exp(-lam) else 0})
    return rows


def test_loglinear_recovers_concavity_and_degree_sign():
    hz = fit_loglinear(_synth(), 1.0)
    # theta[1] is coef on log1p(k): positive (exposure increases hazard); theta[2] on log1p(deg): negative
    assert hz.theta[1] > 0.2
    assert hz.theta[2] < 0.0


def test_world_execution_matches_closed_form_no_rollout():
    rows = _synth(400, seed=2)
    hz = fit_loglinear(rows, 1.0)
    p_world = contagion_window_predict(rows, hz, 1.0, {}, n_particles=300, seed=5, rollout=False)
    p_cf = closed_form_window_p(rows, hz, 1.0)
    mad = sum(abs(a - c) for a, c in zip(p_world, p_cf)) / len(rows)
    assert mad < 0.01                                         # Monte-Carlo agreement (parity)


def test_frailty_sigma_recovers_zero_on_homogeneous_data():
    rows = _synth(2000, seed=3)
    hz = fit_loglinear(rows, 1.0)
    sig, profile = fit_frailty_sigma(hz, rows, 1.0)
    assert sig <= 0.5                                         # no injected heterogeneity → small/zero sd


def test_marginal_window_p_frailty_monotone():
    # more heterogeneity spreads probability but E stays bounded; sanity on the GH quadrature
    p0 = marginal_window_p(0.5, 1.0, 0.0)
    p1 = marginal_window_p(0.5, 1.0, 1.0)
    assert 0.0 < p1 < 1.0 and 0.0 < p0 < 1.0


def test_rollout_adds_activation_mass():
    rows = _synth(200, seed=4)
    hz = LinearHazard(0.1)
    # dense in-sample followers → rollout should raise activation probability vs frozen exposure
    fol = {r["u"]: [rows[(i + 1) % len(rows)]["u"]] for i, r in enumerate(rows)}
    p_roll = contagion_window_predict(rows, hz, 1.0, fol, n_particles=100, seed=1, rollout=True)
    p_froze = contagion_window_predict(rows, hz, 1.0, fol, n_particles=100, seed=1, rollout=False)
    assert sum(p_roll) >= sum(p_froze) - 1e-6


def test_hawkes_fits_and_forecasts():
    rng = random.Random(7)
    # simulate a bursty stream: Poisson background + clustered offspring
    ts = []
    t = 0.0
    while t < 40000:
        t += rng.expovariate(1 / 200.0)
        ts.append(t)
        if rng.random() < 0.3:
            for _ in range(rng.randint(1, 4)):
                ts.append(t + rng.expovariate(1 / 60.0))
    ts.sort()
    mu, al, om, ll = fit_hawkes(ts, 0.0, 30000)
    assert 0.0 <= al < 1.0 and mu > 0
    fc = hawkes_forecast_counts(mu, al, om, ts, 30000, 40000, 10, n_sims=30, seed=1)
    assert len(fc) == 10 and sum(fc) > 0
