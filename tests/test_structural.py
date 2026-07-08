"""Tests for the structural simulation engine (EXP-063): Monte Carlo, calibrated time, decomposition."""
import math

from swm.simulation.structural import (StructuralModel, SVar, montecarlo, prob_of,
                                        variance_decomposition)


def test_montecarlo_numeric_and_categorical():
    num = montecarlo(lambda rng: rng.gauss(0, 1), n=4000)
    assert num["kind"] == "numeric" and abs(num["mean"]) < 0.1 and num["p05"] < num["p95"]
    cat = montecarlo(lambda rng: "a" if rng.random() < 0.7 else "b", n=4000)
    assert cat["kind"] == "categorical" and cat["mode"] == "a" and abs(cat["distribution"]["a"] - 0.7) < 0.05


def test_prob_of():
    p = prob_of(lambda rng: rng.random(), lambda x: x < 0.3, n=5000)
    assert abs(p - 0.3) < 0.03


def test_diffusion_time_scaling():
    # a pure random walk: sd of the outcome must grow as sqrt(horizon) (Wiener scaling = calibrated time)
    m = StructuralModel(variables={"x": SVar("x", 0.5, est_sd=0.0, vol=0.05, lo=-9, hi=9)},
                        outcome_fn=lambda s: s["x"])
    sd4 = montecarlo(m.simulate_once(4, dt=1.0), n=6000)["sd"]
    sd16 = montecarlo(m.simulate_once(16, dt=1.0), n=6000, seed=3)["sd"]
    assert abs(sd4 - 0.05 * math.sqrt(4)) < 0.01           # matches closed-form diffusion
    assert abs(sd16 / sd4 - 2.0) < 0.15                    # 4x horizon -> 2x spread


def test_step_size_invariance():
    # the diffusion result must not depend on the integration step dt (only on elapsed horizon)
    m = StructuralModel(variables={"x": SVar("x", 0.5, vol=0.04, lo=-9, hi=9)}, outcome_fn=lambda s: s["x"])
    coarse = montecarlo(m.simulate_once(8, dt=2.0), n=6000)["sd"]
    fine = montecarlo(m.simulate_once(8, dt=0.5), n=6000, seed=4)["sd"]
    assert abs(coarse - fine) < 0.02


def test_variance_decomposition_separates_sources():
    # only epistemic uncertainty -> irreducible fraction ~ 0
    epi = StructuralModel(variables={"x": SVar("x", 0.5, est_sd=0.1, vol=0.0, lo=-9, hi=9)},
                          outcome_fn=lambda s: s["x"])
    d_epi = variance_decomposition(epi, horizon=5, dt=1.0, n=3000)
    assert d_epi["irreducible_frac"] < 0.2 and d_epi["forecastable"]
    # only aleatoric noise -> irreducible fraction ~ 1
    alea = StructuralModel(variables={"x": SVar("x", 0.5, est_sd=0.0, vol=0.1, lo=-9, hi=9)},
                           outcome_fn=lambda s: s["x"])
    d_alea = variance_decomposition(alea, horizon=5, dt=1.0, n=3000)
    assert d_alea["irreducible_frac"] > 0.8 and not d_alea["forecastable"]


def test_structural_coupling_moves_target():
    # a variable pulled toward a high target by the drift should end higher than it started
    m = StructuralModel(
        variables={"y": SVar("y", 0.3, vol=0.0, lo=0, hi=1)},
        drift_fn=lambda s, dt: {"y": 0.3 * (0.9 - s["y"])}, outcome_fn=lambda s: s["y"])
    out = montecarlo(m.simulate_once(10, dt=1.0), n=1000)
    assert out["mean"] > 0.6                                # coupling drove it up toward 0.9
