"""Pins mean-of-K aggregation (opt-in stability fix for the ~0.6 single-run variance): K seeded runs are
averaged into one forecast with per-run/sd/spread provenance; the default single-run path is untouched."""
from types import SimpleNamespace

import swm.world_model_v2.unified_runtime as U


def _stub(seed_to_p):
    def fake(question, *, seed=0, **kw):
        return SimpleNamespace(raw_probability=seed_to_p(seed), calibrated_probability=None,
                               provenance={}, limitations=[], simulation_status="completed_with_degradation")
    return fake


def test_mean_of_k_averages_and_records_spread(monkeypatch):
    monkeypatch.setattr(U, "simulate_world", _stub(lambda s: [0.1, 0.7, 0.4][s % 3]))
    res = U.simulate_world_stable("q", n_runs=3, as_of="2026-05-07", horizon="2026-06-12")
    mk = res.provenance["mean_of_k"]
    assert mk["n_runs"] == 3 and mk["per_run"] == [0.1, 0.7, 0.4]
    assert abs(mk["mean"] - 0.4) < 1e-9 and mk["spread"] == 0.6
    assert res.calibrated_probability == 0.4                   # the scored value is the K-run mean


def test_seeds_are_varied(monkeypatch):
    seen = []
    monkeypatch.setattr(U, "simulate_world", lambda q, *, seed=0, **k: (seen.append(seed) or
                        SimpleNamespace(raw_probability=0.5, calibrated_probability=None, provenance={},
                                        limitations=[], simulation_status="completed")))
    U.simulate_world_stable("q", n_runs=4, as_of="2026-05-07", horizon="2026-06-12", seed=10)
    assert seen == [10, 11, 12, 13]                            # base_seed + i


def test_runs_without_forecast_excluded_from_mean(monkeypatch):
    monkeypatch.setattr(U, "simulate_world", _stub(lambda s: {0: 0.8, 1: None, 2: 0.6}[s % 3]))
    res = U.simulate_world_stable("q", n_runs=3, as_of="2026-05-07", horizon="2026-06-12")
    mk = res.provenance["mean_of_k"]
    assert mk["n_valid"] == 2 and abs(mk["mean"] - 0.7) < 1e-9   # None excluded; mean of 0.8,0.6
