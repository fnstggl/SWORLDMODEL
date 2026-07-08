"""Tests for the future-event model + distributional rollout."""
import statistics

from swm.transition.event_model import EventModel, _feats


def test_feats_capture_level_and_volatility():
    f = _feats([0.5, 0.5, 0.5])
    assert f[0] == 0.5 and abs(f[3] - 0.25) < 1e-9        # level, p*(1-p)
    calm = _feats([0.5, 0.5, 0.5, 0.5])
    choppy = _feats([0.5, 0.7, 0.4, 0.6])
    assert choppy[2] > calm[2]                            # volatility feature higher for choppy


def test_heteroskedastic_sigma_reflects_state_volatility():
    # calm markets vs volatile markets -> the model should predict larger sigma for volatile state
    calm = [[0.5 + 0.005 * ((i % 3) - 1) for i in range(10)] for _ in range(40)]
    volatile = [[0.5 + 0.08 * ((i % 3) - 1) for i in range(10)] for _ in range(40)]
    em = EventModel().fit(calm + volatile)
    s_calm = em.step_sigma([0.5, 0.5, 0.5])
    s_vol = em.step_sigma([0.5, 0.65, 0.4, 0.6])
    assert s_vol > s_calm                                 # variance placed where turbulence is


def test_forecast_band_widens_and_stays_in_unit_interval():
    seqs = [[0.5 + 0.03 * ((i % 5) - 2) for i in range(12)] for _ in range(40)]
    em = EventModel().fit(seqs)
    f = em.forecast([0.5, 0.5, 0.5], horizon=6, n_samples=200, seed=0)
    assert len(f) == 6
    assert (f[5]["hi"] - f[5]["lo"]) >= (f[0]["hi"] - f[0]["lo"])     # widens with horizon
    for step in f:
        assert 0.0 <= step["lo"] <= step["mean"] <= step["hi"] <= 1.0


def test_sigma_mult_scales_the_band():
    seqs = [[0.5 + 0.03 * ((i % 5) - 2) for i in range(12)] for _ in range(40)]
    em = EventModel().fit(seqs)
    em.sigma_mult = 0.5
    narrow = em.forecast([0.5, 0.5], 4, n_samples=200, seed=0)
    em.sigma_mult = 1.5
    wide = em.forecast([0.5, 0.5], 4, n_samples=200, seed=0)
    assert (wide[3]["hi"] - wide[3]["lo"]) > (narrow[3]["hi"] - narrow[3]["lo"])
