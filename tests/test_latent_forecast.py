"""Tests for the re-architected latent-state forecaster — the fixes, verified by construction (no LLM)."""
from swm.api.latent_forecast import LatentSpec, parse_latent, simulate_latent


def test_no_drivers_returns_base_rate_exactly():
    # the anchor: with no evidence, the forecast IS the reference-class base rate (fixes the coin flip)
    assert abs(simulate_latent(LatentSpec(base_rate=0.5, kind="event", drivers=[]), 0.01, n=4000) - 0.5) < 0.02
    # a rare event with no drivers stays near its (shrunk) low base rate — NOT pulled to 0.5
    p = simulate_latent(LatentSpec(base_rate=0.1, kind="event", drivers=[]), 0.01, n=4000)
    assert p < 0.35 and p > 0.05


def test_drivers_move_but_honest_uncertainty_bounds_it():
    up = [{"direction": 1.0, "strength": 1.0, "grounded": True}]
    down = [{"direction": -1.0, "strength": 1.0, "grounded": True}]
    p_up = simulate_latent(LatentSpec(base_rate=0.5, kind="event", drivers=up), 0.1, n=4000)
    p_dn = simulate_latent(LatentSpec(base_rate=0.5, kind="event", drivers=down), 0.1, n=4000)
    assert p_up > 0.5 > p_dn                                  # evidence moves the forecast in the right direction
    assert p_up < 0.9 and p_dn > 0.1                          # but honest uncertainty keeps it from the extremes


def test_ungrounded_driver_is_weaker_than_grounded():
    g = simulate_latent(LatentSpec(base_rate=0.5, kind="event",
                                   drivers=[{"direction": 1.0, "strength": 1.0, "grounded": True}]), 0.1, n=6000)
    u = simulate_latent(LatentSpec(base_rate=0.5, kind="event",
                                   drivers=[{"direction": 1.0, "strength": 1.0, "grounded": False}]), 0.1, n=6000)
    assert g > u                                              # a guess is integrated out toward the anchor


def test_evidence_decays_over_horizon():
    d = [{"direction": 1.0, "strength": 1.0, "grounded": True}]
    near = simulate_latent(LatentSpec(base_rate=0.5, kind="event", drivers=d), 0.05, n=6000)
    far = simulate_latent(LatentSpec(base_rate=0.5, kind="event", drivers=d), 5.0, n=6000)
    assert near > far                                         # a far-off event regresses toward the base rate
    assert abs(far - 0.5) < abs(near - 0.5)


def test_metric_threshold_is_time_accurate_and_shrinks_when_unsure():
    # current 100, threshold 200, "high" confidence, SHORT horizon, low vol -> very unlikely to double
    short = simulate_latent(LatentSpec(base_rate=0.5, kind="metric", current_value=100.0, threshold=200.0,
                                       direction=">", annual_vol_pct=30.0, grounded_conf="high"), 0.02, n=6000)
    # same but a LONG horizon -> far more time to reach the threshold -> higher P (time-accuracy)
    long = simulate_latent(LatentSpec(base_rate=0.5, kind="metric", current_value=100.0, threshold=200.0,
                                      direction=">", annual_vol_pct=30.0, grounded_conf="high"), 5.0, n=6000)
    assert short < long
    # "low" confidence in the current value shrinks the sim toward the base rate (honest without live grounding)
    lowconf = simulate_latent(LatentSpec(base_rate=0.5, kind="metric", current_value=100.0, threshold=200.0,
                                         direction=">", annual_vol_pct=30.0, grounded_conf="low"), 0.02, n=6000)
    assert lowconf > short                                    # pulled up toward base rate 0.5 from ~0


def test_parse_latent_clamps_and_defaults():
    s = parse_latent('{"base_rate": 1.5, "kind": "event", "drivers": [{"direction": 5, "strength": 9}]}')
    assert s.base_rate <= 0.98 and s.kind == "event"
    assert s.drivers[0]["direction"] == 1.0 and s.drivers[0]["strength"] == 1.0   # clamped
    assert parse_latent("not json") is None
