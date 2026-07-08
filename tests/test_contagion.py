"""Tests for the contagion/tipping dynamic (EXP-072)."""
from experiments.exp072_contagion import _contagion_roll, _score


def test_contagion_rises_then_reverses():
    """A rising name (positive momentum) must keep rising for a while, PEAK, and turn down — the coupled
    bandwagon+fatigue behavior a linear trend cannot produce. Roll the full trajectory from one start."""
    path = [_contagion_roll(1.0, 0.5, rho=0.8, lam=0.05, steps=k) for k in range(1, 21)]
    assert max(path) > 1.1                        # momentum carries it up past the start
    assert path[-1] < max(path) - 0.05            # then fatigue reverses it — a turning point


def test_contagion_beats_persistence_at_a_turn():
    # a name at its peak (high, mild momentum) whose truth is a post-peak decline
    turn = [{"p": 4.0, "g": 0.1, "truth": 2.0}]   # near peak, will fall to 2.0 in H years
    persist = _score(turn, lambda s: s["p"])                                  # says 4.0 -> err 2.0
    contagion = _score(turn, lambda s: _contagion_roll(s["p"], s["g"], 0.4, 0.05, 10))
    assert contagion < persist                    # the coupled dynamic anticipates the fall


def test_flat_low_name_stays_low():
    # a low, flat name should not be driven anywhere dramatic
    assert _contagion_roll(0.2, 0.0, rho=0.6, lam=0.1, steps=10) < 0.3
