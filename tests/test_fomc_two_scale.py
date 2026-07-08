"""Guard tests for the FOMC two-scale world (EXP-071): the environment entity must drive the members."""
import importlib

exp = importlib.import_module("experiments.exp071_fomc_two_scale")


def test_environment_drives_members_coupled():
    """With coupling on, a sustained macro pressure must produce a non-constant committee output (the env
    entity absorbs the external pressure into its STATE, which the coupling reads). Guards the fix where a
    step-less environment entity silently dropped the external input."""
    press = [2.0] * 30                                    # sustained hawkish pressure
    out = exp._two_scale(press, rho=0.85, k=1.0, tau=0.5, bias_spread=1.0)
    assert max(out) > 0.05                                # the committee actually moves (hawkish)
    assert any(abs(a - b) > 1e-6 for a, b in zip(out, out[1:]))   # non-constant -> members accumulate


def test_coupling_cut_collapses_to_hold():
    press = [2.0] * 30
    cut = exp._two_scale(press, rho=0.85, k=1.0, tau=0.5, bias_spread=1.0, coupled=False)
    assert all(abs(x) < 1e-9 for x in cut)                # frozen members -> committee always holds


def test_score_direction_and_mae():
    s = exp._score([0.3, -0.2, 0.0], [0.4, -0.1, 0.05])
    assert s["n"] == 3 and 0.0 <= s["direction_acc"] <= 1.0 and s["mae"] > 0
