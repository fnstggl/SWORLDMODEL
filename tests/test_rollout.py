"""Tests for the multi-step belief rollout operator."""
from swm.transition.rollout import MultiStepRollout, _traj_features


def test_traj_features_shape_and_momentum():
    f = _traj_features([0.2, 0.3, 0.4, 0.5])
    assert len(f) == 4
    assert f[0] == 0.5 and f[1] > 0                  # level, positive momentum
    assert abs(f[3] - 0.25) < 1e-9                   # p*(1-p) at p=0.5


def test_rollout_produces_widening_band():
    # random-walk-ish sequences -> nonzero innovation -> band widens with horizon
    seqs = [[0.5 + 0.02 * ((i % 5) - 2) for i in range(12)] for _ in range(40)]
    roll = MultiStepRollout().fit(seqs)
    out = roll.rollout([0.5, 0.5, 0.5], horizon=6, n_samples=101)
    assert len(out) == 6
    width1 = out[0]["hi"] - out[0]["lo"]
    width6 = out[5]["hi"] - out[5]["lo"]
    assert width6 >= width1                          # uncertainty grows with horizon
    for step in out:
        assert 0.0 <= step["lo"] <= step["mean"] <= step["hi"] <= 1.0


def test_first_step_impact_moves_forecast():
    seqs = [[0.5] * 10 for _ in range(20)]
    roll = MultiStepRollout().fit(seqs)
    up = roll.rollout([0.5, 0.5], horizon=3, first_step_impact=1.0, impact_scale=0.1, n_samples=51)
    down = roll.rollout([0.5, 0.5], horizon=3, first_step_impact=-1.0, impact_scale=0.1, n_samples=51)
    assert up[0]["mean"] > down[0]["mean"]           # positive event impact raises the step-1 forecast


def test_rollout_clamps_to_unit_interval():
    seqs = [[0.95, 0.96, 0.97, 0.98] for _ in range(20)]
    roll = MultiStepRollout().fit(seqs)
    out = roll.rollout([0.98, 0.99], horizon=5, first_step_impact=1.0, impact_scale=0.5, n_samples=51)
    assert all(0.0 <= s["mean"] <= 1.0 and s["hi"] <= 1.0 for s in out)
