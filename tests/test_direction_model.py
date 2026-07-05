"""Tests for the direction model (forecasting which way a belief moves)."""
from swm.transition.direction_model import DirectionModel, direction_features, FEATURE_NAMES


def test_direction_features_shape_and_lean():
    f = direction_features([0.4, 0.5, 0.7])
    assert len(f) == len(FEATURE_NAMES)
    assert abs(f[0] - 0.2) < 1e-9                    # lean = p - 0.5 = 0.2
    assert f[1] == abs(f[0])                          # abs_lean
    down = direction_features([0.4, 0.3, 0.2])
    assert down[0] < 0                               # negative lean when below 0.5


def test_result_cue_detected_in_news():
    f = direction_features([0.5, 0.6], news=[{"title": "Exit poll: challenger wins in a landslide"}])
    assert f[4] > 0                                  # result_cue fires on "exit poll"/"wins"
    f0 = direction_features([0.5, 0.6], news=[{"title": "a calm day in markets"}])
    assert f0[4] == 0.0


def test_learns_lean_predicts_up():
    # synthetic: up-moves happen when lean>0 (belief resolves toward its side)
    import random
    rng = random.Random(0)
    ex = []
    for _ in range(400):
        p = rng.random()
        move = (0.1 if rng.random() < (0.5 + 0.4 * (p - 0.5)) else -0.1)   # P(up) rises with lean
        ex.append((direction_features([0.5, p]), move))
    dm = DirectionModel().fit(ex)
    assert dm.p_up(direction_features([0.5, 0.9])) > dm.p_up(direction_features([0.5, 0.1]))


def test_direction_call_thresholds():
    dm = DirectionModel()
    # unfitted -> 0.5 -> no confident call
    assert dm.direction(direction_features([0.5, 0.6])) == 0
