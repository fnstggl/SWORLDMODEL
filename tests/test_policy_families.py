"""Choice + learning family transitions: QRE, FS utility, mixture recovery, RL/EWA/habit updates."""
import math
import random

from swm.world_model_v2.policy import logit_choice, tremble_mix, w1_dist_sample, PopulationPreferences, PreferenceAtom
from swm.world_model_v2.registry.families.choice import (
    GAME_GRID, GamePolicyModel, fs_published_pop, fs_utility, responder_threshold, fit_game_policy,
    sample_policy_action)
from swm.world_model_v2.registry.families.learning import (
    EWAState, belief_probs, belief_update_counts, habit_update, reinforcement_update)


def test_logit_choice_limits():
    assert logit_choice([1, 1, 1], 0.0) == [1 / 3, 1 / 3, 1 / 3]        # λ=0 uniform
    p = logit_choice([0, 0, 10], 5.0)
    assert p[2] > 0.99                                                   # high λ → best reply


def test_fs_utility_and_responder_threshold():
    assert fs_utility(50, 50, 1.0, 0.5) == 50                            # equal split, no inequity
    assert fs_utility(80, 20, 1.0, 0.5) < 80                             # advantageous inequity penalized
    assert 0 < responder_threshold(1.0) < 50                             # positive min acceptable offer


def test_fs_mixture_recovers_on_synthetic_population():
    rng = random.Random(3)
    m0 = GamePolicyModel(fs_published_pop(),
                         lam={g: 0.1 * 100.0 / {**{x: 100.0 for x in GAME_GRID}, "trust_banker": 150.0,
                                                "public_goods": 20.0}[g] for g in GAME_GRID},
                         sd={g: 0.08 * (GAME_GRID[g][1] - GAME_GRID[g][0]) for g in GAME_GRID})
    train = {g: [sample_policy_action(m0, g, rng) for _ in range(120)] for g in GAME_GRID}
    m = fit_game_policy(train)
    # in-sample W1 should be small on the games the mixture governs
    for g in ("dictator", "ultimatum_responder"):
        lo, hi, _ = GAME_GRID[g]
        assert w1_dist_sample(m.population_dist(g), train[g], lo, hi) / (hi - lo) < 0.1


def test_reinforcement_moves_toward_reward():
    q = 0.0
    for _ in range(50):
        q = reinforcement_update(q, 1.0, alpha=0.2)
    assert q > 0.9


def test_belief_learning_tracks_frequencies():
    counts = {"a": 0.0, "b": 0.0}
    for _ in range(30):
        counts = belief_update_counts(counts, "a")
    for _ in range(10):
        counts = belief_update_counts(counts, "b")
    p = belief_probs(counts)
    assert p["a"] > p["b"] and abs(p["a"] - 0.75) < 0.05


def test_ewa_updates_and_chooses():
    ewa = EWAState(A={"x": 0.0, "y": 0.0})
    ewa = ewa.update("x", {"x": 1.0, "y": 0.0})
    probs = ewa.choice_probs(5.0)
    assert probs["x"] > probs["y"]


def test_habit_accumulates_and_saturates():
    h = 0.0
    for _ in range(50):
        h = habit_update(h, True, gamma=0.2)
    assert h > 0.9
    h2 = habit_update(1.0, False, gamma=0.2)
    assert h2 < 1.0                                                      # decays when not acted
