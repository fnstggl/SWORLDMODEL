"""Tests for the action-layer scoreboard (precision@1, policy regret, CATE-sign, IPS / doubly-robust)."""
import random

from swm.eval.policy_regret import (cate_sign_accuracy, doubly_robust_value, ips_value, policy_regret,
                                     precision_at_1)


def test_precision_at_1_and_regret():
    chosen = ["A", "B", "C", "A"]
    oracle = ["A", ["B", "C"], "A", "A"]          # instance 2 has tied-best {B,C}; instance 3 model missed
    assert abs(precision_at_1(chosen, oracle) - 0.75) < 1e-9
    assert abs(policy_regret([1.0, 0.8, 0.5, 1.0], [1.0, 0.8, 1.0, 1.0]) - 0.125) < 1e-9


def test_cate_sign_accuracy_excludes_ties():
    pred = [0.3, -0.2, 0.05, -0.4]
    true = [0.5, -0.1, 0.0, 0.2]                    # last: model says B worse, truth says better -> miss; 3rd tie
    assert abs(cate_sign_accuracy(pred, true, tol=0.0) - 2 / 3) < 1e-9


def test_ips_is_unbiased_on_random_logging():
    """Under uniform random logging, IPS of a target policy recovers its true value in expectation."""
    rng = random.Random(0)
    true_reward = {"A": 0.7, "B": 0.3}             # deterministic expected reward per action
    n = 20000
    logged, rewards, props, target = [], [], [], []
    for _ in range(n):
        a = "A" if rng.random() < 0.5 else "B"     # behavior policy: uniform
        logged.append(a)
        rewards.append(1.0 if rng.random() < true_reward[a] else 0.0)
        props.append(0.5)
        target.append("A")                          # target policy always picks A
    v = ips_value(logged, rewards, props, target)
    assert abs(v - true_reward["A"]) < 0.02         # ~ 0.70


def test_doubly_robust_matches_with_good_model():
    rng = random.Random(1)
    true_reward = {"A": 0.7, "B": 0.3}
    n = 8000
    logged, rewards, props, target, qt, ql = [], [], [], [], [], []
    for _ in range(n):
        a = "A" if rng.random() < 0.5 else "B"
        r = 1.0 if rng.random() < true_reward[a] else 0.0
        logged.append(a); rewards.append(r); props.append(0.5); target.append("A")
        qt.append(true_reward["A"]); ql.append(true_reward[a])   # a (near-)perfect reward model
    v = doubly_robust_value(logged, rewards, props, target, qt, ql)
    assert abs(v - true_reward["A"]) < 0.02
