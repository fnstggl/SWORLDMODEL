"""Balanced sampler: anti-dominance, caps, rare-task minimums."""
from machine_learning.sampling.balanced_sampler import SamplingConfig, apply_caps, compute_weights


def test_weights_sum_to_one():
    counts = {("big", "T1"): 100000, ("small", "T1"): 100}
    w = compute_weights(counts, SamplingConfig(temperature=0.5))
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_max_dominance_enforced():
    counts = {("big", "T1"): 1_000_000, ("small", "T1"): 1000}
    w = compute_weights(counts, SamplingConfig(temperature=1.0, max_dataset_dominance=0.5))
    big = sum(v for k, v in w.items() if k[0] == "big")
    assert big <= 0.5 + 1e-6


def test_temperature_flattens():
    counts = {("big", "T1"): 1_000_000, ("small", "T1"): 1000}
    hot = compute_weights(counts, SamplingConfig(temperature=1.0, max_dataset_dominance=1.0))
    cold = compute_weights(counts, SamplingConfig(temperature=0.2, max_dataset_dominance=1.0))
    big_hot = sum(v for k, v in hot.items() if k[0] == "big")
    big_cold = sum(v for k, v in cold.items() if k[0] == "big")
    assert big_cold < big_hot  # lower temperature gives the giant less share


def test_per_dataset_cap():
    counts = {("big", "T1"): 500000, ("big", "T2"): 500000}
    capped = apply_caps(counts, SamplingConfig(per_dataset_cap=100000))
    assert sum(capped.values()) <= 100000 + 2


def test_rare_task_minimum():
    counts = {("d", "common"): 100000, ("d", "rare"): 100}
    w = compute_weights(counts, SamplingConfig(temperature=1.0, rare_task_min_fraction=0.2,
                                               max_dataset_dominance=1.0))
    rare = sum(v for k, v in w.items() if k[1] == "rare")
    assert rare >= 0.2 - 1e-6
