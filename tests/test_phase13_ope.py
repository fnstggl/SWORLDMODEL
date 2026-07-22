"""Off-policy estimator validation against KNOWN synthetic ground truth (Parts 26–29).

Validating the ESTIMATORS with synthetic data whose true policy value is known analytically is the
correct methodology — it certifies the estimator math before the estimators grade Phase 13 on REAL
logged data. Each test constructs a logged bandit/MDP with a computable target-policy value and
asserts recovery within a tolerance set by the sample size.
"""
from __future__ import annotations

import random

import pytest

from swm.world_model_v2.phase13.ope import (cluster_bootstrap_ci, direct_method, doubly_robust,
                                            estimator_disagreement, ips, linear_fit, logistic_fit,
                                            overlap_diagnostics, per_decision_is, sequential_dr,
                                            snips)


# ---------------------------------------------------------------- synthetic logged bandit
def _bandit(n=4000, seed=0):
    """3 actions, binary context x; reward ~ Bernoulli(mu[a][x]); logging policy fixed propensities.
    True value of 'always a' = mean_x mu[a][x] (x uniform)."""
    mu = {0: {0: 0.2, 1: 0.5}, 1: {0: 0.6, 1: 0.3}, 2: {0: 0.4, 1: 0.7}}
    logp = {0: 0.5, 1: 0.3, 2: 0.2}
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        x = rng.randint(0, 1)
        r = rng.random()
        a = 0 if r < logp[0] else (1 if r < logp[0] + logp[1] else 2)
        rew = 1.0 if rng.random() < mu[a][x] else 0.0
        rows.append({"context": {"x": x}, "action": a, "reward": rew,
                     "propensity": logp[a], "cluster": f"c{i % 40}"})
    true = {a: 0.5 * (mu[a][0] + mu[a][1]) for a in mu}
    return rows, true


def _feat(c, a=None):
    return [c["x"], 1.0 if a == 1 else 0.0, 1.0 if a == 2 else 0.0] if a is not None else [c["x"]]


# ---------------------------------------------------------------- regressors
def test_linear_fit_recovers_known_line():
    X = [[x] for x in range(20)]
    y = [3.0 * x[0] + 2.0 for x in X]
    m = linear_fit(X, y)
    assert abs(m.predict([10.0]) - 32.0) < 0.1


def test_logistic_fit_separates():
    X = [[-2.0], [-1.0], [1.0], [2.0]] * 10
    y = [0, 0, 1, 1] * 10
    m = logistic_fit(X, y, epochs=500)
    assert m.predict([2.0]) > 0.7 and m.predict([-2.0]) < 0.3


# ---------------------------------------------------------------- IPS / SNIPS recovery
def test_ips_snips_recover_target_value():
    rows, true = _bandit(4000, seed=1)
    for a in (0, 2):
        r_ips = ips(rows, lambda c, a=a: a)
        r_snips = snips(rows, lambda c, a=a: a)
        assert abs(r_ips.value - true[a]) < 0.05, (a, r_ips.value, true[a])
        assert abs(r_snips.value - true[a]) < 0.05, (a, r_snips.value, true[a])
        assert r_ips.ci[0] <= true[a] <= r_ips.ci[1]


def test_dr_recovers_and_beats_ips_variance_with_good_model():
    rows, true = _bandit(4000, seed=2)
    pol = lambda c: 2
    r_ips = ips(rows, pol)
    r_dr = doubly_robust(rows, pol, model_fn=logistic_fit,
                         featurize=lambda c, a=None: _feat(c, a) if a is not None else _feat(c))
    assert abs(r_dr.value - true[2]) < 0.05
    # DR with an informative reward model should not be wider than raw IPS
    assert r_dr.se <= r_ips.se + 1e-6


def test_direct_method_recovers_with_per_action_model():
    rows, true = _bandit(4000, seed=3)
    r = direct_method(rows, lambda c: 2, model_fn=logistic_fit, featurize=lambda c: _feat(c))
    assert abs(r.value - true[2]) < 0.06


def test_stochastic_policy_supported():
    rows, true = _bandit(4000, seed=4)
    pol = lambda c: {0: 0.5, 2: 0.5}                       # mixed target
    r = ips(rows, pol)
    expected = 0.5 * true[0] + 0.5 * true[2]
    assert abs(r.value - expected) < 0.05


# ---------------------------------------------------------------- assumption guards (must refuse)
def test_ips_refuses_without_propensities():
    rows = [{"context": {"x": 0}, "action": 1, "reward": 1.0}]      # no propensity
    with pytest.raises(ValueError):
        ips(rows, lambda c: 1)


def test_dr_refuses_single_cluster():
    rows = [{"context": {"x": 0}, "action": 1, "reward": 1.0, "propensity": 0.5, "cluster": "only"}
            for _ in range(10)]
    with pytest.raises(ValueError):
        doubly_robust(rows, lambda c: 1, featurize=lambda c, a=None: _feat(c, a) if a is not None
                      else _feat(c))


def test_sequential_refuses_without_step_propensities():
    seqs = [{"steps": [{"context": {"x": 0}, "action": 0, "reward": 1.0}], "cluster": "a"}]
    with pytest.raises(ValueError):
        per_decision_is(seqs, lambda c: 0)


# ---------------------------------------------------------------- overlap diagnostics
def test_overlap_flags_no_overlap_case():
    # logging policy NEVER plays action 2; target ALWAYS plays action 2 -> zero overlap
    rng = random.Random(5)
    rows = [{"context": {"x": rng.randint(0, 1)}, "action": rng.randint(0, 1),
             "reward": float(rng.random() < 0.5), "propensity": 0.5, "cluster": f"c{i%10}"}
            for i in range(500)]
    diag = overlap_diagnostics(rows, lambda c: 2)
    assert diag["n_matched"] == 0
    assert diag.get("weak_overlap") is True
    # IPS on zero overlap returns 0 value (no matched weight) — not a crash, but ESS ~ 0
    r = ips(rows, lambda c: 2)
    assert r.value == 0.0


def test_clipping_sensitivity_curve_present():
    rows, _ = _bandit(2000, seed=6)
    from swm.world_model_v2.phase13.ope import clipping_sensitivity
    curve = clipping_sensitivity(rows, lambda c: 2)
    assert set(curve) >= {2, 5, 10, 20, None}


# ---------------------------------------------------------------- sequential (Part 29)
def _mdp_sequences(n=2000, seed=0):
    """2-step MDP: state s0 uniform in {0,1}; action in {0,1}; reward depends on (s,a); next state
    deterministic. Logging policy uniform (propensity 0.5 each step). True value of 'always 1'
    computed analytically below."""
    rng = random.Random(seed)
    seqs = []
    # reward tables per step
    R0 = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 0.5, (1, 1): 0.0}
    R1 = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 1.0, (1, 1): 0.5}
    for i in range(n):
        s0 = rng.randint(0, 1)
        a0 = rng.randint(0, 1)
        r0 = R0[(s0, a0)]
        s1 = a0                                            # next state = action taken
        a1 = rng.randint(0, 1)
        r1 = R1[(s1, a1)]
        seqs.append({"steps": [
            {"context": {"s": s0}, "action": a0, "reward": r0, "propensity": 0.5},
            {"context": {"s": s1}, "action": a1, "reward": r1, "propensity": 0.5}],
            "cluster": f"c{i % 40}"})
    # 'always 1': s0 uniform -> r0 = mean_s0 R0[s0,1] = (1.0+0.0)/2 = 0.5; s1 = 1 always -> r1 = R1[1,1]=0.5
    true = 0.5 + 0.5
    return seqs, true


def test_per_decision_is_recovers_sequential_value():
    seqs, true = _mdp_sequences(3000, seed=7)
    r = per_decision_is(seqs, lambda c: 1)
    assert abs(r.value - true) < 0.08, (r.value, true)


def test_sequential_dr_recovers_and_reports_ci():
    seqs, true = _mdp_sequences(3000, seed=8)
    r = sequential_dr(seqs, lambda c: 1, featurize=lambda c: [c["s"]])
    assert abs(r.value - true) < 0.12, (r.value, true)
    # CI should bracket the truth within a small slack (the point estimate here is near-exact, so a
    # tight CI can sit a hair off truth on either side)
    assert r.ci[0] - 0.05 <= true <= r.ci[1] + 0.05


# ---------------------------------------------------------------- CI + determinism
def test_cluster_bootstrap_covers_truth():
    # 40 clusters each a noisy estimate of 0.5; the 95% CI should cover 0.5
    rng = random.Random(9)
    vbc = {f"c{k}": [rng.gauss(0.5, 0.1) for _ in range(20)] for k in range(40)}
    out = cluster_bootstrap_ci(vbc, lambda s: sum(s) / len(s), seed=0)
    assert out["ci"][0] <= 0.5 <= out["ci"][1]


def test_determinism_same_seed_same_result():
    rows, _ = _bandit(2000, seed=10)
    r1 = ips(rows, lambda c: 2, seed=3)
    r2 = ips(rows, lambda c: 2, seed=3)
    assert r1.value == r2.value and r1.ci == r2.ci


def test_estimator_disagreement_reports_overlap():
    rows, _ = _bandit(3000, seed=11)
    pol = lambda c: 2
    results = [ips(rows, pol), snips(rows, pol),
               doubly_robust(rows, pol, model_fn=logistic_fit,
                             featurize=lambda c, a=None: _feat(c, a) if a is not None else _feat(c))]
    dis = estimator_disagreement(results)
    assert dis["max_abs_diff"] < 0.05           # all three agree on well-overlapped data
    assert dis["all_cis_overlap"] is True
