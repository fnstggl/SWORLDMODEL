"""Phase 9 — production population & multilayer-network inference (scripted, no network).

Foundation tests: the Phase-3 posterior engine, EXTENDED (not replaced) with a compositional (simplex)
representation and an edge-existence (Bernoulli/log-odds) representation, must recover known ground truth,
keep segment weights normalized, collapse dependent evidence, and be deterministic — the same guarantees the
Phase-3 outcome-rate posterior already meets.
"""
import random

import pytest

from swm.world_model_v2.phase3_observation import (EDGE_OBS_MODELS, RELATION_LAYERS, EdgeObservation,
                                                   collapse_edge_observations)
from swm.world_model_v2.phase3_posterior import (infer_compositional_posterior, infer_edge_posterior)


# ============================================================ compositional (simplex) posterior
def _multinomial_draw(rng, probs, n):
    counts = [0] * len(probs)
    for _ in range(n):
        r, acc = rng.random(), 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    return counts


def test_compositional_posterior_recovers_true_simplex():
    segs = ["A", "B", "C", "D"]
    true = [0.5, 0.25, 0.15, 0.10]
    rng = random.Random(1)
    counts = _multinomial_draw(rng, true, 800)
    obs = [{"counts": dict(zip(segs, counts)), "reliability": 1.0, "source": "survey"}]
    post = infer_compositional_posterior(segs, [1.0] * 4, obs, seed=0)
    # sums to one (compositional, not independent scalars)
    assert abs(sum(post.posterior_mean) - 1.0) < 1e-6
    # recovers the true simplex within tolerance and beats the flat prior
    err = sum(abs(m - t) for m, t in zip(post.posterior_mean, true))
    prior_err = sum(abs(0.25 - t) for t in true)
    assert err < 0.08 and err < prior_err
    # conjugate summary agrees with the particle posterior mean
    ca = post.conjugate_alpha
    cm = [a / sum(ca) for a in ca]
    assert all(abs(a - b) < 0.05 for a, b in zip(cm, post.posterior_mean))


def test_compositional_weights_are_nonnegative_and_normalized_per_particle():
    segs = ["x", "y", "z"]
    obs = [{"counts": {"x": 60, "y": 30, "z": 10}, "reliability": 0.9}]
    post = infer_compositional_posterior(segs, [1.0, 1.0, 1.0], obs, seed=2, n_particles=200)
    for vec, w in post.particles:
        assert all(v >= 0 for v in vec) and abs(sum(vec) - 1.0) < 1e-6


def test_compositional_dependence_collapse_reduces_confidence():
    segs = ["a", "b"]
    one = {"counts": {"a": 70, "b": 30}, "reliability": 0.9, "dependence_group": "poll-42"}
    syndicated = [dict(one) for _ in range(4)]                 # 4 re-reports of ONE poll
    dep = infer_compositional_posterior(segs, [1, 1], syndicated, seed=0, use_dependence=True)
    indep = infer_compositional_posterior(segs, [1, 1], syndicated, seed=0, use_dependence=False)
    assert dep.n_effective_observations == 1 and indep.n_effective_observations == 4
    # collapsing to one poll leaves MORE posterior spread than counting it four times
    assert sum(dep.posterior_sd) > sum(indep.posterior_sd)


def test_compositional_is_deterministic():
    segs = ["a", "b", "c"]
    obs = [{"counts": {"a": 40, "b": 40, "c": 20}, "reliability": 0.8}]
    a = infer_compositional_posterior(segs, [1, 1, 1], obs, seed=7)
    b = infer_compositional_posterior(segs, [1, 1, 1], obs, seed=7)
    assert a.posterior_mean == b.posterior_mean


def test_no_evidence_leaves_compositional_prior():
    segs = ["a", "b", "c"]
    post = infer_compositional_posterior(segs, [2, 1, 1], [], seed=0)
    assert post.n_effective_observations == 0
    # posterior mean ≈ prior mean (Dirichlet(2,1,1) → 0.5,0.25,0.25)
    assert abs(post.posterior_mean[0] - 0.5) < 0.06


# ============================================================ edge-existence (Bernoulli/log-odds) posterior
def test_at_least_ten_relation_layers():
    assert len(set(RELATION_LAYERS)) >= 10
    assert len({m["layer"] for m in EDGE_OBS_MODELS.values()}) >= 8


def test_strong_communication_record_raises_edge_posterior():
    obs = [EdgeObservation("alice", "bob", "direct_communication_record", strength="strong", reliability=0.95)]
    post = infer_edge_posterior("alice", "bob", "communication", obs, prior_p=0.1)
    # one strong-but-imperfect record against a skeptical 0.1 prior updates well past the prior (honest
    # Bayesian: a single noisy observation should NOT alone reach near-certainty)
    assert post.posterior_p > 0.75 and post.posterior_p > 0.1
    assert post.observed_status == "observed"
    assert post.log_odds_shift > 0


def test_repeated_records_reach_near_certainty():
    obs = [EdgeObservation("alice", "bob", "direct_communication_record", strength="strong", reliability=0.95,
                           dependence_group=f"src{i}") for i in range(3)]     # 3 INDEPENDENT records
    post = infer_edge_posterior("alice", "bob", "communication", obs, prior_p=0.1)
    assert post.posterior_p > 0.95                            # independent corroboration → near-certain


def test_absence_evidence_lowers_edge_posterior():
    obs = [EdgeObservation("a", "b", "absence_of_expected_interaction", strength="strong", reliability=0.9)]
    post = infer_edge_posterior("a", "b", "communication", obs, prior_p=0.5)
    assert post.posterior_p < 0.5                              # absence evidence pushes existence DOWN


def test_weak_evidence_moves_edge_less_than_strong():
    weak = [EdgeObservation("a", "b", "social_follow", strength="weak", reliability=0.6)]
    strong = [EdgeObservation("a", "b", "org_chart_relationship", strength="strong", reliability=0.95)]
    pw = infer_edge_posterior("a", "b", "influence", weak, prior_p=0.1)
    ps = infer_edge_posterior("a", "b", "reporting", strong, prior_p=0.1)
    assert ps.posterior_p > pw.posterior_p


def test_edge_dependence_collapse_counts_once():
    obs = [EdgeObservation("a", "b", "direct_communication_record", strength="strong", reliability=0.9,
                           dependence_group="wire") for _ in range(5)]
    collapsed = collapse_edge_observations(obs)
    assert len(collapsed) == 1 and collapsed[0].n_collapsed == 5
    single = infer_edge_posterior("a", "b", "communication",
                                  [EdgeObservation("a", "b", "direct_communication_record", "strong", 0.9)],
                                  prior_p=0.1)
    grouped = infer_edge_posterior("a", "b", "communication", obs, prior_p=0.1)
    # 5 syndicated copies must not out-update a single independent observation
    assert abs(grouped.posterior_p - single.posterior_p) < 1e-6


def test_no_edge_evidence_returns_prior():
    post = infer_edge_posterior("a", "b", "communication", [], prior_p=0.15)
    assert abs(post.posterior_p - 0.15) < 1e-6 and post.observed_status == "hypothesized"
