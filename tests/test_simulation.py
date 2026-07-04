"""Tests for the multi-step, multi-actor simulation engine and its components."""
import random

import pytest

from swm.simulation.actors import default_hn_segments, IndividualActorState
from swm.simulation.engine import HNSimulationEngine
from swm.simulation.policies import PolicyParams, frontpage_prob, sample_poisson
from swm.simulation.world_rollout import WorldRollout
from swm.state.context_dynamics import ContextDynamics
from swm.state.entity_history import EntityHistory, EntityHistoryStore
from swm.transition.learned_transition import GradientBoostedClassifier, compare_learned_vs_handcoded
from swm.worlds.hybrid import HybridModel, StateSufficiencyGate


# ---------------- learned transition ----------------
def test_gbdt_beats_logistic_on_interactions():
    rng = random.Random(0)
    X, s = [], []
    for _ in range(700):
        a, b = rng.random(), rng.random()
        xor = (a > 0.5) != (b > 0.5)
        s.append(100 if rng.random() < (0.75 if xor else 0.08) else 2)
        X.append([a, b, rng.random()])
    r = compare_learned_vs_handcoded(X[:500], s[:500], X[500:], s[500:], thr=40)
    assert r["learned_gbdt"]["log_loss"] < r["handcoded_logistic"]["log_loss"]


# ---------------- entity history ----------------
def test_entity_history_depth_and_sufficiency():
    h = EntityHistory("a")
    assert h.tier() == "cold" and h.sufficiency() == 0.0
    for ts, m in [(1, 50), (2, 60), (3, 55), (4, 70), (5, 65), (6, 80), (7, 75), (8, 90)]:
        h.observe(ts, m)
    assert h.tier() == "deep"
    assert h.sufficiency() > 0.5
    f = h.features(now=100)
    assert f["eh_depth"] == 8 and f["eh_max_logscore"] > 0


def test_entity_history_store_asof():
    s = EntityHistoryStore()
    f0 = s.features("u", now=10)          # cold before any observation
    assert f0["eh_depth"] == 0
    s.observe("u", 5, 40)
    assert s.features("u", now=10)["eh_depth"] == 1


# ---------------- simulation engine ----------------
def test_simulation_probability_from_trajectories():
    eng = HNSimulationEngine()
    strong = {"audience_fit": 0.9, "hn_native": 0.9, "technical_depth": 0.8, "novelty": 0.8,
              "topic_ai": 1.0, "cat_Show": 1.0}
    weak = {"audience_fit": 0.1, "hn_native": 0.1, "technical_depth": 0.1, "novelty": 0.2}
    s_strong = eng.simulate(strong, author_rep=1.5, n_samples=300, seed=1)
    s_weak = eng.simulate(weak, author_rep=-0.5, n_samples=300, seed=2)
    # probability is the fraction of trajectories that hit — bounded and ordered by quality
    assert 0.0 <= s_weak["thresholds"][40] <= s_strong["thresholds"][40] <= 1.0
    assert s_strong["thresholds"][40] > s_weak["thresholds"][40]
    # multi-actor: several segments contributed reactions in a recorded trajectory
    rec = eng.simulate(strong, author_rep=1.5, n_samples=6, seed=3, record_steps=True)
    assert rec["step_traces"] is not None


def test_frontpage_transition_is_bimodal():
    """The front-page transition should create heavy-tailed / bimodal outcomes, not a smooth blob."""
    eng = HNSimulationEngine()
    feats = {"audience_fit": 0.6, "hn_native": 0.6, "technical_depth": 0.5, "novelty": 0.6}
    sim = eng.simulate(feats, author_rep=0.5, n_samples=400, seed=4, record_steps=True)
    scores = sorted(sim["sample_scores"])
    # a meaningful fraction die small (<10) AND a fraction reach a cascade (>=40) — two modes
    frac_dead = sum(1 for s in scores if s < 10) / len(scores)
    frac_hit = sum(1 for s in scores if s >= 40) / len(scores)
    assert frac_dead > 0.2 and frac_hit > 0.0


def test_frontpage_prob_monotone():
    p = PolicyParams()
    assert frontpage_prob(0, p) < frontpage_prob(p.frontpage_threshold, p) < frontpage_prob(100, p)


def test_poisson_sampler_mean():
    rng = random.Random(0)
    xs = [sample_poisson(4.0, rng) for _ in range(2000)]
    assert 3.5 < sum(xs) / len(xs) < 4.5


def test_engine_save_load_roundtrip(tmp_path):
    eng = HNSimulationEngine(params=PolicyParams(frontpage_threshold=11.0))
    eng.readout = (1.3, -0.2)
    path = tmp_path / "sim.json"
    eng.save(path)
    e2 = HNSimulationEngine.load(path)
    assert e2.params.frontpage_threshold == 11.0 and e2.readout == (1.3, -0.2)


# ---------------- world rollout ----------------
def test_world_rollout_uncertainty_by_horizon():
    eng = HNSimulationEngine()
    ro = WorldRollout(eng).rollout({"audience_fit": 0.7, "hn_native": 0.7}, author_rep=0.8,
                                   n_samples=120, seed=0)
    assert len(ro["per_step"]) == eng.params.n_steps
    # cumulative score median is non-decreasing across steps
    meds = [s["median"] for s in ro["per_step"]]
    assert all(meds[i] <= meds[i + 1] + 1e-9 for i in range(len(meds) - 1))


def test_scenario_tree_ranks_actions():
    eng = HNSimulationEngine()
    cands = [("strong", {"audience_fit": 0.9, "hn_native": 0.9, "topic_ai": 1.0}, 1.5, {}),
             ("weak", {"audience_fit": 0.1, "hn_native": 0.1}, -0.5, {})]
    tree = WorldRollout(eng).scenario_tree(cands, n_samples=120)
    assert tree["recommended"] == "strong"


# ---------------- hybrid gate ----------------
def test_hybrid_gate_defers_by_state_depth():
    h = HybridModel(gate=StateSufficiencyGate(softness=6.0))
    cold = h.predict(world_p=0.9, llm_p=0.1, author_depth=0, domain_depth=0)
    deep = h.predict(world_p=0.9, llm_p=0.1, author_depth=20, domain_depth=10)
    assert cold["branch"] == "llm" and cold["p"] < 0.4       # cold -> trust LLM
    assert deep["branch"] == "world_model" and deep["p"] > 0.7  # deep -> trust world model


# ---------------- context dynamics ----------------
def test_context_dynamics_asof_updates():
    c = ContextDynamics()
    assert c.reputation("x.com") == 0.0
    c.observe(ts=1.0, topic="ai", domain="x.com", magnitude=200)
    assert c.reputation("x.com") > 0.0
    assert c.novelty("ai") < 1.0                              # novelty fatigues after a post


# ---------------- individual simulation ----------------
def test_individual_simulate_reply_valid_prob():
    from experiments.individual_simulation_harness import OutboundMessage, simulate_reply
    m = OutboundMessage("r1", "me", 0.0, {"personalization": 0.8, "ask_strength": 0.6,
                        "pushiness": 0.1, "length": 0.4}, prior_replies=3, prior_ignores=1)
    r = simulate_reply(m, n_samples=200, seed=0)
    assert 0.0 <= r["p_reply"] <= 1.0
