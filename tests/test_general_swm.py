"""Tests for the general world-model architecture (Phase 2): state, transitions, retrieval,
worlds, simulation, eval, and — critically — the leakage guarantees."""
import math
import random

import pytest

from swm.eval.leakage import (LeakageError, check_content_dedup, check_label_separation,
                              check_temporal, full_gate)
from swm.eval.decision_lift import decision_lift, hit_capture
from swm.eval.individual_response_eval import evaluate
from swm.eval.market_comparison import compare, retrieval_gap
from swm.eval.raw_llm_vs_world_model import run_benchmark
from swm.retrieval.asof_store import AsOfStore, ContextItem
from swm.retrieval.entity_context import EntityContext
from swm.simulation.counterfactuals import best_of, contrast
from swm.simulation.rollout import calibration_by_horizon, simulate
from swm.simulation.scenario_tree import ScenarioTree, expected_magnitude
from swm.state.incentives import incentives_from_title
from swm.state.latent import BetaHierarchical, HierarchicalPosterior, LatentField
from swm.state.population import PopulationState
from swm.state.state import Action
from swm.transition.aggregate_transition import AggregateTransition
from swm.transition.diffusion import HawkesProcess, independent_cascade, linear_threshold
from swm.state.graph import Graph
from swm.transition.nonstationarity import DriftTracker
from swm.worlds.aggregate_world import AggregateWorld
from swm.worlds.individual_world import IndividualWorld


# ---------------- latent / pooling ----------------

def test_hierarchical_partial_pooling_shrinks_with_evidence():
    h = HierarchicalPosterior(segment_mean=0.3, prior_strength=4.0)
    assert h.mean == pytest.approx(0.3)           # cold -> segment prior
    assert h.shrinkage == pytest.approx(1.0)
    for _ in range(20):
        h.observe(1.0)
    assert h.mean > 0.7                            # strong evidence -> individual
    assert h.shrinkage < 0.2
    assert h.uncertainty < HierarchicalPosterior(segment_mean=0.3).uncertainty


def test_beta_hierarchical_bounds_and_pooling():
    b = BetaHierarchical(segment_rate=0.2, prior_strength=4.0)
    assert b.mean == pytest.approx(0.2)
    for _ in range(10):
        b.observe(1)
    assert 0.2 < b.mean < 1.0
    lo, hi = b.interval()
    assert 0.0 <= lo <= hi <= 1.0


# ---------------- nonstationarity ----------------

def test_drift_tracker_detects_shift():
    d = DriftTracker(fast_halflife=10, slow_halflife=200)
    for _ in range(200):
        d.observe(0.1)
    calm = abs(d.indicator())
    for _ in range(60):
        d.observe(0.9)
    assert abs(d.indicator()) > calm               # regime shift shows up
    assert d.inflation() >= 1.0


# ---------------- diffusion ----------------

def test_independent_cascade_spreads_and_is_bounded():
    g = Graph()
    for i in range(20):
        g.add_edge("0", str(i + 1), weight=1.0)    # star from seed 0, certain activation
    out = independent_cascade(g, ["0"], rng=random.Random(0))
    assert out["final_active"] == 21               # all activate under w=1


def test_hawkes_branching_ratio_and_sim():
    h = HawkesProcess(mu=0.2, alpha=0.4, beta=1.0)
    assert h.branching_ratio == pytest.approx(0.4)
    ev = h.simulate(20.0, rng=random.Random(1))
    assert all(0 <= t <= 20 for t in ev)


# ---------------- as-of retrieval + leakage (THE guarantee) ----------------

def test_asof_store_never_returns_future():
    s = AsOfStore()
    s.add(ContextItem("a", 100.0, "news", text="past"))
    s.add(ContextItem("b", 200.0, "news", text="future"))
    got = s.query(as_of=150.0)
    assert [i.item_id for i in got] == ["a"]
    s.assert_no_leak(150.0, got)


def test_asof_store_rejects_untimestamped_and_missing_asof():
    s = AsOfStore()
    with pytest.raises(LeakageError):
        s.add(ContextItem("x", None, "news"))       # type: ignore[arg-type]
    s.add(ContextItem("y", 10.0, "news"))
    with pytest.raises(LeakageError):
        s.query(as_of=None)                          # type: ignore[arg-type]


def test_leakage_gate_catches_future_and_label():
    with pytest.raises(LeakageError):
        check_temporal([{"timestamp": 300.0}], as_of=150.0)
    with pytest.raises(LeakageError):
        check_label_separation(["a", "replied"], "replied")
    items = [ContextItem("c", 10.0, "news", text="the answer is YES")]
    with pytest.raises(LeakageError):
        check_content_dedup("the answer is YES", items)
    rep = full_gate(as_of=150.0, used_items=[{"timestamp": 100.0}],
                    feature_names=["a", "b"], label_name="y")
    assert rep["leakage_gate"] == "PASS"


def test_entity_context_is_asof():
    ec = EntityContext()
    pairs = [(10.0, 1.0), (20.0, 0.0), (30.0, 1.0)]
    f = ec.from_pairs(pairs, as_of=25.0)
    assert f["n_past"] == 2                          # 30.0 excluded


# ---------------- aggregate world: state must actually help on structured data ----------------

def _agg_samples(n=800, seed=0):
    rng = random.Random(seed)
    dq = {"good.com": 0.5, "bad.com": 0.04}
    out = []
    for i in range(n):
        topic = rng.choice(["ai", "other"])
        dom = rng.choice(list(dq))
        base = dq[dom] * (1.5 if topic == "ai" else 1.0)
        mag = 200.0 if rng.random() < base * 0.5 else (50.0 if rng.random() < base else float(rng.randrange(8)))
        a = Action(action_id=f"a{i}", actor_id=f"u{i%30}",
                   content_features={"title_len": 0.5, "is_text": 0.0, "topic": topic},
                   timing={"hour": rng.randrange(24), "weekday": i % 7, "ts": 1_700_000_000 + i * 3600},
                   meta={"domain": dom, "title": f"{topic} {i}"})
        out.append((a, mag))
    return out


def test_aggregate_state_transition_beats_content_only():
    aw = AggregateWorld(domain="synthetic")
    bt = aw.backtest(_agg_samples())
    assert "error" not in bt
    assert bt["state_helps_logloss"] > 0            # state genuinely enters and helps
    assert bt["state_transition"]["log_loss"] < bt["content_only"]["log_loss"]


def test_aggregate_predict_conditions_on_state():
    """Same action, different states -> different predictions (state is NOT cosmetic)."""
    aw = AggregateWorld(domain="synthetic").fit_stream(_agg_samples())
    a = Action(action_id="q", actor_id="u1",
               content_features={"title_len": 0.5, "topic": "ai"},
               timing={"hour": 12, "weekday": 2, "ts": 1_700_100_000}, meta={"domain": "good.com", "title": "ai q"})
    p_good = aw.predict(a)["thresholds"][40]
    a_bad = Action(action_id="q", actor_id="u1", content_features={"title_len": 0.5, "topic": "ai"},
                   timing={"hour": 12, "weekday": 2, "ts": 1_700_100_000}, meta={"domain": "bad.com", "title": "ai q"})
    p_bad = aw.predict(a_bad)["thresholds"][40]
    assert p_good != p_bad                           # domain reputation state changes the prediction


# ---------------- individual world: partial pooling beats segment ----------------

def _ind_samples(n=1200, seed=1):
    rng = random.Random(seed)
    people = {f"p{j}": rng.betavariate(2, 5) for j in range(50)}
    out = []
    for i in range(n):
        pid = rng.choice(list(people))
        out.append((pid, {"log_words": rng.uniform(2, 5)}, int(rng.random() < people[pid])))
    return out


def test_individual_pooling_beats_segment():
    res = evaluate(_ind_samples(), ["log_words"])
    assert res["regimes"]["+person"]["log_loss"] <= res["regimes"]["segment"]["log_loss"]
    assert res["regimes"]["raw_llm"]["status"].startswith("BLOCKED")   # honest about missing LLM


# ---------------- simulation ----------------

def test_simulate_returns_distribution_and_is_unvalidated():
    aw = AggregateWorld(domain="synthetic").fit_stream(_agg_samples())
    pop = PopulationState(timestamp=1_700_100_000)
    plan = [Action(action_id=f"p{i}", actor_id="u1",
                   content_features={"title_len": 0.5, "topic": "ai"},
                   timing={"hour": 12, "weekday": 2, "ts": 1_700_100_000 + i * 86400},
                   meta={"domain": "good.com", "title": f"ai {i}"}) for i in range(3)]
    ro = simulate(aw.transition, pop, plan, n_samples=50)
    assert ro.report_type == "simulation"
    assert ro.calibration_grade == "unvalidated"     # honesty gate holds for multi-step
    assert len(ro.per_step) == 3


def test_scenario_tree_and_counterfactual():
    aw = AggregateWorld(domain="synthetic").fit_stream(_agg_samples())
    good = Action(action_id="good", actor_id="u1", content_features={"title_len": 0.5, "topic": "ai"},
                  timing={"hour": 12, "weekday": 2, "ts": 1_700_100_000}, meta={"domain": "good.com", "title": "ai"})
    bad = Action(action_id="bad", actor_id="u1", content_features={"title_len": 0.5, "topic": "other"},
                 timing={"hour": 3, "weekday": 6, "ts": 1_700_100_000}, meta={"domain": "bad.com", "title": "other"})
    tree = ScenarioTree(aw.transition, aw.pop)
    out = tree.evaluate([good, bad])
    assert out["recommended_action_id"] == "good"
    cf = contrast(aw.transition, aw.pop, good, bad)
    assert cf["prefer"] == "good" and cf["delta_p_hit"] >= 0


# ---------------- decision lift + benchmark plumbing ----------------

def test_decision_lift_curve():
    y = [1 if i % 4 == 0 else 0 for i in range(100)]
    good = [1.0 if v else 0.0 for v in y]           # perfect ranker
    rand = [0.5] * 100
    dl = decision_lift(y, good, rand, target_k=0.25)
    assert dl["curve"][2]["model"] >= dl["curve"][2]["baseline"]


def test_run_benchmark_verdict():
    y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    res = run_benchmark(y, {
        "raw_llm": [0.6] * 10, "raw_llm_context": [0.55] * 10,
        "aggregate_world": [0.9 if v else 0.1 for v in y], "individual_world": None,
    }, target="hit")
    assert "world model" in res.verdict.lower()
    assert res.tiers["individual_world"]["status"].startswith("BLOCKED")


def test_market_comparison_and_retrieval_gap():
    truth = [{"id": f"m{i}", "resolution": i % 2, "market_at_T": 0.5, "bettors": 10 + i}
             for i in range(20)]
    no_ret = {f"m{i}": 0.5 for i in range(20)}
    with_ret = {f"m{i}": (0.9 if i % 2 else 0.1) for i in range(20)}   # informed
    cmp = compare(truth, with_ret)
    assert cmp["segments"][0]["model_brier"] < cmp["segments"][0]["market_brier"]
    gap = retrieval_gap(truth, no_ret, with_ret)
    assert gap["ALL"]["gap_closed"] > 0              # retrieval helped
