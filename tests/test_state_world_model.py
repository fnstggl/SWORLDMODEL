"""Tests for the state-transition world model (spec section 10)."""
import os

from swm.retrieval.context import PostRecord, as_of, author_context
from swm.state.ablation import run_ablation
from swm.state.factors import build_hn_registry
from swm.state.state import Action, WorldState
from swm.state.transition import OutcomeHead, PriorHead, TransitionModel
from swm.state.trajectory import rollout

T0 = 1_700_000_000.0
DAY = 86400.0


def _records():
    return [PostRecord("alice", T0 + k * DAY, [5, 80, 3][k], f"post {k}", "a.com") for k in range(3)]


def test_no_post_asof_leakage_in_retrieval():
    recs = _records() + [PostRecord("alice", T0 + 10 * DAY, 999, "future", "a.com")]
    visible = as_of(recs, T0 + 3 * DAY)
    assert all(r.timestamp < T0 + 3 * DAY for r in visible)
    assert 999 not in [r.score for r in visible]          # the future post is unreachable
    ctx = author_context(recs, "alice", T0 + 3 * DAY)
    assert ctx["n_past"] == 3 and ctx["max_past"] == 80    # not 999


def _samples(n=120):
    """Synthetic: a few authors with a stable per-author hit propensity -> state should help."""
    import random
    rng = random.Random(0)
    reg = build_hn_registry()
    world = WorldState(timestamp=T0)
    rows, scores = [], []
    authors = {f"u{i}": rng.uniform(0.05, 0.6) for i in range(12)}
    ts = T0
    for _ in range(n):
        a = rng.choice(list(authors))
        ts += DAY
        action = Action(action_id=f"{a}-{ts}", actor_id=a,
                        content_features={"title_len": 0.5, "is_show": 0.0, "is_ask": 0.0,
                                          "is_text": 0.0, "topic": "ai"},
                        timing={"hour": 15, "weekday": 2, "ts": ts}, meta={"domain": "x.com"})
        e = world.entity(a)
        rows.append({f.name: f.extract(e, action, world.context_state) for f in reg.active()})
        score = 60 if rng.random() < authors[a] else 3
        scores.append(score)
        reg.apply_update(e, world.context_state, action, score)
    return reg, rows, scores


def test_state_updates_after_action():
    reg = build_hn_registry()
    world = WorldState(timestamp=T0)
    action = Action("a1", "bob", content_features={"topic": "ai"},
                    timing={"hour": 12, "ts": T0}, meta={"domain": "z.com"})
    e = world.entity("bob")
    before = e.stable_traits.get("quality")
    reg.apply_update(e, world.context_state, action, 200)   # a hit
    after = e.stable_traits["quality"].mean
    assert before is None or after != before.mean
    assert e.history_features["n_posts"] == 1
    assert world.context_state.domain_reputation["z.com"].n > 1   # domain state transitioned


def test_ablation_runs_and_sets_status():
    reg, rows, scores = _samples()
    results = run_ablation(reg, rows, scores, thr_key=40)
    assert results and all(r.verdict in ("KEEP", "DROP", "EXPERIMENTAL") for r in results)
    assert all(reg._f[r.factor].status == r.verdict for r in results)  # statuses updated in place


def test_rollout_returns_multiple_distinct_trajectories():
    reg = build_hn_registry()
    model = TransitionModel(reg, PriorHead())
    plan = [Action(f"p{i}", "anon", content_features={"topic": "ai"},
                   timing={"hour": 12, "ts": T0 + i * DAY}, meta={"domain": "d.com"})
            for i in range(3)]
    ro = rollout(model, WorldState(timestamp=T0), plan, n_samples=100, seed=1)
    assert ro.n_samples == 100 and len(ro.trajectories) == 100
    assert len(ro.per_step) == 3
    distinct = {tuple(round(x, 1) for x in t) for t in ro.trajectories}
    assert len(distinct) > 1                                # a distribution, not one prophecy


def test_unvalidated_domain_is_labelled_unvalidated():
    reg = build_hn_registry()
    model = TransitionModel(reg, PriorHead())
    plan = [Action("p0", "anon", content_features={"topic": "ai"},
                   timing={"hour": 12, "ts": T0}, meta={"domain": "d.com"})]
    ro = rollout(model, WorldState(timestamp=T0), plan, n_samples=20)  # no validated_grade
    assert ro.report_type == "simulation"
    assert ro.calibration_grade == "unvalidated" and ro.warning


def test_predict_and_rollout_are_distinct():
    os.environ["SWM_DB"] = ":memory:"
    from fastapi.testclient import TestClient

    from api.app import app
    c = TestClient(app)
    r = c.post("/v1/rollout", json={"world_id": "unknown_domain",
               "action_plan": [{"actor_id": "a", "title": "Show HN: thing", "domain": "d.com"}],
               "n_samples": 30})
    j = r.json()
    assert j["report_type"] == "simulation"               # rollout is never "prediction"
    assert j["calibration_grade"] == "unvalidated" and j["warning"]
    assert "trajectory_distribution" in j
    # /predict without a fitted world returns an error, but is a DIFFERENT, prediction-typed path
    r2 = c.post("/v1/predict", json={"contact_id": "x", "text": "hi"})
    assert r2.status_code == 200 and "trajectory_distribution" not in r2.json()
