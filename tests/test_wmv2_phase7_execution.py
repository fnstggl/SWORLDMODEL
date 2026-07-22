"""Phase 7 — shared-world EXECUTION: operators emit StateDelta + future events, history/context leakage,
Phase-3 posterior propagation through nonlinear forms."""
import random
import time

from swm.world_model_v2.state import WorldState, SimulationClock, Entity
from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.transitions import StateDelta, _OPERATORS
from swm.world_model_v2.nonlinear import operators as ops
from swm.world_model_v2.nonlinear import history
from swm.world_model_v2.nonlinear.posterior import ParamPosterior, propagate, delta_method_gap
from swm.world_model_v2.nonlinear.forms import get_form


def _world():
    now = 1_400_000_000.0
    return WorldState(world_id="t", branch_id="b0", clock=SimulationClock(now=now, as_of=now)), now


def test_operators_registered():
    assert "nonlinear_mechanism" in _OPERATORS
    assert "nonlinear_contagion" in _OPERATORS
    assert "nonlinear_state_step" in _OPERATORS


def test_mechanism_emits_statedelta_and_writes_quantity():
    world, now = _world()
    world.entities["a"] = Entity(identity="a", entity_type="person")
    spec = {"form_id": "logistic", "params": {"weights": {"x": 3.0}, "intercept": -1.0},
            "features": {"x": 1.0}, "outcome_var": "adopt", "actor": "a", "output": "prob"}
    op = ops.NonlinearMechanismOperator()
    ev = Event(ts=now, etype="nonlinear_transition", participants=["a"], payload={"nonlinear_spec": spec})
    delta, vr = op.run(world, ev, random.Random(1))
    assert isinstance(delta, StateDelta)
    assert any(c["path"] == "quantities[adopt]" for c in delta.changes)
    assert "adopt" in world.quantities


def test_recurrent_followup_event_scheduled():
    world, now = _world()
    world.entities["a"] = Entity(identity="a", entity_type="person")
    nxt = {"form_id": "logistic", "params": {"weights": {}, "intercept": 5.0}, "outcome_var": "again",
           "actor": "a", "output": "prob"}
    spec = {"form_id": "logistic", "params": {"weights": {}, "intercept": 9.0}, "outcome_var": "adopt",
            "actor": "a", "output": "prob", "recurrent": {"etype": "nonlinear_transition", "delay_h": 24,
                                                          "next_spec": nxt}}
    op = ops.NonlinearMechanismOperator()
    ev = Event(ts=now, etype="nonlinear_transition", participants=["a"], payload={"nonlinear_spec": spec})
    # force occurrence: intercept 9 → p≈1
    delta, _ = op.run(world, ev, random.Random(0))
    assert delta.follow_up_events, "a recurrent occurrence must schedule the next event"
    assert delta.follow_up_events[0]["etype"] == "nonlinear_transition"


def test_contagion_retransmission_generates_future_exposures():
    world, now = _world()
    for e in ("alice", "bob", "carol"):
        world.entities[e] = Entity(identity=e, entity_type="person")
    # many exposures → near-certain activation
    for i in range(10):
        history.record_exposure(world.entities["alice"], at=now - 3600 * i, source=f"s{i}")
    rspec = {"form_id": "exposure_response_hazard", "params": {"theta": [-1, 1, 0.5, 0.1, 0]},
             "outcome_var": "active", "window_days": 1.0, "deg": 5}
    spec = {"form_id": "exposure_response_hazard", "params": {"theta": [2.0, 1, 0.5, 0.1, 0]},
            "outcome_var": "active", "window_days": 1.0, "deg": 10, "k0": 5,
            "followers": ["bob", "carol"], "retransmit_spec": rspec}
    op = ops.NonlinearContagionOperator()
    ev = Event(ts=now, etype="contagion_exposure", participants=["alice"], payload={"contagion_spec": spec})
    delta, _ = op.run(world, ev, random.Random(0))
    if world.entities["alice"].value("outcome", key="active") == "True":
        assert len(delta.follow_up_events) == 2   # retransmission to bob + carol
        # followers got a typed exposure recorded (history propagation)
        assert history.history_features(world.entities["bob"], now=now)["cum_count"] >= 1


def test_state_step_executes_trajectory_and_schedules_next():
    world, now = _world()
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    register_quantity_type("share", units="state")
    world.quantities["share"] = Quantity(name="share", qtype="share", value=0.02, timestamp=now)
    horizon = now + 5.5 * 86400.0
    spec = {"state_var": "share", "form_id": "logistic_growth", "params": {"r": 0.5, "L": 0.1},
            "dt": 1.0, "horizon_ts": horizon, "mode": "increment", "clamp": [0, 1]}
    q = EventQueue(horizon_ts=horizon)
    q.schedule(Event(ts=now + 86400.0, etype="state_step", payload={"step_spec": spec}))
    engine = RolloutEngine(operators=[ops.NonlinearStateStepOperator()])
    branch = engine.run_branch(world, q, seed=0, max_events=10)
    assert len(branch.log) >= 4                      # multiple years stepped
    assert world.quantities["share"].value > 0.02    # grew
    assert world.quantities["share"].value <= 0.1 + 1e-9   # saturates at L (bounded, no overshoot)


def test_history_features_are_leakage_free():
    world, now = _world()
    e = Entity(identity="a", entity_type="person")
    history.record_exposure(e, at=now - 3600, source="s1")
    history.record_exposure(e, at=now + 99999, source="future")   # a FUTURE event
    hf = history.history_features(e, now=now)
    assert hf["cum_count"] == 1.0, "future exposure must NOT enter the feature"


def test_posterior_propagation_beats_naive_on_curved_form():
    # Hill is convex near k; E[f(X)] must differ from f(E[X]) under uncertainty in k
    form = get_form("hill")
    post = {"k": ParamPosterior("k", envelope={"mean": 4.0, "sd": 2.0, "lo": 0.5})}
    pm = lambda s: {"theta": 1.0, "n": 3.0, "k": s["k"]}
    pr = propagate(form, post, {"x": 4.0}, n=4000, rng=random.Random(1), param_map=pm)
    assert abs(pr.jensen_gap) > 0.01                 # a real, measured gap
    assert pr.mean != pr.naive


def test_delta_method_flags_when_per_particle_required():
    form = get_form("hill")
    post = {"k": ParamPosterior("k", envelope={"mean": 4.0, "sd": 2.5, "lo": 0.5})}
    pm = lambda s: {"theta": 1.0, "n": 3.0, "k": s["k"]}
    dm = delta_method_gap(form, post, {"x": 4.0}, param_map=pm)
    assert "delta_method_gap" in dm


def test_operator_records_posterior_propagation_flag():
    world, now = _world()
    world.entities["a"] = Entity(identity="a", entity_type="person")
    spec = {"form_id": "hill", "params": {"theta": 1.0, "n": 3.0}, "outcome_var": "adopt", "actor": "a",
            "param_posteriors": {"k": {"envelope": {"mean": 4.0, "sd": 2.0, "lo": 0.5}}},
            "param_map": {"k": "k"}, "inputs": {"x": 4.0}, "output": "rate", "window_days": 1.0}
    op = ops.NonlinearMechanismOperator()
    ev = Event(ts=now, etype="nonlinear_transition", participants=["a"], payload={"nonlinear_spec": spec})
    delta, _ = op.run(world, ev, random.Random(2))
    assert delta.uncertainty.get("posterior_propagated") is True
    assert "jensen_gap" in delta.uncertainty


def test_refractory_suppresses_response():
    world, now = _world()
    world.entities["a"] = Entity(identity="a", entity_type="person")
    history.record_exposure(world.entities["a"], at=now - 600, source="s")   # 10 min ago
    spec = {"form_id": "logistic", "params": {"weights": {}, "intercept": 9.0}, "outcome_var": "adopt",
            "actor": "a", "output": "prob", "history_window": {"refractory_h": 24.0}}
    op = ops.NonlinearMechanismOperator()
    ev = Event(ts=now, etype="nonlinear_transition", participants=["a"], payload={"nonlinear_spec": spec})
    delta, _ = op.run(world, ev, random.Random(0))
    assert "refractory_suppressed" in delta.reason_codes
