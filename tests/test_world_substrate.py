"""Tests for the persistent World substrate (EXP-070): entities, couplings, shared clock, cross-scale feedback."""
from swm.world.substrate import Entity, World, rollout


def test_advance_steps_state_on_shared_clock():
    w = World().add(Entity("x", "environment", {"v": 0.0},
                           step_fn=lambda s, inp, dt, rng: {"v": s["v"] + dt}))
    w.advance(1.0); w.advance(2.0)
    assert w.entities["x"].state["v"] == 3.0 and w.clock == 3.0


def test_coupling_wires_output_into_input():
    w = (World()
         .add(Entity("src", "individual", {"out": 0.7}))
         .add(Entity("dst", "institution", {"seen": 0.0},
                     step_fn=lambda s, inp, dt, rng: {"seen": inp.get("signal", -1)}))
         .couple("src", "dst", lambda s: {"signal": s["out"]}))
    w.advance(1.0)
    assert w.entities["dst"].state["seen"] == 0.7          # src's output reached dst's input


def test_query_reads_outcome_without_mutation():
    w = World().add(Entity("bank", "institution", {"distress": 0.8},
                           readout_fn=lambda s, inp, rng: "FAILED" if s["distress"] > 0.5 else "ok"))
    assert w.query("bank") == "FAILED" and w.entities["bank"].state["distress"] == 0.8


def test_without_couplings_isolates_scales():
    w = (World().add(Entity("a", "x", {"v": 1.0}))
         .add(Entity("b", "y", {"seen": 0.0}, step_fn=lambda s, inp, dt, rng: {"seen": inp.get("sig", -9)}))
         .couple("a", "b", lambda s: {"sig": s["v"]}))
    sep = w.without_couplings()
    sep.advance(1.0)
    assert sep.entities["b"].state["seen"] == -9           # edge cut -> no input reaches b
    assert len(sep.couplings) == 0 and set(sep.entities) == {"a", "b"}


def _bank_world(coupled):
    w = World().add(Entity("rumor", "environment", {"level": 0.0},
                    step_fn=lambda s, inp, dt, rng: {"level": min(1.0, 0.95 * s["level"]
                            + 0.6 * inp.get("bank_distress", 0.0) + inp.get("shock", 0.0))}))
    for i in range(8):
        w.add(Entity(f"d{i}", "individual", {"intent": 0.1},
                     step_fn=lambda s, inp, dt, rng: {"intent": min(1.0, max(0.0, s["intent"] + 0.4 * (
                         inp.get("rumor", 0.0) + 0.7 * inp.get("bank_distress", 0.0) - s["intent"])))}))
    w.add(Entity("bank", "institution", {"distress": 0.0},
                 step_fn=lambda s, inp, dt, rng: {"distress": (sum(v > 0.5 for k, v in inp.items()
                         if k.startswith("d")) / 8) if any(k.startswith("d") for k in inp) else 0.0},
                 readout_fn=lambda s, inp, rng: "FAILED" if s["distress"] > 0.5 else "stable"))
    if coupled:
        for i in range(8):
            w.couple("rumor", f"d{i}", lambda s: {"rumor": s["level"]})
            w.couple("bank", f"d{i}", lambda s: {"bank_distress": s["distress"]})
            w.couple(f"d{i}", "bank", (lambda j: (lambda s: {f"d{j}": s["intent"]}))(i))
        w.couple("bank", "rumor", lambda s: {"bank_distress": s["distress"]})
    return w


def test_cross_scale_feedback_only_with_coupling():
    """The SAME shock cascades to failure in the coupled world and fizzles when the scales are separate —
    emergent cross-scale contagion the substrate exists to capture."""
    def outcome(coupled):
        w = _bank_world(coupled)
        for step in range(14):
            w.advance(1.0, external=({"rumor": {"shock": 0.8}} if step == 0 else None))
        return w.query("bank")
    assert outcome(coupled=True) == "FAILED"
    assert outcome(coupled=False) == "stable"


def test_rollout_advances_to_horizon():
    w = World().add(Entity("x", "e", {"v": 0.0}, step_fn=lambda s, inp, dt, rng: {"v": s["v"] + dt},
                           readout_fn=lambda s, inp, rng: s["v"]))
    assert rollout(w, "x", horizon=5.0, dt=1.0) == 5.0
