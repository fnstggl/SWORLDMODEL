"""The capstone wiring: WorldModel.simulate auto-grounds a compiled spec's high-leverage variables through the
grounder before running — so any question simulates on the measured world, not the LLM's guessed state.
Deterministic: a cached compile_fn + a mock grounder (no network)."""
from swm.api.compiler import StructuralCompiler
from swm.api.state_grounding import DataGrounder, StateGrounder
from swm.api.world_model import WorldModel, general_world_model

SPEC = {"mechanism": "calibrated_readout",
        "variables": [{"name": "inflation", "value": 0.5, "est_sd": 0.3, "weight": 4.0, "weight_sd": 0.5,
                       "center": 0.5, "lo": 0.0, "hi": 1.0},
                      {"name": "noise", "value": 0.5, "est_sd": 0.3, "weight": 0.001, "weight_sd": 0.1,
                       "center": 0.5, "lo": 0.0, "hi": 1.0}],
        "extra": {"intercept": 0.0}, "outcome": {"event": {"op": ">", "value": 0.5}}}


def _wm(grounder):
    return WorldModel(compiler=StructuralCompiler(lambda prompt: dict(SPEC)), grounder=grounder, validate=False)


def test_simulate_auto_grounds_high_leverage_variable():
    # the router measures inflation=0.9 (high-leverage); noise (negligible weight) is left at its prior
    fetch = lambda var, as_of: (0.9, 0.02) if var == "inflation" else None
    res = _wm(StateGrounder(default=DataGrounder(fetch, name="web"))).simulate("Will it clear the bar?")
    g = res["grounding"]
    assert g is not None and g["grounded"] == 1                       # exactly the high-leverage var grounded
    vals = {v[0]: v[1] for v in res["spec"]["variables"]}
    assert vals["inflation"] == 0.9 and vals["noise"] == 0.5          # measured value flowed into the run spec
    rec = {r["var"]: r for r in g["detail"]}
    assert rec["inflation"]["grounded"] and rec["inflation"]["source"] == "web"


def test_grounding_shifts_the_forecast():
    # grounding the high-leverage variable up (0.9 vs the 0.5 guess) must raise P(event) vs the ungrounded run
    ungrounded = _wm(None).simulate("Q")
    grounded = _wm(StateGrounder(default=DataGrounder(lambda v, a: (0.9, 0.02) if v == "inflation" else None))
                   ).simulate("Q")
    assert ungrounded["grounding"] is None
    assert grounded["forecast"]["p_event"] > ungrounded["forecast"]["p_event"]


def test_ungroundable_variable_leaves_spec_at_prior():
    res = _wm(StateGrounder(default=DataGrounder(lambda v, a: None))).simulate("Q")   # nothing measurable
    assert res["grounding"]["grounded"] == 0
    assert {v[0]: v[1] for v in res["spec"]["variables"]}["inflation"] == 0.5          # honest prior, no fake


def test_grounder_failure_does_not_break_the_run():
    class Boom:
        def ground_spec(self, spec, question=None, *, as_of=None):
            raise RuntimeError("grounder exploded")
    res = _wm(Boom()).simulate("Q")
    assert "error" in res["grounding"] and res["forecast"].get("p_event") is not None   # still forecasts


def test_general_world_model_factory_wiring():
    # ground=False keeps it offline/deterministic; the factory still assembles a working compile->run pipeline
    wm = general_world_model(compile_fn=lambda p: dict(SPEC), ground=False, validate=False)
    assert wm.grounder is None
    res = wm.simulate("anything")
    assert res["grounding"] is None and res["forecast"]["mechanism"] == "calibrated_readout"
