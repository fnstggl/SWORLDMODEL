"""Tests for the ActionWorldModel front door: question -> compile -> do(a) over the world -> best action."""
from swm.api.action_simulate import ActionWorldModel
from swm.api.compiler import StructuralCompiler, cached_compile_fn
from swm.decision.action import set_message
from swm.decision.space import enumerate_actions, grid
from swm.decision.utility import Mean, identity


def _awm(cache):
    return ActionWorldModel(compiler=StructuralCompiler(cached_compile_fn(cache)))


def test_front_door_best_message_single_agent():
    q = "What should I send to re-engage this lapsed user?"
    cache = {q: {"mechanism": "single_agent",
                 "extra": {"person": {"trait_openness": 0.55}, "est_sd": {"trait_openness": 0.05},
                           "message": {"clarity": 0.5}}, "rationale": "one person's reply decision"}}
    awm = _awm(cache)
    space = enumerate_actions([set_message({"clarity": 0.2}, label="vague"),
                              set_message({"clarity": 0.95, "relevance": 0.9}, label="sharp")])
    res = awm.best_action(q, space, identity(), key=q, max_per_arm=3000)
    assert res.best.label == "sharp" and res.navigable is not None


def test_front_door_pricing_generic_scm():
    q = "What price maximizes profit for the pro tier?"
    cache = {q: {"mechanism": "generic_scm",
                 "variables": [{"name": "price", "value": 10.0, "lo": 0.0, "hi": 100.0},
                               {"name": "demand", "value": 1.0, "volatility": 0.01, "lo": 0.0, "hi": 1.0}],
                 "equations": {"price": "0.0", "demand": "0.6*((1 - price/100) - demand)"},
                 "outcome": {"expr": "price*demand"}, "horizon": 8, "rationale": "demand curve"}}
    awm = _awm(cache)
    res = awm.best_action(q, grid("price", 10, 90, 9), identity(), objective=Mean(), key=q, max_per_arm=4000)
    assert 40 <= res.best.action.meta["value"] <= 60
