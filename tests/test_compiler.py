"""Tests for the world-model compiler (EXP-064): safe eval, spec parsing, mechanism dispatch, front door."""
import pytest

from swm.api.compiler import CompiledModel, StructuralCompiler, cached_compile_fn
from swm.api.model_spec import parse_spec, safe_eval
from swm.api.world_model import WorldModel


# ---- the safe structural-equation evaluator ----
def test_safe_eval_arithmetic_and_funcs():
    ns = {"a": 0.6, "b": 0.2}
    assert abs(safe_eval("0.3*(a - b)", ns) - 0.12) < 1e-9
    assert abs(safe_eval("max(a, b)", ns) - 0.6) < 1e-9
    assert abs(safe_eval("a if a > b else b", ns) - 0.6) < 1e-9


def test_safe_eval_rejects_unsafe():
    for bad in ("__import__('os').system('ls')", "a.__class__", "open('x')", "().__class__.__bases__"):
        with pytest.raises(ValueError):
            safe_eval(bad, {"a": 1.0})


def test_parse_spec():
    spec = parse_spec({"mechanism": "generic_scm",
                       "variables": [{"name": "x", "value": 0.4, "est_sd": 0.02, "volatility": 0.01}],
                       "equations": {"x": "0.1*(0.8 - x)"}, "outcome": {"variable": "x"}, "horizon": 5})
    assert spec.mechanism == "generic_scm" and spec.var("x").value == 0.4 and spec.horizon == 5.0


# ---- mechanism dispatch ----
def test_generic_scm_runs_and_decomposes():
    spec = parse_spec({"mechanism": "generic_scm",
                       "variables": [{"name": "v", "value": 0.49, "est_sd": 0.01, "volatility": 0.02}],
                       "equations": {"v": "0.2*(0.55 - v)"},
                       "outcome": {"variable": "v", "event": {"op": ">", "value": 0.5}}, "horizon": 10})
    out = CompiledModel(spec).run(n=3000)
    assert out["mechanism"] == "generic_scm" and 0.0 <= out["p_event"] <= 1.0
    assert "irreducible_frac" in out["uncertainty"]


def test_bracket_reproduces_calibrated_odds():
    spec = parse_spec({"mechanism": "bracket", "outcome": {"target": "A"},
                       "extra": {"competitors": [{"name": "A", "strength": 1660, "est_sd": 40},
                                                 {"name": "B", "strength": 1600, "est_sd": 40},
                                                 {"name": "C", "strength": 1560, "est_sd": 40},
                                                 {"name": "D", "strength": 1520, "est_sd": 40}],
                                 "series_length": 7, "home_advantage": 100}})
    out = CompiledModel(spec).run(n=4000)
    assert out["mechanism"] == "bracket" and out["favorite"] == "A"
    # the strongest team is favored but far from certain (irreducible tournament variance)
    assert 0.30 < out["p_target"] < 0.75


def test_committee_and_electorate_and_single_agent_run():
    comm = parse_spec({"mechanism": "committee", "outcome": {"event": {"op": ">", "value": 0.5}},
                       "extra": {"agents": [{"id": f"a{i}", "position": p} for i, p in
                                            enumerate([0.8, 0.7, 0.3, 0.2, 0.55])], "rounds": 4}})
    assert 0.0 <= CompiledModel(comm).run(n=1500)["p_event"] <= 1.0

    elec = parse_spec({"mechanism": "electorate", "outcome": {"event": {"op": ">", "value": 0.5}},
                       "extra": {"cells": [{"stance": 0.6, "weight": 3, "est_sd": 0.03},
                                           {"stance": 0.4, "weight": 2, "est_sd": 0.03}]}})
    assert 0.0 <= CompiledModel(elec).run(n=1000)["p_event"] <= 1.0

    sa = parse_spec({"mechanism": "single_agent",
                     "extra": {"person": {"trait_openness": 0.7}, "message": {"clarity": 0.8}}})
    assert 0.0 <= CompiledModel(sa).run(n=1000)["p_respond_mean"] <= 1.0


# ---- the front door ----
def test_world_model_simulate_dispatches():
    q = "Will it pass?"
    cache = {q: {"mechanism": "committee", "outcome": {"event": {"op": ">", "value": 0.5}},
                 "extra": {"agents": [{"id": "a", "position": 0.9}, {"id": "b", "position": 0.8}],
                           "rounds": 3}, "rationale": "a committee vote"}}
    wm = WorldModel(compiler=StructuralCompiler(cached_compile_fn(cache)))
    out = wm.simulate(q, key=q)
    assert out["mechanism"] == "committee" and out["forecast"]["p_event"] is not None
    assert out["spec"]["rationale"] == "a committee vote" and "headline" in out


def test_cached_backend_raises_on_miss():
    wm = WorldModel(compiler=StructuralCompiler(cached_compile_fn({})))
    with pytest.raises(KeyError):
        wm.simulate("unknown question", key="unknown question")
