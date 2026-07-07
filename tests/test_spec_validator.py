"""Tests for the spec validator + repair loop (EXP-067)."""
from swm.api.compiler import CompiledModel
from swm.api.model_spec import parse_spec
from swm.api.spec_validator import ValidatingCompiler, validate


def _codes(spec_json):
    return {i.code for i in validate(parse_spec(spec_json))}


def test_catches_the_inflation_bug():
    # the real EXP-066 bug: mean-reversion term with an equilibrium (~35) far outside the [0,10] bound
    buggy = {"mechanism": "generic_scm",
             "variables": [{"name": "cpi", "value": 4.2, "est_sd": 0.5, "volatility": 0.3, "lo": 0, "hi": 10}],
             "equations": {"cpi": "0.01*(100 - cpi) - 0.02*(cpi - 3)"},
             "outcome": {"variable": "cpi", "event": {"op": ">", "value": 3}}, "horizon": 12}
    codes = _codes(buggy)
    assert "saturates_bound" in codes or "equilibrium_out_of_bounds" in codes   # the equation bug is caught
    assert "degenerate_outcome" in codes or "trivial_event" in codes            # and its downstream effect


def test_clean_spec_passes():
    good = {"mechanism": "generic_scm",
            "variables": [{"name": "v", "value": 0.45, "est_sd": 0.02, "volatility": 0.02, "lo": 0, "hi": 1}],
            "equations": {"v": "-0.3*(v - 0.5)"},
            "outcome": {"variable": "v", "event": {"op": ">", "value": 0.5}}, "horizon": 8}
    assert not [i for i in validate(parse_spec(good)) if i.severity == "error"]


def test_equilibrium_out_of_bounds_fires():
    spec = {"mechanism": "generic_scm",
            "variables": [{"name": "x", "value": 0.5, "volatility": 0.01, "lo": 0, "hi": 1}],
            "equations": {"x": "-0.5*(x - 2.0)"},                 # pulled toward 2.0, outside [0,1]
            "outcome": {"variable": "x"}, "horizon": 5}
    assert "equilibrium_out_of_bounds" in _codes(spec)


def test_event_threshold_outside_support_fires():
    spec = {"mechanism": "generic_scm",
            "variables": [{"name": "x", "value": 0.5, "volatility": 0.02, "lo": 0, "hi": 1}],
            "equations": {"x": "-0.2*(x - 0.5)"},
            "outcome": {"variable": "x", "event": {"op": ">", "value": 5}}, "horizon": 5}   # 5 outside [0,1]
    assert "event_threshold_outside_support" in _codes(spec)


def test_value_and_volatility_checks():
    oob = {"mechanism": "generic_scm",
           "variables": [{"name": "x", "value": 3.0, "volatility": 0.01, "lo": 0, "hi": 1}],
           "equations": {"x": "-0.2*(x-0.5)"}, "outcome": {"variable": "x"}}
    assert "value_out_of_bounds" in _codes(oob)
    wide = {"mechanism": "generic_scm",
            "variables": [{"name": "x", "value": 0.5, "volatility": 2.0, "lo": 0, "hi": 1}],
            "equations": {"x": "-0.2*(x-0.5)"}, "outcome": {"variable": "x"}, "horizon": 9}
    assert "volatility_too_large" in _codes(wide)


def test_validating_compiler_repairs_to_clean():
    buggy = {"mechanism": "generic_scm",
             "variables": [{"name": "cpi", "value": 4.2, "est_sd": 0.5, "volatility": 0.3, "lo": 0, "hi": 10}],
             "equations": {"cpi": "0.01*(100 - cpi) - 0.02*(cpi - 3)"},
             "outcome": {"variable": "cpi", "event": {"op": ">", "value": 3}}, "horizon": 12}
    fixed = {**buggy, "equations": {"cpi": "-0.3*(cpi - 3)"}}

    class _Stub:
        def compile(self, q, c="", *, key=None):
            return CompiledModel(parse_spec(buggy))

    vc = ValidatingCompiler(compiler=_Stub(), repair_fn=lambda prompt: fixed, max_repairs=2)
    compiled = vc.compile("Will inflation exceed 3%?")
    assert vc.last_report["clean"] and vc.last_report["repairs"] == 1
    out = compiled.run(n=2000)
    assert 0.0 < out["p_event"] < 1.0 and (out["interval_80"][1] - out["interval_80"][0]) > 0.1


def test_no_repair_fn_reports_unclean():
    buggy = {"mechanism": "generic_scm",
             "variables": [{"name": "x", "value": 0.5, "volatility": 0.01, "lo": 0, "hi": 1}],
             "equations": {"x": "0.1*(5 - x)"}, "outcome": {"variable": "x"}}   # pulled to 5, outside [0,1]

    class _Stub:
        def compile(self, q, c="", *, key=None):
            return CompiledModel(parse_spec(buggy))

    vc = ValidatingCompiler(compiler=_Stub(), repair_fn=None)
    vc.compile("q")
    assert not vc.last_report["clean"] and vc.last_report["repairs"] == 0
