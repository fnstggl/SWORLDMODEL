"""Pins the quarantine guardrail: deprecated inner entries warn loudly and name the canonical path, while
the canonical path and pipeline's non-deprecated exports stay fully importable/usable."""
import warnings

import pytest

from swm.world_model_v2._quarantine import CANONICAL_ENTRY, quarantined


def test_quarantined_decorator_warns_once_and_passes_through():
    calls = {"n": 0}

    @quarantined(reason="test")
    def f(x):
        calls["n"] += 1
        return x * 2

    assert getattr(f, "__quarantined__") is True
    assert f.__use_instead__ == CANONICAL_ENTRY
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert f(3) == 6                       # return value unchanged (never a hard block)
        assert f(4) == 8
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep) == 1                        # warns ONCE per process, not per call
    assert "canonical" in str(dep[0].message).lower() or "simulate_world" in str(dep[0].message)
    assert calls["n"] == 2


def test_pipeline_simulate_is_quarantined_but_helpers_are_not():
    # the deprecated ENTRY is marked; the module's shared helpers (used by the canonical path) are not
    from swm.world_model_v2.pipeline import simulate, result_from_run, _operator_delta_census
    assert getattr(simulate, "__quarantined__", False) is True
    assert "simulate_world" in simulate.__use_instead__
    assert not getattr(result_from_run, "__quarantined__", False)
    assert not getattr(_operator_delta_census, "__quarantined__", False)


def test_canonical_entry_still_imports():
    from swm.world_model_v2.unified_runtime import simulate_world
    assert callable(simulate_world)
    assert not getattr(simulate_world, "__quarantined__", False)
