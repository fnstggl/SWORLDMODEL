"""Regression: the Stage-B executability probe must verify operator instantiation with the SAME
backend configuration the rollout will use.

Since the §19 no-numeric-substitution contract (PR #124, reconciled by #127), instantiating the
qualitative actor policy without a backend RAISES (CompilerExecutionError, taxonomy
unavailable_service) instead of silently serving a numeric psychology. The executability check used
to probe `operators_from_plan(plan, llm=None)`, which under the strict contract rejects EVERY
candidate on any live run ("no executable structural candidate remained ..."), while the test suite
never saw it because conftest sets SWM_ALLOW_NUMERIC_BASELINE=1 (the numeric arena masks the raise).
This file pins both sides: strict mode with a backend passes; strict mode without one fails with the
actor-backend refusal, not a silent numeric substitution. (EXP-107 discovery.)"""
from __future__ import annotations

import pytest

from swm.world_model_v2.ensemble_compiler import _executability_check
from tests.test_structural_ensemble import decomp_payload, four_way_llm


@pytest.fixture
def strict_actor_integrity(monkeypatch):
    """Live-run configuration: the §19 numeric-baseline arena is CLOSED."""
    monkeypatch.delenv("SWM_ALLOW_NUMERIC_BASELINE", raising=False)
    monkeypatch.delenv("SWM_LLM_ACTORS", raising=False)
    monkeypatch.delenv("SWM_ACTOR_POLICY", raising=False)


def _plan_with_actor_decisions(llm, seed=3):
    """A candidate plan that DECLARES an actor decision — the live-question shape that forces
    operator instantiation to construct the qualitative actor runtime (a plan with no decisions
    never touches the actor backend, which is why the mini fixtures could not catch this)."""
    from swm.world_model_v2.compiler import compile_world
    decision = {"actor": "avery", "role": "principal", "at": "2025-07-01",
                "candidate_actions": [{"name": "approve", "family": "communication",
                                       "target": {"target_type": "actor", "target_id": "blake"},
                                       "mechanisms_triggered": ["record_action"],
                                       "inclusion_reason": "core"}]}
    llm.decomp_by_model["m0_actor_relationship"] = decomp_payload(
        ["avery", "blake"], lean="weak_yes", hyp="h_a", actor_decisions=[decision])
    return compile_world("Will the initiative be approved?", llm=llm, evidence="",
                         as_of="2025-06-01", horizon="2025-09-01", seed=seed, persist=False,
                         structural_directive="STRUCTURAL DIRECTIVE — independent candidate model "
                                              "'m0_actor_relationship' (perspective: actor).")


def test_executability_passes_with_backend_under_strict_contract(strict_actor_integrity):
    llm = four_way_llm()
    plan = _plan_with_actor_decisions(llm)
    ok, why = _executability_check(plan, llm=llm)
    assert ok, f"a live-configured candidate must verify executable, got: {why}"


def test_executability_without_backend_fails_loudly_not_numerically(strict_actor_integrity):
    llm = four_way_llm()
    plan = _plan_with_actor_decisions(llm)
    ok, why = _executability_check(plan, llm=None)
    assert not ok
    assert "LLM backend" in why or "unavailable" in why, (
        "the strict contract must surface the actor-backend refusal, never substitute a numeric "
        f"psychology silently; got: {why}")


def test_probe_makes_no_llm_calls(strict_actor_integrity):
    """Operator construction must be construction only — the probe may not spend provider calls."""
    llm = four_way_llm()
    plan = _plan_with_actor_decisions(llm)
    calls = []

    def counting(prompt):
        calls.append(prompt)
        return llm(prompt)

    ok, why = _executability_check(plan, llm=counting)
    assert ok, why
    assert calls == [], f"executability probe spent {len(calls)} LLM call(s); must be zero"
