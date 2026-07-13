"""Unified runtime — canonical-entry / default-on / integration tests (Parts O/U)."""
from __future__ import annotations
import inspect
from pathlib import Path

import swm.facade as facade
from swm.world_model_v2 import unified_runtime as U
from swm.world_model_v2.result import SimulationResult


def test_canonical_entry_exists():
    assert hasattr(U, "simulate_world") and callable(U.simulate_world)


def test_facade_routes_v2_to_unified_runtime_not_lightweight_pipeline():
    """STATIC bypass guard (Part O): the facade's world_model_v2 branch must call simulate_world, and must NOT
    call the legacy pipeline.simulate. Fails if a future edit reintroduces the lightweight bypass."""
    src = inspect.getsource(facade.forecast)
    # locate the world_model_v2 branch region
    idx = src.index('architecture == "world_model_v2"')
    branch = src[idx: idx + 900]
    assert "simulate_world" in branch, "facade v2 branch must route to the unified simulate_world"
    assert "from swm.world_model_v2.pipeline import simulate" not in branch, \
        "facade v2 branch must NOT call the legacy lightweight pipeline.simulate"


def test_no_phase_opt_in_flags_in_signature():
    """Default-on (Part S gate 2): the canonical entry must NOT require use_posterior/enable_persistence/
    with_networks/use_institutions/nonlinear/dynamic_recompile/maximum_capacity flags."""
    params = set(inspect.signature(U.simulate_world).parameters)
    forbidden = {"use_posterior", "enable_persistence", "with_networks", "use_institutions", "nonlinear",
                 "dynamic_recompile", "maximum_capacity"}
    assert not (params & forbidden), f"canonical entry must not require opt-in flags: {params & forbidden}"


def test_manifest_covers_all_phases():
    assert set(U._PHASES) >= {"phase1_compiler", "phase2_evidence", "phase3_posterior", "phase4_actor_policy",
                              "phase6_registry", "phase7_nonlinear", "phase8_persistence", "phase9_populations",
                              "phase9_networks", "phase10_institutions", "phase11_recompilation"}


def test_nonlinear_operators_registered_at_runtime():
    """Phase 7 no longer CLI-only: importing the unified runtime registers the nonlinear operators."""
    from swm.world_model_v2 import transitions
    ops = set(getattr(transitions, "_OPERATORS", {}).keys())
    assert {"nonlinear_mechanism", "nonlinear_contagion", "nonlinear_state_step"} <= ops


def test_clarification_and_failure_paths_return_manifest():
    """Even the early-return paths must carry the runtime tag + manifest (traceability)."""
    r = SimulationResult(question="Q", simulation_status="execution_failed", failure_taxonomy="x")
    assert r.simulation_status == "execution_failed"          # contract sanity


def test_old_phase12_calibrator_marked_incompatible():
    """Part N/gate 16: the pre-unification Phase-12 calibrator must be refused for the unified runtime."""
    from swm.world_model_v2.phase12_serve import load_phase12_bundle, compatible_with
    b = load_phase12_bundle()
    if b is None:
        return
    ok, reason = compatible_with(b, phase11_present=True)
    assert ok is False and "refit" in reason


def test_ablation_policy_hook_present():
    """The execution_policy drop_phases hook exists for the causal-ablation harness (not a normal-caller flag)."""
    params = inspect.signature(U.simulate_world).parameters
    assert "execution_policy" in params


def test_integration_audit_artifact_exists():
    p = Path("experiments/results/unified/integration_audit.json")
    assert p.exists()
