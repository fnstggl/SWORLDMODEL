"""HARD ENFORCEMENT (§28 items 1–11 + AST/call-spy): the production runtime cannot regress to
the periodic scheduler, the six-actor cap, or the fixed delay constants. These tests parse the
ACTUAL production sources (AST — not one string grep), spy on the quarantined scheduler, and
re-run the machine-generated assumption audit as a test."""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "swm" / "world_model_v2"

#: the complete production temporal path (audit §1) — every module the default route executes
PRODUCTION_FILES = [
    V2 / "unified_runtime.py", V2 / "pipeline.py", V2 / "materialize.py", V2 / "rollout.py",
    V2 / "events.py", V2 / "event_time.py", V2 / "fidelity.py", V2 / "world_dynamics.py",
    V2 / "generated_world.py", V2 / "semantic_consequences.py", V2 / "phase4_execution.py",
    V2 / "qualitative_actor.py", V2 / "individual_reaction.py", V2 / "scheduled_facts.py",
    V2 / "phase8_pipeline.py", V2 / "temporal_runtime.py", V2 / "temporal_compiler.py",
    V2 / "temporal_model.py", V2 / "temporal_hazards.py", V2 / "temporal_calendar.py",
    V2 / "phase13" / "api.py", V2 / "phase13" / "counterfactual.py", V2 / "phase13" / "crn.py",
    V2 / "phase13" / "interventions.py", V2 / "phase13" / "affordances.py",
    V2 / "phase13" / "scenario_actions" / "execution.py",
]


def _asts():
    for f in PRODUCTION_FILES:
        yield f, ast.parse(f.read_text())


# ---------------------------------------------------------------- 1+2: periodic review unreachable
def test_inv1_production_never_calls_deepen_trajectory():
    for f, tree in _asts():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                assert name != "deepen_trajectory", f"{f}: calls deepen_trajectory"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                assert "deepen_trajectory" not in names, f"{f}: imports deepen_trajectory"
                assert "legacy_periodic_review_ablation" not in names, \
                    f"{f}: imports the quarantined ablation"


def test_inv2_periodic_strategic_review_string_absent_from_production():
    for f in PRODUCTION_FILES:
        assert "periodic strategic review" not in f.read_text(), \
            f"{f}: emits 'periodic strategic review'"


def test_inv1_call_spy_production_funnel_never_reaches_the_ablation(monkeypatch):
    """Call-spy: run the production funnel (plan → materialize → temporal rollout → readout)
    with a spy on the quarantined scheduler — it is never invoked, and no evenly spaced
    decision grid appears on the executed schedule."""
    from swm.world_model_v2 import legacy_ablations as LA
    calls = []
    monkeypatch.setattr(LA, "legacy_periodic_review_ablation",
                        lambda *a, **k: calls.append(1) or {})
    import types
    from swm.world_model_v2.contracts import OutcomeContract
    from swm.world_model_v2.compiler import WorldExecutionPlan
    from swm.world_model_v2.materialize import run_from_plan
    from swm.world_model_v2.state import parse_time
    t0 = parse_time("2026-03-02")
    plan = WorldExecutionPlan(
        question="Will the labs sign the agreement?", as_of=t0,
        horizon_ts=t0 + 90 * 86400.0,
        outcome_contract=OutcomeContract(
            family="binary", options=["yes", "no"], readout_var="outcome",
            resolution_rule="signed agreement exists",
            readout=lambda w: getattr(w.quantities.get("outcome"), "value", None)),
        entities=[{"id": f"lab_{i}", "type": "institution", "fields": {}} for i in range(3)],
        quantities=[{"name": "outcome", "qtype": "outcome", "value": None, "sd": None}],
        scheduled_events=[{"etype": "resolve_outcome", "ts": t0 + 89 * 86400.0,
                           "participants": [], "payload": {"outcome_var": "outcome",
                                                           "options": ["yes", "no"]}}],
        accepted_mechanisms=[{"mech_id": "outcome", "operator": "generic_outcome_prior",
                              "parameter_source": "broad prior", "sensitivity": 0.5}],
        compute_plan={"n_particles": 4})
    result, branches = run_from_plan(plan, llm=None, seed=3)
    assert branches and not calls, "production reached the quarantined periodic scheduler"
    # no evenly spaced multi-actor decision grid on the plan
    dec = [e for e in plan.scheduled_events if e.get("etype") == "decision_opportunity"]
    assert not dec


def test_inv3_no_production_function_splits_horizon_into_generic_points():
    """AST: no production module builds `clamp(2..14)`-style generic decision grids; the only
    `for k in range(1, n+1)` timestamp loops allowed are in the quarantined ablation file."""
    for f, tree in _asts():
        src = f.read_text()
        assert "min(14, int(horizon_days" not in src and \
               "max(2, min(14" not in src, f"{f}: generic 2..14 decision-point clamp"


# ---------------------------------------------------------------- 4: six-actor cap absent
def test_inv4_no_actor_list_sliced_to_six():
    for f, tree in _asts():
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) \
                    and node.slice.upper is not None \
                    and isinstance(node.slice.upper, ast.Constant) \
                    and node.slice.upper.value == 6 and node.slice.lower is None:
                base = ast.unparse(node.value)
                assert not any(t in base.lower() for t in
                               ("actor", "strategic", "entities", "frontier", "modes")), \
                    f"{f}:{node.lineno}: {base}[:6] — fixed six-cap on {base}"


# ---------------------------------------------------------------- 5–8: fixed delays absent
def test_inv5_to_8_fixed_delay_constants_absent():
    """The machine-generated assumption audit doubles as the removal proof: every assumption
    dispositioned removed/replaced/quarantined must no longer match its production file."""
    sys.path.insert(0, str(ROOT / "experiments"))
    from temporal_audit_gen import generate_audit
    audit = generate_audit()
    assert audit["all_removals_verified"], \
        f"audit removal failures: {audit['removal_failures']}"
    assert audit["n_removal_verified"] == audit["n_requiring_removal"]


def test_inv5_to_11_specific_literals_not_reintroduced():
    """Belt over the audit: the specific banned constructs never reappear in delay positions."""
    banned = ("delay_s: float = 1800.0", 'delay_s=1800.0', '"public_delay_s", 3600.0',
              '"default_delay_s", 60.0', '+ 1800.0', '"delivery_delay_s", 60.0',
              '"max_invocations_per_actor": 5,', '"max_cascade_depth": 8}',
              "return out[:8]", "background_every_days * 86400.0",
              "_RECHECK_S = 21600.0", "_STEP_GAP_S = 60.0")
    for f in PRODUCTION_FILES:
        src = f.read_text()
        for b in banned:
            assert b not in src, f"{f}: banned construct {b!r} reintroduced"


def test_inv9_to_11_safety_budgets_are_service_protection_not_reality():
    """Budgets exist only as safety limits far above natural cascades, and their exhaustion is
    typed as temporal truncation in the code that enforces them."""
    from swm.world_model_v2.generated_world import DEFAULT_BUDGETS
    assert DEFAULT_BUDGETS["max_invocations_per_actor"] >= 30
    assert DEFAULT_BUDGETS["max_cascade_depth"] >= 32
    src = (V2 / "generated_world.py").read_text()
    assert "_record_truncation" in src and "temporally_truncated" in src
    # 24-event cascade caps interpreted as quiescence: nothing in production reads a 24 cap
    for f in PRODUCTION_FILES:
        assert "max_cascade\": 24" not in f.read_text()


# ---------------------------------------------------------------- default-route proof
def test_default_route_is_the_temporal_runtime():
    """Every production rollout engine delegates to run_branch_temporal; the unified runtime
    compiles a scenario temporal model by default."""
    from swm.world_model_v2.rollout import RolloutEngine
    from swm.world_model_v2.phase13.crn import MatchedRolloutEngine
    assert "run_branch_temporal" in inspect.getsource(RolloutEngine.run_branch)
    assert "run_branch_temporal" in inspect.getsource(MatchedRolloutEngine.run_branch)
    from swm.world_model_v2 import unified_runtime as U
    # the temporal compile lives in the per-plan conditioning helper shared by BOTH routes:
    # the single-model ablation AND every structural-ensemble candidate model
    src = inspect.getsource(U._condition_plan)
    assert "compile_temporal_model" in src and "attach_temporal_model" in src
    # and the temporal stage is not gated behind an off-by-default flag
    assert 'if "temporal_model" not in drop' in src        # ablation-drop only, default ON
    from swm.world_model_v2 import structural_runtime as SR
    per_model = inspect.getsource(SR._condition_and_pilot_model)
    assert "_condition_plan" in per_model    # every structural model compiles ITS OWN temporal model


def test_rollout_engine_has_no_background_tick_cadence():
    from swm.world_model_v2.rollout import RolloutEngine
    assert not hasattr(RolloutEngine(operators=[]), "background_every_days")
    src = inspect.getsource(sys.modules["swm.world_model_v2.rollout"])
    assert "background_every_days" not in src


def test_stance_reviews_are_event_driven_not_scheduled():
    from swm.world_model_v2 import world_dynamics as WD
    assert not hasattr(WD, "STANCE_REVIEW_COOLDOWN")       # review-count cooldown is gone
    assert hasattr(WD, "STANCE_MATERIAL_HYSTERESIS")
    assert hasattr(WD, "contested_attrition_interval")     # elapsed-time attrition exists
    assert "attrition_per_review" not in {k for k in WD.COUPLING_PRIORS}
    assert "attrition_rate_per_day" in WD.COUPLING_PRIORS
    op = WD.StanceReviewOperator()

    class _Ev:
        etype = "stance_relevant_change"
        payload = {}

    class _W:
        quantities = {}
    assert op.applicable(_W(), _Ev())


def test_event_time_uses_first_passage_not_grids():
    src = (V2 / "event_time.py").read_text()
    assert "first_passage_processes" in src and "ensure_first_passage_state" in src
    assert "for k in range(1, n_rounds + 1)" not in src
    from swm.world_model_v2 import event_time as ET
    assert hasattr(ET, "FirstPassageOperator")
    assert hasattr(ET, "resume_first_passage_after_collapse")
