"""Core integration tests: the scenario-generated action layer end to end, offline.

The candidate is compiled once into kernel ops, executes through the canonical rollout
(control plane routes observations; a scripted actor reacts through the SAME kernel), the
goal contract reads the evolved records, and the API returns one honest result.
"""
import copy
import json

import pytest

from swm.world_model_v2.phase13.contracts import DecisionProblem
from swm.world_model_v2.phase13.scenario_actions.api import (evaluate_actions_generated,
                                                             evaluate_proposed_actions,
                                                             is_generated_context)
from swm.world_model_v2.phase13.scenario_actions.candidates import (ConcreteAction,
                                                                    ConditionSpec, PlanStep)
from swm.world_model_v2.phase13.scenario_actions.execution import (observable_projection,
                                                                   plan_intervention)
from tests.scenario_fixtures import T0, DAY, build_context, council_schema

MAKER, OFFICER = "rivera", "chen"


def problem(**kw):
    kw.setdefault("decision_id", "d1")
    kw.setdefault("decision_maker", MAKER)
    kw.setdefault("authority", ["petitioner"])
    kw.setdefault("horizon", "2023-12-31T00:00:00Z")
    return DecisionProblem(**kw)


def filing_candidate(cid="file_petition"):
    """File the petition (kernel ops precompiled — the offline path with no LLM compiler),
    which emits a public notice; the scripted officer reacts by granting."""
    step = PlanStep(
        step_id=f"{cid}_s1",
        intent="file the variance petition for parcel 12 at the clerk window",
        target_ids=[OFFICER], channel="clerk_window",
        exact_content="Petition: reduce setback on parcel 12 from 20ft to 12ft.",
        visibility="public")
    step.compiled_ops = [
        {"op": "create_or_update_record", "record_type": "variance_petition",
         "record_id": "pet_12", "status": "filed",
         "fields": {"parcel": "12", "request": "setback reduction",
                    "status": "filed", "matter": "pet_12"}},
        {"op": "emit_semantic_event", "semantic_type_id": "petition_filed_notice",
         "exact_content": "Petition: reduce setback on parcel 12 from 20ft to 12ft.",
         "structured_fields": {"parcel": "12"}, "direct_targets": [OFFICER],
         "intended_visibility": "public"}]
    step.compile_meta = {"compiler": "test_precompiled"}
    return ConcreteAction(candidate_id=cid, actor_id=MAKER, title="file the petition",
                          strategy_class="direct_filing",
                          causal_theory="filing opens the panel's decision", steps=[step])


def officer_grants(world, situation):
    if "petition_filed_notice" not in situation:
        return None
    return [{"op": "create_or_update_record", "record_type": "panel_member_decision",
             "fields": {"position": "approve", "matter": "pet_12"}, "status": "decided"},
            {"op": "create_or_update_record", "record_type": "variance_grant",
             "record_id": "grant_12", "status": "issued",
             "fields": {"parcel": "12", "status": "issued"}}]


def ctx(script=None, **kw):
    schema = council_schema()
    return build_context(schema, [MAKER, OFFICER],
                         script=script if script is not None
                         else {OFFICER: officer_grants}, **kw)


# ------------------------------------------------------------------ end to end
def test_end_to_end_action_changes_world_through_canonical_runtime():
    world_context, report, runtime = ctx(n_particles=4)
    res = evaluate_actions_generated(problem(), [filing_candidate()], world_context,
                                     goal_text="obtain the variance", seed=3)
    sr = res.provenance["scenario_report"]
    ev = sr["evaluations"]["file_petition"]
    assert ev["success_count"] == ev["n_particles"] == 4       # officer granted everywhere
    ref = sr["evaluations"]["do_nothing"]
    assert ref["success_count"] == 0                            # reference unchanged
    assert res.recommended == "file_petition"
    assert runtime.invocations, "the affected actor was never invoked"
    assert report["scenario_events_emitted"] >= 4
    assert report["observations_delivered"] >= 1


def test_do_nothing_reference_is_matched_and_inert():
    world_context, report, _ = ctx(n_particles=3)
    res = evaluate_actions_generated(problem(), [filing_candidate()], world_context,
                                     goal_text="obtain the variance", seed=1)
    sr = res.provenance["scenario_report"]
    assert sr["evaluations"]["do_nothing"]["success_count"] == 0
    assert sr["evaluations"]["do_nothing"]["forbidden_count"] == 0


def test_exact_content_survives_into_the_world():
    world_context, report, _ = ctx(n_particles=2)
    cand = filing_candidate()
    res = evaluate_actions_generated(problem(), [cand], world_context,
                                     goal_text="obtain the variance", seed=0)
    compiled = res.provenance["scenario_report"]["compiled_effects"]["file_petition"]
    blob = json.dumps(compiled)
    assert "reduce setback on parcel 12 from 20ft to 12ft" in blob


def test_infeasible_resources_gate_before_simulation():
    world_context, report, _ = ctx(n_particles=3, maker_resources={"filing_credits": 0.0})
    cand = filing_candidate()
    cand.steps[0].resource_commitments = {"filing_credits": 5.0}
    res = evaluate_actions_generated(problem(), [cand], world_context,
                                     goal_text="obtain the variance", seed=0)
    sr = res.provenance["scenario_report"]
    assert any(r["candidate_id"] == "file_petition" for r in sr["rejected"])
    assert "file_petition" not in sr["evaluations"]
    assert res.recommendation_kind in ("abstain", "action")


def test_contingent_step_fires_only_when_observed_condition_holds():
    """Step 2 is gated on OBSERVING the grant record; in the no-grant world it lapses."""
    def officer_silent(world, situation):
        return None

    world_context, report, _ = ctx(script={OFFICER: officer_silent}, n_particles=2)
    cand = filing_candidate("contingent_plan")
    thanks = PlanStep(
        step_id="contingent_plan_s2",
        intent="send a public thank-you statement after the grant issues",
        conditions=[ConditionSpec(kind="record", record_type="variance_grant",
                                  op="exists", description="grant exists")],
        after_steps=["contingent_plan_s1"], visibility="public", max_condition_checks=2)
    thanks.compiled_ops = [{"op": "emit_semantic_event",
                            "semantic_type_id": "petition_filed_notice",
                            "exact_content": "thank you",
                            "intended_visibility": "public"}]
    thanks.compile_meta = {"compiler": "test_precompiled"}
    cand.steps.append(thanks)
    res = evaluate_actions_generated(problem(), [cand], world_context,
                                     goal_text="obtain the variance", seed=0)
    diag = res.provenance["scenario_report"]["trajectory_summaries"]["contingent_plan"]
    assert diag["step_stats"]["contingent_plan_s2"]["lapsed"] == 2      # never fired
    assert diag["step_stats"]["contingent_plan_s1"]["completed"] == 2


def test_policy_never_reads_hidden_state():
    """The observable projection exposes only visible records/info/own resources."""
    world_context, _, _ = ctx(n_particles=1)
    w = world_context["initial"].sample_particles(1)[0]
    from swm.world_model_v2.generated_world import execute_kernel_ops, generated_report
    from swm.world_model_v2.transitions import StateDelta
    ctx_k = {"actor_id": OFFICER, "action_id": "x", "now": w.clock.now,
             "report": generated_report(), "events": [], "quarantined": [],
             "compiler": "test"}
    d = StateDelta(at=w.clock.now, event_type="t", operator="t")
    execute_kernel_ops(w, [{"op": "create_or_update_record",
                            "record_type": "panel_member_decision",
                            "record_id": "secret_note", "visibility": "participants",
                            "audience": [OFFICER],
                            "fields": {"position": "draft", "matter": "pet_12"}}], ctx_k, d)
    proj = observable_projection(w, MAKER)
    assert all(r["record_id"] != "secret_note" for r in proj["records"])
    proj_officer = observable_projection(w, OFFICER)
    assert any(r["record_id"] == "secret_note" for r in proj_officer["records"])


def test_natural_language_action_enters_runtime_without_catalog(monkeypatch):
    """A user phrase absent from all source verb lists compiles (offline: scaffold path,
    visibly classified) and executes — never coerced into a registered verb."""
    world_context, report, _ = ctx(n_particles=2)
    res = evaluate_proposed_actions(
        "obtain the variance",
        ["hand-deliver a revised parcel drawing to chen with a note offering to narrow "
         "the driveway"],
        world_context, problem=problem(), seed=0)
    sr = res.provenance["scenario_report"]
    user = [c for c in sr["candidates"] if c["candidate_id"].startswith("user_")]
    assert user and "hand-deliver a revised parcel drawing" in user[0]["original_text"]
    ev = sr["evaluations"].get("user_1")
    assert ev is not None, "user action did not reach simulation"
    compiled = sr["compiled_effects"]["user_1"]
    blob = json.dumps(compiled)
    assert "unmodeled_actor_action" in blob        # offline: visible scaffold, not a verb
    assert "hand-deliver a revised parcel drawing" in blob   # exact content preserved


def test_input_problem_never_mutated():
    world_context, _, _ = ctx(n_particles=2)
    p = problem()
    before = copy.deepcopy(p.candidate_actions)
    evaluate_actions_generated(p, [filing_candidate()], world_context,
                               goal_text="obtain the variance", seed=0)
    assert p.candidate_actions == before
    assert p.generated_action_permission is True


def test_is_generated_context_detection():
    world_context, _, _ = ctx(n_particles=1)
    assert is_generated_context(world_context)


def test_matched_worlds_share_exogenous_identity():
    """The same seed produces identical reference trajectories run-to-run (replayable)."""
    wc1, _, _ = ctx(n_particles=3)
    wc2, _, _ = ctx(n_particles=3)
    r1 = evaluate_actions_generated(problem(), [filing_candidate()], wc1,
                                    goal_text="obtain the variance", seed=9)
    r2 = evaluate_actions_generated(problem(), [filing_candidate()], wc2,
                                    goal_text="obtain the variance", seed=9)
    e1 = r1.provenance["scenario_report"]["evaluations"]
    e2 = r2.provenance["scenario_report"]["evaluations"]
    assert e1["file_petition"]["success_count"] == e2["file_petition"]["success_count"]
    assert e1["do_nothing"]["predicate_counts"] == e2["do_nothing"]["predicate_counts"]
