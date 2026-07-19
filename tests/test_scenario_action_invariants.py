"""§18 architectural invariants of the scenario-generated action layer, as hard tests.

Everything runs OFFLINE against the shared generated-world harness (tests/scenario_fixtures).
Where a prompt must be inspected, a *scripted* llm — a plain ``function(prompt) -> str`` — is
used; where an actor must react, it reacts through the SAME kernel production uses. No test
here touches production source; the two places a genuine production gap is documented are
marked xfail with a precise reason and listed in the final report.
"""
import copy
import json
import random
import re

import pytest

from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.generated_world import (execute_kernel_ops, generated_report,
                                                run_institutional_aggregation)
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.state import SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta

from swm.world_model_v2.phase13.contracts import DecisionProblem
from swm.world_model_v2.phase13 import ontology
from swm.world_model_v2.phase13.scenario_actions import candidates as candidates_mod
from swm.world_model_v2.phase13.scenario_actions.api import (_decide_recommendation,
                                                            _make_evaluator, _schema_of,
                                                            discover_best_action,
                                                            evaluate_actions_generated,
                                                            evaluate_proposed_actions)
from swm.world_model_v2.phase13.scenario_actions.candidates import (ConcreteAction, ConditionSpec,
                                                                    PlanStep, do_nothing_action,
                                                                    merge_equivalent,
                                                                    single_step_action)
from swm.world_model_v2.phase13.scenario_actions.compiler import (_static_violations,
                                                                  ScenarioActionCompiler)
from swm.world_model_v2.phase13.scenario_actions.execution import (PLAN_STEP_ETYPE,
                                                                   ScenarioPlanOperator,
                                                                   condition_holds,
                                                                   observable_projection,
                                                                   plan_intervention)
from swm.world_model_v2.phase13.scenario_actions.feasibility import check_across_particles
from swm.world_model_v2.phase13.scenario_actions.goals import (GoalContract, GoalContractGenerator,
                                                               GoalPredicate, compare_candidates)
from swm.world_model_v2.phase13.scenario_actions.generated_search import ScenarioActionSearch
from swm.world_model_v2.phase13.scenario_actions.language import ActionLanguageGenerator
from swm.world_model_v2.phase13.scenario_actions.roles import (RoleRunner, RoleTrace, blind_labels,
                                                              blind_candidate_view)

from tests.scenario_fixtures import (T0, DAY, build_context, build_world, council_schema)

MAKER, OFFICER = "rivera", "chen"


def problem(**kw):
    kw.setdefault("decision_id", "d1")
    kw.setdefault("decision_maker", MAKER)
    kw.setdefault("authority", ["petitioner"])
    kw.setdefault("horizon", "2023-12-31T00:00:00Z")
    return DecisionProblem(**kw)


def filing_candidate(cid="file_petition",
                     content="Petition: reduce setback on parcel 12 from 20ft to 12ft."):
    step = PlanStep(step_id=f"{cid}_s1", intent="file the variance petition for parcel 12",
                    target_ids=[OFFICER], channel="clerk_window", exact_content=content,
                    visibility="public")
    step.compiled_ops = [
        {"op": "create_or_update_record", "record_type": "variance_petition", "record_id": "pet_12",
         "status": "filed", "fields": {"parcel": "12", "request": "setback reduction",
                                       "status": "filed", "matter": "pet_12"}},
        {"op": "emit_semantic_event", "semantic_type_id": "petition_filed_notice",
         "exact_content": content, "structured_fields": {"parcel": "12"},
         "direct_targets": [OFFICER], "intended_visibility": "public"}]
    step.compile_meta = {"compiler": "test_precompiled"}
    return ConcreteAction(candidate_id=cid, actor_id=MAKER, title="file the petition",
                          strategy_class="direct_filing", steps=[step])


def officer_grants(world, situation):
    if "petition_filed_notice" not in situation:
        return None
    return [{"op": "create_or_update_record", "record_type": "panel_member_decision",
             "fields": {"position": "approve", "matter": "pet_12"}, "status": "decided"},
            {"op": "create_or_update_record", "record_type": "variance_grant", "record_id": "grant_12",
             "status": "issued", "fields": {"parcel": "12", "status": "issued"}}]


def officer_silent(world, situation):
    return None


def ctx(script=None, **kw):
    return build_context(council_schema(), [MAKER, OFFICER],
                         script=script if script is not None else {OFFICER: officer_grants}, **kw)


# ====================================================================== 1
def test_generated_mode_never_lists_the_global_operation_catalog_in_a_prompt():
    """No LLM prompt on the generated path presents the legacy verb catalog as a menu."""
    legacy_verbs = set(ontology._OPERATIONS)          # imported ONLY for the reference set
    assert len(legacy_verbs) >= 50
    recorded = []

    def recording_llm(prompt):
        recorded.append(str(prompt))
        return "{}"

    wc1, _, _ = ctx(n_particles=2)
    evaluate_proposed_actions("obtain the variance",
                              ["hand-deliver a revised parcel drawing to chen"],
                              wc1, problem=problem(), seed=0, llm=recording_llm)
    wc2, _, _ = ctx(n_particles=2)
    discover_best_action("obtain the variance", wc2, problem=problem(), seed=0, llm=recording_llm)
    assert recorded, "no prompts were recorded — the scripted llm was never called"

    def distinct_verbs(prompt):
        return {v for v in legacy_verbs if re.search(r"\b" + re.escape(v) + r"\b", prompt)}

    worst = max((len(distinct_verbs(p)) for p in recorded), default=0)
    assert worst < 8, (f"a prompt listed {worst} legacy catalog verbs as a menu; the generated "
                       f"path must never expose the global operation catalog")


# ====================================================================== 2
def test_generated_mode_never_calls_operation_registered(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("operation_registered must never be consulted on the generated path")

    monkeypatch.setattr(ontology, "operation_registered", boom)
    wc, _, _ = ctx(n_particles=2)
    res = evaluate_actions_generated(problem(), [filing_candidate()], wc,
                                     goal_text="obtain the variance", seed=0)
    assert "scenario_report" in res.provenance
    wc2, _, _ = ctx(n_particles=2)
    evaluate_proposed_actions("obtain the variance", ["walk a drawing over to chen"], wc2,
                              problem=problem(), seed=0)


# ====================================================================== 3
def test_novel_action_phrase_absent_from_source_compiles_and_executes():
    rng = random.Random(20240718)
    words = ["quixotic", "zorbing", "gambit", "flumox", "verdigris", "cascade", "nimbus",
             "obelisk", "tessellate", "gossamer"]
    marker = f"NOVEL{rng.randint(10_000_000, 99_999_999)}"
    phrase = " ".join(rng.choice(words) for _ in range(6)) + " " + marker
    wc, _, _ = ctx(n_particles=2)
    res = evaluate_proposed_actions("obtain the variance", [phrase], wc, problem=problem(), seed=0)
    sr = res.provenance["scenario_report"]
    assert "user_1" in sr["evaluations"], "the novel phrase never reached simulation"
    blob = json.dumps(sr["compiled_effects"]["user_1"])
    assert marker in blob and phrase in blob, "the exact novel text did not survive compilation"


# ====================================================================== 4
def test_cannot_directly_write_another_actors_beliefs():
    """A precompiled mind-write op is quarantined at execution; the world is left unchanged."""
    w = build_world(council_schema(), [MAKER, OFFICER])
    rep = generated_report()
    op = ScenarioPlanOperator(report=rep)
    step = PlanStep(step_id="mw_s1", intent="secretly set chen's stance", visibility="participants")
    step.compiled_ops = [{"op": "create_or_update_record", "record_type": "panel_member_decision",
                          "record_id": "mw", "fields": {"position": "approve",
                                                        "belief_of_chen": "now supportive"}}]
    cand = ConcreteAction(candidate_id="mw", actor_id=MAKER, steps=[step])
    ev = Event(ts=w.clock.now, etype=PLAN_STEP_ETYPE, participants=[MAKER],
               payload={"candidate_id": "mw", "step_id": "mw_s1", "plan": cand})
    delta, _vr = op.run(w, ev, random.Random(0))
    assert rep["steps_failed_at_execution"] == 1
    assert any("all_ops_quarantined" in rc for rc in delta.reason_codes)
    assert rep.get("human_reactions_attempted_directly", 0) >= 1
    assert w.objects == {}, "a mind-write op leaked state into the world"

    # end-to-end through the search/execution path: fails loudly, every particle
    wc, rep2, _ = ctx(script={OFFICER: officer_silent}, n_particles=2)
    bad = ConcreteAction("mind_write", MAKER, title="seize chen's vote", steps=[step])
    evaluate_actions_generated(problem(), [bad], wc, goal_text="obtain the variance", seed=0)
    assert rep2["steps_failed_at_execution"] >= 2
    assert any(fr.get("kind") == "step_ops_quarantined" for fr in rep2["fallback_reasons"])


# ====================================================================== 5
def test_cannot_directly_write_another_actors_choice():
    schema = council_schema()                          # holders of zoning_panel = [chen]
    op = {"op": "create_or_update_record", "record_type": "panel_member_decision",
          "fields": {"position": "approve", "matter": "pet_12"}}
    v = _static_violations(op, schema=schema, maker=MAKER)   # rivera is NOT a holder
    assert any(x.startswith("institutional_decision_write") for x in v)
    # the holder writing their OWN decision record is fine
    assert not any(x.startswith("institutional_decision_write")
                   for x in _static_violations(op, schema=schema, maker=OFFICER))

    # at the world-mechanism level: rivera's forged decision record is NOT counted by the
    # institution, and the kernel refuses to let rivera overwrite chen's real record
    w = build_world(schema, [MAKER, OFFICER])
    d = StateDelta(at=w.clock.now, event_type="t", operator="t")
    ctxk = {"actor_id": MAKER, "action_id": "x", "now": w.clock.now, "report": generated_report(),
            "events": [], "quarantined": [], "compiler": "t"}
    execute_kernel_ops(w, [{"op": "create_or_update_record", "record_type": "panel_member_decision",
                            "record_id": "forged", "fields": {"position": "approve",
                                                              "matter": "pet_12"}}], ctxk, d)
    agg = run_institutional_aggregation(w, "zoning_panel", matter_record_id="pet_12")
    assert agg["cast"] == 0 and agg["passed"] is False, "a non-holder's forged vote was counted"

    # chen writes its OWN real decision; the kernel refuses to let rivera overwrite it
    ctx_chen = {"actor_id": OFFICER, "action_id": "x", "now": w.clock.now,
                "report": generated_report(), "events": [], "quarantined": [], "compiler": "t"}
    execute_kernel_ops(w, [{"op": "create_or_update_record", "record_type": "panel_member_decision",
                            "record_id": "chen_vote", "fields": {"position": "approve",
                                                                 "matter": "pet_12"}}], ctx_chen, d)
    ctx_riv = {"actor_id": MAKER, "action_id": "x", "now": w.clock.now, "report": generated_report(),
               "events": [], "quarantined": [], "compiler": "t"}
    execute_kernel_ops(w, [{"op": "create_or_update_record", "record_type": "panel_member_decision",
                            "record_id": "chen_vote", "fields": {"position": "reject"}}], ctx_riv, d)
    assert ctx_riv["quarantined"], "kernel let rivera overwrite chen's decision record"
    assert w.objects["chen_vote"].attributes["position"] == "approve"   # unchanged


# ====================================================================== 6
def test_cannot_directly_write_a_terminal_outcome():
    import types
    schema = council_schema()                          # outcome predicate: variance_grant exists
    op = {"op": "create_or_update_record", "record_type": "variance_grant",
          "fields": {"parcel": "12", "status": "issued"}}
    v = _static_violations(op, schema=schema, maker=MAKER, language=None)
    assert any(x.startswith("terminal_outcome_write") for x in v)
    # with sole verified authority over that exact record it is permitted (mechanism-owned)
    lang = types.SimpleNamespace(controllable_objects=["grant_12"])
    op2 = dict(op, record_id="grant_12")
    assert not any(x.startswith("terminal_outcome_write")
                   for x in _static_violations(op2, schema=schema, maker=MAKER, language=lang))


# ====================================================================== 7
def test_downstream_social_effects_travel_through_observation_and_actors():
    wc, rep, runtime = ctx(script={OFFICER: officer_grants}, n_particles=3)
    res = evaluate_actions_generated(problem(), [filing_candidate()], wc,
                                     goal_text="obtain the variance", seed=1)
    ev = res.provenance["scenario_report"]["evaluations"]["file_petition"]
    assert ev["success_count"] > 0
    assert rep["observations_delivered"] >= 1
    assert rep["actors_invoked"] >= 1
    assert rep["actor_actions_executed"] >= 1
    # an actor can only ACT after being INVOKED (which follows an observation delivery)
    assert rep["actors_invoked"] >= rep["actor_actions_executed"]

    wc2, rep2, _ = ctx(script={OFFICER: officer_silent}, n_particles=3)
    res2 = evaluate_actions_generated(problem(), [filing_candidate()], wc2,
                                      goal_text="obtain the variance", seed=1)
    assert res2.provenance["scenario_report"]["evaluations"]["file_petition"]["success_count"] == 0
    assert rep2["actor_actions_executed"] == 0, "a grant appeared with no actor reaction"


# ====================================================================== 8
def test_exact_content_terms_targets_timing_observability_survive_compilation():
    terms = {"deposit_usd": 12345, "close_date": "2023-11-05", "contingencies": ["inspection"]}
    timing = T0 + 7 * DAY
    marker = "SETBACK-VERBATIM-9c1f reduce to 12ft exactly"
    cand = filing_candidate("exact", content=marker)
    step = cand.steps[0]
    step.terms = dict(terms)
    step.timing_ts = timing
    step.visibility = "public"
    step.target_ids = [OFFICER]
    d = cand.as_dict()
    s0 = d["steps"][0]
    assert s0["exact_content"] == marker                       # byte-exact content
    assert s0["terms"] == terms                                # value-exact terms
    assert s0["timing_ts"] == timing                           # value-exact timing
    assert s0["target_ids"] == [OFFICER] and s0["visibility"] == "public"
    wc, _, _ = ctx(n_particles=1)
    res = evaluate_actions_generated(problem(), [cand], wc, goal_text="obtain the variance", seed=0)
    compiled = res.provenance["scenario_report"]["compiled_effects"]["exact"]
    assert marker in json.dumps(compiled), "exact content was altered during compilation"


# ====================================================================== 9
def test_unsupported_actions_are_real_world_events_not_history_only_records():
    from swm.world_model_v2.scenario_schema import UNMODELED_EVENT_TYPE
    wc, _, _ = ctx(n_particles=1)
    ev = _make_evaluator(wc, n_particles=1, seed=0, llm=None)
    w0 = ev.particles()[0]
    schema = _schema_of(ev)
    lang = ActionLanguageGenerator(None).generate(problem(), w0, schema)
    phrase = "hand-deliver a revised parcel drawing offering to narrow the driveway MARKER5521"
    cand = single_step_action("user_1", MAKER, phrase, original_text=phrase, source="user")
    report = ScenarioActionCompiler(None).compile_candidate(w0, lang, cand)
    assert report.classification in ("unmodeled", "partially_modeled")
    arm = ev.evaluate_arm("user_1", plan_intervention(cand))
    world = arm.branches[0].world
    hits = [e for e in world.semantic_log
            if e["semantic_type_id"] == UNMODELED_EVENT_TYPE and phrase in e["exact_content"]]
    assert hits, "the unsupported action left no real world event carrying its exact content"
    # the maker's past_actions did NOT silently absorb it as the ONLY trace
    assert world.entities[MAKER].value("past_actions") == []

    # classification is visible in the report
    wc2, _, _ = ctx(n_particles=2)
    res = evaluate_proposed_actions("obtain the variance", [phrase], wc2, problem=problem(), seed=0)
    cand_dict = next(c for c in res.provenance["scenario_report"]["candidates"]
                     if c["candidate_id"] == "user_1")
    cls = cand_dict["provenance"]["compile_report"]["classification"]
    assert cls in ("unmodeled", "partially_modeled")
    assert cand_dict["provenance"].get("model_support") in ("unmodeled", "partially_modeled")


# ====================================================================== 10
def test_every_action_yields_effects_visible_partial_or_hard_rejection():
    infeasible = filing_candidate("starved")
    infeasible.steps[0].resource_commitments = {"filing_credits": 99.0}
    nl = "quietly lobby the neighbors over coffee ZZ8842"
    wc, _, _ = ctx(script={OFFICER: officer_grants}, n_particles=3,
                   maker_resources={"filing_credits": 0.0})
    res = evaluate_actions_generated(problem(), [filing_candidate(), nl, infeasible], wc,
                                     goal_text="obtain the variance", seed=0)
    sr = res.provenance["scenario_report"]
    # (1) fully compiled → real effects
    assert sr["evaluations"]["file_petition"]["success_count"] > 0
    # (2) scaffold → visible partial/unmodeled classification, still simulated
    user = next(c for c in sr["candidates"] if c["candidate_id"].startswith("user_"))
    assert user["provenance"]["compile_report"]["classification"] in ("unmodeled", "partially_modeled")
    assert user["candidate_id"] in sr["evaluations"]
    # (3) infeasible → hard rejection with a typed gate, never silently simulated
    gated = next(r for r in sr["rejected"] if r["candidate_id"] == "starved")
    assert gated["gates"] and all("code" in g for g in gated["gates"])
    assert "starved" not in sr["evaluations"]


# ====================================================================== 11
def test_no_arbitrary_scalar_progress_and_evaluations_are_counts_not_utilities():
    import swm.world_model_v2.phase13.scenario_actions as pkg
    import pathlib
    pkg_dir = pathlib.Path(pkg.__file__).parent
    for f in sorted(pkg_dir.glob("*.py")):
        src = f.read_text()
        assert "pathway_progress" not in src, f"{f.name} writes a scalar pathway_progress"
        assert "mode_progress" not in src, f"{f.name} writes a scalar mode_progress"
    wc, _, _ = ctx(n_particles=3)
    res = evaluate_actions_generated(problem(), [filing_candidate()], wc,
                                     goal_text="obtain the variance", seed=0)
    for cid, e in res.provenance["scenario_report"]["evaluations"].items():
        assert "utility" not in e, f"{cid} evaluation minted a utility scalar"
        assert "predicate_counts" in e and "success_count" in e     # counted evidence


# ====================================================================== 12
def test_no_llm_minted_utility_weight_or_failure_probability_in_ranking():
    import dataclasses
    banned = {"failure_prob", "direct_cost", "indirect_cost"}
    for dc in (ConcreteAction, PlanStep):
        names = {f.name for f in dataclasses.fields(dc)}
        assert not (names & banned), f"{dc.__name__} carries a banned cost/probability field"
    goal = GoalContract(predicates=[GoalPredicate("g", role="desired_terminal",
                                                  record_type="variance_grant", op="exists")])
    evals = {"a": {"n_particles": 2, "success_count": 2, "forbidden_count": 0, "near_miss_count": 0,
                   "by_hypothesis": {}, "quantities": {}},
             "b": {"n_particles": 2, "success_count": 0, "forbidden_count": 0, "near_miss_count": 0,
                   "by_hypothesis": {}, "quantities": {}}}
    cmp = compare_candidates(goal, evals)
    blob = json.dumps(cmp)
    for k in banned:
        assert f'"{k}"' not in blob, f"ranking output carries a {k} field"
    # ranking is by counted frequencies, explicitly NOT a minted utility weight
    assert "no minted utility weights" in cmp["ranking_basis"]
    for item in cmp["ranked"]:
        assert not (set(item) & banned)


# ====================================================================== 13
def test_user_actions_not_coerced_into_registered_verbs():
    phrase = "swing by chen's office and float the driveway idea informally MARK771"
    wc, _, _ = ctx(n_particles=2)
    res = evaluate_proposed_actions("obtain the variance", [phrase], wc, problem=problem(), seed=0)
    cand = next(c for c in res.provenance["scenario_report"]["candidates"]
                if c["candidate_id"] == "user_1")
    assert "operation" not in cand and "family" not in cand
    assert cand["original_text"] == phrase                     # preserved verbatim
    assert not hasattr(ConcreteAction("x", MAKER), "operation")


# ====================================================================== 14
def test_materially_different_actions_are_not_falsely_deduplicated():
    a = filing_candidate("a", content="Offer: narrow the driveway to 9ft.")
    b = filing_candidate("b", content="Offer: raise the fence and keep the driveway.")
    b.steps[0].terms = {"fence_height_ft": 6}
    kept, merges = merge_equivalent([a, b])
    assert len(kept) == 2 and merges == [], "distinct interventions were wrongly merged"
    assert a.identity() != b.identity()


# ====================================================================== 15
def test_paraphrase_merges_only_with_recorded_evidence():
    a = filing_candidate("a", content="Offer A")
    b = filing_candidate("b", content="Offer B (a paraphrase)")
    assert a.identity() != b.identity()
    # no judge → no tier-2 merge
    kept, merges = merge_equivalent([copy.deepcopy(a), copy.deepcopy(b)])
    assert len(kept) == 2 and merges == []
    # a judge that approves → merge WITH claim + evidence + method, recorded on the survivor
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    kept2, merges2 = merge_equivalent([a2, b2],
                                      judge=lambda x, y: (True, "same underlying offer, reworded"))
    assert len(kept2) == 1 and len(merges2) == 1
    rec = merges2[0]
    assert rec["claim"] and rec["evidence"] and rec["method"] == "llm_judged"
    survivor = kept2[0]
    assert any(m["method"] == "llm_judged" for m in survivor.provenance.get("merged_candidates", []))


# ====================================================================== 16
def test_feasibility_across_hypotheses_and_at_execution():
    # (A) per-hypothesis feasibility frequencies when a resource is missing in half the worlds
    def vary(w, i):
        have = 5.0 if i % 2 == 0 else 0.0
        from swm.world_model_v2.state import F
        w.entities[MAKER].set("resources", F(have, status="observed"), key="filing_credits")

    wc, _, _ = build_context(council_schema(), [MAKER, OFFICER], script={OFFICER: officer_silent},
                             maker_resources={"filing_credits": 5.0}, n_particles=4, vary=vary)
    ev = _make_evaluator(wc, n_particles=4, seed=0, llm=None)
    parts = ev.particles()
    assignment = ["res_present" if i % 2 == 0 else "res_absent" for i in range(len(parts))]
    lang = ActionLanguageGenerator(None).generate(problem(), parts[0], _schema_of(ev))
    cand = filing_candidate("hyp")
    cand.steps[0].resource_commitments = {"filing_credits": 5.0}
    feas = check_across_particles(parts, assignment, lang, problem(), cand)
    assert feas["n_feasible"] < feas["n_particles"]            # infeasible somewhere
    assert feas["by_hypothesis"]["res_present"]["feasible"] > 0
    assert feas["by_hypothesis"]["res_absent"]["feasible"] == 0

    # (B) resources present at t0 but consumed by an earlier step → loud execution failure
    wc2, rep2, _ = build_context(council_schema(), [MAKER, OFFICER], script={OFFICER: officer_silent},
                                 maker_resources={"filing_credits": 5.0}, n_particles=2)
    drain = ConcreteAction("drain", MAKER, title="drain then need", steps=[])
    s1 = PlanStep(step_id="drain_s1", intent="transfer the credits away")
    s1.compiled_ops = [{"op": "transfer_conserved_quantity", "resource": "filing_credits",
                        "amount": 5.0, "to": OFFICER}]
    s1.compile_meta = {"compiler": "test"}
    s2 = PlanStep(step_id="drain_s2", intent="file, needing credits", after_steps=["drain_s1"],
                  resource_commitments={"filing_credits": 3.0})
    s2.compiled_ops = [{"op": "create_or_update_record", "record_type": "variance_petition",
                        "record_id": "p2", "status": "filed", "fields": {"status": "filed"}}]
    s2.compile_meta = {"compiler": "test"}
    drain.steps = [s1, s2]
    evaluate_actions_generated(problem(), [drain], wc2, goal_text="obtain the variance", seed=0)
    assert rep2["steps_failed_at_execution"] >= 2
    assert any(fr.get("kind") == "step_infeasible_at_execution" for fr in rep2["fallback_reasons"])


# ====================================================================== 17
def test_matched_rollouts_preserve_exogenous_streams_and_are_deterministic():
    wc1, _, _ = ctx(n_particles=3)
    wc2, _, _ = ctx(n_particles=3)
    r1 = evaluate_actions_generated(problem(), [filing_candidate()], wc1,
                                    goal_text="obtain the variance", seed=9)
    r2 = evaluate_actions_generated(problem(), [filing_candidate()], wc2,
                                    goal_text="obtain the variance", seed=9)
    e1 = r1.provenance["scenario_report"]["evaluations"]
    e2 = r2.provenance["scenario_report"]["evaluations"]
    # reference arm identical run-to-run
    assert e1["do_nothing"]["predicate_counts"] == e2["do_nothing"]["predicate_counts"]
    assert e1["do_nothing"]["success_count"] == e2["do_nothing"]["success_count"]
    # and the intervention arm is reproducible under the same seed
    assert e1["file_petition"]["success_count"] == e2["file_petition"]["success_count"]


# ====================================================================== 18
def _record_adjudicator_prompts():
    recorded = []

    def recording_llm(prompt):
        recorded.append(str(prompt))
        return "{}"

    wc, _, _ = ctx(script={OFFICER: officer_grants}, n_particles=2)
    evaluate_actions_generated(problem(), [filing_candidate("file_petition"),
                                           filing_candidate("second_route", content="a different offer")],
                               wc, goal_text="obtain the variance", seed=0, llm=recording_llm)
    return [p for p in recorded if "final independent adjudicator" in p]


def test_blind_comparison_hides_candidate_provenance_and_source():
    # blind view never carries identity/provenance
    c = filing_candidate("secret_file_petition_id")
    c.source = "goal_backward"
    view = json.dumps(blind_candidate_view(c))
    assert "secret_file_petition_id" not in view and "goal_backward" not in view
    labeled, mapping = blind_labels([filing_candidate("x"), do_nothing_action(MAKER)], seed=3)
    assert all(lab.startswith("OPTION_") for lab, _ in labeled)
    assert set(mapping.values()) == {"x", "do_nothing"}
    # generator/source names never enter the adjudicator prompt
    adj = _record_adjudicator_prompts()
    assert adj, "no adjudication prompt was recorded"
    for p in adj:
        for forbidden in ('"source"', "goal_backward", "baseline", "revision", "affordance"):
            assert forbidden not in p, f"adjudicator prompt leaked generator identity {forbidden!r}"


def test_adjudicator_prompt_never_reveals_candidate_id():
    """Regression for a real leak: adjudicate() embedded the unsanitized TrajectoryDiagnosis
    (candidate_id + step-id-keyed step_stats) into the blind prompt. _blind_diagnosis now
    drops the id and re-keys steps positionally."""
    adj = _record_adjudicator_prompts()
    assert adj, "no adjudication prompt was recorded"
    for p in adj:
        assert "file_petition" not in p and "second_route" not in p, \
            "adjudicator prompt leaked a candidate_id"


# ====================================================================== 19
def _scripted_officer_trigger(world, situation):
    if "GRANT_TRIGGER" in situation:
        return [{"op": "create_or_update_record", "record_type": "variance_grant", "record_id": "g",
                 "status": "issued", "fields": {"parcel": "12", "status": "issued"}}]
    return None


def _content_candidate(cid, content, title):
    st = PlanStep(step_id=f"{cid}_s1", intent="file", target_ids=[OFFICER],
                  exact_content=content, visibility="public")
    st.compiled_ops = [{"op": "emit_semantic_event", "semantic_type_id": "petition_filed_notice",
                        "exact_content": content, "structured_fields": {"parcel": "12"},
                        "direct_targets": [OFFICER], "intended_visibility": "public"}]
    st.compile_meta = {"compiler": "test"}
    c = ConcreteAction(cid, MAKER, title=title, steps=[st])
    return c


def _search(wc, runner_llm, cands, *, seed=0, max_rounds=1, goal=None):
    ev = _make_evaluator(wc, n_particles=wc["n_particles"], seed=seed, llm=None)
    schema = _schema_of(ev)
    w0 = ev.particles()[0]
    lang = ActionLanguageGenerator(None).generate(problem(), w0, schema)
    goal = goal or GoalContractGenerator(None).generate(problem(), schema, "obtain the variance")
    runner = RoleRunner(runner_llm, trace=RoleTrace(), max_calls=80)
    search = ScenarioActionSearch(ev, language=lang, goal=goal, problem=problem(),
                                  compiler=ScenarioActionCompiler(None), runner=runner,
                                  max_revision_rounds=max_rounds)
    return search, search.run(cands, seed=seed)


def test_critics_cannot_select_the_final_action():
    """The adjudicator's disagreement is surfaced as a note, but the deterministic comparison
    governs the recommendation."""
    def adjudicator_prefers_loser(prompt):
        if "final independent adjudicator" in prompt:
            labels = [(m.group(0), m.start()) for m in re.finditer(r"OPTION_[A-Z]", prompt)]
            ti = prompt.find("LOSER_TITLE")
            pick = "OPTION_A"
            for lab, idx in labels:
                if idx < ti:
                    pick = lab
            return json.dumps({"best_supported_label": pick, "why": "vibes",
                               "not_separable": [], "highest_value_information": "x"})
        return "{}"

    wc, _, _ = build_context(council_schema(), [MAKER, OFFICER],
                             script={OFFICER: _scripted_officer_trigger}, n_particles=3)
    winner = _content_candidate("winner_id", "GRANT_TRIGGER now", "WINNER_TITLE")
    loser = _content_candidate("loser_id", "nothing happens here", "LOSER_TITLE")
    _, sr = _search(wc, adjudicator_prefers_loser, [winner, loser], max_rounds=1)
    assert sr.comparison["order"][0] == "winner_id"            # deterministic winner
    assert sr.adjudication.get("adjudicator_pick") == "loser_id"
    assert sr.adjudication.get("note"), "adjudicator disagreement was not surfaced"
    goal = GoalContractGenerator(None).generate(problem(), council_schema(), "obtain")
    recommended, kind, _ = _decide_recommendation(problem(), goal, sr)
    assert recommended == "winner_id" and kind == "action"


# ====================================================================== 20
def test_revisions_rerun_through_the_simulator():
    def runner_llm(prompt):
        if "MATERIALLY different repairs" in prompt:
            return json.dumps({"revisions": [{"op": "change_content",
                                              "addressed_break": "actor_rejection",
                                              "changes": [{"step_id": "B_s1",
                                                           "exact_content": "GRANT_TRIGGER please"}],
                                              "title": "revised"}]})
        return "{}"

    wc, _, _ = build_context(council_schema(), [MAKER, OFFICER],
                             script={OFFICER: _scripted_officer_trigger}, n_particles=2)
    B = _content_candidate("B", "no trigger present", "B")
    _, sr = _search(wc, runner_llm, [B], max_rounds=1)
    children = [r["child"] for r in sr.revisions]
    assert children, "no revision child was produced"
    assert all(c in sr.evaluations for c in children), "a revision child was never simulated"


# ====================================================================== 21
def test_locally_improved_step_that_worsens_the_whole_is_rejected():
    def sanction_officer(world, situation):
        if "SANCTION_ME" in situation:
            return [{"op": "create_or_update_record", "record_type": "panel_member_decision",
                     "fields": {"position": "sanction", "matter": "pet_12"}, "status": "decided"}]
        return None

    def runner_llm(prompt):
        if "MATERIALLY different repairs" in prompt:
            return json.dumps({"revisions": [{"op": "change_content",
                                              "addressed_break": "actor_rejection",
                                              "changes": [{"step_id": "flawed_s1",
                                                           "exact_content": "SANCTION_ME now"}],
                                              "title": "worse"}]})
        return "{}"

    goal = GoalContract(predicates=[
        GoalPredicate("granted", role="desired_terminal", record_type="variance_grant", op="exists"),
        GoalPredicate("sanctioned", role="forbidden", record_type="panel_member_decision",
                      field="position", op="eq", value="sanction")])
    wc, _, _ = build_context(council_schema(), [MAKER, OFFICER],
                             script={OFFICER: sanction_officer}, n_particles=2)
    flawed = _content_candidate("flawed", "quiet neutral filing", "flawed")
    _, sr = _search(wc, runner_llm, [flawed], max_rounds=1, goal=goal)
    worsened = [g for g in sr.screened_out
                if any(gate["code"] == "revision_worsened_forbidden" for gate in g["gates"])]
    assert worsened, "a revision that increased forbidden-state frequency was not rejected"


# ====================================================================== 22
def test_underspecified_goal_yields_pareto_or_abstention():
    a = _content_candidate("route_a", "no reaction A", "route A")
    b = _content_candidate("route_b", "no reaction B", "route B")
    wc, _, _ = build_context(council_schema(), [MAKER, OFFICER],
                             script={OFFICER: officer_silent}, n_particles=3)
    res = evaluate_actions_generated(problem(), [a, b], wc, goal_text="obtain the variance", seed=0)
    # the schema-projection goal declares missing preferences → never a fabricated single winner
    assert res.provenance["scenario_report"]["goal_contract"]["missing_preferences"]
    assert res.recommendation_kind in ("pareto", "abstain")
    assert res.recommended is None
    assert len(res.pareto_frontier) >= 2


# ====================================================================== 23
def test_phase13_evaluate_actions_does_not_mutate_input_on_the_generated_route():
    from swm.world_model_v2.phase13 import api as phase13_api
    wc, _, _ = ctx(n_particles=2)
    p = problem()
    before_candidates = copy.deepcopy(p.candidate_actions)
    before_perm = p.generated_action_permission
    res = phase13_api.evaluate_actions(p, [filing_candidate()], wc, mode="auto",
                                       goal_text="obtain the variance", seed=0)
    assert "scenario_report" in res.provenance                 # took the generated route
    assert p.candidate_actions == before_candidates            # caller's contract untouched
    assert p.generated_action_permission == before_perm


# ====================================================================== 24
def test_policy_conditions_observe_only_the_canonical_boundary():
    w = build_world(council_schema(), [MAKER, OFFICER])
    d = StateDelta(at=w.clock.now, event_type="t", operator="t")
    ctxk = {"actor_id": OFFICER, "action_id": "x", "now": w.clock.now, "report": generated_report(),
            "events": [], "quarantined": [], "compiler": "t"}
    execute_kernel_ops(w, [{"op": "create_or_update_record", "record_type": "panel_member_decision",
                            "record_id": "private_note", "visibility": "participants",
                            "audience": [OFFICER], "fields": {"position": "draft", "matter": "pet_12"}}],
                       ctxk, d)
    assert "private_note" in w.objects                         # the record really exists
    cond = ConditionSpec(kind="record", record_type="panel_member_decision", op="exists")
    # rivera cannot condition on chen's private record even though it exists
    assert condition_holds(cond, observable_projection(w, MAKER)) is False
    # chen (the audience) can
    assert condition_holds(cond, observable_projection(w, OFFICER)) is True


# ====================================================================== 25
def test_legacy_fixed_v1_unreachable_in_generated_mode(monkeypatch):
    from swm.world_model_v2.phase13 import api as phase13_api

    def boom(name):
        raise AssertionError("operation_spec must never be consulted on the generated path")

    monkeypatch.setattr(ontology, "operation_spec", boom)
    wc, _, _ = ctx(script={OFFICER: officer_grants}, n_particles=2)
    res = phase13_api.recommend_action(problem(), wc, mode="auto",
                                       goal_text="obtain the variance", seed=0)
    assert "scenario_report" in res.provenance
    for e in res.evaluated:
        assert "operation" not in e and "family" not in e


# ====================================================================== 26
def test_missing_generated_action_semantics_fail_loudly():
    class BareInitial:
        def sample_particles(self, n, seed=0):
            return [WorldState("g", f"b{i}", SimulationClock(T0, T0), network=RelationGraph(),
                               information=InformationLedger()) for i in range(n)]

    context = {"initial": BareInitial(), "queue_builder": lambda w: EventQueue(horizon_ts=T0 + DAY),
               "operators": [], "contract": None, "n_particles": 2, "hypotheses": [], "max_events": 50}
    ev = _make_evaluator(context, n_particles=2, seed=0, llm=None)
    with pytest.raises(RuntimeError, match="under-modeled"):
        _schema_of(ev)


# ====================================================================== 29
def test_canonical_runtime_used_for_every_step_candidate():
    wc, _, _ = ctx(script={OFFICER: officer_grants}, n_particles=2)
    ev = _make_evaluator(wc, n_particles=2, seed=0, llm=None)
    cand = filing_candidate()
    arm = ev.evaluate_arm(cand.candidate_id, plan_intervention(cand))
    operators = {dl.operator for b in arm.branches for dl in b.log}
    assert "scenario_plan_step" in operators, "a step candidate was evaluated outside the engine"
    # the status-quo reference legitimately schedules no plan step
    ref = ev.evaluate_arm("do_nothing", plan_intervention(do_nothing_action(MAKER)))
    assert "scenario_plan_step" not in {dl.operator for b in ref.branches for dl in b.log}


# ====================================================================== 30
def test_no_silent_numeric_actor_fallback_in_the_plan_path():
    wc, rep, runtime = ctx(script={OFFICER: officer_grants}, n_particles=3)
    evaluate_actions_generated(problem(), [filing_candidate()], wc,
                               goal_text="obtain the variance", seed=0)
    assert rep["numeric_fallbacks"] == 0
    assert rep["mechanistic_fallbacks"] == 0
    # every invocation went through the scripted actor runtime — no numeric stand-in
    assert rep["actors_invoked"] == len(runtime.invocations)
    assert rep["actor_actions_executed"] >= 1
