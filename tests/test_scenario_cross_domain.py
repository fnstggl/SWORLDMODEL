"""§19 generality suite — the scenario-generated action layer across fifteen materially
different decision domains, plus a fully randomized scenario whose type names are invented at
test time (proving no memorized literals are needed).

Coverage rotates across the proofs the spec enumerates; each proof is exercised in at least
three scenarios (fixture-optimal recovery in at least four):
  (a) the generated action language differs meaningfully by scenario
  (b) actions are not drawn from one shared catalog (disjoint vocab, no verb field)
  (c) several genuinely different strategies (scripted-llm planner)
  (d) user-proposed actions are evaluable
  (e) the known fixture-optimal plan is recovered from full rollout outcomes
  (f) impossible actions are rejected for the correct typed reason
  (g) the exact action content changes the trajectory
  (h) revision after diagnosis improves a deliberately flawed candidate
  (i) the final recommendation follows real simulated success, not the prettiest title
"""
import json
import re

import pytest

from swm.world_model_v2.phase13.scenario_actions.api import (evaluate_actions_generated,
                                                            evaluate_proposed_actions,
                                                            _make_evaluator, _schema_of)
from swm.world_model_v2.phase13.scenario_actions.compiler import ScenarioActionCompiler
from swm.world_model_v2.phase13.scenario_actions.feasibility import check_across_particles
from swm.world_model_v2.phase13.scenario_actions.generated_search import ScenarioActionSearch
from swm.world_model_v2.phase13.scenario_actions.goals import GoalContractGenerator
from swm.world_model_v2.phase13.scenario_actions.language import ActionLanguageGenerator
from swm.world_model_v2.phase13.scenario_actions.planner import GoalBackwardPlanner
from swm.world_model_v2.phase13.scenario_actions.roles import RoleRunner, RoleTrace

from tests.scenario_domain_fixtures import (all_scenarios, composite_escrow, random_scenario,
                                            scenario_by_key)

SCENARIOS = all_scenarios()
BY_KEY = {s.key: s for s in SCENARIOS}


def _search_pieces(s, wc, n_particles):
    ev = _make_evaluator(wc, n_particles=n_particles, seed=0, llm=None)
    schema = _schema_of(ev)
    w0 = ev.particles()[0]
    lang = ActionLanguageGenerator(None).generate(s.problem(), w0, schema)
    goal = GoalContractGenerator(None).generate(s.problem(), schema, s.goal_text)
    return ev, schema, lang, goal


# ====================================================================== (a)
def test_generated_action_language_differs_by_scenario():
    """Each scenario's deterministic-projection language is a materially different object."""
    sigs = {}
    for s in SCENARIOS:
        lang = ActionLanguageGenerator(None).generate(s.problem(), s.world(), s.schema)
        sigs[s.key] = (frozenset(lang.compiler_contract.get("record_types", [])),
                       frozenset(lang.resources),
                       frozenset(i.get("institution_id") for i in lang.institutions),
                       frozenset(lang.relevant_actors))
    assert len(set(sigs.values())) == len(sigs), "two scenarios produced identical action languages"
    # and pairwise the record vocabularies genuinely differ (not just actor names)
    keys = list(sigs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert sigs[keys[i]][0] != sigs[keys[j]][0], \
                f"{keys[i]} and {keys[j]} share a record-type vocabulary"


# ====================================================================== (b)
def test_actions_are_not_from_one_shared_catalog():
    """Record vocabularies are pairwise disjoint and no candidate step carries a verb field."""
    vocab = {s.key: set(s.schema.record_types()) for s in SCENARIOS}
    keys = list(vocab)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            shared = vocab[keys[i]] & vocab[keys[j]]
            assert not shared, f"{keys[i]}/{keys[j]} share record types {shared}"
    for s in SCENARIOS:
        cand = s.optimal() if s.key != "composite_escrow" else s.composite()
        d = cand.as_dict()
        assert "operation" not in d and "family" not in d
        for step in d["steps"]:
            assert "operation" not in step and "family" not in step
            for op in step["compiled_ops"]:
                assert op["op"] in ("create_or_update_record", "emit_semantic_event",
                                    "schedule_semantic_event", "transfer_conserved_quantity",
                                    "create_or_remove_relation", "remove_record",
                                    "declare_schema_definition")


# ====================================================================== (c)
def _planner_llm(prompt):
    if "strategies whose causal theories create" in prompt:
        return json.dumps({"strategies": [{"title": "precondition first",
                                           "causal_theory": "build the required precondition, then act",
                                           "strategy_class": "precondition_first",
                                           "key_steps": ["a"], "requires": []}]})
    if "ONLY from what the decision maker verifiably controls" in prompt:
        return json.dumps({"strategies": [{"title": "direct authority",
                                           "causal_theory": "use direct authority immediately",
                                           "strategy_class": "direct_authority",
                                           "key_steps": ["b"], "requires": []}]})
    if "ORTHOGONAL to the obvious" in prompt:
        return json.dumps({"strategies": [{"title": "gather then act",
                                           "causal_theory": "gather information before committing",
                                           "strategy_class": "information_gathering",
                                           "key_steps": ["c"], "requires": []}]})
    if "Convert this strategy into ONE executable concrete plan" in prompt:
        m = re.search(r'"strategy_class": "([^"]+)"', prompt)
        sc = m.group(1) if m else "generic"
        return json.dumps({"title": f"plan {sc}",
                           "steps": [{"intent": f"execute {sc}", "targets": [], "channel": "direct",
                                      "exact_content": f"distinct content for {sc}",
                                      "visibility": "participants"}],
                           "fallback": "continue", "assumptions": []})
    return "{}"


@pytest.mark.parametrize("key", ["founder_launch", "negotiation", "product_pricing", "legal_regulatory"])
def test_several_genuinely_different_strategies(key):
    s = BY_KEY[key]
    wc, _, _ = s.context(n_particles=2)
    ev, schema, lang, goal = _search_pieces(s, wc, 2)
    runner = RoleRunner(_planner_llm, trace=RoleTrace(), max_calls=120)
    planner = GoalBackwardPlanner(runner, language=lang, goal=goal, problem=s.problem(), schema=schema)
    out = planner.generate([], seed=0)
    assert out.diversity["n_concrete"] >= 3
    assert out.diversity["n_strategy_classes"] >= 3, "planner did not produce distinct strategies"
    # the strategies are materially different causal theories, not paraphrases
    assert len(set(out.diversity["strategy_classes"])) >= 3


# ====================================================================== (d)
@pytest.mark.parametrize("key", ["partnership_outreach", "hiring_team", "crisis_response",
                                 "personal_relationship"])
def test_user_proposed_actions_are_evaluable(key):
    s = BY_KEY[key]
    wc, _, _ = s.context(n_particles=2)
    text = s.nl_action()
    res = evaluate_proposed_actions(s.goal_text, [text], wc, problem=s.problem(), seed=0)
    sr = res.provenance["scenario_report"]
    user = [c for c in sr["candidates"] if c["candidate_id"].startswith("user_")]
    assert user, "the user action was dropped"
    uid = user[0]["candidate_id"]
    assert uid in sr["evaluations"], "the user action never reached simulation"
    assert user[0]["original_text"] == text                    # preserved verbatim
    assert text[:24] in json.dumps(sr["compiled_effects"][uid])   # exact content carried


# ====================================================================== (e) + (i)
@pytest.mark.parametrize("key", ["founder_launch", "negotiation", "hiring_team", "product_pricing",
                                 "operational_allocation", "coalition_group"])
def test_fixture_optimal_plan_is_recovered_from_rollout_outcomes(key):
    s = BY_KEY[key]
    wc, rep, _ = s.context(n_particles=3)
    # exactly one candidate carries the substantive trigger; the decoys wear the fancy titles
    optimal = s.optimal("optimal", title="the plain workmanlike plan")
    decoy1 = s.decoy("decoy1", title="Transformational North-Star Play")
    decoy2 = s.content_twin("decoy2", "an inspiring but empty rallying cry", title="Grand Vision")
    res = evaluate_actions_generated(s.problem(), [decoy1, optimal, decoy2], wc,
                                     goal_text=s.goal_text, seed=2)
    sr = res.provenance["scenario_report"]
    succ = {k: v["success_count"] for k, v in sr["evaluations"].items()}
    assert succ["optimal"] == 3, f"the optimal plan did not succeed in every world: {succ}"
    assert succ["decoy1"] == 0 and succ["decoy2"] == 0
    # (i) the recommendation follows simulated success, not the prettiest title
    assert res.recommended == "optimal" and res.recommendation_kind == "action"


# ====================================================================== (f)
@pytest.mark.parametrize("key,reason,build", [
    ("negotiation", "wrong_actor", "wrong_actor_candidate"),
    ("operational_allocation", "insufficient_resources", "resource_hog"),
    ("timing_sensitive", "timing_after_horizon", "after_horizon"),
    ("crisis_response", "wrong_actor", "wrong_actor_candidate"),
])
def test_impossible_actions_rejected_for_the_correct_reason(key, reason, build):
    s = BY_KEY[key]
    maker_resources = {s.resource: 1.0} if s.resource and build == "resource_hog" else None
    wc, _, _ = s.context(n_particles=2, maker_resources=maker_resources)
    ev, schema, lang, goal = _search_pieces(s, wc, 2)
    cand = getattr(s, build)()
    feas = check_across_particles(ev.particles(), list(getattr(ev, "_assignment", [])),
                                  lang, s.problem(), cand, goal=goal)
    codes = {r["code"] for r in feas["rejection_reasons"]}
    assert reason in codes, f"{key}: expected {reason}, got {codes}"
    # and end-to-end it is screened out of simulation with a typed gate, never silently run
    res = evaluate_actions_generated(s.problem(), [cand], wc, goal_text=s.goal_text, seed=0)
    sr = res.provenance["scenario_report"]
    assert any(r["candidate_id"] == cand.candidate_id for r in sr["rejected"])
    assert cand.candidate_id not in sr["evaluations"]


# ====================================================================== (g)
@pytest.mark.parametrize("key", ["founder_launch", "partnership_outreach", "hiring_team",
                                 "legal_regulatory"])
def test_exact_action_content_changes_the_trajectory(key):
    s = BY_KEY[key]
    wc, _, _ = s.context(n_particles=3)
    # two candidates identical in shape; only the exact content differs (one carries the trigger)
    reacts = s.content_twin("reacts", f"{s.trigger}: the concrete substantive ask")
    inert = s.content_twin("inert", "the very same shape of message, minus the substance")
    res = evaluate_actions_generated(s.problem(), [reacts, inert], wc, goal_text=s.goal_text, seed=1)
    ev = res.provenance["scenario_report"]["evaluations"]
    assert ev["reacts"]["success_count"] > ev["inert"]["success_count"], \
        "the exact action content did not change the simulated trajectory"
    assert ev["inert"]["success_count"] == 0


# ====================================================================== (h)
@pytest.mark.parametrize("key", ["founder_launch", "partnership_outreach", "crisis_response",
                                 "institutional_procedure"])
def test_revision_after_diagnosis_improves_a_flawed_candidate(key):
    s = BY_KEY[key]
    assert s.distractor, f"{key} has no distractor actor for a wrong-target flaw"
    # a deliberately flawed candidate: right message, WRONG recipient (the decider never sees it)
    flawed = s.content_twin("flawed", f"{s.trigger}: the substantive ask")
    flawed.steps[0].target_ids = [s.distractor]
    flawed.steps[0].compiled_ops[0]["direct_targets"] = [s.distractor]
    wc, _, _ = s.context(n_particles=2)
    ev, schema, lang, goal = _search_pieces(s, wc, 2)

    def runner_llm(prompt):
        if "MATERIALLY different repairs" in prompt:
            m = re.search(r'"id":\s*"([^"]+)"', prompt)
            sid = m.group(1) if m else "flawed_s1"
            return json.dumps({"revisions": [{"op": "change_target", "addressed_break": "wrong_target",
                                              "changes": [{"step_id": sid, "targets": [s.decider]}],
                                              "title": "retargeted to the decider"}]})
        return "{}"

    runner = RoleRunner(runner_llm, trace=RoleTrace(), max_calls=80)
    search = ScenarioActionSearch(ev, language=lang, goal=goal, problem=s.problem(),
                                  compiler=ScenarioActionCompiler(None), runner=runner,
                                  max_revision_rounds=1)
    sr = search.run([flawed], seed=0)
    assert sr.evaluations["flawed"]["success_count"] == 0      # the flaw really fails
    children = {r["child"]: r for r in sr.revisions}
    assert children, "diagnosis produced no revision"
    best_child = max((c for c in children if c in sr.evaluations),
                     key=lambda c: sr.evaluations[c]["success_count"], default=None)
    assert best_child is not None
    assert sr.evaluations[best_child]["success_count"] > 0, \
        "the diagnosis-directed revision did not repair the flawed candidate"


# ====================================================================== novel composite act
def test_single_composite_act_carries_dual_institutional_semantics():
    """One act that escrows the deposit AND files a conditional withdrawal at once — not
    representable by any single legacy verb — and it must satisfy BOTH terminal predicates."""
    s = composite_escrow()
    wc, _, _ = s.context(n_particles=2)
    res = evaluate_actions_generated(s.problem(), [s.composite("composite"), s.composite_partial("partial")],
                                     wc, goal_text=s.goal_text, seed=0)
    sr = res.provenance["scenario_report"]
    assert sr["evaluations"]["composite"]["success_count"] == 2      # both instruments lodged
    assert sr["evaluations"]["partial"]["success_count"] == 0        # escrow alone is not enough
    assert res.recommended == "composite"
    # the single step really carries two institutional direct effects
    ops = sr["compiled_effects"]["composite"][0]["ops"]
    record_types = {o.get("record_type") for o in ops}
    assert record_types == {"escrow_deposit_record", "conditional_withdrawal_notice"}


# ====================================================================== randomized (no literals)
@pytest.mark.parametrize("seed", [1, 7, 42, 100, 2024, 99999])
def test_randomized_scenario_runs_end_to_end(seed):
    """Type names invented at test time — the pipeline needs no memorized vocabulary."""
    s = random_scenario(seed)
    from swm.world_model_v2.scenario_schema import validate_scenario_schema
    ok, issues = validate_scenario_schema(s.schema)
    assert ok, f"random scenario schema invalid: {issues[:3]}"
    wc, rep, _ = s.context(n_particles=3)
    res = evaluate_actions_generated(s.problem(), [s.optimal("optimal"), s.decoy("decoy")], wc,
                                     goal_text=s.goal_text, seed=0)
    sr = res.provenance["scenario_report"]
    assert sr["evaluations"]["optimal"]["success_count"] == 3
    assert sr["evaluations"]["decoy"]["success_count"] == 0
    assert res.recommended == "optimal"
    assert rep["actors_invoked"] >= 1 and rep["actor_actions_executed"] >= 1
