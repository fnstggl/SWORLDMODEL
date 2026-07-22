"""Executable acceptance report for the scenario-generated action layer.

Every row is produced by RUNNING something — a pytest node, an AST scan, or a live
structural probe through the public API. Nothing here is hand-asserted; re-run with:

    PYTHONPATH=. python benchmarks/phase13/scenario_action_acceptance.py

Writes artifacts/phase13/action_language/acceptance_report.json.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "artifacts", "phase13", "action_language",
                   "acceptance_report.json")
PKG = os.path.join(ROOT, "swm", "world_model_v2", "phase13", "scenario_actions")

#: §18 invariant -> the pytest expression that proves it (run live below)
INVARIANT_TESTS = {
    "1_no_catalog_in_prompts":
        "test_scenario_action_invariants.py::test_generated_mode_never_lists_the_global_operation_catalog_in_a_prompt",
    "2_no_operation_registered":
        "test_scenario_action_invariants.py::test_generated_mode_never_calls_operation_registered",
    "3_novel_phrase_executes":
        "test_scenario_action_invariants.py::test_novel_action_phrase_absent_from_source_compiles_and_executes",
    "4_no_belief_writes":
        "test_scenario_action_invariants.py::test_cannot_directly_write_another_actors_beliefs",
    "5_no_choice_writes":
        "test_scenario_action_invariants.py::test_cannot_directly_write_another_actors_choice",
    "6_no_terminal_writes":
        "test_scenario_action_invariants.py::test_cannot_directly_write_a_terminal_outcome",
    "7_effects_via_observation_actors":
        "test_scenario_action_invariants.py::test_downstream_social_effects_travel_through_observation_and_actors",
    "8_exact_content_survives":
        "test_scenario_action_invariants.py::test_exact_content_terms_targets_timing_observability_survive_compilation",
    "9_no_history_only_execution":
        "test_scenario_action_invariants.py::test_unsupported_actions_are_real_world_events_not_history_only_records",
    "10_real_partial_or_rejected":
        "test_scenario_action_invariants.py::test_every_action_yields_effects_visible_partial_or_hard_rejection",
    "11_no_progress_scalar":
        "test_scenario_action_invariants.py::test_no_arbitrary_scalar_progress_and_evaluations_are_counts_not_utilities",
    "12_no_minted_numbers_in_ranking":
        "test_scenario_action_invariants.py::test_no_llm_minted_utility_weight_or_failure_probability_in_ranking",
    "13_user_actions_not_coerced":
        "test_scenario_action_invariants.py::test_user_actions_not_coerced_into_registered_verbs",
    "14_no_false_dedup":
        "test_scenario_action_invariants.py::test_materially_different_actions_are_not_falsely_deduplicated",
    "15_merges_carry_evidence":
        "test_scenario_action_invariants.py::test_paraphrase_merges_only_with_recorded_evidence",
    "16_feasibility_hypotheses_and_execution":
        "test_scenario_action_invariants.py::test_feasibility_across_hypotheses_and_at_execution",
    "17_matched_exogenous_streams":
        "test_scenario_action_invariants.py::test_matched_rollouts_preserve_exogenous_streams_and_are_deterministic",
    "18_blind_comparison":
        "test_scenario_action_invariants.py::test_blind_comparison_hides_candidate_provenance_and_source"
        " or test_adjudicator_prompt_never_reveals_candidate_id",
    "19_critics_cannot_select":
        "test_scenario_action_invariants.py::test_critics_cannot_select_the_final_action",
    "20_revisions_rerun_simulated":
        "test_scenario_action_invariants.py::test_revisions_rerun_through_the_simulator",
    "21_local_improvement_whole_worse_rejected":
        "test_scenario_action_invariants.py::test_locally_improved_step_that_worsens_the_whole_is_rejected",
    "22_underspecified_pareto_or_abstain":
        "test_scenario_action_invariants.py::test_underspecified_goal_yields_pareto_or_abstention",
    "23_no_input_mutation":
        "test_scenario_action_invariants.py::test_phase13_evaluate_actions_does_not_mutate_input_on_the_generated_route",
    "24_policy_observable_boundary":
        "test_scenario_action_invariants.py::test_policy_conditions_observe_only_the_canonical_boundary",
    "25_legacy_unreachable_in_generated":
        "test_scenario_action_invariants.py::test_legacy_fixed_v1_unreachable_in_generated_mode",
    "26_missing_semantics_fail_loudly":
        "test_scenario_action_invariants.py::test_missing_generated_action_semantics_fail_loudly",
    "29_canonical_runtime_every_candidate":
        "test_scenario_action_invariants.py::test_canonical_runtime_used_for_every_step_candidate",
    "30_no_silent_numeric_fallback":
        "test_scenario_action_invariants.py::test_no_silent_numeric_actor_fallback_in_the_plan_path",
}

SUITES = {
    "27_pr115_message_battery": ["tests/test_reply_first.py", "tests/test_llm_moves.py",
                                 "tests/test_iterative_editor.py",
                                 "tests/test_outreach_funnel.py",
                                 "tests/test_persona_response.py"],
    "28_semantic_world_suites": ["tests/test_generated_world.py",
                                 "tests/test_semantic_consequences.py"],
    "core_integration": ["tests/test_scenario_action_layer.py"],
    "cross_domain_generality": ["tests/test_scenario_cross_domain.py"],
    "source_enforcement": ["tests/test_scenario_action_enforcement.py"],
    "phase13_legacy_regression": ["tests/test_phase13_engine.py", "tests/test_phase13_ope.py"],
}


def _pytest(args: list) -> dict:
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=no", *args],
                       capture_output=True, text=True, cwd=ROOT, timeout=1800)
    tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
    return {"command": "pytest -q " + " ".join(args), "exit_code": p.returncode,
            "summary": tail[0][:200], "wall_s": round(time.time() - t0, 1),
            "passed": p.returncode == 0}


def _structural_checks() -> list:
    """AST + import probes executed directly (independent of the test suite)."""
    rows = []
    # S1: scenario_actions never imports the legacy catalog symbols
    banned = {"_OPERATIONS", "operation_registered", "operation_spec", "OPERATION_FAMILIES"}
    hits = []
    for fn in sorted(os.listdir(PKG)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(PKG, fn)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "ontology" in str(node.module or ""):
                for a in node.names:
                    if a.name in banned:
                        hits.append(f"{fn}:{node.lineno}:{a.name}")
    rows.append({"check": "S1_no_legacy_catalog_imports_in_package", "passed": not hits,
                 "evidence": hits or "no scenario_actions module imports the legacy registry"})
    # S2: the kernel stays semantically empty (<= 8 storage-mechanics ops)
    from swm.world_model_v2.generated_world import KERNEL_OPS
    rows.append({"check": "S2_kernel_semantically_empty",
                 "passed": len(KERNEL_OPS) <= 8,
                 "evidence": list(KERNEL_OPS)})
    # S3: default routing — a generated context reaches the scenario layer via the DEFAULT api
    try:
        sys.path.insert(0, ROOT)
        from tests.scenario_fixtures import build_context, council_schema
        from tests.test_scenario_action_layer import filing_candidate, officer_grants, problem
        from swm.world_model_v2.phase13.api import evaluate_actions
        wc, _, _ = build_context(council_schema(), ["rivera", "chen"],
                                 script={"chen": officer_grants}, n_particles=2)
        res = evaluate_actions(problem(), [filing_candidate()], wc,
                               goal_text="obtain the variance", seed=0)
        ok = "scenario_report" in res.provenance and res.recommended == "file_petition"
        rows.append({"check": "S3_default_api_routes_generated_mode", "passed": ok,
                     "evidence": {"recommendation": res.recommended,
                                  "kind": res.recommendation_kind}})
        # S4: no candidate in generated mode carries a catalog operation
        cands = res.provenance["scenario_report"]["candidates"]
        has_op = [c["candidate_id"] for c in cands if "operation" in c]
        rows.append({"check": "S4_no_operation_field_on_generated_candidates",
                     "passed": not has_op, "evidence": has_op or "clean"})
        # S5: human summary carries the honest support claim, no invented percentages
        hs = res.provenance["human_summary"]
        claim_ok = "best-supported among the considered" in hs.get("support", "")
        rows.append({"check": "S5_honest_support_claim", "passed": claim_ok,
                     "evidence": hs.get("support", "")[:160]})
    except Exception as e:  # noqa: BLE001 — a crashed probe is a FAILED row, loudly
        rows.append({"check": "S3_S5_live_structural_probe", "passed": False,
                     "evidence": f"{type(e).__name__}: {e}"[:300]})
    return rows


def main():
    t0 = time.time()
    report = {"kind": "scenario_action_acceptance.v1",
              "generated_by": "benchmarks/phase13/scenario_action_acceptance.py",
              "generated_at_unix": time.time(), "rows": [], "suites": {}, "invariants": {}}
    for name, rows in _suite_runs().items():
        report["suites"][name] = rows
    for inv, expr in INVARIANT_TESTS.items():
        file, sel = expr.split("::", 1)
        r = _pytest([os.path.join("tests", file), "-k", sel.replace('" "', " ")])
        # pytest exits 5 when no test matched — record as NOT COVERED, never as a pass
        r["covered"] = r["exit_code"] != 5
        r["passed"] = r["passed"] and r["covered"]
        report["invariants"][inv] = r
    report["rows"] = _structural_checks()
    n_inv = len(report["invariants"])
    report["totals"] = {
        "invariants_passed": sum(1 for r in report["invariants"].values() if r["passed"]),
        "invariants_total": n_inv,
        "invariants_not_covered": [k for k, r in report["invariants"].items()
                                   if not r.get("covered")],
        "suites_passed": sum(1 for r in report["suites"].values() if r["passed"]),
        "suites_total": len(report["suites"]),
        "structural_passed": sum(1 for r in report["rows"] if r["passed"]),
        "structural_total": len(report["rows"]),
        "wall_s": round(time.time() - t0, 1)}
    report["all_pass"] = (report["totals"]["invariants_passed"] == n_inv
                          and report["totals"]["suites_passed"] == len(report["suites"])
                          and report["totals"]["structural_passed"] == len(report["rows"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report["totals"], indent=1))
    print("all_pass:", report["all_pass"], "->", os.path.relpath(OUT, ROOT))
    return 0 if report["all_pass"] else 1


def _suite_runs() -> dict:
    return {name: _pytest(files) for name, files in SUITES.items()}


if __name__ == "__main__":
    sys.exit(main())
