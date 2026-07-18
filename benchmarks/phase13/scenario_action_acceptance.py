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
    "1_no_catalog_in_prompts": "test_scenario_action_invariants.py::test_inv1",
    "2_no_operation_registered": "test_scenario_action_invariants.py::test_inv2",
    "3_novel_phrase_executes": "test_scenario_action_invariants.py::test_inv3",
    "4_no_belief_writes": "test_scenario_action_invariants.py::test_inv4",
    "5_no_choice_writes": "test_scenario_action_invariants.py::test_inv5",
    "6_no_terminal_writes": "test_scenario_action_invariants.py::test_inv6",
    "7_effects_via_observation_actors": "test_scenario_action_invariants.py::test_inv7",
    "8_exact_content_survives": "test_scenario_action_invariants.py::test_inv8",
    "9_no_history_only_execution": "test_scenario_action_invariants.py::test_inv9",
    "10_real_partial_or_rejected": "test_scenario_action_invariants.py::test_inv10",
    "11_no_progress_scalar": "test_scenario_action_invariants.py::test_inv11",
    "12_no_minted_numbers_in_ranking": "test_scenario_action_invariants.py::test_inv12",
    "13_user_actions_not_coerced": "test_scenario_action_invariants.py::test_inv13",
    "14_no_false_dedup": "test_scenario_action_invariants.py::test_inv14",
    "15_merges_carry_evidence": "test_scenario_action_invariants.py::test_inv15",
    "16_feasibility_hypotheses_and_execution": "test_scenario_action_invariants.py::test_inv16",
    "17_matched_exogenous_streams": "test_scenario_action_invariants.py::test_inv17",
    "18_blind_comparison": "test_scenario_action_invariants.py::test_inv18",
    "19_critics_cannot_select": "test_scenario_action_invariants.py::test_inv19",
    "20_revisions_rerun_simulated": "test_scenario_action_invariants.py::test_inv20",
    "21_local_improvement_whole_worse_rejected":
        "test_scenario_action_invariants.py::test_inv21",
    "22_underspecified_pareto_or_abstain": "test_scenario_action_invariants.py::test_inv22",
    "23_no_input_mutation": "test_scenario_action_invariants.py::test_inv23",
    "24_policy_observable_boundary": "test_scenario_action_invariants.py::test_inv24",
    "25_legacy_unreachable_in_generated": "test_scenario_action_invariants.py::test_inv25",
    "26_missing_semantics_fail_loudly": "test_scenario_action_invariants.py::test_inv26",
    "29_canonical_runtime_every_candidate": "test_scenario_action_invariants.py::test_inv29",
    "30_no_silent_numeric_fallback": "test_scenario_action_invariants.py::test_inv30",
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
        r = _pytest([os.path.join("tests", expr.split("::")[0]), "-k",
                     expr.split("::")[1]])
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
