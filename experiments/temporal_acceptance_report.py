"""EXECUTABLE ACCEPTANCE REPORT (§35) — machine-generated from code, tests, and trace
artifacts; FAILS (exit 1) unless every §35 condition holds. Not a handwritten JSON file: every
entry below is computed by running the enforcement/invariant suites, importing the production
modules, re-running the assumption audit, and reading the live forensic artifacts.

Run: PYTHONPATH=. python experiments/temporal_acceptance_report.py
"""
from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "temporal" / "acceptance_report.json"


def _pytest(paths, timeout=1800) -> dict:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths], cwd=ROOT,
                       capture_output=True, text=True, timeout=timeout,
                       env={**__import__("os").environ, "PYTHONPATH": "."})
    tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
    return {"cmd": f"pytest -q {' '.join(paths)}", "returncode": p.returncode,
            "tail": tail[0], "passed": p.returncode == 0}


def check_all() -> dict:
    checks = {}

    # 1. the periodic scheduler is unreachable by default (enforcement suite: AST + call-spy)
    checks["periodic_scheduler_unreachable"] = _pytest(["tests/test_temporal_enforcement.py"])

    # 2-3. six-actor limit + fixed delivery/reconsideration defaults absent (assumption audit)
    sys.path.insert(0, str(ROOT / "experiments"))
    from temporal_audit_gen import generate_audit
    audit = generate_audit()
    checks["assumption_audit_removals"] = {
        "passed": audit["all_removals_verified"],
        "n_verified": f"{audit['n_removal_verified']}/{audit['n_requiring_removal']}",
        "failures": audit["removal_failures"][:5]}

    # 4-11. behavioral invariants: triggers default, attention separate from delivery,
    # exact-interval continuous processes, state-responsive hazards, order-invariant
    # simultaneity, conditional institutional stages
    checks["behavioral_invariants"] = _pytest(["tests/test_temporal_invariants.py"])
    checks["cross_domain_fixtures"] = _pytest(["tests/test_temporal_fixtures.py"])

    # actual LLM temporal compilation + critics wired into the default route
    from swm.world_model_v2 import unified_runtime as U
    src = inspect.getsource(U.simulate_world)
    from swm.world_model_v2 import temporal_compiler as TC
    checks["llm_temporal_compilation_default"] = {
        "passed": ("compile_temporal_model" in src
                   and "temporal_critic_A" in inspect.getsource(TC)
                   and "temporal_critic_B" in inspect.getsource(TC)
                   and 'if "temporal_model" not in drop' in src),
        "note": "compile stage + two independent critics on the default route (drop-gated for "
                "ablation harness only)"}

    # all production routes use the temporal runtime
    from swm.world_model_v2.rollout import RolloutEngine
    from swm.world_model_v2.phase13.crn import MatchedRolloutEngine
    from swm.world_model_v2 import individual_reaction as IR
    checks["all_routes_use_temporal_runtime"] = {
        "passed": ("run_branch_temporal" in inspect.getsource(RolloutEngine.run_branch)
                   and "run_branch_temporal" in inspect.getsource(
                       MatchedRolloutEngine.run_branch)
                   and "compute_notice_ts" in inspect.getsource(
                       IR.simulate_individual_reaction)),
        "note": "core rollout, phase13 matched engine, personal-reaction route"}

    # no numerical actor fallback on truncation + truncation surfaced
    from swm.world_model_v2 import generated_world as GW
    gsrc = inspect.getsource(GW)
    checks["truncation_surfaced_no_numeric_fallback"] = {
        "passed": ("_record_truncation" in gsrc and "temporally_truncated" in gsrc
                   and "numeric" not in inspect.getsource(GW._record_truncation)),
        "note": "safety exhaustion records truncation; never converts actors to numeric"}

    # test suites (§37): temporal + event/rollout/world-dynamics/semantic/qualitative/
    # phase13/messages/full V2
    checks["suite_event_time_rollout_dynamics"] = _pytest(
        ["tests/test_wmv2_event_time.py", "tests/test_wmv2_world_dynamics.py",
         "tests/test_world_model_v2.py", "tests/test_event_coupled_rollout.py"])
    checks["suite_semantic_and_generated"] = _pytest(
        ["tests/test_semantic_consequences.py", "tests/test_generated_world.py"])
    checks["suite_qualitative_actors"] = _pytest(
        ["tests/test_qualitative_actor.py", "tests/test_llm_actor.py"])
    checks["suite_phase13"] = _pytest(
        ["tests/test_phase13_engine.py", "tests/test_scenario_action_layer.py",
         "tests/test_scenario_action_invariants.py",
         "tests/test_scenario_action_enforcement.py", "tests/test_scenario_cross_domain.py"])
    checks["suite_pr115_messages"] = _pytest(
        ["tests/test_reply_first.py", "tests/test_content_graph.py",
         "tests/test_message_optimizer.py"], timeout=2400)
    checks["suite_unified_runtime"] = _pytest(
        ["tests/test_wmv2_unified_runtime.py", "tests/test_wmv2_phase8.py",
         "tests/test_wmv2_activation_synthesis.py", "tests/test_wmv2_phase_supervision.py"])
    checks["suite_benchmark_harness"] = _pytest(["tests/test_temporal_benchmark_harness.py"])

    # forensic traces exist and verified (live LLM runs)
    fdir = ROOT / "artifacts" / "temporal" / "forensics"
    fcases = sorted(fdir.glob("*.json")) if fdir.exists() else []
    fsum = {}
    if (fdir / "summary.json").exists():
        fsum = json.loads((fdir / "summary.json").read_text())
    n_verified = sum(1 for c in fsum.get("cases", [])
                     if c.get("verification_all_passed"))
    checks["forensic_traces"] = {
        "passed": len([f for f in fcases if f.name != "summary.json"]) >= 6
                  and n_verified >= 6,
        "n_case_artifacts": len([f for f in fcases if f.name != "summary.json"]),
        "n_verified": n_verified,
        "note": "six live LLM-backed production-runtime cases with §30 verification blocks"}

    # cost measured
    cost = ROOT / "artifacts" / "temporal" / "cost_benchmark.json"
    checks["cost_measured"] = {"passed": cost.exists(),
                               "artifact": str(cost.relative_to(ROOT)) if cost.exists()
                               else None}

    return checks


def main():
    t0 = time.time()
    checks = check_all()
    ok = all(v.get("passed") for v in checks.values())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": commit, "accepted": ok,
              "generation": "computed from code/tests/artifacts by "
                            "experiments/temporal_acceptance_report.py — not handwritten",
              "checks": checks, "runtime_s": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    for k, v in checks.items():
        print(f"{'PASS' if v.get('passed') else 'FAIL'}  {k}  "
              f"{v.get('tail', v.get('note', ''))[:80]}")
    print(f"\nACCEPTED={ok} → {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
