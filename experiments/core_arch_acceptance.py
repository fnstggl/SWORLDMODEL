"""§42 executable acceptance report — GENERATED from tests and runtime traces, never handwritten.

Runs the enforcement/contract suites and inspects the live forensic artifacts, then emits
artifacts/core_arch/acceptance_report.{json,md}. The process exits NONZERO unless every gate
passes — a red gate is a red report, not a footnote.

Run:  PYTHONPATH=. python experiments/core_arch_acceptance.py [--skip-tests]
"""
import glob
import json
import os
import subprocess
import sys
import time

OUT_DIR = "artifacts/core_arch"
os.makedirs(OUT_DIR, exist_ok=True)
PYTEST = os.environ.get("SWM_PYTEST", "/root/.local/bin/pytest")

#: gate -> (kind, source, checker description). Kinds: suite (pytest file must pass),
#: trace (predicate over forensic artifacts), code (predicate over the tree).
SUITES = {
    "boundary_contract_suite": "tests/test_world_boundary.py",
    "outside_world_suite": "tests/test_outside_world.py",
    "bounded_cognition_suite": "tests/test_bounded_cognition_contracts.py",
    "truncation_suite": "tests/test_truncation_contracts.py",
    "mechanism_spec_suite": "tests/test_mechanism_spec.py",
    "invariant_enforcement_suite": "tests/test_core_arch_invariants.py",
    "cross_domain_fixture_suite": "tests/test_core_arch_fixtures.py",
    "phase13_integration_suite": "tests/test_phase13_core_arch_integration.py",
    "qualitative_actor_suite": "tests/test_qualitative_actor.py",
    "generated_world_suite": "tests/test_generated_world.py",
    "structural_ensemble_suite": "tests/test_structural_ensemble.py",
    "combined_runtime_suite": "tests/test_combined_runtime_integration.py",
}


def run_suite(path: str) -> dict:
    if not os.path.exists(path):
        return {"ok": False, "detail": "suite file missing"}
    t0 = time.time()
    p = subprocess.run([PYTEST, path, "-q", "--no-header"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "."})
    tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
    return {"ok": p.returncode == 0, "detail": tail[0][:160],
            "seconds": round(time.time() - t0, 1)}


def forensic_records() -> list:
    out = []
    for path in sorted(glob.glob("artifacts/core_arch_forensics/case*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                out.append((path, json.load(f)))
        except Exception:  # noqa: BLE001
            continue
    return out


def trace_gates(records: list) -> dict:
    """§42 trace-derived gates over the ACTUAL live artifacts."""
    gates = {}
    live = [(p, r) for p, r in records if isinstance(r, dict) and r.get("verification")]
    gates["forensic_traces_exist"] = {"ok": len(live) >= 1,
                                      "detail": f"{len(live)} live case artifact(s)"}
    gates["actual_llm_calls_used"] = {
        "ok": any((r.get("provider_cost") or {}).get("calls", 0) > 0 for _, r in live),
        "detail": f"calls per case: "
                  f"{[(os.path.basename(p), (r.get('provider_cost') or {}).get('calls')) for p, r in live]}"}
    gates["explicit_boundaries_default"] = {
        "ok": all(bool(r.get("world_boundaries")) for _, r in live
                  if not str(r.get("case", "")).startswith("case2")) and bool(live),
        "detail": "world_boundaries section present on ensemble-route live results"}
    gates["residual_outside_world_default"] = {
        "ok": any(bool(r.get("outside_world")) for _, r in live),
        "detail": "outside_world section present on live results"}
    gates["boundary_criticism_ran"] = {
        "ok": any(any(("critic" in str(t.get("stage", "")))
                      for b in (r.get("world_boundaries") or {}).values()
                      for t in (b.get("generation_trace") or []))
                  for _, r in live),
        "detail": "critic stages present in live boundary generation traces"}
    gates["no_numeric_actor_fallback_live"] = {
        "ok": all((r.get("verification") or {}).get("numeric_actor_fallbacks", 1) == 0
                  for _, r in live) and bool(live),
        "detail": "verification.numeric_actor_fallbacks == 0 on every live case"}
    gates["no_generic_prior_write_live"] = {
        "ok": all((r.get("verification") or {}).get("generic_prior_writes", 1) == 0
                  for _, r in live) and bool(live),
        "detail": "no broad-prior terminal writes on live cases (suppressions are legal and "
                  "surface as under_modeled)"}
    gates["truncated_weight_visible"] = {
        "ok": any(((r.get("truncation_report") or {}).get("truncated_weight") or 0) > 0
                  and not (r.get("verification") or {}).get(
                      "branch_continuation_after_truncation", True)
                  for _, r in live),
        "detail": "≥1 live case carries visible truncated weight with halted branches "
                  "(case6 is designed to)"}
    gates["monoculture_surfaced"] = {
        "ok": all((r.get("model_family_report") or {}).get("model_family_monoculture")
                  is not None for _, r in live) and bool(live),
        "detail": "model_family_monoculture reported on every live case"}
    gates["cognition_stage_traces_live"] = {
        "ok": any((r.get("verification") or {}).get("cognition_records", 0) > 0
                  for _, r in live),
        "detail": "bounded-cognition records present on live cases"}
    gates["under_modeled_surfacing_works"] = {
        "ok": any(r.get("under_modeled_subtypes") for _, r in live)
              or all((r.get("verification") or {}).get("generic_prior_suppressions", 0) == 0
                     and not (r.get("world_boundaries") or {}).get("under_modeled")
                     for _, r in live),
        "detail": "under_modeled subtypes appear when gaps exist (or no gaps existed)"}
    return gates


def code_gates() -> dict:
    gates = {}
    prod = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-E",
         r"environ\[.SWM_ALLOW_(NUMERIC_BASELINE|GENERIC_PRIOR).\]\s*=", "swm/"],
        capture_output=True, text=True)
    gates["production_never_sets_baseline_markers"] = {
        "ok": prod.stdout.strip() == "",
        "detail": prod.stdout.strip()[:200] or "no production assignment of the §19/§28 markers"}
    try:
        from swm.world_model_v2.result import SIMULATION_STATUSES, UNDER_MODELED_SUBTYPES
        gates["result_contract_statuses"] = {
            "ok": "under_modeled" in SIMULATION_STATUSES and "truncated" in SIMULATION_STATUSES
                  and len(UNDER_MODELED_SUBTYPES) >= 7,
            "detail": f"{SIMULATION_STATUSES}"}
    except Exception as e:  # noqa: BLE001
        gates["result_contract_statuses"] = {"ok": False, "detail": str(e)[:160]}
    try:
        from swm.world_model_v2.truncation import BRANCH_STATUSES
        gates["branch_status_vocabulary"] = {"ok": len(BRANCH_STATUSES) >= 12,
                                             "detail": str(BRANCH_STATUSES)}
    except Exception as e:  # noqa: BLE001
        gates["branch_status_vocabulary"] = {"ok": False, "detail": str(e)[:160]}
    try:
        from swm.world_model_v2.mechanism_spec import build_spec_index
        idx = build_spec_index()
        missing = [k for k, s in idx.items() if not (s.read_set or s.write_set)]
        gates["mechanisms_declare_read_write_sets"] = {
            "ok": len(idx) >= 30 and len(missing) <= max(2, len(idx) // 10),
            "detail": f"{len(idx)} specs; {len(missing)} without any declared I/O"}
    except Exception as e:  # noqa: BLE001
        gates["mechanisms_declare_read_write_sets"] = {"ok": False, "detail": str(e)[:160]}
    return gates


def main():
    skip_tests = "--skip-tests" in sys.argv
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "generator": "experiments/core_arch_acceptance.py (machine-generated; a "
                           "handwritten acceptance file is not acceptable — §42)",
              "gates": {}}
    if not skip_tests:
        for gate, path in SUITES.items():
            report["gates"][gate] = {"kind": "suite", "source": path, **run_suite(path)}
            print(f"[suite] {gate}: {'PASS' if report['gates'][gate]['ok'] else 'FAIL'} "
                  f"({report['gates'][gate]['detail']})", flush=True)
    records = forensic_records()
    for gate, val in trace_gates(records).items():
        report["gates"][gate] = {"kind": "trace", **val}
        print(f"[trace] {gate}: {'PASS' if val['ok'] else 'FAIL'} ({val['detail'][:120]})",
              flush=True)
    for gate, val in code_gates().items():
        report["gates"][gate] = {"kind": "code", **val}
        print(f"[code]  {gate}: {'PASS' if val['ok'] else 'FAIL'} ({val['detail'][:120]})",
              flush=True)
    n_ok = sum(1 for g in report["gates"].values() if g["ok"])
    report["summary"] = {"passed": n_ok, "total": len(report["gates"]),
                         "all_green": n_ok == len(report["gates"])}
    with open(os.path.join(OUT_DIR, "acceptance_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    lines = [f"# Core-architecture acceptance report (§42) — machine-generated",
             f"", f"Generated {report['generated_at']} by `{sys.argv[0]}`. "
             f"**{n_ok}/{len(report['gates'])} gates green.**", ""]
    for gate, g in report["gates"].items():
        lines.append(f"- {'✅' if g['ok'] else '❌'} `{gate}` ({g['kind']}): "
                     f"{str(g.get('detail', ''))[:200]}")
    with open(os.path.join(OUT_DIR, "acceptance_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{n_ok}/{len(report['gates'])} gates green → "
          f"{'ACCEPTED' if report['summary']['all_green'] else 'REJECTED'}", flush=True)
    sys.exit(0 if report["summary"]["all_green"] else 1)


if __name__ == "__main__":
    main()
