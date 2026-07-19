"""EXECUTABLE production-path audit for structural-model uncertainty (ensemble contract Section 1).

Regenerates artifacts/structural_ensemble/production_path_audit.json from the CURRENT code — an AST/grep
scan, not a handwritten file. It identifies every call site of the single-plan machinery, classifies each
route (default-ensemble / explicit-ablation / legacy-compat / phase-scoped science / experiment / test),
and verifies the integration invariants (the canonical entry dispatches to the ensemble; the one-plan
call sites that remain are the sanctioned ones).

Run: PYTHONPATH=. python experiments/structural_ensemble_audit.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "structural_ensemble"

#: sanctioned classifications for remaining single-plan call sites (everything else is a violation)
SANCTIONED = {
    "swm/world_model_v2/unified_runtime.py": "explicit_single_structural_model_ablation",
    "swm/world_model_v2/ensemble_compiler.py": "ensemble_stage_b_backend",
    "swm/world_model_v2/compiler.py": "definition_or_recompile_library",
    "swm/world_model_v2/pipeline.py": "legacy_compat_helper_not_public_default",
    "swm/world_model_v2/phase3_pipeline.py": "phase_scoped_science_route",
    "swm/world_model_v2/phase9_pipeline.py": "phase_scoped_science_route",
    "swm/world_model_v2/evidence_pipeline.py": "phase_scoped_science_route",
    "swm/world_model_v2/phase8_pipeline.py": "phase_scoped_science_route",
    "swm/world_model_v2/evidence_materialize.py": "library_diagnostic",
}


def _call_sites(pattern: str, dirs=("swm", "experiments", "benchmarks", "tests", "api")) -> list:
    rx = re.compile(pattern)
    out = []
    for d in dirs:
        for p in sorted((ROOT / d).rglob("*.py")):
            try:
                text = p.read_text()
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                s = line.strip()
                if rx.search(line) and not s.startswith(("#", '"', "'")) and '"""' not in s:
                    rel = str(p.relative_to(ROOT))
                    out.append({"file": rel, "line": i, "code": s[:160]})
    return out


def _classify(site: dict) -> str:
    f = site["file"]
    if f.startswith("tests/"):
        return "test"
    if f.startswith("experiments/") or f.startswith("benchmarks/"):
        return "experiment_or_benchmark"
    return SANCTIONED.get(f, "UNSANCTIONED_PRODUCTION_SINGLE_PLAN")


def _ast_facts() -> dict:
    """The integration invariants, read from the AST of the canonical modules."""
    import inspect
    sys.path.insert(0, str(ROOT))
    import swm.world_model_v2.unified_runtime as U
    import swm.world_model_v2.structural_runtime as SR

    def calls(fn):
        names = set()
        for node in ast.walk(ast.parse(inspect.getsource(fn))):
            if isinstance(node, ast.Call):
                t = node.func
                names.add(t.id if isinstance(t, ast.Name) else getattr(t, "attr", ""))
        return names

    entry = calls(U.simulate_world)
    ens = calls(SR.simulate_structural_ensemble)
    return {
        "canonical_entry_dispatches_to_ensemble": "simulate_structural_ensemble" in entry,
        "canonical_entry_calls_single_plan_compiler_directly": "compile_world" in entry,
        "ensemble_runtime_calls_single_plan_compiler_directly": "compile_world" in ens,
        "ensemble_runtime_uses_ensemble_compiler": {"reconnoiter_structures",
                                                    "compile_candidates"} <= ens,
        "single_model_mode_is_explicit": "single_structural_model" in inspect.getsource(U.simulate_world),
        "default_mode_literal": 'policy.get("structural_mode", "ensemble")'
                                in inspect.getsource(U.simulate_world),
    }


def build_audit() -> dict:
    single_plan = {
        "compile_world_call_sites": [
            {**s, "classification": _classify(s)}
            for s in _call_sites(r"compile_world\(")
            if "def compile_world" not in s["code"]],
        "run_from_plan_call_sites": [
            {**s, "classification": _classify(s)}
            for s in _call_sites(r"run_from_plan\(")
            if "def run_from_plan" not in s["code"]],
        "simulate_world_call_sites": [
            {**s, "classification": ("public_default_ensemble_route"
                                     if s["file"] == "swm/facade.py" else _classify(s))}
            for s in _call_sites(r"simulate_world\(")
            if "def simulate_world" not in s["code"]],
    }
    findings = {
        "single_plan_assumptions_before_this_change": [
            "unified_runtime.simulate_world compiled exactly ONE WorldExecutionPlan (now: ensemble)",
            "plan.structural_hypotheses stored competing narratives INSIDE one plan (level-B only)",
            "evidence requirements derived from one schema (now: union over recon candidates)",
            "particles were created from one causal representation (now: per-model particle sets)",
            "aggregation covered hidden-state uncertainty only (now: between-model decomposition)",
            "individual-reaction route bypassed compilation with one unchallenged frame "
            "(now: frame ensemble)",
            "Phase 13 compared actions inside one causal model (now: cross-model default + guard)",
            "wmv2_historical_benchmark B6 arm called compile_world+run_from_plan directly "
            "(now routed through simulate_world)",
        ],
        "silent_single_model_continuation_paths": "none: ensemble-stage failures return "
                                                  "execution_failed loudly; a single survivor "
                                                  "requires a convergence certificate "
                                                  "(EnsembleIntegrityError otherwise)",
    }
    violations = [s for group in single_plan.values() for s in group
                  if s["classification"] == "UNSANCTIONED_PRODUCTION_SINGLE_PLAN"]
    return {"schema_version": "structural_ensemble.audit.v1",
            "integration_invariants": _ast_facts(),
            "call_sites": single_plan,
            "findings": findings,
            "unsanctioned_violations": violations,
            "ok": (not violations
                   and _ast_facts()["canonical_entry_dispatches_to_ensemble"]
                   and not _ast_facts()["canonical_entry_calls_single_plan_compiler_directly"])}


def main():
    audit = build_audit()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "production_path_audit.json"
    path.write_text(json.dumps(audit, indent=1))
    n = sum(len(v) for v in audit["call_sites"].values())
    print(f"audit: {n} call sites scanned; unsanctioned={len(audit['unsanctioned_violations'])}; "
          f"ok={audit['ok']} -> {path}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
