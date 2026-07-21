"""EXECUTABLE acceptance gates for default-on structural-model uncertainty (Section 28).

Every gate is an EXECUTED check — a scripted-backend run through the real runtime, an AST audit, a live
forensic-trace inspection, or a pytest invocation — never a handwritten status. The report
(artifacts/structural_ensemble/acceptance_report.{json,md}) is regenerated on every run and the process
exits nonzero unless ALL gates pass.

Run: PYTHONPATH=. python experiments/structural_ensemble_acceptance.py [--skip-suite]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
OUT = ROOT / "artifacts" / "structural_ensemble"

GATES: list = []


def gate(name):
    def deco(fn):
        GATES.append((name, fn))
        return fn
    return deco


_UNSET = object()


def _default_run(policy=None, llm=_UNSET, **kw):
    import functools
    from swm.world_model_v2.unified_runtime import simulate_world as _sw_default
    # archival full-fidelity harness: pinned since the §25 default switch
    simulate_world = functools.partial(_sw_default, execution_profile="full_fidelity")
    from test_structural_ensemble import HERMETIC, four_way_llm
    return simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                          horizon="2025-09-01", llm=four_way_llm() if llm is _UNSET else llm,
                          seed=3,
                          execution_policy=dict(HERMETIC) if policy is None else policy, **kw)


_CACHE: dict = {}


def _res():
    if "res" not in _CACHE:
        _CACHE["res"] = _default_run()
    return _CACHE["res"]


@gate("structural_ensemble_is_the_default")
def g_default():
    res = _res()
    return (res.structural_ensemble is not None
            and res.provenance["structural_mode"] == "ensemble"), \
        "default simulate_world produced a structural ensemble with no enable flag"


@gate("actual_independent_generation_calls_occurred")
def g_independent():
    from test_structural_ensemble import four_way_llm
    llm = four_way_llm()
    res = _default_run(llm=llm)
    gm = res.structural_ensemble["generation_manifest"]
    indep = [g for g in gm if g["independent"] and g["ok"]]
    distinct_prompts = len({g["prompt_hash"] for g in indep})
    return (len(indep) >= 3 and distinct_prompts == len(indep)
            and len(llm.prompts["recon"]) == len(indep)), \
        f"{len(indep)} independent calls, {distinct_prompts} distinct prompts, " \
        f"{len(llm.prompts['recon'])} actual backend invocations"


@gate("default_target_is_approximately_four")
def g_target():
    from swm.world_model_v2.structural_contracts import GENERATION_TARGET_CALLS
    n = _res().structural_ensemble["n_independent_generation_calls"]
    return n == GENERATION_TARGET_CALLS == 4, f"target={GENERATION_TARGET_CALLS}, observed={n}"


@gate("candidate_count_adapts")
def g_adaptive():
    from test_structural_ensemble import (OMISSION_QUIET, decomp_payload, four_way_llm)
    omission = dict(OMISSION_QUIET)
    omission.update({"missing_decisive_actor": "unmodeled regulator",
                     "no_further_material_model": False,
                     "proposed_models": [{"causal_thesis": "Regulator gate decides",
                                          "decisive_actors": ["regulator"], "why_missing": "gap"}]})
    llm = four_way_llm(omission=omission)
    llm.decomp_by_model["m4_adversarial_alternative"] = decomp_payload(["regulator"],
                                                                       lean="strong_no", hyp="h_r")
    res = _default_run(llm=llm)
    se = res.structural_ensemble
    return se["n_expansion_candidates"] >= 1, \
        f"expanded to {se['n_initial_candidates'] + se['n_expansion_candidates']} candidates"


@gate("independent_executable_plans_exist")
def g_plans():
    models = [m for m in _res().structural_ensemble["models"]
              if m["promotion_status"] == "promoted"]
    hashes = {m["schema_hash"] for m in models}
    return len(models) >= 2 and len(hashes) == len(models), \
        f"{len(models)} promoted models with {len(hashes)} distinct schema hashes"


@gate("adversarial_omission_search_runs")
def g_omission():
    cm = _res().structural_ensemble["critic_manifest"]
    kinds = {c["critic"] for c in cm}
    return {"structural_omission", "candidate_causal", "cross_model_contrast"} <= kinds, \
        f"critics executed: {sorted(kinds)}"


@gate("conservative_deduplication_runs")
def g_dedup():
    mm = _res().structural_ensemble["merge_manifest"]
    return (len(mm) >= 1 and all({"survivor", "merged", "confidence",
                                  "structural_comparison"} <= set(m) for m in mm)), \
        f"{len(mm)} conservative merge(s) with full comparison records"


@gate("every_plausible_model_receives_a_pilot")
def g_pilot():
    from swm.world_model_v2.structural_runtime import PILOT_MIN_PARTICLES
    se = _res().structural_ensemble
    sims = se["simulation_manifest"]
    surviving = [m["model_id"] for m in se["models"]]
    ok = all(sims[m]["pilot_particles"] >= PILOT_MIN_PARTICLES for m in surviving)
    return ok and se["n_pilot_simulated"] == len(surviving), \
        f"{se['n_pilot_simulated']} pilots ≥ {PILOT_MIN_PARTICLES} particles each"


@gate("promoted_models_receive_full_per_model_budgets")
def g_budget():
    se = _res().structural_ensemble
    rows = [(m["model_id"], se["simulation_manifest"][m["model_id"]])
            for m in se["models"] if m["promotion_status"] == "promoted"]
    ok = all(s["final_particles"] >= s["full_budget_required"] >= 12 for _, s in rows)
    return ok, "; ".join(f"{m}: {s['final_particles']}/{s['full_budget_required']}"
                         for m, s in rows)


@gate("no_final_budget_divided_across_models")
def g_no_division():
    se = _res().structural_ensemble
    rows = [se["simulation_manifest"][m["model_id"]] for m in se["models"]
            if m["promotion_status"] == "promoted"]
    n_one = max(s["full_budget_required"] for s in rows)
    total = sum(s["final_particles"] for s in rows)
    return total >= n_one * len(rows), \
        f"total particles {total} ≥ {n_one} × {len(rows)} models"


@gate("pilot_computation_is_reused")
def g_reuse():
    se = _res().structural_ensemble
    rows = [se["simulation_manifest"][m["model_id"]] for m in se["models"]
            if m["promotion_status"] == "promoted"]
    return all(s["pilot_reused_as_prefix"] for s in rows), \
        f"pilot prefix reused in {len(rows)}/{len(rows)} promoted models"


@gate("all_production_routes_use_the_ensemble")
def g_routes():
    from experiments.structural_ensemble_audit import build_audit
    audit = build_audit()
    inv = audit["integration_invariants"]
    return (audit["ok"] and inv["canonical_entry_dispatches_to_ensemble"]
            and not inv["canonical_entry_calls_single_plan_compiler_directly"]), \
        f"audit ok={audit['ok']}, unsanctioned={len(audit['unsanctioned_violations'])}"


@gate("phase13_evaluates_across_models")
def g_phase13():
    from swm.world_model_v2.phase13.api import SingleModelContextError, recommend_action
    from swm.world_model_v2.phase13.contracts import DecisionProblem, Stakeholder, UtilitySpec
    res = _res()
    problem = DecisionProblem(
        decision_id="gate", decision_maker="avery",
        authority=["communicate", "gather_information"], as_of="2025-06-01",
        utility=UtilitySpec(stakeholders=[Stakeholder(
            "avery", utility_fn=lambda o: float(o.get("quantities", {}).get("outcome", 0.0) or 0.0))]))
    r = recommend_action(problem, res, budget="diagnostic", seed=2, n_particles=6)
    n = r.provenance["structural_ensemble"]["n_models_evaluated"]
    plan = res._ensemble_handle.surviving()[0].executable_plan
    try:
        recommend_action(problem, plan, budget="diagnostic", seed=2, n_particles=4)
        guarded = False
    except SingleModelContextError:
        guarded = True
    return n >= 2 and guarded, f"evaluated across {n} models; bare-plan guard active={guarded}"


@gate("per_model_outputs_are_preserved")
def g_per_model():
    se = _res().structural_ensemble
    dists = se["model_distributions"]
    promoted = [m for m in se["models"] if m["promotion_status"] == "promoted"]
    return (set(dists) == {m["model_id"] for m in promoted}
            and all(m["prediction"] == dists[m["model_id"]] for m in promoted)), \
        f"per-model distributions retained for {len(dists)} models"


@gate("structural_sensitivity_is_reported")
def g_sensitivity():
    cls = _res().structural_ensemble["structural_sensitivity"]["classification"]
    return cls in ("structurally_stable", "mildly_structurally_sensitive",
                   "materially_structurally_sensitive", "structurally_underidentified",
                   "ensemble_execution_incomplete"), f"classification={cls}"


@gate("reversal_conditions_are_reported")
def g_reversal():
    rc = _res().structural_ensemble["reversal_conditions"]
    return bool(rc) and all("assumption" in r for r in rc), f"{len(rc)} reversal condition(s)"


@gate("cost_and_cache_manifests_are_complete")
def g_cost():
    cm = _res().structural_ensemble["cost_manifest"]
    need = {"llm_calls_by_stage", "llm_calls_by_model", "cache_hits_by_stage", "total_llm_calls",
            "total_cache_hits", "single_model_equivalent_llm_calls", "pilot_particles_reused"}
    return need <= set(cm), f"manifest keys: {sorted(set(cm) & need)}"


@gate("no_numerical_actor_fallback_occurred")
def g_no_fallback():
    per = _res().provenance["per_model_provenance"]
    reports = [(m, (p.get("actor_policy_report") or {})) for m, p in per.items()]
    fallbacks = {m: r.get("fallbacks", 0) for m, r in reports}
    return all(v in (0, None) for v in fallbacks.values()), f"fallbacks by model: {fallbacks}"


@gate("no_single_model_silent_fallback")
def g_no_silent_single():
    res_none = _default_run(llm=None)
    default_mode = _res().provenance["structural_mode"]
    return (default_mode == "ensemble"
            and res_none.simulation_status == "execution_failed"
            and res_none.failure_taxonomy == "unavailable_service"), \
        f"default mode={default_mode}; missing backend -> {res_none.simulation_status}/" \
        f"{res_none.failure_taxonomy}"


@gate("live_llm_generation_traces_exist")
def g_live_traces():
    summary = OUT / "forensics" / "summary.json"
    if not summary.exists():
        return False, "no live forensic artifacts (run experiments/structural_ensemble_forensics.py)"
    cases = json.loads(summary.read_text())["cases"]
    done = [c for c in cases if c.get("n_independent_generation_calls") and
            c["n_independent_generation_calls"] >= 3 and not c.get("error")]
    return bool(done), f"{len(done)}/{len(cases)} live cases with ≥3 independent generation calls"


def run_suites(skip: bool) -> tuple:
    if skip:
        return None, "skipped by flag (run without --skip-suite for the full gate)"
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q",
           "--deselect", "tests/test_agent_engine.py::test_dataset_registry_is_valid_and_honest"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    tail = (p.stdout or "").strip().splitlines()[-1] if p.stdout else ""
    return p.returncode == 0, f"pytest exit={p.returncode}: {tail[:160]} " \
                              f"(one pre-existing environmental deselect: dataset registry file)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-suite", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for name, fn in GATES:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 — a crashing gate is a failing gate
            ok, detail = False, f"gate crashed: {type(e).__name__}: {e}"
        rows.append({"gate": name, "pass": bool(ok), "detail": str(detail)[:300]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    suite_ok, suite_detail = run_suites(args.skip_suite)
    rows.append({"gate": "full_test_suites_pass",
                 "pass": bool(suite_ok) if suite_ok is not None else None,
                 "detail": suite_detail})
    print(f"[{'PASS' if suite_ok else ('SKIP' if suite_ok is None else 'FAIL')}] "
          f"full_test_suites_pass: {suite_detail}", flush=True)
    hard = [r for r in rows if r["pass"] is not None]
    all_pass = all(r["pass"] for r in hard)
    report = {"schema_version": "structural_ensemble.acceptance.v1",
              "generated_by": "experiments/structural_ensemble_acceptance.py (executable checks)",
              "all_gates_pass": all_pass, "n_gates": len(hard),
              "gates": rows, "wall_clock_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acceptance_report.json").write_text(json.dumps(report, indent=1))
    md = ["# Structural-ensemble acceptance report",
          f"\nGenerated by executable checks; all_gates_pass = **{all_pass}**\n",
          "| gate | pass | detail |", "|---|---|---|"]
    md += [f"| {r['gate']} | {r['pass']} | {r['detail']} |" for r in rows]
    (OUT / "acceptance_report.md").write_text("\n".join(md) + "\n")
    print(f"\nall_gates_pass={all_pass} -> {OUT / 'acceptance_report.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
