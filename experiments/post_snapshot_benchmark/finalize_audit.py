"""Build the final, machine-readable post-snapshot benchmark audit.

This reporting step never reads a resolution store.  It consumes only the
already-scored locked artifact and its single-open ledger, then reconciles the
prospectively frozen forecasts, baselines, phase records, isolation manifests,
causal diagnostic, and test report.  A failed invariant remains failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/post_snapshot_benchmark"
PHASES = (
    "phase1_compiler", "phase2_evidence", "phase3_posterior", "phase4_actor_policy",
    "phase6_registry", "phase7_nonlinear", "phase8_persistence", "phase9_populations",
    "phase9_networks", "phase10_institutions", "phase11_recompilation",
)
SPLITS = ("calibration", "validation", "locked_test")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _gate(required, achieved, evidence: list[str], passed: bool | None = None) -> dict:
    return {
        "required": required,
        "achieved": achieved,
        "evidence": evidence,
        "pass": bool(achieved == required) if passed is None else bool(passed),
    }


def build() -> dict:
    forecasts_by_split = {
        split: _jsonl(RESULTS / f"{split}_forecasts.jsonl") for split in SPLITS
    }
    baselines_by_split = {
        split: _jsonl(RESULTS / f"{split}_baselines.jsonl") for split in SPLITS
    }
    forecasts = [row for split in SPLITS for row in forecasts_by_split[split]]
    baselines = [row for split in SPLITS for row in baselines_by_split[split]]
    score = _json(RESULTS / "locked_test_score.json")
    ledger = _json(RESULTS / "locked_outcome_access_ledger.json")
    causal = _json(RESULTS / "causal_coverage_results.json")
    relevance = _json(RESULTS / "phase_relevance_audit.json")
    capsules = _json(RESULTS / "evidence_capsule_manifest.json")
    split_manifest = _json(RESULTS / "temporal_split_manifest.json")
    selection = _json(RESULTS / "frozen_selection_manifest.json")
    leakage = _json(RESULTS / "leakage_probe_summary.json")
    suite = _json(RESULTS / "full_test_suite_report.json")
    suite_current = suite.get("final_merged_tree", suite)
    temporal = _json(RESULTS / "model_temporal_safety_audit.json")
    phase12_fit = _json(RESULTS / "phase12_calibration_fit.json")
    phase12_selection = _json(RESULTS / "phase12_validation_selection.json")
    freeze = _json(RESULTS / "benchmark_code_freeze_manifest.json")

    worlds = {row["event_world_cluster"] for row in forecasts}
    phase_internal_degradation = []
    phase_table = {}
    causal_by_phase = {
        record["phase"]: record for record in causal["aggregate"]["per_category"].values()
    }
    relevance_metrics = relevance.get("metrics", {})
    for phase in PHASES:
        records = [row["phase_execution_records"][phase] for row in forecasts]
        relevant = [record for record in records if record.get("relevant") is True]
        irrelevant = [record for record in records if record.get("relevant") is False]
        blocked = [record for record in records if str(record.get("execution_status", "")).startswith("blocked_")]
        internal = [
            record for record in records
            if "_error:" in str(record.get("validation_status", ""))
            or "_error:" in str(record.get("relevance_reasons", ""))
            or bool(record.get("errors"))
        ]
        if internal:
            phase_internal_degradation.append({"phase": phase, "affected_rows": len(internal)})
        causal_record = causal_by_phase.get(phase)
        metric = relevance_metrics.get(phase, {})
        phase_table[phase] = {
            "invoked_rows": len(records),
            "invoked_on_every_row": len(records) == len(forecasts),
            "representative_relevant_rows": len(relevant),
            "relevant_causally_active_rate": (
                sum(record.get("execution_status") == "causally_active" for record in relevant) /
                len(relevant) if relevant else None
            ),
            "irrelevant_explicit_noop_rate": (
                sum(record.get("execution_status") == "no_op_causally_irrelevant" for record in irrelevant) /
                len(irrelevant) if irrelevant else None
            ),
            "blocked_rate": len(blocked) / len(records),
            "internal_degradation_rows": len(internal),
            "development_relevance_recall": metric.get("recall_against_preserved_authored_labels"),
            "development_false_causal_activation": metric.get(
                "false_activation_against_preserved_authored_labels"),
            "causal_coverage_independent_worlds": (
                causal_record.get("independent_worlds") if causal_record else None),
            "meaningful_ablation_effect_rate": (
                causal_record.get("any_meaningful_effect_rate") if causal_record else None),
            "status": (
                "fail_internal_degradation" if internal else
                "fail_partial_causal_activation" if causal_record and
                    causal_record.get("full_causally_active_rate") != 1.0 else
                "fail_no_meaningful_ablation_effect" if causal_record and
                    causal_record.get("any_meaningful_effect_rate") == 0 else
                "pass" if causal_record or phase in ("phase1_compiler", "phase3_posterior") else
                "not_separately_ablated"
            ),
        }

    record_coverage = all(
        set(row.get("phase_execution_records", {})) == set(PHASES) for row in forecasts
    )
    formal_relevant_execution = all(
        record.get("execution_status") == "causally_active"
        for row in forecasts for record in row["phase_execution_records"].values()
        if record.get("relevant") is True
    )
    irrelevant_noops = all(
        record.get("execution_status") == "no_op_causally_irrelevant"
        for row in forecasts for record in row["phase_execution_records"].values()
        if record.get("relevant") is False
    )
    baseline_parity = all(
        row.get("all_required_model_arms_complete") is True
        and row.get("identical_evidence_for_all_model_arms") is True
        and row.get("call_matched_ensemble_exactly_v2_budget") is True
        for row in baselines
    )
    all_isolation = all(
        _json(path).get("all_isolation_probes_pass") is True
        for path in RESULTS.glob("isolation_manifest*.json")
    )
    split_counts = {split: len(rows) for split, rows in forecasts_by_split.items()}
    locked_counts = score["counts"]
    no_internal_degradation = not phase_internal_degradation
    causal_effects_all = all(
        record.get("any_meaningful_effect_rate", 0) > 0
        for record in causal["aggregate"]["per_category"].values()
    )
    causal_full_activation = all(
        record.get("full_causally_active_rate") == 1.0
        for record in causal["aggregate"]["per_category"].values()
    )
    suite_clean = suite_current["summary"]["failed"] == 0
    market_informed_complete = score.get("market_informed_comparison", {}).get("status") == "complete"

    gates = {
        "representative_worlds": _gate(100, len(worlds), ["frozen_selection_manifest.json"]),
        "cutoffs_per_world": _gate(4, min(Counter(row["event_world_cluster"] for row in forecasts).values()),
                                    ["calibration_forecasts.jsonl", "validation_forecasts.jsonl",
                                     "locked_test_forecasts.jsonl"]),
        "primary_v2_forecasts": _gate(400, len(forecasts), ["*_forecasts.jsonl"]),
        "temporal_safety_tier": _gate(
            "causally_blinded_historical", temporal["selected_temporal_safety_tier"],
            ["model_temporal_safety_audit.json", "leakage_probe_summary.json"]),
        "phase_record_coverage": _gate(True, record_coverage, ["*_forecasts.jsonl"]),
        "formal_relevant_phase_execution": _gate(True, formal_relevant_execution,
                                                  ["*_forecasts.jsonl"]),
        "internal_phase_degradations_zero": _gate(0, sum(
            row["affected_rows"] for row in phase_internal_degradation), ["*_forecasts.jsonl"],
            passed=no_internal_degradation),
        "irrelevant_phases_explicit_noop": _gate(True, irrelevant_noops, ["*_forecasts.jsonl"]),
        "blocked_relevant_phases": _gate(0, sum(len(row.get("blocked_relevant_phases", []))
                                                for row in forecasts), ["*_forecasts.jsonl"]),
        "terminal_world_state_source": _gate(
            True, all(row.get("terminal_source") == "terminal_world_states" for row in forecasts),
            ["*_forecasts.jsonl"]),
        "evidence_cutoff_safety": _gate(True, capsules.get("all_capsules_cutoff_safe"),
                                        ["evidence_capsule_manifest.json"]),
        "forecaster_outcome_isolation": _gate(
            True, all(row.get("resolution_inaccessible") is True for row in forecasts),
            ["*_forecasts.jsonl", "isolation_manifest*.json"]),
        "forecaster_network_isolation": _gate(True, all_isolation,
                                               ["isolation_manifest*.json"]),
        "world_split_integrity": _gate(True, split_manifest.get("world_integrity"),
                                       ["temporal_split_manifest.json"]),
        "required_baseline_rows": _gate(400, len(baselines), ["*_baselines.jsonl"]),
        "baseline_evidence_and_call_parity": _gate(True, baseline_parity, ["*_baselines.jsonl"]),
        "phase12_fit_governance": _gate(
            "calibration", phase12_fit.get("governance", {}).get("fit_split"),
            ["phase12_calibration_fit.json"]),
        "phase12_selection_governance": _gate(
            "validation", phase12_selection.get("governance", {}).get("selection_split"),
            ["phase12_validation_selection.json"]),
        "locked_test_single_open": _gate(
            1, ledger.get("read_count"), ["locked_outcome_access_ledger.json"]),
        "locked_score_rows": _gate(160, locked_counts.get("forecast_rows"),
                                   ["locked_test_score.json", "locked_test_scored_rows.jsonl"]),
        "market_informed_v2_comparison": _gate(
            True, market_informed_complete, ["locked_test_score.json"]),
        "causal_coverage_protocol": _gate(
            True, causal["aggregate"].get("all_completion_gates_pass"),
            ["causal_coverage_results.json"]),
        "causal_full_phase_activation": _gate(
            True, causal_full_activation, ["causal_coverage_results.json"]),
        "all_covered_phases_show_meaningful_ablation": _gate(
            True, causal_effects_all, ["causal_coverage_results.json"]),
        "full_repository_test_suite": _gate(0, suite_current["summary"]["failed"],
                                            ["full_test_suite_report.json"], passed=suite_clean),
        "stacked_pr_test_file_diff": _gate(0, suite_current.get("stacked_pr_test_file_diff_count"),
                                           ["full_test_suite_report.json"]),
    }
    strict_completion = all(gate["pass"] for gate in gates.values())
    production_eligible = strict_completion and temporal.get("selected_tier_code") in ("A", "B")
    comparison_by_baseline = {
        f"{row['left']}_vs_{row['right']}": row for row in score["comparisons"]
    }
    payload = {
        "schema_version": 1,
        "generated_at": _now(),
        "benchmark_status": (
            "strict_completion_passed" if strict_completion
            else "executed_and_scored_but_strict_completion_failed"),
        "strict_completion_passed": strict_completion,
        "production_eligible": production_eligible,
        "production_judgment": (
            "ineligible: Phase 2 degraded on every primary row; Tier C is mutable; "
            "one Phase 4 causal row was blocked; Phase 8 and Phase 11 had zero meaningful controlled "
            "ablation effect; the market-informed comparison was not run; full suite has failures"
        ),
        "exact_counts": {
            "independent_worlds": len(worlds),
            "forecast_rows_by_split": split_counts,
            "primary_v2_forecasts": len(forecasts),
            "baseline_rows": len(baselines),
            "phase_execution_records": sum(len(row["phase_execution_records"]) for row in forecasts),
            "evidence_capsules": capsules.get("n_capsules"),
            "locked_scored_rows": locked_counts.get("forecast_rows"),
            "causal_worlds": causal["aggregate"].get("worlds"),
            "causal_cutoff_rows": causal["aggregate"].get("cutoff_rows"),
        },
        "required_completion_gates": gates,
        "phase_table": phase_table,
        "phase_internal_degradation": phase_internal_degradation,
        "locked_metrics": score["metrics"],
        "locked_comparisons": comparison_by_baseline,
        "model_memory_sensitivity": score["model_memory_sensitivity"],
        "causal_coverage": causal["aggregate"],
        "temporal_safety": {
            "tier": temporal["selected_temporal_safety_tier"],
            "tier_code": temporal["selected_tier_code"],
            "immutable_hosted_version": temporal.get("hosted_api_immutable_version_identifier"),
            "leakage_probe_summary": leakage,
        },
        "governance": {
            "selection_sha256": selection.get("selection_sha256"),
            "benchmark_code_freeze_sha256": freeze.get("freeze_sha256"),
            "benchmark_code_source_commit": freeze.get("benchmark_code_source_commit"),
            "locked_outcome_ledger": ledger,
            "negative_results_preserved": True,
        },
        "test_suite": {"current": suite_current, "preserved_history": suite},
    }
    payload["artifact_sha256"] = _sha(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "exact_completion_gate_report.json")
    args = parser.parse_args()
    payload = build()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact_sha256": payload["artifact_sha256"],
        "benchmark_status": payload["benchmark_status"],
        "failed_gates": [name for name, gate in payload["required_completion_gates"].items()
                         if not gate["pass"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
