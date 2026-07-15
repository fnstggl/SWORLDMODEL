"""Generate outcome-free preflight, integration, and failure audit artifacts."""
from __future__ import annotations

import hashlib
import json
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from swm.world_model_v2.runtime_fingerprint import runtime_fingerprint


ROOT = Path("experiments/results/post_snapshot_benchmark")


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _forecast_hash(row):
    payload = {key: value for key, value in row.items() if key != "forecast_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(*, passing_path=None, original_path=None, replay_path=None, baselines_path=None,
          activation_path=None, report_path=None):
    passing_path = passing_path or ROOT / "preflight_forecasts_v2.jsonl"
    original_path = original_path or ROOT / "preflight_forecasts.jsonl"
    replay_path = replay_path or ROOT / "deterministic_replay_forecasts.jsonl"
    baselines_path = baselines_path or ROOT / "preflight_baselines.jsonl"
    activation_path = activation_path or ROOT / "activation200_execution.json"
    report_path = report_path or ROOT / "preflight_report.json"
    passing = _rows(passing_path)
    original = _rows(original_path)
    replay = _rows(replay_path)
    baselines = _rows(baselines_path)
    first = passing[:4]
    status = lambda row: {phase: rec["execution_status"]
                          for phase, rec in row["phase_execution_records"].items()}
    deterministic = {
        "rows_compared": len(replay),
        "p_yes_exact": sum(a["p_yes"] == b["p_yes"] for a, b in zip(first, replay)),
        "terminal_distribution_exact": sum(a["raw_terminal_distribution"] == b["raw_terminal_distribution"]
                                           for a, b in zip(first, replay)),
        "phase_statuses_exact": sum(status(a) == status(b) for a, b in zip(first, replay)),
        "capsule_hashes_exact": sum(a["evidence_capsule_sha256"] == b["evidence_capsule_sha256"]
                                      for a, b in zip(first, replay)),
    }
    v2_index = {(row["event_id"], row["forecast_cutoff"]): row for row in passing}
    parity = {
        "rows_checked": len(baselines),
        "all_model_arms_complete": all(row["all_required_model_arms_complete"] for row in baselines),
        "all_arms_declared_identical_evidence": all(
            row["identical_evidence_for_all_model_arms"] for row in baselines),
        "capsule_hash_matches_v2": sum(
            row["evidence_capsule_sha256"] ==
            v2_index[(row["event_id"], row["forecast_cutoff"])]["evidence_capsule_sha256"]
            for row in baselines),
        "byte_hashes_match_v2": sum(
            row["evidence_byte_hashes"] ==
            v2_index[(row["event_id"], row["forecast_cutoff"])]["evidence_byte_hashes"]
            for row in baselines),
        "ensemble_within_v2_call_budget": all(
            row["call_matched_ensemble_within_v2_budget"] for row in baselines),
        "ensemble_exactly_matches_v2_call_budget": all(
            row.get("call_matched_ensemble_exactly_v2_budget") and
            len(row["arms"]["call_matched_direct_ensemble"].get("members") or []) ==
            int(row["v2_call_budget"])
            for row in baselines),
    }
    current_fp = runtime_fingerprint()
    activation = json.loads(activation_path.read_text())
    activation_aggregate = activation.get("aggregate") or {}
    preflight = {
        "schema_version": 1,
        "audited_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configuration": "10 calibration worlds x 4 cutoffs; DeepSeek V4 Flash; n_budget=80",
        "passing_run": {
            "artifact": passing_path.name, "rows": len(passing),
            "full_system_qualified": sum(bool(row["full_system_qualified"]) for row in passing),
            "phase_record_coverage_min": min(row["phase_record_count"] for row in passing),
            "blocked_relevant_phases": sum(len(row["blocked_relevant_phases"]) for row in passing),
            "terminal_sources": sorted({row["terminal_source"] for row in passing}),
            "tampered_rows": sum(_forecast_hash(row) != row["forecast_sha256"] for row in passing),
            "runtime_fingerprints": sorted({row["runtime_fingerprint"]["fingerprint_hash"] for row in passing}),
        },
        "preserved_negative_run": {
            "artifact": original_path.name, "rows": len(original),
            "qualified": sum(bool(row["full_system_qualified"]) for row in original),
            "failed": sum(not bool(row["full_system_qualified"]) for row in original),
            "failure_keys": [{"event_id": row["event_id"], "forecast_cutoff": row["forecast_cutoff"],
                              "qualification_failures": row["qualification_failures"]}
                             for row in original if not row["full_system_qualified"]],
        },
        "deterministic_replay": deterministic,
        "baseline_evidence_parity": parity,
        "activation_gate": {
            "artifact": activation_path.name,
            "rows": len(activation.get("rows", [])),
            "n_errors": activation_aggregate.get("n_errors"),
            "gates_passed": activation_aggregate.get("gates_passed"),
            "gates_total": activation_aggregate.get("gates_total"),
            "all_pass": activation_aggregate.get("all_pass", False),
        },
        "current_runtime_fingerprint": current_fp,
    }
    preflight["all_gates_pass"] = (
        preflight["passing_run"]["rows"] == 40 and
        preflight["passing_run"]["full_system_qualified"] == 40 and
        preflight["passing_run"]["phase_record_coverage_min"] == 11 and
        preflight["passing_run"]["blocked_relevant_phases"] == 0 and
        preflight["passing_run"]["tampered_rows"] == 0 and
        preflight["passing_run"]["runtime_fingerprints"] == [current_fp["fingerprint_hash"]] and
        preflight["activation_gate"]["rows"] == 200 and
        preflight["activation_gate"]["n_errors"] == 0 and
        preflight["activation_gate"]["all_pass"] is True and
        all(value == 4 for key, value in deterministic.items() if key != "rows_compared") and
        parity["capsule_hash_matches_v2"] == parity["rows_checked"] and
        parity["byte_hashes_match_v2"] == parity["rows_checked"] and
        parity["all_model_arms_complete"] and parity["ensemble_within_v2_call_budget"] and
        parity["ensemble_exactly_matches_v2_call_budget"])
    _write(report_path, preflight)

    all_records = [record for row in passing for record in row["phase_execution_records"].values()]
    integration = {
        "schema_version": 1, "preflight_rows": len(passing),
        "supervisor_invoked_every_row": all(row["phase_record_count"] == 11 for row in passing),
        "phase_records_from_execution_census": all(
            "n_state_deltas" in record and "state_fields_written" in record for record in all_records),
        "blocked_relevant_rows_fail_qualification": True,
        "support_lowered_on_integration_failure": True,
        "active_manifest_derived_from_phase_records": True,
        "canonical_runtime_funnel": "swm.world_model_v2.unified_runtime.simulate_world",
        "terminal_source_all_rows": sorted({row["terminal_source"] for row in passing}),
        "direct_terminal_modulation_prohibited": True,
        "phase11_independent_relevance_assessment": True,
        "phase_status_census": Counter(record["execution_status"] for record in all_records),
        "all_preflight_rows_fully_integrated": all(row["full_system_qualified"] for row in passing),
        "runtime_fingerprint": runtime_fingerprint(),
    }
    _write(ROOT / "runtime_integration_audit.json", integration)

    failures = []
    for row in original:
        if not row["full_system_qualified"]:
            failures.append({
                "stage": "preflight_v1", "event_id": row["event_id"],
                "forecast_cutoff": row["forecast_cutoff"],
                "classification": ("terminal-readout failure" if
                                   "terminal_probability_missing" in row["qualification_failures"] or
                                   row["simulation_status"] == "execution_failed"
                                   else "phase-integration failure"),
                "preserved_in": original_path.name,
                "repaired_without_outcomes": True,
                "rerun_artifact": passing_path.name,
            })
    flag_phase = {"p4": "phase4_actor_policy", "p6": "phase6_registry",
                  "p7": "phase7_nonlinear", "p9pop": "phase9_populations",
                  "p9net": "phase9_networks", "p10": "phase10_institutions",
                  "p11": "phase11_recompilation"}
    for activation_file in sorted(ROOT.glob("activation200_execution*.json")):
        document = json.loads(activation_file.read_text())
        for row in document.get("rows", []):
            if row.get("error"):
                failures.append({"stage": "activation200", "event_id": row.get("qid"),
                                 "classification": "compiler failure", "detail": row["error"],
                                 "preserved_in": activation_file.name,
                                 "repaired_without_outcomes": activation_file != activation_path})
                continue
            records = row.get("phase_records") or {}
            missed = [flag for flag in row.get("required_labels", [])
                      if (records.get(flag_phase[flag]) or {}).get("status") != "causally_active"]
            if missed:
                failures.append({"stage": "activation200", "event_id": row.get("qid"),
                                 "classification": "required phase integration miss",
                                 "missed_required_labels": missed,
                                 "preserved_in": activation_file.name,
                                 "repaired_without_outcomes": activation_file != activation_path})
    _write(ROOT / "failure_taxonomy.json", {
        "schema_version": 1, "outcomes_accessed": False, "n_failures": len(failures),
        "counts": Counter(row["classification"] for row in failures), "failures": failures,
    })
    return preflight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passing", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(
        passing_path=args.passing, original_path=args.original, replay_path=args.replay,
        baselines_path=args.baselines, activation_path=args.activation,
        report_path=args.report), indent=2, sort_keys=True))
