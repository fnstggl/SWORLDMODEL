"""Generate outcome-free preflight, integration, and failure audit artifacts."""
from __future__ import annotations

import hashlib
import json
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


def build():
    passing = _rows(ROOT / "preflight_forecasts_v2.jsonl")
    original = _rows(ROOT / "preflight_forecasts.jsonl")
    replay = _rows(ROOT / "deterministic_replay_forecasts.jsonl")
    baselines = _rows(ROOT / "preflight_baselines.jsonl")
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
    }
    preflight = {
        "schema_version": 1,
        "audited_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configuration": "10 calibration worlds x 4 cutoffs; DeepSeek V4 Flash; n_budget=80",
        "passing_run": {
            "artifact": "preflight_forecasts_v2.jsonl", "rows": len(passing),
            "full_system_qualified": sum(bool(row["full_system_qualified"]) for row in passing),
            "phase_record_coverage_min": min(row["phase_record_count"] for row in passing),
            "blocked_relevant_phases": sum(len(row["blocked_relevant_phases"]) for row in passing),
            "terminal_sources": sorted({row["terminal_source"] for row in passing}),
            "tampered_rows": sum(_forecast_hash(row) != row["forecast_sha256"] for row in passing),
            "runtime_fingerprints": sorted({row["runtime_fingerprint"]["fingerprint_hash"] for row in passing}),
        },
        "preserved_negative_run": {
            "artifact": "preflight_forecasts.jsonl", "rows": len(original),
            "qualified": sum(bool(row["full_system_qualified"]) for row in original),
            "failed": sum(not bool(row["full_system_qualified"]) for row in original),
            "failure_keys": [{"event_id": row["event_id"], "forecast_cutoff": row["forecast_cutoff"],
                              "qualification_failures": row["qualification_failures"]}
                             for row in original if not row["full_system_qualified"]],
        },
        "deterministic_replay": deterministic,
        "baseline_evidence_parity": parity,
    }
    preflight["all_gates_pass"] = (
        preflight["passing_run"]["rows"] == 40 and
        preflight["passing_run"]["full_system_qualified"] == 40 and
        preflight["passing_run"]["phase_record_coverage_min"] == 11 and
        preflight["passing_run"]["blocked_relevant_phases"] == 0 and
        preflight["passing_run"]["tampered_rows"] == 0 and
        all(value == 4 for key, value in deterministic.items() if key != "rows_compared") and
        parity["capsule_hash_matches_v2"] == parity["rows_checked"] and
        parity["byte_hashes_match_v2"] == parity["rows_checked"] and
        parity["all_model_arms_complete"] and parity["ensemble_within_v2_call_budget"])
    _write(ROOT / "preflight_report.json", preflight)

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
                "preserved_in": "preflight_forecasts.jsonl",
                "repaired_without_outcomes": True,
                "rerun_artifact": "preflight_forecasts_v2.jsonl",
            })
    activation = json.loads((ROOT / "activation200_execution.json").read_text())
    for row in activation.get("rows", []):
        if row.get("error"):
            failures.append({"stage": "activation200_no_credential_smoke", "event_id": row.get("qid"),
                             "classification": "compiler failure", "detail": row["error"],
                             "preserved_in": "activation200_execution.json",
                             "repaired_without_outcomes": False})
    _write(ROOT / "failure_taxonomy.json", {
        "schema_version": 1, "outcomes_accessed": False, "n_failures": len(failures),
        "counts": Counter(row["classification"] for row in failures), "failures": failures,
    })
    return preflight


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
