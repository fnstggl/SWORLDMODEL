"""OS-sealed, resumable full-runtime forecast worker.

This module is invoked only by ``sealed_exec.py``.  It reads an ephemeral API
credential from an inherited pipe, consumes only blinded capsules, invokes the
canonical unified runtime once per row, and freezes every attempt before any
scoring process is allowed to open outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from swm.api.deepseek_backend import deepseek_chat_fn
from swm.world_model_v2.runtime_fingerprint import runtime_fingerprint
from swm.world_model_v2.unified_runtime import simulate_world


PHASES = (
    "phase1_compiler", "phase2_evidence", "phase3_posterior", "phase4_actor_policy",
    "phase6_registry", "phase7_nonlinear", "phase8_persistence", "phase9_populations",
    "phase9_networks", "phase10_institutions", "phase11_recompilation",
)
ALLOWED_STATUSES = {
    "causally_active", "no_op_causally_irrelevant", "blocked_missing_state",
    "blocked_no_mechanism", "blocked_invalid_contract", "execution_failed",
}


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _freeze_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class CapsuleBundle:
    """Minimal evidence-bundle contract backed only by one frozen capsule."""

    def __init__(self, capsule: dict):
        self.question_id = capsule["event_id"]
        self.question = capsule["question"]
        self.as_of = _timestamp(capsule["cutoff"])
        self.slack_s = 0.0
        self.claims = []
        self.included_claim_ids = []
        self.quarantine = []
        self.items = []
        self.documents = []
        for index, item in enumerate(capsule["items"]):
            claim_id = f"capsule_claim_{index}"
            text = str(item.get("blinded_text") or "")
            self.claims.append({
                "claim_id": claim_id, "text": text, "title": item["source_type"],
                "claim_class": "archived_document", "subject": "blinded event state",
                "predicate": "available_at_cutoff", "object": text[:240], "quote": text[:300],
                "publication_time": _timestamp(item["first_proven_available_at"]),
                "source": item["archive_source"], "raw_sha256": item["source_raw_sha256"],
                "archive_retrieval_id": item["archive_retrieval_id"],
            })
            self.included_claim_ids.append(claim_id)
        self._hash = capsule["capsule_sha256"]

    def bundle_hash(self):
        return self._hash

    def included_claims(self):
        return list(self.claims)

    def render(self, max_chars=4000):
        return "\n\n".join(claim["text"] for claim in self.claims)[:max_chars]


def _read_key(fd: int) -> str:
    chunks = []
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(fd)
    key = b"".join(chunks).decode().strip()
    if not key.startswith("sk-") or not key.isascii():
        raise RuntimeError("invalid ephemeral DeepSeek credential on inherited pipe")
    return key


def _p_yes(result) -> float | None:
    if isinstance(result.raw_probability, (int, float)):
        return float(result.raw_probability)
    distribution = result.raw_distribution or {}
    for key, value in distribution.items():
        if str(key).lower() in ("yes", "true", "occurred", "success"):
            return float(value)
    return None


def qualify(result) -> tuple[bool, list[str]]:
    failures = []
    if not result.has_forecast():
        failures.append(f"simulation_status:{result.simulation_status}")
    records = (result.provenance or {}).get("phase_execution_records") or {}
    if set(records) != set(PHASES):
        failures.append(f"phase_record_coverage:{len(records)}/11")
    for phase in PHASES:
        record = records.get(phase) or {}
        status = record.get("execution_status")
        if status not in ALLOWED_STATUSES:
            failures.append(f"{phase}:invalid_status:{status}")
        if record.get("relevant") and status != "causally_active":
            failures.append(f"{phase}:relevant_not_active:{status}")
        if not record.get("relevant") and status != "no_op_causally_irrelevant":
            failures.append(f"{phase}:irrelevant_not_explicit_noop:{status}")
    if (result.provenance or {}).get("fully_integrated") is not True:
        failures.append("fully_integrated:false")
    if _p_yes(result) is None:
        failures.append("terminal_probability_missing")
    return not failures, failures


def _capsule_path(capsule_root: Path, event_id: str, cutoff: str) -> Path:
    safe = "".join(ch for ch in cutoff if ch.isalnum())
    return capsule_root / f"{event_id}__{safe}.json"


def _load_completed(path: Path) -> set[tuple[str, str]]:
    completed = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        row = json.loads(line)
        frozen = {k: v for k, v in row.items() if k != "forecast_sha256"}
        if row.get("forecast_sha256") != _freeze_hash(frozen):
            continue
        if row.get("full_system_qualified"):
            completed.add((row["event_id"], row["forecast_cutoff"]))
    return completed


def _atomic_write_attempt(output: Path, row: dict, *, canonical: bool = False) -> None:
    """Preserve an attempt shard and atomically rebuild its JSONL view.

    The default view retains every attempt for forensic tools.  Forecast and
    baseline workers pass ``canonical=True`` so their benchmark JSONL contains
    exactly the latest successful attempt for each key (or the latest failed
    attempt if no success exists); all attempts still remain in ``*_rows``.
    """
    shard_dir = output.parent / f"{output.stem}_rows"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cutoff = "".join(ch for ch in row["forecast_cutoff"] if ch.isalnum())
    prefix = f"{row['event_id']}__{cutoff}__"
    attempt = len(list(shard_dir.glob(prefix + "*.json"))) + 1
    shard = shard_dir / f"{prefix}{attempt:03d}.json"
    temporary = shard.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary.replace(shard)
    aggregate_tmp = output.with_suffix(output.suffix + ".tmp")
    with aggregate_tmp.open("w") as handle:
        if not canonical:
            for path in sorted(shard_dir.glob("*.json")):
                handle.write(path.read_text())
        else:
            attempts_by_key = {}
            for path in sorted(shard_dir.glob("*.json")):
                attempt_row = json.loads(path.read_text())
                key = (attempt_row["event_id"], attempt_row["forecast_cutoff"])
                attempts_by_key.setdefault(key, []).append(attempt_row)
            for key in sorted(attempts_by_key):
                attempts = attempts_by_key[key]
                successful = [attempt for attempt in attempts
                              if attempt.get("full_system_qualified") is True or
                              attempt.get("all_required_model_arms_complete") is True]
                selected = (successful or attempts)[-1]
                handle.write(json.dumps(selected, sort_keys=True, default=str) + "\n")
    aggregate_tmp.replace(output)


def run(*, credential_fd: int, forecast_input: Path, capsule_root: Path, output: Path,
        split: str, world_limit: int | None = None) -> dict:
    key = _read_key(credential_fd)
    base_llm = deepseek_chat_fn(model="deepseek-v4-flash", api_key=key,
                               system="Reply ONLY with valid JSON.", max_tokens=2400,
                               temperature=0.0, thinking="disabled")
    calls = {"n": 0}

    def llm(prompt):
        calls["n"] += 1
        return base_llm(prompt)

    data = json.loads(forecast_input.read_text())
    worlds = [world for world in data["worlds"] if world["split"] == split]
    if world_limit is not None:
        worlds = worlds[:world_limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(output)
    fp = runtime_fingerprint()
    attempted = qualified = skipped = 0
    for world in worlds:
        for cutoff_index, cutoff in enumerate(world["forecast_cutoffs"]):
            key_tuple = (world["event_id"], cutoff)
            if key_tuple in completed:
                skipped += 1
                continue
            attempted += 1
            capsule_path = _capsule_path(capsule_root, world["event_id"], cutoff)
            capsule = json.loads(capsule_path.read_text())
            call_start = calls["n"]
            started = time.time()
            row = {
                "schema_version": 1, "event_id": world["event_id"],
                "event_world_cluster": world["event_world_cluster"], "domain": world["domain"],
                "split": split, "forecast_cutoff": cutoff, "cutoff_index": cutoff_index,
                "question_open_time": world["question_open_time"],
                "resolution_time": world["resolution_time"],
                "temporal_safety_tier": "causally_blinded_historical",
                "model_identifier": "deepseek-v4-flash", "model_revision": "mutable_api_alias",
                "model_hash_manifest": "model_hash_manifest.json",
                "evidence_capsule_id": capsule_path.name,
                "evidence_capsule_sha256": capsule["capsule_sha256"],
                "evidence_byte_hashes": [item["source_raw_sha256"] for item in capsule["items"]],
                "first_proven_availability_times": [item["first_proven_available_at"]
                                                      for item in capsule["items"]],
                "internet_policy": "os_sandbox_local_allowlist_proxy_only_api.deepseek.com_443",
                "resolution_inaccessible": True, "pseudonym_mapping_inaccessible": True,
                "runtime_fingerprint": fp, "failure_taxonomy": None,
            }
            try:
                runtime_question = (
                    "Binary YES/NO forecasting target. Simulate whether the stated condition is satisfied; "
                    "the first outcome option is YES and the second is NO. " + world["question"])
                result = simulate_world(
                    runtime_question, as_of=cutoff, horizon=world["resolution_time"],
                    seed=int(hashlib.sha256(f"{world['event_id']}:{cutoff}".encode()).hexdigest()[:8], 16),
                    llm=llm, prebuilt_bundle=CapsuleBundle(capsule),
                    compute_budget={"n_budget": 80}, trace_level="forensic")
                is_qualified, qualification_failures = qualify(result)
                records = (result.provenance or {}).get("phase_execution_records") or {}
                row.update({
                    "simulation_status": result.simulation_status,
                    "engine_failure_taxonomy": result.failure_taxonomy,
                    "support_grade": result.support_grade, "raw_terminal_distribution": result.raw_distribution,
                    "p_yes": _p_yes(result), "terminal_source": "terminal_world_states",
                    "terminal_state_evidence": {
                        "n_particles": (result.provenance or {}).get("n_particles"),
                        "n_state_deltas": (result.provenance or {}).get("n_deltas"),
                        "readout_var": (result.provenance or {}).get("readout_var"),
                    },
                    "phase_execution_records": records,
                    "phase_record_count": len(records),
                    "blocked_relevant_phases": [phase for phase, record in records.items()
                                                if record.get("relevant") and
                                                str(record.get("execution_status", "")).startswith("blocked")],
                    "full_system_qualified": is_qualified,
                    "qualification_failures": qualification_failures,
                    "limitations": result.limitations,
                })
                if not is_qualified:
                    row["failure_taxonomy"] = "phase-integration failure"
            except Exception as exc:  # noqa: BLE001 - every attempt must be preserved
                row.update({"simulation_status": "execution_failed", "full_system_qualified": False,
                            "qualification_failures": [f"{type(exc).__name__}: {str(exc)[:240]}"],
                            "failure_taxonomy": "simulation failure"})
            row["latency_s"] = round(time.time() - started, 3)
            row["model_calls"] = calls["n"] - call_start
            row["forecast_frozen_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            row["forecast_sha256"] = _freeze_hash(row)
            _atomic_write_attempt(output, row, canonical=True)
            if row["full_system_qualified"]:
                qualified += 1
            print(f"row {attempted}: qualified={row['full_system_qualified']} ", flush=True)
    return {"split": split, "attempted": attempted, "qualified_this_run": qualified,
            "skipped_frozen": skipped, "expected_rows": len(worlds) * 4,
            "runtime_fingerprint": fp["fingerprint_hash"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-fd", type=int, required=True)
    parser.add_argument("--forecast-input", type=Path, required=True)
    parser.add_argument("--capsule-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "validation", "locked_test"), required=True)
    parser.add_argument("--world-limit", type=int)
    args = parser.parse_args()
    report = run(credential_fd=args.credential_fd, forecast_input=args.forecast_input,
                 capsule_root=args.capsule_root, output=args.output, split=args.split,
                 world_limit=args.world_limit)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
