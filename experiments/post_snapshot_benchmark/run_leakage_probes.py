"""Run the six Tier-C model-memory probes in the scorer-side environment.

Full prompts can contain original event wording and therefore never enter the
repository or the sealed forecasting mount. Immutable attempt shards live in
the external scorer root; the repository receives only a count/hash summary.
No resolution file is opened by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.post_snapshot_benchmark.credentials import read_deepseek_key
from experiments.post_snapshot_benchmark.forecast import _atomic_write_attempt, _capsule_path
from swm.api.deepseek_backend import default_chat_fn
from swm.replay.probes2 import classify_row_v2, run_probes_v2


RESULTS = Path("experiments/results/post_snapshot_benchmark")
DEFAULT_SCORER_ROOT = Path("/private/tmp/wmv2-post-snapshot-scorer-0c5a869")
DEFAULT_OUTPUT = DEFAULT_SCORER_ROOT / "leakage_probes.jsonl"
DEFAULT_SUMMARY = RESULTS / "leakage_probe_summary.json"
_WRITE_LOCK = threading.Lock()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def _probe_call_count(value) -> int:
    if isinstance(value, dict):
        here = int("prompt" in value and "output" in value)
        return here + sum(_probe_call_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_probe_call_count(item) for item in value)
    return 0


def _has_error(value) -> bool:
    if isinstance(value, dict):
        if value.get("error") or value.get("parse_failed"):
            return True
        return any(_has_error(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_error(item) for item in value)
    return False


def _completed(output: Path) -> set[tuple[str, str]]:
    done = set()
    if not output.exists():
        return done
    for line in output.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("probe_complete"):
            done.add((str(row["event_id"]), str(row["forecast_cutoff"])))
    return done


def _one(job: dict, llm) -> dict:
    capsule = json.loads(job["capsule_path"].read_text())
    evidence = "\n".join(str(item.get("blinded_text", ""))
                           for item in capsule.get("items", []))
    started = time.time()
    probes = run_probes_v2(
        llm, real_question=job["real_question"], blinded_question=job["blinded_question"],
        mapping=job["mapping"], cutoff=job["cutoff"], evidence_text=evidence)
    classification = classify_row_v2(
        probes, arm="causally_blinded_historical", name_only_correct=None)
    has_error = _has_error(probes)
    return {
        "schema_version": 1,
        "event_id": job["event_id"],
        "forecast_cutoff": job["cutoff"],
        "split": job["split"],
        "model_identifier": "deepseek-v4-flash",
        "model_revision": "mutable_api_alias",
        "temporal_safety_tier": "causally_blinded_historical",
        "mapping_sha256": job["mapping_sha256"],
        "capsule_sha256": capsule["capsule_sha256"],
        "six_probe_contract": ["name_only", "no_evidence_blinded", "recognition",
                               "identity_permutation", "counterfactual_evidence", "temporal_fact"],
        "probes": probes,
        "n_model_calls": _probe_call_count(probes),
        "preopen_classification": classification,
        "name_only_outcome_validation_pending": True,
        "probe_complete": not has_error,
        "latency_s": round(time.time() - started, 3),
    }


def _summary(output: Path, summary_path: Path, expected: int) -> dict:
    attempts = []
    if output.exists():
        attempts = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    latest = {}
    for row in attempts:
        latest[(row["event_id"], row["forecast_cutoff"])] = row
    complete = [row for row in latest.values() if row.get("probe_complete")]
    classes = {}
    for row in complete:
        key = row.get("preopen_classification", "unknown")
        classes[key] = classes.get(key, 0) + 1
    shard_dir = output.parent / f"{output.stem}_rows"
    hashes = []
    for path in sorted(shard_dir.glob("*.json")) if shard_dir.exists() else []:
        hashes.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_hash = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "full_prompts_outside_forecast_mount": True,
        "full_prompts_outside_repository": True,
        "scorer_artifact_path": str(output),
        "expected_rows": expected,
        "completed_rows": len(complete),
        "attempts_preserved": len(attempts),
        "preopen_classification_counts": classes,
        "name_only_outcome_validation_pending": True,
        "attempt_manifest_sha256": manifest_hash,
        "all_rows_complete": len(complete) == expected,
    }
    _atomic_json(summary_path, summary)
    return summary


def run(*, credential_source: Path, forecast_input: Path, selection_manifest: Path,
        mapping_path: Path, capsule_root: Path, output: Path, summary_path: Path,
        workers: int = 6, limit: int | None = None) -> dict:
    key = read_deepseek_key(credential_source)
    llm = default_chat_fn(system="Reply ONLY with valid JSON.", max_tokens=500,
                          temperature=0.0, api_key=key, model="deepseek-v4-flash",
                          thinking="disabled")
    blinded = json.loads(forecast_input.read_text())["worlds"]
    selected = {row["event_id"]: row for row in
                json.loads(selection_manifest.read_text())["worlds"]}
    mapping_doc = json.loads(mapping_path.read_text())
    mappings = {row["event_id"]: row for row in mapping_doc["mappings"]}
    jobs = []
    for world in blinded:
        event_id = world["event_id"]
        mapping_row = mappings[event_id]
        if mapping_row["mapping_sha256"] != world["mapping_sha256"]:
            raise ValueError(f"mapping hash mismatch for {event_id}")
        for cutoff in world["forecast_cutoffs"]:
            jobs.append({
                "event_id": event_id, "cutoff": cutoff, "split": world["split"],
                "real_question": selected[event_id]["question"],
                "blinded_question": world["question"], "mapping": mapping_row["mapping"],
                "mapping_sha256": mapping_row["mapping_sha256"],
                "capsule_path": _capsule_path(capsule_root, event_id, cutoff),
            })
    jobs.sort(key=lambda row: (row["event_id"], row["cutoff"]))
    if limit is not None:
        jobs = jobs[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(output)
    pending = [job for job in jobs if (job["event_id"], job["cutoff"]) not in done]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_one, job, llm): job for job in pending}
        for future in as_completed(futures):
            row = future.result()
            with _WRITE_LOCK:
                _atomic_write_attempt(output, row)
            print(f"{row['event_id']} {row['forecast_cutoff']} "
                  f"calls={row['n_model_calls']} class={row['preopen_classification']} "
                  f"complete={row['probe_complete']}", flush=True)
    return _summary(output, summary_path, expected=len(jobs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-source", type=Path, required=True)
    parser.add_argument("--forecast-input", type=Path,
                        default=RESULTS / "blinded_forecast_input.json")
    parser.add_argument("--selection-manifest", type=Path,
                        default=RESULTS / "frozen_selection_manifest.json")
    parser.add_argument("--mapping-path", type=Path,
                        default=DEFAULT_SCORER_ROOT / "pseudonym_mappings.json")
    parser.add_argument("--capsule-root", type=Path, default=RESULTS / "capsules")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(
        credential_source=args.credential_source, forecast_input=args.forecast_input,
        selection_manifest=args.selection_manifest, mapping_path=args.mapping_path,
        capsule_root=args.capsule_root, output=args.output, summary_path=args.summary,
        workers=args.workers, limit=args.limit), indent=1))


if __name__ == "__main__":
    main()
