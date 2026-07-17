"""Freeze the outcome-blind runtime/configuration before benchmark forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from swm.world_model_v2.runtime_fingerprint import runtime_fingerprint


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/post_snapshot_benchmark"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def freeze(*, activation_path: Path, preflight_path: Path, isolation_path: Path,
           output: Path) -> dict:
    activation = _load(activation_path)
    preflight = _load(preflight_path)
    isolation = _load(isolation_path)
    selection = RESULTS / "frozen_selection_manifest.json"
    capsules = RESULTS / "evidence_capsule_manifest.json"
    model_audit = RESULTS / "model_temporal_safety_audit.json"
    model_hash = RESULTS / "model_hash_manifest.json"
    fp = runtime_fingerprint()
    source_paths = list(fp["source_sha256"])
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *source_paths], cwd=ROOT, text=True).strip()
    blockers = []
    aggregate = activation.get("aggregate") or {}
    if len(activation.get("rows", [])) != 200 or aggregate.get("n_errors") != 0 or not aggregate.get("all_pass"):
        blockers.append("activation gate is not 200 rows / zero errors / all-pass")
    if not preflight.get("all_gates_pass"):
        blockers.append("sealed preflight report is not all-pass")
    fingerprints = (preflight.get("passing_run") or {}).get("runtime_fingerprints") or []
    if fingerprints != [fp["fingerprint_hash"]]:
        blockers.append("preflight fingerprint does not match current runtime")
    if not isolation.get("all_isolation_probes_pass"):
        blockers.append("OS isolation probes are not all-pass")
    if dirty:
        blockers.append("runtime source files have uncommitted changes")
    if blockers:
        raise RuntimeError("runtime freeze blocked: " + "; ".join(blockers))
    payload = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "outcomes_accessed": False,
        "runtime_fingerprint": fp,
        "activation": {"artifact": activation_path.name, "sha256": _sha(activation_path),
                       "rows": 200, "gates_passed": aggregate["gates_passed"],
                       "gates_total": aggregate["gates_total"]},
        "preflight": {"artifact": preflight_path.name, "sha256": _sha(preflight_path),
                      "all_gates_pass": True},
        "isolation": {"artifact": isolation_path.name, "sha256": _sha(isolation_path),
                      "sandbox_profile_sha256": isolation["sandbox_profile_sha256"]},
        "selection": {"artifact": selection.name, "sha256": _sha(selection),
                      "worlds": 100, "cutoffs_per_world": 4,
                      "chronological_splits": {"calibration": 40, "validation": 20,
                                               "locked_test": 40}},
        "capsules": {"artifact": capsules.name, "sha256": _sha(capsules), "rows": 400},
        "model_temporal_safety": {"tier": "causally_blinded_historical",
                                  "model_identifier": "deepseek-v4-flash",
                                  "model_revision": "mutable_api_alias",
                                  "audit_sha256": _sha(model_audit),
                                  "model_hash_manifest_sha256": _sha(model_hash)},
        "forecast_environment": {"open_internet": False,
                                 "allowed_destination": "api.deepseek.com:443",
                                 "resolution_store_mounted": False,
                                 "pseudonym_mapping_mounted": False},
        "locked_outcomes_opened": False,
    }
    payload["freeze_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=Path,
                        default=RESULTS / "activation200_execution_v4.json")
    parser.add_argument("--preflight", type=Path, default=RESULTS / "preflight_report.json")
    parser.add_argument("--isolation", type=Path, default=RESULTS / "isolation_manifest_v3.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "runtime_freeze_manifest.json")
    args = parser.parse_args()
    print(json.dumps(freeze(activation_path=args.activation, preflight_path=args.preflight,
                            isolation_path=args.isolation, output=args.output), indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
