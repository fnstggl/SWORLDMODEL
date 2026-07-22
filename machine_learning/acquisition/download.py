"""Acquisition orchestrator.

One entry point — :func:`acquire` — that, for a single dataset:

1. reads the registry entry (source, method, license, role),
2. refuses to start if the dataset is blocked/manual/infrastructure (records why),
3. runs a **storage guard**: estimates size, checks free disk, and defers large pulls
   rather than filling the volume,
4. dispatches the right source adapter with bounded exponential-backoff retries,
5. classifies auth/gated failures as *blocked* (never retried) with the exact human
   action required,
6. records a resumable, checksummed source manifest + a redacted committed copy.

One dataset failing NEVER aborts the batch — the failure is recorded and the caller
moves on. Secrets never appear in manifests, notes, or errors.
"""
from __future__ import annotations

from pathlib import Path

from ..config import (DATA_ROOT, disk_status, ensure_working_dirs, raw_dir,
                      DISK_STOP_FRACTION, REPORTS_DIR)
from ..io_utils import human_bytes, read_json, retry, write_json, RetryError, RetryExhausted
from ..registry_io import get_dataset
from .source_adapters.base import AccessBlocked, AcquisitionError, SourceUnavailable, noop_progress
from .source_adapters.git import GitAdapter
from .source_adapters.hf import HFAdapter
from .source_adapters.http import HTTPAdapter
from .verify import source_manifest_path

ADAPTERS = {"hf": HFAdapter, "git": GitAdapter, "http": HTTPAdapter}


def _now(timestamp: str | None) -> str | None:
    # Timestamp is injected by the caller (CLI) so the pipeline stays deterministic in tests.
    return timestamp


def plan(dataset_id: str, *, allow_large: bool = False) -> dict:
    """Return a storage plan for a dataset WITHOUT downloading anything.

    Implements the pre-acquisition safety checklist: estimated size, current disk, and a
    go / defer decision. Printed by the CLI before any large pull.
    """
    entry = get_dataset(dataset_id)
    acq = entry.get("acquire", {}) or {}
    est = acq.get("estimated_bytes")
    ds = disk_status(DATA_ROOT)
    decision = "go"
    reason = "ok"
    if entry["download_method"] in ("none", "manual") or entry["dataset_role"] in ("ACCESS_BLOCKED",):
        decision, reason = "skip", f"not downloadable ({entry['dataset_role']}/{entry['download_method']})"
    elif ds.used_fraction >= DISK_STOP_FRACTION:
        decision, reason = "defer", f"disk already at {ds.used_fraction:.0%} (>= {DISK_STOP_FRACTION:.0%})"
    elif est and not allow_large:
        projected_free = ds.free_bytes - est
        if projected_free < 0 or (ds.used_bytes + est) / max(ds.total_bytes, 1) >= DISK_STOP_FRACTION:
            decision, reason = "defer", (
                f"estimated {human_bytes(est)} would exceed the {DISK_STOP_FRACTION:.0%} "
                f"disk threshold (free {human_bytes(ds.free_bytes)})")
    return {
        "dataset_id": dataset_id,
        "download_method": entry["download_method"],
        "estimated_bytes": est,
        "estimated_human": human_bytes(est) if est else "unknown",
        "disk": ds.as_dict(),
        "decision": decision,
        "reason": reason,
        "allow_large": allow_large,
    }


def _skeleton(dataset_id: str, entry: dict, timestamp: str | None) -> dict:
    return {
        "dataset_id": dataset_id,
        "dataset_version": entry.get("dataset_version"),
        "status": "pending",
        "download_method": entry["download_method"],
        "source_urls": _source_urls(entry),
        "created_at": _now(timestamp),
        "updated_at": _now(timestamp),
        "files": [],
        "total_bytes": 0,
        "estimated_total_bytes": (entry.get("acquire", {}) or {}).get("estimated_bytes"),
        "license_snapshot": {"license": entry.get("license"), "license_file": None,
                             "verified_from": "", "verified": False},
        "access": {"gated": bool(entry.get("gated")), "requires_token": False,
                   "requirement": entry.get("access_requirements")},
        "resume_state": {},
        "notes": [],
        "errors": [],
    }


def _source_urls(entry: dict) -> list[str]:
    urls = []
    for k in ("official_data_source", "official_code_source"):
        if entry.get(k):
            urls.append(entry[k])
    return urls


def acquire(dataset_id: str, *, limit: int | None = None, force: bool = False,
            allow_large: bool = False, timestamp: str | None = None,
            progress=noop_progress) -> dict:
    """Acquire one dataset. Returns the (updated) source manifest dict. Never raises for
    an expected outcome (blocked/deferred/failed) — those are encoded in the manifest."""
    ensure_working_dirs()
    entry = get_dataset(dataset_id)
    method = entry["download_method"]
    role = entry["dataset_role"]

    man = read_json(source_manifest_path(dataset_id)) or _skeleton(dataset_id, entry, timestamp)
    man["updated_at"] = _now(timestamp)

    def save(status: str) -> dict:
        man["status"] = status
        write_json(source_manifest_path(dataset_id), man)
        _commit_redacted(dataset_id, man)
        return man

    # -- non-downloadable short circuits -------------------------------------------------
    if role == "ACCESS_BLOCKED" or method == "manual":
        man["notes"].append(f"not auto-downloadable: {entry.get('blockers') or role}")
        man["access"]["requirement"] = entry.get("access_requirements")
        return save("blocked")
    if role == "INFRASTRUCTURE_ONLY" or method == "none":
        man["notes"].append(f"infrastructure/no-data role: {role}")
        return save("skipped")

    # -- resume: already acquired --------------------------------------------------------
    if man.get("status") == "acquired" and not force:
        man["notes"].append("already acquired (use force=True to re-fetch)")
        return save("acquired")

    # -- storage guard -------------------------------------------------------------------
    p = plan(dataset_id, allow_large=allow_large)
    if p["decision"] == "defer":
        man["notes"].append(f"deferred for storage: {p['reason']}")
        return save("deferred_storage")

    # -- dispatch adapter with retries ---------------------------------------------------
    adapter_cls = ADAPTERS.get(method)
    if adapter_cls is None:
        man["errors"].append({"error_type": "config", "message": f"no adapter for method {method!r}"})
        return save("failed")
    spec = (entry.get("acquire", {}) or {}).get(method)
    if not spec:
        man["errors"].append({"error_type": "config", "message": f"missing acquire.{method} spec"})
        return save("failed")

    adapter = adapter_cls()
    dest = raw_dir(dataset_id)

    def _on_err(rec: RetryError) -> None:
        man["errors"].append({"attempt": rec.attempt, "error_type": rec.error_type,
                              "message": rec.message, "next_action": rec.next_action})

    man["status"] = "acquiring"
    write_json(source_manifest_path(dataset_id), man)
    try:
        result = retry(
            lambda: adapter.fetch(spec, dest, max_bytes=None, progress=progress, limit=limit),
            attempts=5, base_delay=2.0, max_delay=60.0,
            retry_on=(AcquisitionError,),
            give_up_on=(AccessBlocked, SourceUnavailable),
            on_error=_on_err,
        )
    except AccessBlocked as e:
        man["access"] = {"gated": True, "requires_token": e.requires_token, "requirement": e.requirement}
        man["notes"].append(f"access blocked: {e.requirement}")
        return save("blocked")
    except SourceUnavailable as e:
        man["notes"].append(f"source unavailable: {e}")
        return save("blocked")
    except RetryExhausted:
        man["notes"].append("acquisition failed after retries; partial progress preserved")
        return save("failed")

    # -- success -------------------------------------------------------------------------
    man["files"] = [f.as_dict() for f in result.files]
    man["total_bytes"] = result.total_bytes
    man["resume_state"] = result.resume_state
    man["notes"].extend(result.notes)
    man["license_snapshot"] = _license_snapshot(entry, dest, result.files)
    return save("acquired" if result.resume_state.get("complete", True) else "partial")


def _license_snapshot(entry: dict, dest: Path, files) -> dict:
    """Record what the source itself says about the license (from a shipped LICENSE file)."""
    lic_file = None
    verified_from = ""
    verified = False
    for f in files:
        if f.role == "license":
            lic_file = f.path
            verified_from = str(dest / f.path)
            verified = True
            break
    return {
        "license": entry.get("license"),
        "license_class": entry.get("license_class"),
        "license_file": lic_file,
        "verified_from": verified_from or entry.get("official_data_source", ""),
        "verified": verified,
        "commercial_use_allowed": entry.get("commercial_use_allowed"),
        "derivatives_allowed": entry.get("derivatives_allowed"),
        "redistribution_allowed": entry.get("redistribution_allowed"),
    }


def _commit_redacted(dataset_id: str, man: dict) -> None:
    """Write a small, secret-free copy of the manifest under committed reports/acquisition."""
    out = REPORTS_DIR / "acquisition" / f"{dataset_id}.json"
    redacted = {k: v for k, v in man.items() if k != "resume_state"}
    # keep only file names + hashes (no absolute paths), cap the list length in the copy
    redacted["files"] = [{"path": f["path"], "sha256": f["sha256"][:16], "size_bytes": f["size_bytes"],
                          "role": f.get("role", "data")} for f in man.get("files", [])[:200]]
    redacted["n_files"] = len(man.get("files", []))
    write_json(out, redacted)
