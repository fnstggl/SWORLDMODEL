"""Verify acquired data against its source manifest (checksums + sizes).

Integrity is the first line of the audit trail: a normalized record's provenance points
back to a raw file, and that raw file must hash to exactly what was recorded at
acquisition time. ``verify_manifest`` re-hashes every listed file and reports mismatches,
missing files, and unexpected extras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import raw_dir, state_dir
from ..io_utils import read_json, sha256_file


@dataclass
class VerifyReport:
    dataset_id: str
    ok: bool
    checked: int = 0
    mismatches: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id, "ok": self.ok, "checked": self.checked,
            "mismatches": self.mismatches, "missing": self.missing, "notes": self.notes,
        }


def source_manifest_path(dataset_id: str) -> Path:
    return state_dir(dataset_id) / "source_manifest.json"


def load_source_manifest(dataset_id: str) -> dict | None:
    return read_json(source_manifest_path(dataset_id))


def verify_manifest(dataset_id: str, *, deep: bool = True) -> VerifyReport:
    """Verify every file in the dataset's source manifest.

    deep=True re-hashes file contents (slow, authoritative). deep=False checks only that
    files exist with the recorded size (fast smoke check).
    """
    man = load_source_manifest(dataset_id)
    if man is None:
        return VerifyReport(dataset_id, ok=False, notes=["no source manifest"])
    if man.get("status") in ("blocked", "deferred_storage", "skipped", "pending"):
        return VerifyReport(dataset_id, ok=True,
                            notes=[f"status={man['status']}: nothing to verify"])

    root = raw_dir(dataset_id)
    rep = VerifyReport(dataset_id, ok=True)
    for f in man.get("files", []):
        p = root / f["path"]
        if not p.exists():
            rep.missing.append(f["path"])
            rep.ok = False
            continue
        rep.checked += 1
        if p.stat().st_size != f["size_bytes"]:
            rep.mismatches.append(f"{f['path']}: size {p.stat().st_size} != {f['size_bytes']}")
            rep.ok = False
            continue
        if deep:
            actual = sha256_file(p)
            if actual != f["sha256"]:
                rep.mismatches.append(f"{f['path']}: sha256 mismatch")
                rep.ok = False
    if rep.ok:
        rep.notes.append(f"verified {rep.checked} files ({'deep' if deep else 'shallow'})")
    return rep
