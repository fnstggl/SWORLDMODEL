"""Provenance completeness + lineage tracing.

Every normalized record must trace back to an exact raw source record. This module both
*verifies* that lineage is complete and *renders* it: ``trace(record_id)`` returns the full
chain dataset -> raw file(s) -> raw record id/index -> converter+version -> transformation
steps -> content hash, joined with the source acquisition manifest (raw file checksums).
Powers ``python -m machine_learning.cli provenance show <record_id>``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..acquisition.verify import load_source_manifest
from ..config import normalized_dir
from ..normalization.common.parquet_io import iter_records


@dataclass
class ProvenanceReport:
    dataset_id: str
    n_checked: int = 0
    missing: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict:
        return {"dataset_id": self.dataset_id, "n_checked": self.n_checked,
                "n_missing": len(self.missing), "ok": self.ok, "missing": self.missing[:50]}


_REQUIRED = ["converter", "converter_version", "content_hash"]


def check_dataset(dataset_id: str, *, limit: int | None = None) -> ProvenanceReport:
    rep = ProvenanceReport(dataset_id=dataset_id)
    for i, r in enumerate(iter_records(normalized_dir(dataset_id))):
        if limit and i >= limit:
            break
        rep.n_checked += 1
        prov = r.get("provenance", {})
        problems = [k for k in _REQUIRED if not prov.get(k)]
        loc = prov.get("raw_record_locator", {})
        if not (loc.get("files") or loc.get("ids") or loc.get("indices")):
            problems.append("raw_record_locator")
        if not r.get("source", {}).get("dataset_id"):
            problems.append("source.dataset_id")
        if problems:
            rep.missing.append({"record_id": r.get("record_id"), "missing": problems})
    return rep


def trace(record_id: str) -> dict | None:
    """Return the full lineage of a normalized record, or None if not found."""
    parts = record_id.split(":")
    dataset_id = parts[0] if parts else ""
    if not dataset_id:
        return None
    for r in iter_records(normalized_dir(dataset_id)):
        if r.get("record_id") == record_id:
            return _render(r)
    return None


def _render(r: dict) -> dict:
    prov = r.get("provenance", {})
    src = r.get("source", {})
    man = load_source_manifest(src.get("dataset_id", "")) or {}
    file_hashes = {f["path"]: f["sha256"][:16] for f in man.get("files", [])}
    loc = prov.get("raw_record_locator", {})
    return {
        "record_id": r.get("record_id"),
        "dataset_id": src.get("dataset_id"),
        "dataset_version": src.get("dataset_version"),
        "task_type": r.get("task_type"),
        "license_class": src.get("license_class"),
        "citation": src.get("citation"),
        "converter": prov.get("converter"),
        "converter_version": prov.get("converter_version"),
        "conversion_timestamp": prov.get("conversion_timestamp"),
        "code_commit": prov.get("code_commit"),
        "transformation_steps": prov.get("transformation_steps"),
        "raw_source": {
            "files": loc.get("files"),
            "record_ids": loc.get("ids"),
            "indices": loc.get("indices"),
            "raw_file_checksums": {f: file_hashes.get(f, "<not in source manifest>")
                                   for f in (loc.get("files") or [])},
        },
        "content_hash": prov.get("content_hash"),
        "episode_id": r.get("episode", {}).get("episode_id"),
        "split": r.get("split_metadata", {}).get("split"),
        "data_quality": r.get("data_quality", {}),
    }
