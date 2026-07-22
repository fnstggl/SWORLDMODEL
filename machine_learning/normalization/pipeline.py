"""Normalization pipeline: raw source -> validated canonical Parquet shards.

Guarantees:
* every emitted record passes schema validation (envelope + task payload);
* malformed records are **quarantined with a reason**, counted, and reported — never
  silently dropped, and a high quarantine rate is surfaced as a converter bug signal;
* exact duplicate records (same deterministic record_id) are collapsed and counted;
* a small human-review sample (valid + most-suspicious) is written for the audit;
* a normalization manifest records shard layout, counts, task/missing-field distributions,
  and warnings so the whole run is auditable + resumable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..config import HUMAN_REVIEW_DIR, REPORTS_DIR, normalized_dir, raw_dir, state_dir
from ..io_utils import read_json, write_json, write_jsonl
from ..validation.schema_validation import validate_record
from .base import SourceNotAvailable, load_converter
from .common.parquet_io import ShardWriter

SAMPLE_N = 50
SUSPICIOUS_N = 25


@dataclass
class NormalizationReport:
    dataset_id: str
    status: str
    n_emitted: int = 0
    n_valid: int = 0
    n_quarantined: int = 0
    n_duplicates: int = 0
    shards: int = 0
    task_counts: dict = field(default_factory=dict)
    missing_field_counts: dict = field(default_factory=dict)
    n_episodes: int = 0
    n_actors: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["quarantine_rate"] = round(self.n_quarantined / self.n_emitted, 4) if self.n_emitted else 0.0
        return d


def _manifest_path(dataset_id: str) -> Path:
    return state_dir(dataset_id) / "normalization_manifest.json"


def normalize(dataset_id: str, *, limit: int | None = None, timestamp: str | None = None,
              code_commit: str | None = None, shard_size: int = 5000,
              quarantine_threshold: float = 0.20) -> NormalizationReport:
    """Normalize one dataset. Returns a report; also writes shards + manifests + samples."""
    conv = load_converter(dataset_id, timestamp=timestamp, code_commit=code_commit)
    if conv is None:
        return NormalizationReport(dataset_id, status="no_converter",
                                   notes=["registry has no converter for this dataset"])

    src = raw_dir(dataset_id)
    out = normalized_dir(dataset_id)
    out.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(out, shard_size=shard_size)
    quarantine = state_dir(dataset_id) / "quarantine.jsonl"
    quarantine.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    episodes: set[str] = set()
    actors: set[str] = set()
    task_counts: Counter = Counter()
    missing_counts: Counter = Counter()
    q_records: list[dict] = []
    sample: list[dict] = []
    suspicious: list[dict] = []

    rep = NormalizationReport(dataset_id, status="running")

    try:
        it = conv.iter_records(src)
        for rec in it:
            if limit is not None and rep.n_valid >= limit:
                break
            rep.n_emitted += 1
            res = validate_record(rec)
            if not res.ok:
                rep.n_quarantined += 1
                if len(q_records) < 500:
                    q_records.append({"record_id": rec.get("record_id", "<no-id>"),
                                      "errors": res.errors[:8],
                                      "locator": rec.get("provenance", {}).get("raw_record_locator")})
                continue
            rid = rec["record_id"]
            if rid in seen_ids:
                rep.n_duplicates += 1
                continue
            seen_ids.add(rid)
            writer.add(rec)
            rep.n_valid += 1
            task_counts[rec["task_type"]] += 1
            episodes.add(rec["episode"]["episode_id"])
            au = rec["decision_unit"].get("actor_id")
            if au:
                actors.add(au)
            for m in rec["data_quality"].get("missing_fields", []):
                missing_counts[m] += 1
            if len(sample) < SAMPLE_N:
                sample.append(rec)
            dq = rec["data_quality"]
            if (dq.get("possible_leakage") or dq.get("warnings")) and len(suspicious) < SUSPICIOUS_N:
                suspicious.append(rec)
    except SourceNotAvailable as e:
        rep.status = "source_unavailable"
        rep.notes.append(str(e))
        return _finish(rep, writer, dataset_id, task_counts, missing_counts, episodes,
                       actors, q_records, quarantine, sample, suspicious, timestamp)
    except FileNotFoundError as e:
        rep.status = "not_acquired"
        rep.notes.append(f"raw data not found ({e}); acquire the dataset first")
        return _finish(rep, writer, dataset_id, task_counts, missing_counts, episodes,
                       actors, q_records, quarantine, sample, suspicious, timestamp)

    if rep.n_emitted and rep.n_quarantined / rep.n_emitted > quarantine_threshold:
        rep.warnings.append(
            f"HIGH quarantine rate {rep.n_quarantined}/{rep.n_emitted} "
            f"(> {quarantine_threshold:.0%}) — likely a converter bug, investigate before trusting output")
    rep.status = "normalized"
    return _finish(rep, writer, dataset_id, task_counts, missing_counts, episodes,
                   actors, q_records, quarantine, sample, suspicious, timestamp)


def _finish(rep, writer, dataset_id, task_counts, missing_counts, episodes, actors,
            q_records, quarantine_path, sample, suspicious, timestamp) -> NormalizationReport:
    info = writer.close()
    rep.shards = info["shards"]
    rep.task_counts = dict(task_counts)
    rep.missing_field_counts = dict(missing_counts)
    rep.n_episodes = len(episodes)
    rep.n_actors = len(actors)

    if q_records:
        write_jsonl(quarantine_path, q_records)

    # committed human-review sample + suspicious set (small)
    if sample:
        write_jsonl(HUMAN_REVIEW_DIR / f"{dataset_id}.sample.jsonl", sample)
    if suspicious:
        write_jsonl(HUMAN_REVIEW_DIR / f"{dataset_id}.suspicious.jsonl", suspicious)

    manifest = {
        "dataset_id": dataset_id,
        "normalized_at": timestamp,
        **rep.as_dict(),
        "normalized_dir": info["dir"],
    }
    write_json(_manifest_path(dataset_id), manifest)
    write_json(REPORTS_DIR / "normalization" / f"{dataset_id}.json", manifest)
    return rep


def load_normalization_manifest(dataset_id: str) -> dict | None:
    return read_json(_manifest_path(dataset_id))
