"""Sharded Parquet IO for canonical records.

Storage model: each record is stored as a compact JSON string in a ``record_json`` column
alongside flat index columns (record_id, task_type, episode_id, actor_id, split,
content_hash, dedup_hash). This is:

* **lossless** — the full nested record round-trips through ``record_json``;
* **queryable** — index columns support split assignment, dedup, and example views
  without parsing every blob;
* **robust** — heterogeneous task payloads never fight a rigid columnar schema;
* **compact** — Parquet + zstd compresses the repeated JSON keys well.

Shards are deterministic and fixed-size, so normalization is resumable: a shard that
already exists is not recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from ...canonical import canonical_json

INDEX_COLUMNS = ["record_id", "dataset_id", "task_type", "episode_id", "actor_id",
                 "split", "content_hash", "dedup_hash"]


def _row(rec: dict) -> dict:
    return {
        "record_id": rec.get("record_id", ""),
        "dataset_id": rec.get("source", {}).get("dataset_id", ""),
        "task_type": rec.get("task_type", ""),
        "episode_id": rec.get("episode", {}).get("episode_id", ""),
        "actor_id": rec.get("decision_unit", {}).get("actor_id"),
        "split": rec.get("split_metadata", {}).get("split"),
        "content_hash": rec.get("provenance", {}).get("content_hash"),
        "dedup_hash": rec.get("split_metadata", {}).get("dedup_hash"),
        "record_json": canonical_json(rec),
    }


def write_shard(path: str | Path, records: list[dict]) -> int:
    """Write ``records`` to a single Parquet shard atomically. Returns count."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_row(r) for r in records]
    cols = INDEX_COLUMNS + ["record_json"]
    table = pa.table({c: [r[c] for r in rows] for c in cols}) if rows else pa.table(
        {c: pa.array([], type=pa.string()) for c in cols})
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return len(rows)


def read_records(path: str | Path) -> Iterator[dict]:
    """Yield full canonical records from a shard (parses record_json)."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=["record_json"])
    for v in tbl.column("record_json").to_pylist():
        yield json.loads(v)


def read_index(path: str | Path):
    """Return the index columns of a shard as a pyarrow Table (no JSON parsing)."""
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=INDEX_COLUMNS)


def iter_shards(dir_path: str | Path) -> list[Path]:
    d = Path(dir_path)
    if not d.exists():
        return []
    return sorted(d.glob("shard_*.parquet"))


def iter_records(dir_path: str | Path) -> Iterator[dict]:
    for shard in iter_shards(dir_path):
        yield from read_records(shard)


class ShardWriter:
    """Accumulate records and flush deterministic fixed-size Parquet shards.

    Resumable: if a target shard file already exists it is left untouched and the writer
    advances the shard counter, so a re-run continues rather than duplicating work.
    """

    def __init__(self, out_dir: str | Path, shard_size: int = 5000, prefix: str = "shard"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self._buf: list[dict] = []
        self._shard_idx = 0
        self.written_records = 0
        self.written_shards = 0

    def _shard_path(self, idx: int) -> Path:
        return self.out_dir / f"{self.prefix}_{idx:05d}.parquet"

    def add(self, rec: dict) -> None:
        self._buf.append(rec)
        if len(self._buf) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        path = self._shard_path(self._shard_idx)
        if path.exists():
            # resume: shard already materialized; skip re-writing this deterministic slice
            self.written_shards += 1
            self.written_records += len(self._buf)
        else:
            write_shard(path, self._buf)
            self.written_shards += 1
            self.written_records += len(self._buf)
        self._shard_idx += 1
        self._buf = []

    def close(self) -> dict:
        self._flush()
        return {"shards": self.written_shards, "records": self.written_records,
                "shard_size": self.shard_size, "dir": str(self.out_dir)}
