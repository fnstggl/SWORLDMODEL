"""Hugging Face adapter: snapshot_download / hf_hub_download / load_dataset.

Handles both dataset repos (``repo_type=dataset``) and, when a spec asks for streaming,
a bounded number of rows via ``datasets.load_dataset(..., streaming=True)`` written to
Parquet shards. Gated/auth failures become :class:`AccessBlocked` (never retried).

Spec keys (from registry ``acquire.hf``):
  repo_id:        "org/name"                (required)
  repo_type:      "dataset" | "model"       (default "dataset")
  revision:       git revision/tag          (optional)
  allow_patterns: [glob, ...]               (optional; restrict which files to pull)
  config:         dataset config name       (optional; for streaming)
  split:          split name                (optional; for streaming)
  mode:           "snapshot" | "stream"     (default "snapshot")
"""
from __future__ import annotations

import os
from pathlib import Path

from ...io_utils import sha256_file
from .base import Adapter, AccessBlocked, AcquisitionError, FetchResult, FileRecord, ProgressCB, noop_progress


def _classify_hf_error(e: Exception) -> Exception:
    from huggingface_hub.utils import (  # local import
        GatedRepoError, RepositoryNotFoundError, HfHubHTTPError,
    )
    if isinstance(e, GatedRepoError):
        return AccessBlocked("Hugging Face gated repo: accept the license/terms on the "
                             "dataset page and provide HF_TOKEN.", requires_token=True)
    if isinstance(e, RepositoryNotFoundError):
        # Could be private+no-token, or genuinely missing. Surface as blocked-with-token
        # if no token was present, else unavailable.
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            return AccessBlocked("Hugging Face repo not found or private: verify the id "
                                 "and provide HF_TOKEN if it is gated/private.", requires_token=True)
        return AcquisitionError(f"HF repo not found: {e}")
    if isinstance(e, HfHubHTTPError):
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            return AccessBlocked(f"Hugging Face returned {status}: accept terms / provide a valid HF_TOKEN.",
                                 requires_token=True)
    return AcquisitionError(str(e))


class HFAdapter(Adapter):
    name = "huggingface"

    def estimate(self, spec: dict) -> int | None:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.dataset_info(spec["repo_id"], revision=spec.get("revision"),
                                    files_metadata=True, token=_token())
            total = 0
            for s in (info.siblings or []):
                if getattr(s, "size", None):
                    total += s.size
            return total or None
        except Exception:  # noqa: BLE001 - estimation is best-effort
            return None

    def fetch(self, spec: dict, dest: Path, *, token: str | None = None,
              max_bytes: int | None = None, progress: ProgressCB = noop_progress,
              limit: int | None = None) -> FetchResult:
        mode = spec.get("mode", "snapshot")
        if mode == "stream":
            return self._fetch_stream(spec, dest, limit=limit, progress=progress)
        return self._fetch_snapshot(spec, dest, token=token or _token(), progress=progress)

    # ------------------------------------------------------------------ snapshot
    def _fetch_snapshot(self, spec: dict, dest: Path, *, token, progress) -> FetchResult:
        from huggingface_hub import snapshot_download
        dest.mkdir(parents=True, exist_ok=True)
        try:
            local = snapshot_download(
                repo_id=spec["repo_id"],
                repo_type=spec.get("repo_type", "dataset"),
                revision=spec.get("revision"),
                allow_patterns=spec.get("allow_patterns"),
                local_dir=str(dest),
                token=token,
            )
        except Exception as e:  # noqa: BLE001
            raise _classify_hf_error(e) from e
        files = _catalog(Path(local), dest)
        return FetchResult(files=files, resume_state={"mode": "snapshot", "complete": True},
                           notes=[f"snapshot_download {spec['repo_id']}"])

    # ------------------------------------------------------------------ streaming
    def _fetch_stream(self, spec: dict, dest: Path, *, limit, progress) -> FetchResult:
        """Stream a bounded number of rows to a single deterministic Parquet shard.

        This proves the streaming+sharding path for datasets too large to snapshot. The
        orchestrator/normalizer can then request further shards by offset (see
        resume_state)."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import load_dataset

        dest.mkdir(parents=True, exist_ok=True)
        n = int(limit or spec.get("stream_rows", 1000))
        offset = int(spec.get("stream_offset", 0))
        try:
            ds = load_dataset(spec["repo_id"], name=spec.get("config"),
                              split=spec.get("split", "train"), streaming=True,
                              token=_token())
        except Exception as e:  # noqa: BLE001
            raise _classify_hf_error(e) from e

        rows = []
        it = iter(ds)
        for _ in range(offset):
            next(it, None)
        for i, row in enumerate(it):
            if i >= n:
                break
            rows.append({k: _jsonable(v) for k, v in row.items()})
            if i % 200 == 0:
                progress(i, n, f"stream {spec['repo_id']}")
        shard = dest / f"stream_shard_off{offset}_n{len(rows)}.parquet"
        table = pa.Table.from_pylist(rows) if rows else pa.table({})
        pq.write_table(table, shard)
        files = [FileRecord(path=shard.name, sha256=sha256_file(shard),
                            size_bytes=shard.stat().st_size, role="data")]
        return FetchResult(
            files=files,
            resume_state={"mode": "stream", "next_offset": offset + len(rows),
                          "complete": len(rows) < n},
            notes=[f"streamed {len(rows)} rows from offset {offset}"],
        )


def _jsonable(v):
    if isinstance(v, (bytes, bytearray)):
        return f"<bytes:{len(v)}>"
    return v


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _catalog(root: Path, dest: Path) -> list[FileRecord]:
    files: list[FileRecord] = []
    base = dest.resolve()
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".cache" not in p.parts and not p.name.startswith(".huggingface"):
            rel = p.resolve().relative_to(base) if str(p.resolve()).startswith(str(base)) else p.name
            role = "license" if p.name.upper().startswith(("LICENSE", "LICENCE")) else (
                "readme" if p.name.upper().startswith("README") else "data")
            files.append(FileRecord(path=str(rel), sha256=sha256_file(p),
                                    size_bytes=p.stat().st_size, role=role))
    return files
