"""HTTP(S) adapter: resumable direct-URL downloads (curl/wget-style via requests).

Used for datasets hosted at a plain URL (Criteo Uplift .csv.gz, Zenodo tarballs, OSF
files). Supports HTTP Range resume: a partially-downloaded ``.download`` file is
continued rather than restarted.

Spec keys (from registry ``acquire.http``):
  urls:      [ {url, filename, role} ... ]   (required)
  headers:   {..}                            (optional)
"""
from __future__ import annotations

from pathlib import Path

from ...io_utils import human_bytes, sha256_file
from .base import Adapter, AccessBlocked, AcquisitionError, FetchResult, FileRecord, ProgressCB, noop_progress

_CHUNK = 1024 * 256


class HTTPAdapter(Adapter):
    name = "http"

    def estimate(self, spec: dict) -> int | None:
        import requests
        total = 0
        got_any = False
        for item in spec.get("urls", []):
            try:
                r = requests.head(item["url"], allow_redirects=True, timeout=30,
                                  headers=spec.get("headers"))
                cl = r.headers.get("Content-Length")
                if cl:
                    total += int(cl)
                    got_any = True
            except Exception:  # noqa: BLE001
                continue
        return total if got_any else None

    def fetch(self, spec: dict, dest: Path, *, token: str | None = None,
              max_bytes: int | None = None, progress: ProgressCB = noop_progress,
              limit: int | None = None) -> FetchResult:
        import requests
        dest.mkdir(parents=True, exist_ok=True)
        files: list[FileRecord] = []
        notes: list[str] = []
        for item in spec.get("urls", []):
            url = item["url"]
            fname = item.get("filename") or url.rstrip("/").split("/")[-1]
            role = item.get("role", "data")
            out = dest / fname
            if out.exists():
                files.append(FileRecord(path=fname, sha256=sha256_file(out),
                                        size_bytes=out.stat().st_size, role=role))
                notes.append(f"skip existing {fname}")
                continue
            partial = out.with_suffix(out.suffix + ".download")
            resume_from = partial.stat().st_size if partial.exists() else 0
            headers = dict(spec.get("headers") or {})
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"
            try:
                with requests.get(url, stream=True, timeout=120, headers=headers,
                                  allow_redirects=True) as r:
                    if r.status_code in (401, 403):
                        raise AccessBlocked(f"HTTP {r.status_code} for {url}: requires "
                                            f"authentication or accepting terms.")
                    if r.status_code == 416:  # range not satisfiable -> already complete
                        partial.replace(out)
                    elif r.status_code not in (200, 206):
                        raise AcquisitionError(f"HTTP {r.status_code} for {url}")
                    else:
                        mode = "ab" if (resume_from and r.status_code == 206) else "wb"
                        if mode == "wb" and partial.exists():
                            partial.unlink()
                        total = r.headers.get("Content-Length")
                        total_i = (int(total) + resume_from) if total else None
                        done = resume_from
                        with open(partial, mode) as f:
                            for chunk in r.iter_content(_CHUNK):
                                if not chunk:
                                    continue
                                f.write(chunk)
                                done += len(chunk)
                                progress(done, total_i, fname)
                        partial.replace(out)
            except AccessBlocked:
                raise
            except requests.RequestException as e:
                raise AcquisitionError(f"download failed for {url}: {e}") from e
            files.append(FileRecord(path=fname, sha256=sha256_file(out),
                                    size_bytes=out.stat().st_size, role=role))
            notes.append(f"downloaded {fname} ({human_bytes(out.stat().st_size)})")
        return FetchResult(files=files, resume_state={"mode": "http", "complete": True},
                           notes=notes)
