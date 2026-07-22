"""Git adapter: clone an official repository (optionally shallow, optionally LFS).

Used for datasets whose canonical home is a GitHub repo (PersuasionForGood,
CraigslistBargain, Deal-or-No-Deal, CaSiNo, ABCD, Open Bandit sample, ...).

Spec keys (from registry ``acquire.git``):
  url:          "https://github.com/org/repo.git"   (required)
  revision:     commit/tag/branch                    (optional)
  subdir:       only keep this subdir of the repo    (optional, e.g. "data")
  lfs:          bool (default False)                  pull Git LFS objects
  depth:        int (default 1)                       shallow clone depth
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ...io_utils import sha256_file
from .base import Adapter, AccessBlocked, AcquisitionError, FetchResult, FileRecord, ProgressCB, noop_progress


class GitAdapter(Adapter):
    name = "git"

    def fetch(self, spec: dict, dest: Path, *, token: str | None = None,
              max_bytes: int | None = None, progress: ProgressCB = noop_progress,
              limit: int | None = None) -> FetchResult:
        url = spec["url"]
        if shutil.which("git") is None:
            raise AcquisitionError("git is not installed")
        dest.mkdir(parents=True, exist_ok=True)
        clone_dir = dest / "_clone"
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        env = {"GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "0" if spec.get("lfs") else "1"}
        cmd = ["git", "clone", "--quiet"]
        depth = int(spec.get("depth", 1))
        if depth:
            cmd += ["--depth", str(depth)]
        if not spec.get("lfs"):
            cmd += ["--config", "filter.lfs.smudge=git-lfs smudge --skip %f",
                    "--config", "filter.lfs.process=git-lfs filter-process --skip"]
        cmd += [url, str(clone_dir)]
        _run(cmd, env)

        rev = spec.get("revision")
        if rev:
            _run(["git", "-C", str(clone_dir), "fetch", "--depth", "1", "origin", rev], env)
            _run(["git", "-C", str(clone_dir), "checkout", rev], env)
        if spec.get("lfs"):
            _run(["git", "-C", str(clone_dir), "lfs", "pull"], env)

        # Move the requested subdir (or the whole tree minus .git) into dest, then drop clone.
        subdir = spec.get("subdir")
        src_root = clone_dir / subdir if subdir else clone_dir
        if not src_root.exists():
            raise AcquisitionError(f"subdir {subdir!r} not found in {url}")

        files = _relocate(src_root, dest, drop_git=(subdir is None))
        shutil.rmtree(clone_dir, ignore_errors=True)
        rev_hash = _rev_parse(dest)  # best-effort; clone already removed, so from notes
        return FetchResult(files=files,
                           resume_state={"mode": "git", "complete": True, "revision": rev or "HEAD"},
                           notes=[f"git clone {url}" + (f"#{rev}" if rev else "")])


def _run(cmd: list[str], env: dict) -> None:
    import os
    full_env = {**os.environ, **env}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=1800)
    except subprocess.TimeoutExpired as e:
        raise AcquisitionError(f"git timed out: {' '.join(cmd[:3])}") from e
    if proc.returncode != 0:
        err = (proc.stderr or "").lower()
        if any(s in err for s in ("authentication", "permission denied", "could not read",
                                  "403", "terminal prompts disabled")):
            raise AccessBlocked("git clone requires authentication/permission for this repo.")
        if "not found" in err or "repository not found" in err:
            raise AccessBlocked("git repository not found or private.")
        raise AcquisitionError(f"git failed ({proc.returncode}): {(proc.stderr or '').strip()[:200]}")


def _rev_parse(_dest: Path) -> str:
    return "unknown"


def _relocate(src_root: Path, dest: Path, *, drop_git: bool) -> list[FileRecord]:
    files: list[FileRecord] = []
    for p in sorted(src_root.rglob("*")):
        if not p.is_file():
            continue
        if drop_git and ".git" in p.parts:
            continue
        rel = p.relative_to(src_root)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        role = "license" if p.name.upper().startswith(("LICENSE", "LICENCE")) else (
            "readme" if p.name.upper().startswith("README") else "data")
        files.append(FileRecord(path=str(rel), sha256=sha256_file(out),
                                size_bytes=out.stat().st_size, role=role))
    return files
