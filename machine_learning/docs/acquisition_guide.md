# Acquisition guide

Acquisition pulls one dataset from its official source into `$SWM_DATA_ROOT/raw/<id>/`, records a
resumable, checksummed **source manifest**, and never lets one failure abort a batch. It never
fabricates access: gated/auth failures are recorded as `blocked` with the exact human action needed.

## Source adapters

`acquisition/download.py` dispatches one of three adapters based on the registry's `download_method`:

| method | adapter | what it does |
|---|---|---|
| `hf` | `source_adapters/hf.py` | `snapshot_download` (whole repo), or bounded `stream` mode → Parquet shards for sets too large to snapshot |
| `git` | `source_adapters/git.py` | clone an official repo (shallow by default; optional `subdir`, optional Git LFS) |
| `http` | `source_adapters/http.py` | fetch plain HTTP(S) file(s), incl. LFS-media URLs |
| `manual` / `none` | — | not auto-downloadable (blocked / infrastructure) — recorded, skipped |

Adapters are **resumable** (skip files already present with a matching checksum), **honest about
access** (raise `AccessBlocked`, never retried), and **secret-free** (tokens never appear in paths,
notes, errors, or the committed manifest copy).

## Environment

```bash
export SWM_DATA_ROOT=/path/to/large/volume     # bulk working storage — NOT the repo
export HF_HOME=/path/to/hf_home                # HF cache location
export HF_TOKEN=...                            # required for gated/large HF pulls
pip install -r machine_learning/requirements/base.txt -r machine_learning/requirements/data.txt
```

## Commands

```bash
# Inspect the plan first (size estimate + disk + go/defer/skip decision — downloads nothing):
python -m machine_learning.cli datasets inspect casino

# Acquire one dataset:
python -m machine_learning.cli datasets acquire casino

# Large sets are DEFERRED by the storage guard unless you opt in:
python -m machine_learning.cli datasets acquire omnibehavior --allow-large

# Bound a streaming pull (rows) for a huge HF dataset:
python -m machine_learning.cli datasets acquire kuairand --limit 1000

# Acquire everything acquirable, resumably (blocked/deferred are recorded, not fatal):
python -m machine_learning.cli datasets prepare-all
python -m machine_learning.cli datasets prepare-all --only casino,persuasionforgood
```

`acquire` returns a status of `acquired` / `partial` / `blocked` / `deferred_storage` / `skipped` /
`failed` — the first five are all "expected outcomes" and never raise.

## Storage guard (`SWM_DISK_STOP_FRACTION`)

Before any large pull, `plan()` estimates the download size, reads free disk on the working volume,
and decides `go` / `defer` / `skip`. A pull is **deferred** when the volume is already at or above
`SWM_DISK_STOP_FRACTION` (default `0.85`), or when the estimated size would push it over that
threshold — unless you pass `--allow-large`. This prevents filling the volume mid-download.

```bash
export SWM_DISK_STOP_FRACTION=0.90   # allow the volume to fill to 90% before deferring
```

## Gated / blocked handling

- **HF gated repo** → accept the license/terms on the dataset page, set `HF_TOKEN`, re-run. The
  manifest records `access.gated=true` and the exact requirement.
- **HF 401/403 or private/not-found without a token** → recorded as blocked-with-token.
- **git auth/permission or repo-not-found** → recorded as blocked (never retried).
- **`ACCESS_BLOCKED` role / `manual` method** → short-circuited to `blocked` before any network call,
  with the registry `blockers` text as the human action.

## Source manifests + checksums

Each dataset gets `$SWM_DATA_ROOT/state/<id>/` state plus a source manifest at the path from
`acquisition/verify.py`. It lists every fetched file with its `sha256`, size, and role
(`data`/`license`/`readme`), the total bytes, a `license_snapshot` (what a shipped LICENSE file
actually says), and `resume_state`. A small **redacted, secret-free** copy is committed under
`reports/acquisition/<id>.json` (file names + truncated hashes, no absolute paths, no resume state).
These checksums are what `provenance show` joins against to prove a record's raw lineage.
