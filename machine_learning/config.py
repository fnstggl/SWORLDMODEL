"""Central configuration and path resolution for the SWORLDMODEL behaviour-ML system.

This module is the single source of truth for *where things live*. It is imported by
almost every other module, so it has **no heavy dependencies** (stdlib only) and never
imports ``swm`` — the production runtime must stay fully isolated from this package.

Path philosophy
---------------
There are two kinds of storage:

* **Repository storage** (committed): code, configs, registry, schemas, small fixtures,
  reports, manifests summaries, human-review samples. Lives under ``machine_learning/``.
* **Working storage** (NOT committed): raw downloads, extracted archives, normalized
  Parquet shards, generated example shards, caches, checkpoints, logs. Lives under
  ``$SWM_DATA_ROOT`` which defaults to ``machine_learning/data`` but should be pointed at
  a large external volume for real acquisition.

Nothing in working storage is ever required to reconstruct the pipeline — every artifact
there is re-derivable from committed code + the public source data.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Repository-relative anchors (committed)
# --------------------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent            # .../machine_learning
REPO_ROOT = PKG_ROOT.parent                           # .../SWORLDMODEL

REGISTRY_DIR = PKG_ROOT / "registry"
SCHEMAS_DIR = PKG_ROOT / "schemas"
CONFIGS_DIR = PKG_ROOT / "configs"
REPORTS_DIR = PKG_ROOT / "reports"
ARTIFACTS_DIR = PKG_ROOT / "artifacts"
DOCS_DIR = PKG_ROOT / "docs"
FIXTURES_DIR = PKG_ROOT / "tests" / "fixtures"

# Committed report sub-locations
READINESS_DIR = REPORTS_DIR / "readiness"
AUDIT_DIR = REPORTS_DIR / "audit"
HUMAN_REVIEW_DIR = REPORTS_DIR / "human_review"
COMMITTED_SAMPLES_DIR = REPORTS_DIR / "samples"

# Approval file governing which datasets a human has cleared for training.
APPROVALS_FILE = REGISTRY_DIR / "training_approvals.yaml"


# --------------------------------------------------------------------------------------
# Working storage (NOT committed) — resolved from environment
# --------------------------------------------------------------------------------------
def _resolve_data_root() -> Path:
    env = os.environ.get("SWM_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Safe in-repo default. This lands under machine_learning/data which is covered by the
    # repository .gitignore, so accidental commits of bulk data cannot happen.
    return (PKG_ROOT / "data").resolve()


DATA_ROOT = _resolve_data_root()

RAW_DIR = DATA_ROOT / "raw"                 # untouched downloaded/extracted source data
NORMALIZED_DIR = DATA_ROOT / "normalized"   # canonical behaviour-event Parquet shards
EXAMPLES_DIR = DATA_ROOT / "examples"       # task-specific example shards
SPLITS_DIR = DATA_ROOT / "splits"           # split assignment tables
MANIFESTS_DIR = DATA_ROOT / "manifests"     # generated training-view manifests (large)
SAMPLES_DIR = DATA_ROOT / "samples"         # portable JSONL samples for debugging
CACHE_DIR = DATA_ROOT / "cache"             # HF cache, tmp extraction, etc.
LOGS_DIR = DATA_ROOT / "logs"
STATE_DIR = DATA_ROOT / "state"             # resumable acquisition/normalization state

WORKING_DIRS = [
    RAW_DIR, NORMALIZED_DIR, EXAMPLES_DIR, SPLITS_DIR, MANIFESTS_DIR,
    SAMPLES_DIR, CACHE_DIR, LOGS_DIR, STATE_DIR,
]

# Storage-safety threshold. When disk usage on the working volume exceeds this fraction,
# new large acquisitions must not start (see acquisition.download.storage_guard).
DISK_STOP_FRACTION = float(os.environ.get("SWM_DISK_STOP_FRACTION", "0.85"))


def ensure_working_dirs() -> None:
    """Create the working-storage tree if missing. Idempotent."""
    for d in WORKING_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def raw_dir(dataset_id: str) -> Path:
    return RAW_DIR / dataset_id


def normalized_dir(dataset_id: str) -> Path:
    return NORMALIZED_DIR / dataset_id


def examples_dir(dataset_id: str) -> Path:
    return EXAMPLES_DIR / dataset_id


def state_dir(dataset_id: str) -> Path:
    d = STATE_DIR / dataset_id
    return d


# --------------------------------------------------------------------------------------
# Disk accounting
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiskStatus:
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int

    @property
    def used_fraction(self) -> float:
        if self.total_bytes <= 0:
            return 1.0
        return self.used_bytes / self.total_bytes

    @property
    def free_gib(self) -> float:
        return self.free_bytes / (1024 ** 3)

    @property
    def total_gib(self) -> float:
        return self.total_bytes / (1024 ** 3)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "total_gib": round(self.total_gib, 2),
            "free_gib": round(self.free_gib, 2),
            "used_fraction": round(self.used_fraction, 4),
            "stop_fraction": DISK_STOP_FRACTION,
            "over_threshold": self.used_fraction >= DISK_STOP_FRACTION,
        }


def disk_status(path: Path | None = None) -> DiskStatus:
    """Return disk usage for the volume backing ``path`` (defaults to the data root)."""
    target = path or DATA_ROOT
    # walk up to an existing ancestor so this works before dirs are created
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return DiskStatus(
        path=str(target),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
    )


# --------------------------------------------------------------------------------------
# Package versions stamped into provenance
# --------------------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"
CONVERTER_FRAMEWORK_VERSION = "1.0.0"


def summary() -> dict:
    """Human-readable snapshot of the current configuration (used by ``cli`` + reports)."""
    ds = disk_status()
    return {
        "pkg_root": str(PKG_ROOT),
        "repo_root": str(REPO_ROOT),
        "data_root": str(DATA_ROOT),
        "data_root_from_env": bool(os.environ.get("SWM_DATA_ROOT", "").strip()),
        "hf_home": os.environ.get("HF_HOME", "<unset>"),
        "schema_version": SCHEMA_VERSION,
        "disk": ds.as_dict(),
    }
