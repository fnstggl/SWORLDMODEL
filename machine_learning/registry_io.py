"""Load + validate the dataset registry (registry/*.yaml).

The registry is the machine-readable spine of the whole system: it says, for every
dataset, where it comes from, its license, how to acquire it, which tasks it supports,
which leakage-isolation unit to use, and — crucially — its *role* (train / eval-only /
blocked / infrastructure). Nothing downstream is allowed to treat a dataset as training
data unless the registry + the human approval file both permit it.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import REGISTRY_DIR
from .tasks import TASK_TYPE_SET

DATASETS_YAML = REGISTRY_DIR / "datasets.yaml"
LICENSES_YAML = REGISTRY_DIR / "licenses.yaml"
TAXONOMY_YAML = REGISTRY_DIR / "task_taxonomy.yaml"
FIELD_MAPPINGS_YAML = REGISTRY_DIR / "field_mappings.yaml"

# Allowed values (mirrored in docs + tests)
DATASET_ROLES = {
    "TRAIN_CANDIDATE", "VALIDATION_CANDIDATE", "CROSS_DATASET_EVAL_ONLY",
    "LICENSE_RESTRICTED_EVAL_ONLY", "INFRASTRUCTURE_ONLY", "ACCESS_BLOCKED", "UNUSABLE",
}
DOWNLOAD_METHODS = {"hf", "git", "http", "manual", "none"}
CONVERSION_STATES = {
    "PENDING", "CONVERTER_READY_STORAGE_BLOCKED", "SAMPLE_NORMALIZED",
    "NORMALIZED_AND_VALIDATED", "NORMALIZED_EVAL_ONLY", "INFRASTRUCTURE_ONLY",
    "ACCESS_BLOCKED", "LICENSE_BLOCKED", "SOURCE_UNAVAILABLE", "UNUSABLE_AFTER_INSPECTION",
}
TRINARY = {"yes", "no", "unknown"}

_REQUIRED = [
    "dataset_id", "official_name", "license", "license_class", "commercial_use_allowed",
    "derivatives_allowed", "redistribution_allowed", "access_requirements",
    "download_method", "dataset_role", "supported_tasks", "split_unit", "conversion_status",
]


def _load_yaml(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return yaml.safe_load(path.read_text()) or default


@functools.lru_cache(maxsize=1)
def load_licenses() -> dict:
    return _load_yaml(LICENSES_YAML, {}).get("license_classes", {})


@functools.lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    return _load_yaml(TAXONOMY_YAML, {})


@functools.lru_cache(maxsize=1)
def load_field_mappings() -> dict:
    return _load_yaml(FIELD_MAPPINGS_YAML, {})


# Fields whose YAML values may be the bare words yes/no (which YAML 1.1 parses as
# booleans) and must be normalized to the canonical strings yes/no/unknown.
_TRINARY_FIELDS = ("commercial_use_allowed", "derivatives_allowed", "redistribution_allowed")


def _norm_trinary(v) -> str:
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return str(v).lower()


def load_datasets() -> dict[str, dict]:
    """Return {dataset_id: entry}. Not cached so tests can edit + reload."""
    doc = _load_yaml(DATASETS_YAML, {})
    entries = doc.get("datasets", []) if isinstance(doc, dict) else doc
    out: dict[str, dict] = {}
    for e in entries or []:
        for f in _TRINARY_FIELDS:
            if f in e:
                e[f] = _norm_trinary(e[f])
        out[e["dataset_id"]] = e
    return out


def get_dataset(dataset_id: str) -> dict:
    ds = load_datasets()
    if dataset_id not in ds:
        raise KeyError(f"unknown dataset_id {dataset_id!r}; known: {sorted(ds)}")
    return ds[dataset_id]


@dataclass
class RegistryIssue:
    dataset_id: str
    severity: str  # "error" | "warning"
    message: str


def verify_registry() -> list[RegistryIssue]:
    """Structural + cross-reference validation of the registry.

    Errors (not warnings) mean the registry is internally inconsistent and must be fixed
    before the pipeline is trustworthy.
    """
    issues: list[RegistryIssue] = []
    licenses = load_licenses()
    datasets = load_datasets()

    if not datasets:
        issues.append(RegistryIssue("<registry>", "error", "no datasets found"))
        return issues

    for did, e in datasets.items():
        def err(msg: str) -> None:
            issues.append(RegistryIssue(did, "error", msg))

        def warn(msg: str) -> None:
            issues.append(RegistryIssue(did, "warning", msg))

        for key in _REQUIRED:
            if key not in e or e[key] in (None, ""):
                err(f"missing required field: {key}")

        if e.get("license_class") and e["license_class"] not in licenses:
            err(f"license_class {e['license_class']!r} not defined in licenses.yaml")
        if e.get("dataset_role") not in DATASET_ROLES:
            err(f"invalid dataset_role: {e.get('dataset_role')!r}")
        if e.get("download_method") not in DOWNLOAD_METHODS:
            err(f"invalid download_method: {e.get('download_method')!r}")
        if e.get("conversion_status") not in CONVERSION_STATES:
            err(f"invalid conversion_status: {e.get('conversion_status')!r}")
        for tri in ("commercial_use_allowed", "derivatives_allowed", "redistribution_allowed"):
            if str(e.get(tri)).lower() not in TRINARY:
                err(f"{tri} must be yes/no/unknown, got {e.get(tri)!r}")

        for t in e.get("supported_tasks", []) or []:
            if t not in TASK_TYPE_SET:
                err(f"supported_task {t!r} not in taxonomy")

        # Consistency: a TRAIN_CANDIDATE must not be license-restricted for derivatives.
        if e.get("dataset_role") == "TRAIN_CANDIDATE":
            if str(e.get("derivatives_allowed")).lower() == "no":
                err("TRAIN_CANDIDATE but derivatives_allowed=no (cannot train)")
            if str(e.get("commercial_use_allowed")).lower() == "unknown":
                warn("TRAIN_CANDIDATE with unknown commercial-use status")

        # An acquirable dataset should have an acquire spec matching its method.
        method = e.get("download_method")
        acq = e.get("acquire", {})
        if method in ("hf", "git", "http"):
            if not acq or acq.get("method") != method:
                warn(f"download_method={method} but acquire.method={acq.get('method')!r}")

        # last_verified_at present for provenance of the licensing decision.
        if not e.get("last_verified_at"):
            warn("no last_verified_at (licensing/access decision undated)")

    return issues


def summarize() -> dict:
    datasets = load_datasets()
    by_role: dict[str, int] = {}
    for e in datasets.values():
        by_role[e["dataset_role"]] = by_role.get(e["dataset_role"], 0) + 1
    return {"n_datasets": len(datasets), "by_role": by_role}
