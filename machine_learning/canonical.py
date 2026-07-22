"""Constructors + hashing for canonical behaviour-event records.

Every converter builds records through :func:`make_record`, which guarantees:

* a deterministic, reproducible ``record_id`` (independent of wall-clock time),
* a ``content_hash`` over the *semantic* content (used for exact-dedup + integrity),
* all 12 top-level sections present with safe defaults,
* provenance wired up (converter, version, raw locator, transformation steps).

Determinism rule: ``record_id`` and ``content_hash`` MUST NOT depend on timestamps or
any run-specific value. Re-running the converter on the same raw input reproduces the
same ids. Timestamps live only in ``source.normalization_timestamp`` and
``provenance.conversion_timestamp``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import SCHEMA_VERSION
from .tasks import TASK_TYPE_SET

_HASH_LEN = 16


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, non-ASCII preserved.

    Used for both hashing and portable JSONL export so the same content always
    serializes identically.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_default)


def _default(o: Any):
    # tolerate numpy/pandas scalars without importing them at module load
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:  # pragma: no cover
            pass
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, length: int = _HASH_LEN) -> str:
    """Deterministic short hash of the given parts."""
    return sha256_hex(canonical_json(list(parts)))[:length]


# Fields that are excluded from the semantic content hash because they are run-specific
# or assigned by later stages (splitting), not part of the record's meaning.
_NON_SEMANTIC_TOP = {"provenance", "split_metadata"}
_NON_SEMANTIC_SOURCE = {"normalization_timestamp"}


def content_hash(record: dict) -> str:
    """Hash the semantic content of a record (excludes timestamps + split assignment)."""
    semantic = {k: v for k, v in record.items() if k not in _NON_SEMANTIC_TOP}
    src = dict(semantic.get("source", {}))
    for k in _NON_SEMANTIC_SOURCE:
        src.pop(k, None)
    semantic["source"] = src
    return sha256_hex(canonical_json(semantic))


def _default_data_quality() -> dict:
    return {
        "missing_fields": [],
        "inferred_fields": [],
        "weak_label_fields": [],
        "warnings": [],
        "confidence": "",
        "chronology_verified": False,
        "target_verified": False,
        "possible_leakage": False,
        "license_verified": False,
    }


def make_record(
    *,
    dataset_id: str,
    task_type: str,
    payload: dict,
    converter: str,
    converter_version: str,
    license_class: str,
    # episode
    episode_id: str,
    sequence_index: int | None = None,
    participant_ids: list[str] | None = None,
    group_id: str | None = None,
    session_id: str | None = None,
    experiment_id: str | None = None,
    topic_id: str | None = None,
    item_ids: list[str] | None = None,
    environment_id: str | None = None,
    start_time: Any = None,
    end_time: Any = None,
    # cutoff
    cutoff_time: Any = None,
    cutoff_sequence_index: int | None = None,
    # decision unit
    actor_id: str | None = None,
    actor_role: str | None = None,
    population_id: str | None = None,
    decision_maker_id: str | None = None,
    recipient_ids: list[str] | None = None,
    persistent_identity_available: bool = False,
    # context
    context: dict | None = None,
    # causal
    causal_metadata: dict | None = None,
    # source pointers
    raw_record_ids: list[str] | None = None,
    raw_file_paths: list[str] | None = None,
    raw_content_hashes: list[str] | None = None,
    dataset_version: str | None = None,
    citation: str = "",
    source_language: str = "",
    # provenance
    raw_locator: dict | None = None,
    transformation_steps: list[str] | None = None,
    code_commit: str | None = None,
    normalization_timestamp: str | None = None,
    # data quality overrides
    data_quality: dict | None = None,
) -> dict:
    """Build a fully-formed canonical behaviour-event record.

    ``payload`` must be a ``{"input": {...}, "target": {...}}`` dict already shaped for
    ``task_type``. This function does NOT validate against the JSON schema (that is the
    validation stage's job) but it does guarantee structural completeness + determinism.
    """
    if task_type not in TASK_TYPE_SET:
        raise ValueError(f"unknown task_type: {task_type!r}")
    if not isinstance(payload, dict) or "input" not in payload or "target" not in payload:
        raise ValueError("payload must be a dict with 'input' and 'target'")

    dq = _default_data_quality()
    if data_quality:
        dq.update(data_quality)

    context = dict(context or {})

    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": "",  # filled below (deterministic)
        "task_type": task_type,
        "source": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version or "",
            "raw_record_ids": list(raw_record_ids or []),
            "raw_file_paths": list(raw_file_paths or []),
            "license_class": license_class,
            "citation": citation,
            "source_language": source_language,
            "normalization_timestamp": normalization_timestamp,
            "converter_version": converter_version,
            "raw_content_hashes": list(raw_content_hashes or []),
        },
        "episode": {
            "episode_id": episode_id,
            "participant_ids": list(participant_ids or []),
            "group_id": group_id,
            "session_id": session_id,
            "experiment_id": experiment_id,
            "topic_id": topic_id,
            "item_ids": list(item_ids or []),
            "environment_id": environment_id,
            "start_time": start_time,
            "end_time": end_time,
            "sequence_index": sequence_index,
        },
        "cutoff": {
            "cutoff_time": cutoff_time,
            "cutoff_sequence_index": cutoff_sequence_index,
            "future_hidden": True,
        },
        "decision_unit": {
            "actor_id": actor_id,
            "actor_role": actor_role,
            "population_id": population_id,
            "decision_maker_id": decision_maker_id,
            "recipient_ids": list(recipient_ids or []),
            "persistent_identity_available": bool(persistent_identity_available),
        },
        "context": {
            "actor_profile": context.get("actor_profile", {}),
            "private_state_before": context.get("private_state_before", {}),
            "known_history": context.get("known_history", []),
            "current_observation": context.get("current_observation", {}),
            "world_state": context.get("world_state", {}),
            "relationships": context.get("relationships", []),
            "available_actions": context.get("available_actions", None),
            "institutional_constraints": context.get("institutional_constraints", []),
            "language": context.get("language", source_language or ""),
        },
        "payload": payload,
        "causal_metadata": dict(causal_metadata or {}),
        "data_quality": dq,
        "provenance": {
            "converter": converter,
            "converter_version": converter_version,
            "conversion_timestamp": normalization_timestamp,
            "code_commit": code_commit,
            "transformation_steps": list(transformation_steps or []),
            "raw_record_locator": raw_locator or {"files": [], "indices": [], "ids": []},
            "content_hash": None,
        },
        "split_metadata": {
            "split": None,
            "split_policy": None,
            "isolation_keys": {},
            "dedup_hash": None,
            "near_dup_bucket": None,
        },
    }

    # Deterministic record id: dataset + task + stable pointer to the raw source unit.
    rid_basis = [
        dataset_id, task_type, episode_id, sequence_index,
        record["provenance"]["raw_record_locator"], payload.get("target"),
    ]
    record["record_id"] = f"{dataset_id}:{task_type}:{stable_id(*rid_basis)}"

    # Content hash for dedup + integrity.
    ch = content_hash(record)
    record["provenance"]["content_hash"] = ch
    record["split_metadata"]["dedup_hash"] = ch
    return record
