"""Leakage-safe split assignment.

Splits are assigned by a deterministic hash of an *isolation unit* (participant,
conversation, group, experiment, topic, event, ...), so the same unit always lands in the
same split and no unit can straddle the train/eval boundary. Assignment order (first match
wins) — from most to least restrictive:

  1. whole-dataset eval-only  -> test_cross_dataset   (role CROSS_DATASET_EVAL_ONLY / *_EVAL_ONLY)
  2. future-time holdout       -> test_future_time      (latest time fraction, when cutoff_time exists)
  3. unseen-secondary holdout  -> test_unseen_<people|groups|topics|experiments|conditions>
  4. primary split             -> train / validation / test_in_domain

Everything is derived from the record's own fields + the registry, so re-running is
reproducible and auditable. The result is a split table (record_id -> split + the exact
isolation-key values that decided it), NOT a rewrite of the immutable normalized shards.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..config import SPLITS_DIR, normalized_dir
from ..io_utils import read_json, write_json
from ..normalization.common.parquet_io import iter_records, write_shard
from ..registry_io import get_dataset

# unit name -> how to read its key value(s) from a record
_UNIT_READERS = {
    "participant": lambda r: (r["episode"].get("participant_ids") or [r["decision_unit"].get("actor_id")]),
    "platform_user": lambda r: [r["decision_unit"].get("actor_id")] or r["episode"].get("participant_ids"),
    "conversation": lambda r: [r["episode"]["episode_id"]],
    "session": lambda r: [r["episode"].get("session_id") or r["episode"]["episode_id"]],
    "group": lambda r: [r["episode"].get("group_id") or r["episode"]["episode_id"]],
    "experiment": lambda r: [r["episode"].get("experiment_id") or r["episode"]["episode_id"]],
    "study": lambda r: [r["episode"].get("experiment_id") or r["episode"]["episode_id"]],
    "topic": lambda r: [r["episode"].get("topic_id") or r["episode"]["episode_id"]],
    "event": lambda r: [r["episode"].get("group_id") or r["episode"]["episode_id"]],
    "item": lambda r: (r["episode"].get("item_ids") or [r["episode"]["episode_id"]]),
    "product": lambda r: (r["episode"].get("item_ids") or [r["episode"]["episode_id"]]),
    "organization": lambda r: [r["episode"].get("group_id") or r["episode"]["episode_id"]],
    "time_period": lambda r: [str(r["cutoff"].get("cutoff_time"))],
}

# which "unseen" secondary split a unit maps to
_UNSEEN_SPLIT = {
    "participant": "test_unseen_people",
    "platform_user": "test_unseen_people",
    "group": "test_unseen_groups",
    "topic": "test_unseen_topics",
    "experiment": "test_unseen_experiments",
    "study": "test_unseen_experiments",
    "condition": "test_unseen_conditions",
}


def _bucket(key: str, salt: str = "", n: int = 100) -> int:
    h = hashlib.sha1(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


@dataclass
class SplitPolicy:
    dataset_id: str
    primary_unit: str
    ratios: dict = field(default_factory=lambda: {"train": 0.8, "validation": 0.1, "test_in_domain": 0.1})
    unseen_units: list = field(default_factory=list)       # e.g. ["participant","topic"]
    unseen_fraction: float = 0.1
    future_time_fraction: float = 0.0                       # >0 enables time holdout
    eval_only: bool = False

    @classmethod
    def from_registry(cls, dataset_id: str, override: dict | None = None) -> "SplitPolicy":
        entry = get_dataset(dataset_id)
        role = entry.get("dataset_role")
        eval_only = role in ("CROSS_DATASET_EVAL_ONLY", "LICENSE_RESTRICTED_EVAL_ONLY")
        pol = cls(dataset_id=dataset_id, primary_unit=entry.get("split_unit", "conversation"),
                  eval_only=eval_only)
        if override:
            for k, v in override.items():
                if hasattr(pol, k):
                    setattr(pol, k, v)
        return pol


def _unit_values(record: dict, unit: str) -> list[str]:
    reader = _UNIT_READERS.get(unit)
    if reader is None:
        return [record["episode"]["episode_id"]]
    vals = reader(record) or []
    return [str(v) for v in vals if v is not None] or [record["episode"]["episode_id"]]


def _condition_value(record: dict) -> str | None:
    ws = record.get("context", {}).get("world_state", {}) or {}
    for k in ("condition", "condition_num", "treatment_arm"):
        if k in ws:
            return f"{record['episode'].get('experiment_id')}:{ws[k]}"
    cm = record.get("causal_metadata", {}) or {}
    if cm.get("treatment_arm") is not None:
        return f"{record['episode'].get('experiment_id')}:{cm['treatment_arm']}"
    return None


def assign_split(record: dict, policy: SplitPolicy, *, time_threshold=None) -> tuple[str, dict]:
    """Return (split, isolation_keys) for one record under ``policy``."""
    keys: dict = {}

    if policy.eval_only:
        keys["dataset"] = policy.dataset_id
        return "test_cross_dataset", keys

    # future-time holdout
    if policy.future_time_fraction > 0 and time_threshold is not None:
        ct = record["cutoff"].get("cutoff_time")
        if ct is not None and str(ct) >= str(time_threshold):
            keys["cutoff_time"] = ct
            return "test_future_time", keys

    # unseen-secondary holdouts
    for unit in policy.unseen_units:
        if unit == "condition":
            cval = _condition_value(record)
            vals = [cval] if cval else []
        else:
            vals = _unit_values(record, unit)
        for v in vals:
            if _bucket(v, salt=f"unseen:{unit}") < int(policy.unseen_fraction * 100):
                keys[unit] = v
                return _UNSEEN_SPLIT.get(unit, "test_unseen_people"), keys

    # primary split by hashing the primary isolation unit
    pkey = _unit_values(record, policy.primary_unit)[0]
    keys[policy.primary_unit] = pkey
    b = _bucket(pkey, salt="primary")
    train_cut = int(policy.ratios["train"] * 100)
    val_cut = train_cut + int(policy.ratios["validation"] * 100)
    if b < train_cut:
        return "train", keys
    if b < val_cut:
        return "validation", keys
    return "test_in_domain", keys


@dataclass
class SplitReport:
    dataset_id: str
    counts: dict = field(default_factory=dict)
    n_records: int = 0
    primary_unit: str = ""
    eval_only: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def split_dataset(dataset_id: str, *, policy: SplitPolicy | None = None,
                  override: dict | None = None) -> SplitReport:
    """Assign splits for all normalized records of a dataset; write the split table."""
    policy = policy or SplitPolicy.from_registry(dataset_id, override)
    ndir = normalized_dir(dataset_id)

    # future-time threshold (if enabled): the (1-frac) quantile of cutoff_time
    time_threshold = None
    if policy.future_time_fraction > 0:
        times = sorted(str(r["cutoff"].get("cutoff_time")) for r in iter_records(ndir)
                       if r["cutoff"].get("cutoff_time") is not None)
        if times:
            idx = int((1 - policy.future_time_fraction) * len(times))
            time_threshold = times[min(idx, len(times) - 1)]

    counts: dict = {}
    rows: list[dict] = []
    for r in iter_records(ndir):
        split, keys = assign_split(r, policy, time_threshold=time_threshold)
        counts[split] = counts.get(split, 0) + 1
        rows.append({
            "record_id": r["record_id"], "dataset_id": dataset_id, "task_type": r["task_type"],
            "episode_id": r["episode"]["episode_id"], "actor_id": r["decision_unit"].get("actor_id"),
            "split": split, "content_hash": r["provenance"]["content_hash"],
            "dedup_hash": r["split_metadata"]["dedup_hash"],
            "isolation_keys_json": _dumps(keys),
        })

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    _write_split_table(dataset_id, rows)
    rep = SplitReport(dataset_id=dataset_id, counts=counts, n_records=len(rows),
                      primary_unit=policy.primary_unit, eval_only=policy.eval_only)
    write_json(SPLITS_DIR / f"{dataset_id}.manifest.json",
               {**rep.as_dict(), "policy": {"primary_unit": policy.primary_unit,
                "ratios": policy.ratios, "unseen_units": policy.unseen_units,
                "future_time_fraction": policy.future_time_fraction, "eval_only": policy.eval_only}})
    return rep


def _dumps(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _write_split_table(dataset_id: str, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = ["record_id", "dataset_id", "task_type", "episode_id", "actor_id", "split",
            "content_hash", "dedup_hash", "isolation_keys_json"]
    table = pa.table({c: [r.get(c) for r in rows] for c in cols}) if rows else pa.table(
        {c: pa.array([], type=pa.string()) for c in cols})
    out = SPLITS_DIR / f"{dataset_id}.split.parquet"
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out)


def load_split_table(dataset_id: str) -> list[dict]:
    import pyarrow.parquet as pq
    p = SPLITS_DIR / f"{dataset_id}.split.parquet"
    if not p.exists():
        return []
    return pq.read_table(p).to_pylist()
