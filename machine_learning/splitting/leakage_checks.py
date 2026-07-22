"""Empirical leakage checks over an assigned split table.

These are the guardrails that make a split *trustworthy* rather than merely *assigned*:

* **episode isolation** — every record of one episode is in exactly one split;
* **isolation-unit isolation** — no participant/group/topic/... value straddles splits;
* **cross-split exact-dup** — the same content hash must not appear in two splits;
* **train/eval content overlap** — no exact record content shared between train and any
  test/validation split.

A critical violation here MUST block a dataset from a training manifest.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from .policies import load_split_table

_TEST_SPLITS = {"validation", "test_in_domain", "test_unseen_people", "test_unseen_groups",
                "test_unseen_topics", "test_unseen_experiments", "test_unseen_conditions",
                "test_future_time", "test_cross_dataset"}


@dataclass
class LeakageReport:
    dataset_id: str
    ok: bool = True
    episode_violations: list = field(default_factory=list)
    unit_violations: list = field(default_factory=list)
    cross_split_dupes: list = field(default_factory=list)
    n_records: int = 0
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("episode_violations", "unit_violations", "cross_split_dupes"):
            d[k] = d[k][:50]
        return d


def check_dataset(dataset_id: str) -> LeakageReport:
    rows = load_split_table(dataset_id)
    rep = LeakageReport(dataset_id=dataset_id, n_records=len(rows))
    if not rows:
        rep.notes.append("no split table (dataset not split yet)")
        return rep

    # 1) episode -> single split
    ep_splits: dict = defaultdict(set)
    for r in rows:
        ep_splits[r["episode_id"]].add(r["split"])
    for ep, splits in ep_splits.items():
        if len(splits) > 1:
            rep.episode_violations.append({"episode_id": ep, "splits": sorted(splits)})

    # 2) isolation-unit -> single split (from the recorded isolation_keys)
    unit_splits: dict = defaultdict(set)
    for r in rows:
        keys = json.loads(r.get("isolation_keys_json") or "{}")
        for unit, val in keys.items():
            if unit == "dataset":
                continue
            unit_splits[(unit, str(val))].add(r["split"])
    for (unit, val), splits in unit_splits.items():
        if len(splits) > 1:
            rep.unit_violations.append({"unit": unit, "value": val, "splits": sorted(splits)})

    # 3) content hash appearing in >1 split (exact cross-split duplicate)
    hash_splits: dict = defaultdict(set)
    for r in rows:
        h = r.get("content_hash")
        if h:
            hash_splits[h].add(r["split"])
    for h, splits in hash_splits.items():
        if len(splits) > 1 and any(s in _TEST_SPLITS for s in splits):
            rep.cross_split_dupes.append({"content_hash": h[:16], "splits": sorted(splits)})

    rep.ok = not (rep.episode_violations or rep.unit_violations or rep.cross_split_dupes)
    return rep
