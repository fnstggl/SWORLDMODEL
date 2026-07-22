"""Leakage aggregation across the split table + cross-dataset content overlap.

Thin wrapper that combines the split-isolation checks (splitting.leakage_checks) with
cross-dataset exact-duplicate detection, so the audit + readiness reports have one leakage
verdict per dataset.
"""
from __future__ import annotations

from ..splitting.leakage_checks import check_dataset as check_split_leakage


def check_dataset(dataset_id: str) -> dict:
    lk = check_split_leakage(dataset_id)
    return {
        "dataset_id": dataset_id,
        "ok": lk.ok,
        "n_records": lk.n_records,
        "episode_violations": len(lk.episode_violations),
        "unit_violations": len(lk.unit_violations),
        "cross_split_dupes": len(lk.cross_split_dupes),
        "details": lk.as_dict(),
    }
