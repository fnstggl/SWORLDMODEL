"""Run every validation check for a dataset and produce one gated verdict.

Critical checks (any failure BLOCKS the dataset from a training manifest):
  schema validity, chronology/target-leakage, split-isolation leakage, provenance
  completeness, licensing (training permitted).
Advisory checks (warn, don't block): near-duplicate rate, class imbalance.
"""
from __future__ import annotations

from ..config import normalized_dir, REPORTS_DIR
from ..io_utils import write_json
from ..normalization.common.parquet_io import iter_records
from ..registry_io import get_dataset
from . import chronology, deduplication, distributions, leakage, licensing, provenance
from .schema_validation import validate_record


def validate_dataset(dataset_id: str, *, limit: int | None = None, deep: bool = True) -> dict:
    entry = get_dataset(dataset_id)
    # schema
    n_ok = n_bad = 0
    schema_failures = []
    for i, r in enumerate(iter_records(normalized_dir(dataset_id))):
        if limit and i >= limit:
            break
        res = validate_record(r)
        if res.ok:
            n_ok += 1
        else:
            n_bad += 1
            if len(schema_failures) < 20:
                schema_failures.append({"record_id": r.get("record_id"), "errors": res.errors[:5]})
    schema_ok = n_bad == 0 and n_ok > 0

    chrono = chronology.check_dataset(dataset_id, limit=limit)
    prov = provenance.check_dataset(dataset_id, limit=limit)
    lk = leakage.check_dataset(dataset_id)
    dedup = deduplication.check_dataset(dataset_id, limit=limit) if deep else None
    dist = distributions.check_dataset(dataset_id, limit=limit)

    # licensing: is the class training-permitting? (registry-level, dataset independent of records)
    lic_rows = [r for r in licensing.license_matrix() if r["dataset"] == dataset_id]
    lic = lic_rows[0] if lic_rows else {}
    role = entry.get("dataset_role")
    training_role = role in ("TRAIN_CANDIDATE", "VALIDATION_CANDIDATE")
    lic_ok = (not training_role) or lic.get("training_allowed_by_license") == "yes"

    critical = {
        "schema": schema_ok,
        "chronology": chrono.hard_ok,  # hard leakage bugs block; rare coincidental repeats warn
        "leakage": lk["ok"] or lk["n_records"] == 0,  # no split table yet is not a failure
        "provenance": prov.ok,
        "licensing": lic_ok,
    }
    critical_ok = all(critical.values())

    warnings = []
    if chrono.soft_issues:
        warnings.append(f"{len(chrono.soft_issues)} coincidental verbatim message repeats "
                        f"(target text recurs in history; below the systematic-bug threshold)")
    if dedup and dedup.n_near_dup_candidates / max(dedup.n_records, 1) > 0.2:
        warnings.append(f"high near-duplicate rate {dedup.as_dict()['near_dup_rate']:.0%}")
    warnings.extend(dist.warnings)

    result = {
        "dataset_id": dataset_id,
        "n_records": n_ok + n_bad,
        "critical_ok": critical_ok,
        "critical": critical,
        "warnings": warnings,
        "schema": {"n_ok": n_ok, "n_bad": n_bad, "failures": schema_failures},
        "chronology": chrono.as_dict(),
        "provenance": prov.as_dict(),
        "leakage": lk,
        "deduplication": dedup.as_dict() if dedup else None,
        "distributions": dist.as_dict(),
        "licensing": lic,
    }
    (REPORTS_DIR / "leakage").mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / "leakage" / f"{dataset_id}.json",
               {"dataset_id": dataset_id, "leakage": lk, "chronology": chrono.as_dict()})
    write_json(REPORTS_DIR / "normalization" / f"{dataset_id}.validation.json", result)
    return result
