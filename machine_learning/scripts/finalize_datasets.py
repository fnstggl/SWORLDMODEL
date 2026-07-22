"""Normalize (fully) every dataset with raw data present, then split + validate + audit,
then build the global audit, baselines, and readiness reports. Idempotent + resumable.
"""
from __future__ import annotations

import shutil
import sys

from machine_learning.config import normalized_dir, raw_dir
from machine_learning.normalization.pipeline import normalize
from machine_learning.splitting.policies import split_dataset
from machine_learning.splitting.leakage_checks import check_dataset as leak
from machine_learning.validation import validate_dataset
from machine_learning.report_builder import build_dataset_audit, build_global_audit
from machine_learning.evaluation.reports import run_baselines
from machine_learning.readiness import build_readiness

TS = "2026-07-22T00:00:00Z"
# datasets whose normalized dir was produced with a --limit (rebuild fresh, unlimited)
RELIMIT = ["abcd", "dealornodeal", "open_bandit"]
# streaming/sample datasets to normalize from their acquired sample
STREAM = ["criteo_uplift", "psych101", "socsci210", "simbench", "behaviorbench", "some"]


def has_raw(did: str) -> bool:
    d = raw_dir(did)
    if not d.exists():
        return False
    for p in d.rglob("*"):
        if p.is_file() and ".cache" not in p.parts and p.suffix in (".parquet", ".json", ".csv", ".txt", ".jsonl", ".gz"):
            return True
    return False


def main():
    for did in RELIMIT:
        if normalized_dir(did).exists():
            shutil.rmtree(normalized_dir(did))

    from machine_learning.registry_io import load_datasets
    done = []
    for did in load_datasets():
        if not has_raw(did):
            continue
        try:
            rep = normalize(did, timestamp=TS)
            if rep.n_valid == 0:
                print(f"[{did}] normalized 0 (skip split)"); continue
            split_dataset(did)
            res = validate_dataset(did, limit=40000)
            build_dataset_audit(did)
            done.append(did)
            print(f"[{did}] valid={rep.n_valid} tasks={rep.task_counts} "
                  f"critical_ok={res['critical_ok']} leakage_ok={leak(did).ok}")
        except Exception as e:  # noqa: BLE001
            print(f"[{did}] ERROR: {type(e).__name__}: {str(e)[:120]}")

    build_global_audit()
    run_baselines()
    out = build_readiness()
    print("READINESS:", {k: out[k] for k in ("ready", "n_validated", "n_eval_only", "n_blocked", "total_examples")})
    print("finalized datasets:", done)


if __name__ == "__main__":
    sys.exit(main())
