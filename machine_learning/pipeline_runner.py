"""`prepare-all` orchestrator: acquire -> normalize -> split -> validate, per dataset.

Unattended-safe: one dataset failing never aborts the run — the failure is recorded and
the next dataset proceeds. Continuously updates the live-status checkpoint files
(live_status.md, dataset_status.csv, error_log.jsonl, storage_status.json) so progress is
inspectable mid-run. Fully resumable (each stage is itself resumable).
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from .config import READINESS_DIR, disk_status
from .io_utils import human_bytes, write_json
from .registry_io import get_dataset, load_datasets, training_eligibility

_TS = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
_EST_TOKENS_PER_EX = 320


def _append_error(dataset_id: str, stage: str, err: Exception) -> None:
    READINESS_DIR.mkdir(parents=True, exist_ok=True)
    with open(READINESS_DIR / "error_log.jsonl", "a", encoding="utf-8") as f:
        import json
        f.write(json.dumps({"ts": _TS(), "dataset": dataset_id, "stage": stage,
                            "error_type": type(err).__name__, "message": str(err)[:200]}) + "\n")


def _write_status(rows: list[dict], phase: dict) -> None:
    READINESS_DIR.mkdir(parents=True, exist_ok=True)
    # dataset_status.csv
    cols = ["dataset", "access", "license", "acquired", "normalized", "validated",
            "training_eligible", "evaluation_eligible", "examples", "tokens", "blockers"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    (READINESS_DIR / "dataset_status.csv").write_text(buf.getvalue())
    # live_status.md
    lines = [f"# Live status — updated {_TS()}", "",
             f"Phase: **{phase.get('phase','')}** ({phase.get('done',0)}/{phase.get('total',0)})", "",
             "| dataset | state | examples | blockers |", "|---|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r.get('state','')} | {r.get('examples','')} | {str(r.get('blockers',''))[:60]} |")
    (READINESS_DIR / "live_status.md").write_text("\n".join(lines) + "\n")
    # storage_status.json
    write_json(READINESS_DIR / "storage_status.json",
               {"ts": _TS(), "disk": disk_status().as_dict(), "phase": phase})


def prepare_dataset(dataset_id: str, *, allow_large: bool, limit: int | None) -> dict:
    entry = get_dataset(dataset_id)
    row = {"dataset": dataset_id, "access": entry.get("access_requirements", "")[:20],
           "license": entry.get("license_class"), "acquired": "no", "normalized": "no",
           "validated": "no", "examples": 0, "tokens": 0,
           "blockers": entry.get("blockers", "") or "", "state": "pending"}
    elig, ereason = training_eligibility(dataset_id, require_approval=False)
    row["training_eligible"] = "yes(needs approval)" if elig else f"no({ereason[:20]})"
    row["evaluation_eligible"] = "yes" if entry.get("converter") else "no"

    from .acquisition.download import acquire
    from .normalization.pipeline import normalize
    from .splitting.policies import split_dataset
    from .validation import validate_dataset

    # acquire
    try:
        row["state"] = "acquiring"
        man = acquire(dataset_id, allow_large=allow_large, timestamp=_TS())
        row["acquired"] = man["status"]
        if man["status"] not in ("acquired", "partial"):
            row["state"] = man["status"]
            row["blockers"] = (man.get("notes") or [row["blockers"]])[-1][:80]
            return row
    except Exception as e:  # noqa: BLE001
        _append_error(dataset_id, "acquire", e)
        row["state"] = "failed:acquire"
        row["blockers"] = f"acquire error: {str(e)[:60]}"
        return row

    # normalize
    try:
        row["state"] = "normalizing"
        nrep = normalize(dataset_id, timestamp=_TS(), limit=limit)
        row["normalized"] = nrep.status
        row["examples"] = nrep.n_valid
        row["tokens"] = nrep.n_valid * _EST_TOKENS_PER_EX
        if nrep.n_valid == 0:
            row["state"] = "normalized:empty"
            return row
    except Exception as e:  # noqa: BLE001
        _append_error(dataset_id, "normalize", e)
        row["state"] = "failed:normalize"
        row["blockers"] = f"normalize error: {str(e)[:60]}"
        return row

    # split + validate
    try:
        split_dataset(dataset_id)
        res = validate_dataset(dataset_id, limit=limit)
        row["validated"] = "pass" if res["critical_ok"] else "FAIL"
        row["state"] = "validated" if res["critical_ok"] else "validation_failed"
        if not res["critical_ok"]:
            row["blockers"] = f"critical validation failed: {res['critical']}"
    except Exception as e:  # noqa: BLE001
        _append_error(dataset_id, "validate", e)
        row["state"] = "failed:validate"
        row["blockers"] = f"validate error: {str(e)[:60]}"
    return row


def prepare_all(*, allow_large: bool = False, limit: int | None = None,
                only: list[str] | None = None) -> list[dict]:
    ids = only or list(load_datasets().keys())
    rows: list[dict] = []
    total = len(ids)
    for i, did in enumerate(ids):
        _write_status(rows, {"phase": f"processing {did}", "done": i, "total": total})
        rows.append(prepare_dataset(did, allow_large=allow_large, limit=limit))
        _write_status(rows, {"phase": f"done {did}", "done": i + 1, "total": total})
    _write_status(rows, {"phase": "complete", "done": total, "total": total})
    return rows
