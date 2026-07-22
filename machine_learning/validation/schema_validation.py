"""Load + apply the canonical + task-payload JSON schemas.

Two-pass validation:
  1. the outer canonical envelope (``canonical_behavior_event.schema.json``),
  2. ``payload`` against the task-specific schema for ``record["task_type"]``.

Schemas are loaded once and cached. ``jsonschema`` is the only third-party dependency
and it is in ``requirements/base.txt``.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field

from ..config import SCHEMAS_DIR
from ..tasks import TARGET_PRIMARY_KEY

_CANONICAL_PATH = SCHEMAS_DIR / "canonical_behavior_event.schema.json"
_PAYLOAD_DIR = SCHEMAS_DIR / "task_payloads"


@functools.lru_cache(maxsize=1)
def canonical_schema() -> dict:
    return json.loads(_CANONICAL_PATH.read_text())


@functools.lru_cache(maxsize=None)
def payload_schema(task_type: str) -> dict | None:
    p = _PAYLOAD_DIR / f"{task_type}.schema.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@functools.lru_cache(maxsize=1)
def _validators():
    import jsonschema  # local import so importing this module never hard-requires it
    canonical = jsonschema.Draft7Validator(canonical_schema())
    return jsonschema, canonical


@dataclass
class SchemaResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # convenience
        return self.ok


def validate_record(record: dict) -> SchemaResult:
    """Validate one canonical record (envelope + task-specific payload)."""
    jsonschema, canonical = _validators()
    errors: list[str] = []

    for err in canonical.iter_errors(record):
        loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"envelope[{loc}]: {err.message}")

    task_type = record.get("task_type")
    payload = record.get("payload")
    if task_type and isinstance(payload, dict):
        ps = payload_schema(task_type)
        if ps is None:
            errors.append(f"payload: no schema for task_type {task_type!r}")
        else:
            pv = jsonschema.Draft7Validator(ps)
            for err in pv.iter_errors(payload):
                loc = "/".join(str(p) for p in err.absolute_path) or "<payload>"
                errors.append(f"payload[{loc}]: {err.message}")

        # fast, high-signal target-shape check (empty/missing/truncated target)
        primary = TARGET_PRIMARY_KEY.get(task_type)
        tgt = payload.get("target") if isinstance(payload, dict) else None
        if primary and isinstance(tgt, dict):
            if primary not in tgt:
                errors.append(f"target: missing required key {primary!r} for {task_type}")
            else:
                v = tgt[primary]
                if v is None or (isinstance(v, (str, list, dict)) and len(v) == 0):
                    # allow legitimately-empty continuation only when horizon==0
                    if not (task_type == "PREDICT_TRAJECTORY_CONTINUATION"
                            and payload.get("input", {}).get("horizon") == 0):
                        errors.append(f"target[{primary}]: empty/malformed target value")

    return SchemaResult(ok=not errors, errors=errors)


def validate_many(records) -> tuple[int, int, list[tuple[str, list[str]]]]:
    """Validate an iterable of records. Returns (n_ok, n_bad, failures[:100])."""
    n_ok = n_bad = 0
    failures: list[tuple[str, list[str]]] = []
    for rec in records:
        res = validate_record(rec)
        if res.ok:
            n_ok += 1
        else:
            n_bad += 1
            if len(failures) < 100:
                failures.append((rec.get("record_id", "<no-id>"), res.errors))
    return n_ok, n_bad, failures
