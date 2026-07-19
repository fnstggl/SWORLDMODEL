"""Historical checkpoint registry loader + temporal eligibility gate (fail-closed)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REGISTRY = Path(__file__).with_name("historical_model_registry.json")


def _ts(iso) -> float:
    s = str(iso).replace("Z", "+00:00")
    import datetime as dt
    if "T" not in s:
        s += "T00:00:00+00:00"
    return dt.datetime.fromisoformat(s).timestamp()


def load_registry() -> dict:
    d = json.loads(REGISTRY.read_text())
    d["registry_hash"] = hashlib.sha256(
        json.dumps(d["models"], sort_keys=True).encode()).hexdigest()[:16]
    return d


def get_model(registry_model_id: str) -> dict:
    for m in load_registry()["models"]:
        if m["registry_model_id"] == registry_model_id:
            if m.get("approval_status") not in ("APPROVED_FULL_BENCHMARK",
                                                "APPROVED_DIAGNOSTIC_ONLY"):
                raise PermissionError(
                    f"model {registry_model_id!r} not approved: {m.get('approval_status')}")
            return m
    raise KeyError(f"unknown registry model {registry_model_id!r}")


def assert_temporal_ordering(model: dict, *, question_open_ts: float, cutoff_ts: float,
                             resolution_ts: float | None = None) -> dict:
    """The non-negotiable ordering: model_release < question_open <= cutoff < resolution.
    Fails closed (raises) when any bound is unproven or violated. Returns the proof record."""
    rel = _ts(model["effective_temporal_boundary"])
    if not isinstance(question_open_ts, (int, float)) or question_open_ts <= rel:
        raise ValueError(f"temporal gate: question_open {question_open_ts} must be strictly after "
                         f"model release boundary {model['effective_temporal_boundary']}")
    if not isinstance(cutoff_ts, (int, float)) or cutoff_ts < question_open_ts:
        raise ValueError("temporal gate: forecast_cutoff must be >= question_open")
    if resolution_ts is not None and cutoff_ts >= resolution_ts:
        raise ValueError("temporal gate: forecast_cutoff must precede resolution")
    return {"model_release_boundary": model["effective_temporal_boundary"],
            "question_open_ts": question_open_ts, "cutoff_ts": cutoff_ts,
            "resolution_checked": resolution_ts is not None,
            "ordering": "model_release < question_open <= cutoff" +
                        (" < resolution" if resolution_ts is not None else ""),
            "verified_at": time.time()}
