"""Stage-level checkpointing + bounded retries — never restart the world for one failed stage.

Every named stage stores its output in the run's checkpoint store the moment it completes.
A failed IDEMPOTENT stage retries exactly once; a non-idempotent stage never auto-retries.
There are no background monitors, no recursive wrappers, no process relaunch loops — a stage
that fails twice raises to the orchestrator, which finalizes from the checkpoints that exist."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageRecord:
    name: str
    status: str = "pending"                    # completed | failed | retried_then_completed
    attempts: int = 0
    latency_s: float = 0.0
    error: str = ""


class CheckpointStore:
    """In-memory stage checkpoint store (the run IS the process; cross-run persistence is the
    immutable compile cache's job, never this store's — mutable world state is never persisted)."""

    def __init__(self):
        self._data: dict = {}
        self.records: list[StageRecord] = []

    def has(self, name: str) -> bool:
        return name in self._data

    def get(self, name: str, default=None):
        return self._data.get(name, default)

    def put(self, name: str, value):
        self._data[name] = value

    def run_stage(self, name: str, fn, *, idempotent: bool = True, reuse: bool = True):
        """Execute one named stage with checkpointing. `reuse=True` returns the existing
        checkpoint instead of re-running (the resume path). Idempotent stages get ONE retry."""
        if reuse and self.has(name):
            return self.get(name)
        rec = StageRecord(name=name)
        self.records.append(rec)
        t = time.time()
        try:
            rec.attempts = 1
            out = fn()
        except Exception as first:  # noqa: BLE001
            rec.error = f"{type(first).__name__}: {first}"[:220]
            if not idempotent:
                rec.status = "failed"
                rec.latency_s = round(time.time() - t, 3)
                raise
            try:
                rec.attempts = 2
                out = fn()
                rec.status = "retried_then_completed"
            except Exception as second:  # noqa: BLE001
                rec.status = "failed"
                rec.error += f" | retry: {type(second).__name__}: {second}"[:200]
                rec.latency_s = round(time.time() - t, 3)
                raise
        else:
            rec.status = "completed"
        rec.latency_s = round(time.time() - t, 3)
        self.put(name, out)
        return out

    def manifest(self) -> dict:
        return {"stages": [{"name": r.name, "status": r.status, "attempts": r.attempts,
                            "latency_s": r.latency_s, "error": r.error}
                           for r in self.records],
                "checkpointed": sorted(self._data.keys())}
