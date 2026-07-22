"""Chronology + target-leakage validation.

The single most important safety property: an example's INPUT may contain only information
available strictly before the target behaviour. This module checks, per record:

* ``cutoff.future_hidden`` is True;
* every ``known_history`` event is before the cutoff (by index and/or time);
* the target does not appear verbatim in the input (target leakage) — e.g. the message we
  ask the model to predict must not already be in the dialogue history;
* timestamps are sane (no negative / impossible response times).

A dataset with any critical chronology failure must not enter a training manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..normalization.common.parquet_io import iter_records
from ..config import normalized_dir


@dataclass
class ChronologyIssue:
    record_id: str
    kind: str
    detail: str


@dataclass
class ChronologyReport:
    dataset_id: str
    n_checked: int = 0
    issues: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict:
        return {"dataset_id": self.dataset_id, "n_checked": self.n_checked,
                "n_issues": len(self.issues), "ok": self.ok,
                "issues": [i.__dict__ for i in self.issues[:50]]}


def _history_texts(record: dict) -> list[str]:
    ctx = record.get("context", {})
    inp = record.get("payload", {}).get("input", {})
    hist = ctx.get("known_history") or inp.get("dialogue_history") or inp.get("history") or []
    out = []
    for e in hist:
        if isinstance(e, dict) and e.get("text"):
            out.append(e["text"])
    return out


def check_record(record: dict) -> list[ChronologyIssue]:
    issues: list[ChronologyIssue] = []
    rid = record.get("record_id", "<no-id>")
    cutoff = record.get("cutoff", {})
    if not cutoff.get("future_hidden", False):
        issues.append(ChronologyIssue(rid, "future_not_hidden", "cutoff.future_hidden is not True"))

    # index ordering
    cidx = cutoff.get("cutoff_sequence_index")
    ctx = record.get("context", {})
    hist = ctx.get("known_history") or []
    if cidx is not None:
        for e in hist:
            if isinstance(e, dict) and isinstance(e.get("index"), int) and e["index"] >= cidx:
                issues.append(ChronologyIssue(rid, "history_after_cutoff",
                              f"history index {e['index']} >= cutoff {cidx}"))
                break

    # target verbatim leakage into input
    task = record.get("task_type")
    tgt = record.get("payload", {}).get("target", {})
    if task == "PREDICT_NEXT_MESSAGE":
        msg = (tgt.get("message_text") or "").strip()
        if msg and len(msg) > 8 and msg in _history_texts(record):
            issues.append(ChronologyIssue(rid, "target_in_history", "target message appears in dialogue history"))

    # impossible timing
    if task == "PREDICT_TIME_TO_ACTION":
        t = tgt.get("time_to_action_seconds")
        if isinstance(t, (int, float)) and t < 0:
            issues.append(ChronologyIssue(rid, "negative_time", f"time_to_action_seconds={t}"))
    return issues


def check_dataset(dataset_id: str, *, limit: int | None = None) -> ChronologyReport:
    rep = ChronologyReport(dataset_id=dataset_id)
    for i, r in enumerate(iter_records(normalized_dir(dataset_id))):
        if limit and i >= limit:
            break
        rep.n_checked += 1
        rep.issues.extend(check_record(r))
    return rep
