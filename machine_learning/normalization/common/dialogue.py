"""Helpers shared by conversational converters: building history events + cutoff logic.

The golden rule these helpers enforce: for a decision at turn *k*, the model may see only
turns ``0..k-1`` — everything at or after *k* is the label or the future and is hidden.
"""
from __future__ import annotations

from typing import Any

from ...tasks import MISSING_TIMESTAMP


def history_event(index: int, actor_id: str, kind: str, *, text: str | None = None,
                  t: Any = None, action_type: str | None = None,
                  action_content: dict | None = None, meta: dict | None = None) -> dict:
    """Build one canonical history/observation event (see registry/field_mappings.yaml)."""
    return {
        "index": index,
        "t": t if t is not None else MISSING_TIMESTAMP,
        "actor_id": actor_id,
        "kind": kind,
        "text": text,
        "action_type": action_type,
        "action_content": action_content or {},
        "meta": meta or {},
    }


def history_before(events: list[dict], k: int) -> list[dict]:
    """Return the prefix of ``events`` strictly before index ``k`` (leakage-safe)."""
    return [e for e in events if e.get("index", 0) < k]


def observation_at(events: list[dict], k: int) -> dict:
    """The observation the actor is reacting to = the last event strictly before k."""
    prefix = history_before(events, k)
    if not prefix:
        return {"text": None, "kind": "state", "meta": {}}
    last = prefix[-1]
    return {"text": last.get("text"), "kind": last.get("kind", "message"),
            "meta": {"from_actor": last.get("actor_id")}}
