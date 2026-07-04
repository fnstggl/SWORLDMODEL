"""Append-only, timestamped SQLite event store with as-of reads (audit C.1).

The honesty backbone: every read can be restricted to "what was knowable strictly before T",
which is what makes temporal backtests physically leakage-proof rather than convention-proof.

Channel-agnostic: an email thread and a text thread are both sequences of MESSAGE events with
direction, timestamps, and content. Reply labels are DERIVED (not stored) so the labeling window
can change without re-importing.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from swm.ingestion.schema import Event, EventType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id TEXT NOT NULL,          -- who acted (sender)
    timestamp REAL NOT NULL,
    type TEXT NOT NULL,
    channel TEXT NOT NULL,           -- 'email' | 'text' | ...
    direction TEXT NOT NULL,         -- 'out' (we sent) | 'in' (they sent)
    thread_id TEXT,
    content TEXT,                    -- raw text (private store; embeddings computed on demand)
    content_hash TEXT,
    target_ids TEXT,                 -- JSON list
    features TEXT                    -- JSON dict
);
CREATE INDEX IF NOT EXISTS ix_events_actor_ts ON events(actor_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_events_thread ON events(thread_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(timestamp);
"""

# Per-channel reply windows (seconds) for label derivation.
REPLY_WINDOWS = {"email": 7 * 86400.0, "text": 1 * 86400.0}


@dataclass(frozen=True)
class Send:
    """A labeled outbound message: the unit of training/backtesting."""
    event_id: int
    recipient_id: str
    timestamp: float
    channel: str
    thread_id: str | None
    content: str
    replied: bool
    reply_latency: float | None  # seconds, if replied


class EventStore:
    def __init__(self, path: str | Path = ":memory:"):
        # check_same_thread=False: the API serves sync endpoints from a threadpool.
        # Single-connection writes are serialized with a lock; v1 is single-tenant.
        import threading

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def append(
        self,
        *,
        actor_id: str,
        timestamp: float,
        type: EventType | str,
        channel: str,
        direction: str,
        thread_id: str | None = None,
        content: str = "",
        target_ids: tuple[str, ...] = (),
        features: dict | None = None,
    ) -> int:
        if direction not in ("out", "in"):
            raise ValueError("direction must be 'out' or 'in'")
        with self._lock:
            return self._append_locked(actor_id, timestamp, type, channel, direction,
                                       thread_id, content, target_ids, features)

    def _append_locked(self, actor_id, timestamp, type, channel, direction, thread_id,
                       content, target_ids, features) -> int:
        cur = self._conn.execute(
            "INSERT INTO events (actor_id,timestamp,type,channel,direction,thread_id,"
            "content,content_hash,target_ids,features) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                actor_id,
                float(timestamp),
                type.value if isinstance(type, EventType) else str(type),
                channel,
                direction,
                thread_id,
                content,
                _hash(content),
                json.dumps(list(target_ids)),
                json.dumps(features or {}),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # ---------- as-of reads (the leakage guarantee) ----------

    def history_asof(self, entity_id: str, before_ts: float) -> list[Event]:
        """Everything involving entity_id STRICTLY before before_ts — inbound messages they
        sent, and outbound messages sent to them. Nothing at/after before_ts is reachable."""
        rows = self._conn.execute(
            "SELECT actor_id,timestamp,type,channel,direction,content,target_ids,features "
            "FROM events WHERE timestamp < ? AND "
            "(actor_id = ? OR target_ids LIKE ?) ORDER BY timestamp",
            (float(before_ts), entity_id, f'%"{entity_id}"%'),
        ).fetchall()
        out = []
        for actor, ts, typ, channel, direction, content, targets, feats in rows:
            f = json.loads(feats)
            f.update({"direction": direction, "content": content, "targets": json.loads(targets)})
            out.append(
                Event(actor_id=actor, timestamp=ts, type=EventType(typ), channel=channel,
                      content_ref=None, content_hash=_hash(content),
                      target_ids=tuple(json.loads(targets)), features=f)
            )
        return out

    # ---------- label derivation ----------

    def labeled_sends(self, *, exclude_autoreplies: bool = True,
                      censor_at: float | None = None) -> list[Send]:
        """Every outbound message, labeled: did the recipient reply to THIS message?

        censor_at: if set (usually time.time()), sends whose reply window has not yet fully
        elapsed are EXCLUDED — "no reply yet" is right-censored, not a negative.

        A reply counts only if it arrives after this send and before BOTH (a) the channel's
        reply window and (b) our own next outbound in the same thread — otherwise a single
        reply to a later message would falsely credit every earlier send in the thread
        (label-inflation bug; see tests/test_end_to_end.py).

        Auto-reply heuristic: an inbound reply < 60s after the send is almost certainly a
        machine (OOO/bounce); excluded from positive labels when exclude_autoreplies."""
        sends = self._conn.execute(
            "SELECT id,timestamp,channel,thread_id,content,target_ids FROM events "
            "WHERE direction='out' AND type='message' ORDER BY timestamp"
        ).fetchall()
        result: list[Send] = []
        for eid, ts, channel, thread_id, content, targets in sends:
            recipients = json.loads(targets)
            if not recipients:
                continue
            recipient = recipients[0]
            window = REPLY_WINDOWS.get(channel, 7 * 86400.0)
            if censor_at is not None and ts + window > censor_at:
                continue  # right-censored: outcome not yet observable
            window_end = ts + window
            if thread_id:
                nxt = self._conn.execute(
                    "SELECT MIN(timestamp) FROM events WHERE direction='out' AND thread_id=? "
                    "AND timestamp > ?",
                    (thread_id, ts),
                ).fetchone()[0]
                if nxt is not None:
                    window_end = min(window_end, nxt)
                row = self._conn.execute(
                    "SELECT MIN(timestamp) FROM events WHERE direction='in' AND thread_id=? "
                    "AND actor_id=? AND timestamp > ? AND timestamp <= ?",
                    (thread_id, recipient, ts, window_end),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT MIN(timestamp) FROM events WHERE direction='in' AND actor_id=? "
                    "AND timestamp > ? AND timestamp <= ?",
                    (recipient, ts, window_end),
                ).fetchone()
            reply_ts = row[0]
            latency = (reply_ts - ts) if reply_ts is not None else None
            replied = reply_ts is not None
            if replied and exclude_autoreplies and latency is not None and latency < 60.0:
                replied, latency = False, None
            result.append(Send(eid, recipient, ts, channel, thread_id, content, replied, latency))
        return result

    def recipients(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT json_extract(target_ids,'$[0]') FROM events WHERE direction='out'"
        ).fetchall()
        return sorted(r[0] for r in rows if r[0])

    def close(self) -> None:
        self._conn.close()


def _hash(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()[:16]
