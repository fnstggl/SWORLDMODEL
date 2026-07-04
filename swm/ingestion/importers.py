"""Importers: normalized thread JSON, generic CSV, and iMessage chat.db (audit D).

Everything funnels into one normalized shape so email and text are the same object downstream:

    thread = {
      "thread_id": str,
      "channel": "email" | "text",
      "messages": [
        {"from": str, "to": [str], "timestamp": float (unix s), "text": str}, ...
      ]
    }

`owner_id` identifies *you*; direction is derived from it (from == owner -> 'out').
Group messages are skipped for reply-labeling (attribution is ambiguous); 1:1 only in v1.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

from swm.ingestion.schema import EventType
from swm.ingestion.store import EventStore

# Senders that can never "reply" — excluded so labels aren't polluted.
_NOREPLY_RE = re.compile(
    r"(no-?reply|do-?not-?reply|notifications?@|mailer-daemon|noreply|newsletter|"
    r"updates@|info@|support@|billing@|receipts?@|hello@(?!.*\bpersonal\b))",
    re.I,
)


def looks_automated(address: str) -> bool:
    return bool(_NOREPLY_RE.search(address or ""))


def import_threads(store: EventStore, threads: list[dict], owner_id: str) -> dict:
    """Import normalized threads. Returns counts for an honest ingest report."""
    counts = {"messages_in": 0, "messages_out": 0, "skipped_group": 0, "skipped_automated": 0}
    for th in threads:
        channel = th.get("channel", "email")
        tid = th["thread_id"]
        for m in th.get("messages", []):
            sender = _norm(m["from"])
            recipients = [_norm(r) for r in m.get("to", []) if _norm(r) != _norm(owner_id) or sender != _norm(owner_id)]
            outbound = sender == _norm(owner_id)
            others = [r for r in recipients if r != _norm(owner_id)]
            if outbound and len(others) != 1:
                counts["skipped_group"] += 1
                continue
            counterparty = others[0] if outbound else sender
            if looks_automated(counterparty):
                counts["skipped_automated"] += 1
                continue
            store.append(
                actor_id=sender,
                timestamp=float(m["timestamp"]),
                type=EventType.MESSAGE,
                channel=channel,
                direction="out" if outbound else "in",
                thread_id=tid,
                content=m.get("text", "") or "",
                target_ids=(counterparty,) if outbound else (_norm(owner_id),),
            )
            counts["messages_out" if outbound else "messages_in"] += 1
    return counts


def import_threads_json(store: EventStore, path: str | Path, owner_id: str) -> dict:
    threads = json.loads(Path(path).read_text())
    if isinstance(threads, dict):
        threads = threads.get("threads", [])
    return import_threads(store, threads, owner_id)


def import_csv(store: EventStore, path: str | Path, owner_id: str, channel: str = "email") -> dict:
    """Generic export: columns thread_id, from, to, timestamp (unix s or ISO), text."""
    from datetime import datetime

    threads: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = row["timestamp"]
            try:
                ts = float(ts)
            except ValueError:
                ts = datetime.fromisoformat(ts).timestamp()
            th = threads.setdefault(
                row["thread_id"], {"thread_id": row["thread_id"], "channel": channel, "messages": []}
            )
            th["messages"].append(
                {"from": row["from"], "to": [t.strip() for t in row["to"].split(";") if t.strip()],
                 "timestamp": ts, "text": row.get("text", "")}
            )
    return import_threads(store, list(threads.values()), owner_id)


# Apple epoch (2001-01-01) offset from unix epoch; chat.db dates are ns since Apple epoch (modern).
_APPLE_EPOCH = 978307200.0


def import_imessage_db(store: EventStore, chat_db_path: str | Path, owner_id: str = "me") -> dict:
    """Import a *copy* of macOS ~/Library/Messages/chat.db. 1:1 chats only.

    is_from_me gives direction; the handle (phone/email) is the counterparty id.
    """
    conn = sqlite3.connect(str(chat_db_path))
    rows = conn.execute(
        """
        SELECT c.ROWID, h.id, m.is_from_me, m.date, m.text
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        JOIN chat c ON c.ROWID = cmj.chat_id
        JOIN handle h ON h.ROWID = m.handle_id
        WHERE m.text IS NOT NULL
          AND (SELECT COUNT(*) FROM chat_handle_join chj WHERE chj.chat_id = c.ROWID) = 1
        ORDER BY m.date
        """
    ).fetchall()
    conn.close()
    threads: dict[str, dict] = {}
    for chat_id, handle, is_from_me, date, text in rows:
        ts = date / 1e9 + _APPLE_EPOCH if date > 1e12 else date + _APPLE_EPOCH
        tid = f"imsg-{chat_id}"
        th = threads.setdefault(tid, {"thread_id": tid, "channel": "text", "messages": []})
        th["messages"].append(
            {"from": owner_id if is_from_me else handle,
             "to": [handle] if is_from_me else [owner_id],
             "timestamp": ts, "text": text}
        )
    return import_threads(store, list(threads.values()), owner_id)


def _norm(addr: str) -> str:
    """Normalize 'Name <a@b.c>' -> 'a@b.c', lowercase."""
    m = re.search(r"<([^>]+)>", addr or "")
    return (m.group(1) if m else (addr or "")).strip().lower()
