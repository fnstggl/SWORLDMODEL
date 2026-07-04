"""Convert Gmail `search_threads` page JSON into normalized threads (audit D, wedge A).

Works from the thread-list view (subject + snippet per message) — full bodies are not required
for v1: snippets are truncated UNIFORMLY for replied and unreplied sends alike, so length-ish
features are weakened but not differentially biased. Flagged as a known limitation.

Data-integrity rules applied here (the difference between a backtest and a lie):
1. BOUNCES ARE NOT NEGATIVES. If a mailer-daemon/postmaster message lands in a thread within
   BOUNCE_WINDOW of a send, that send was never delivered — the whole 1:1 thread is dropped
   (undelivered != ignored).
2. Automated counterparties (no-reply@, notifications@, ...) are dropped by the importer.
3. RIGHT-CENSORING is handled downstream: exclude sends whose reply window hasn't elapsed
   (see labeled_sends / the exp001 runner), otherwise "no reply YET" pollutes the negatives.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

BOUNCE_WINDOW = 6 * 3600.0
_BOUNCE_RE = re.compile(r"mailer-daemon|postmaster@|delivery status notification", re.I)


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def convert_pages(pages: list[dict], owner_id: str) -> tuple[list[dict], dict]:
    """Gmail search_threads pages -> (normalized threads, integrity report)."""
    threads, report = [], {"threads_in": 0, "dropped_bounced": 0, "messages": 0}
    for page in pages:
        for th in page.get("threads", []):
            report["threads_in"] += 1
            msgs = th.get("messages", [])
            sends = [m for m in msgs if m.get("sender", "").lower() == owner_id.lower()]
            bounced = any(
                _BOUNCE_RE.search(m.get("sender", "") + " " + m.get("subject", ""))
                and any(abs(_ts(m["date"]) - _ts(s["date"])) <= BOUNCE_WINDOW for s in sends)
                for m in msgs
            )
            if bounced:
                report["dropped_bounced"] += 1
                continue
            norm = {"thread_id": th["id"], "channel": "email", "messages": []}
            for m in msgs:
                if _BOUNCE_RE.search(m.get("sender", "")):
                    continue
                body = html.unescape(m.get("snippet", "") or "")
                subject = html.unescape(m.get("subject", "") or "")
                norm["messages"].append({
                    "from": m["sender"],
                    "to": m.get("toRecipients", []),
                    "timestamp": _ts(m["date"]),
                    "text": (subject + "\n" + body).strip(),
                })
                report["messages"] += 1
            if norm["messages"]:
                threads.append(norm)
    return threads, report


def convert_files(paths: list[str | Path], owner_id: str) -> tuple[list[dict], dict]:
    pages = [json.loads(Path(p).read_text()) for p in paths]
    return convert_pages(pages, owner_id)
