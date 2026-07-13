"""Temporal verification — Phase 2.

Google's after:/before: operators and an RSS pubDate are DISCOVERY signals, not proof. Every discovered item
is independently temporally verified here into a typed status:

  verified_pre_asof   an independent server-side signal proves the content existed at/before as_of
  likely_pre_asof     the claimed publication time is at/before as_of, no independent verification
  uncertain           signals conflict, or the claimed time is within a margin of as_of
  likely_post_asof    the claimed publication time is after as_of
  verified_post_asof  an independent signal proves it appeared only after as_of
  undated             no usable timestamp at all

Signals (independent where possible):
  1. feed pubDate            CLAIMED (RSS) — never trusted alone for a `verified` label
  2. Wayback earliest capture SERVER-SIDE — an archive.org snapshot at/before as_of proves prior existence
  3. (extensible) article JSON-LD datePublished / HTTP Last-Modified

Production inference may use verified_pre_asof and (with explicit uncertainty) likely_pre_asof. `uncertain`
is sensitivity-only. likely_post_asof / verified_post_asof are EXCLUDED from the as-of bundle and kept in the
leakage report. Online verification (Wayback) is optional (`verify_online`) so tests run offline on the
claimed signal; the status honestly reflects which signals were available.
"""
from __future__ import annotations

import json
import time as _time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict

TEMPORAL_STATUSES = ("verified_pre_asof", "likely_pre_asof", "uncertain", "likely_post_asof",
                     "verified_post_asof", "undated")
#: statuses admissible to the production as-of bundle (uncertain is sensitivity-only)
ADMISSIBLE_PRODUCTION = ("verified_pre_asof", "likely_pre_asof")
_WAYBACK = "https://archive.org/wayback/available"
_UA = "Mozilla/5.0 (compatible; swm-evidence/1.0)"


@dataclass
class TemporalRecord:
    status: str
    confidence: float
    as_of: float
    claimed_pubdate_ts: float | None = None
    verified_pubdate_ts: float | None = None          # earliest independent server-side time
    signals: list = field(default_factory=list)       # [{signal, ts, source, note}]
    margin_s: float = 0.0

    def admissible(self) -> bool:
        return self.status in ADMISSIBLE_PRODUCTION

    def as_dict(self) -> dict:
        return asdict(self)


class TemporalVerifier:
    """Verify a discovered item's as-of validity from multiple signals. `margin_days` is a symmetric grey
    zone around as_of within which a claimed-only date is `uncertain` rather than confidently pre/post."""

    def __init__(self, *, verify_online: bool = False, margin_days: float = 1.0, timeout: int = 12):
        self.verify_online = verify_online
        self.margin_s = margin_days * 86400.0
        self.timeout = timeout

    def verify(self, *, as_of: float, claimed_ts: float | None, url: str = "") -> TemporalRecord:
        rec = TemporalRecord(status="undated", confidence=0.0, as_of=as_of,
                             claimed_pubdate_ts=claimed_ts, margin_s=self.margin_s)
        if claimed_ts is not None:
            rec.signals.append({"signal": "feed_pubdate", "ts": claimed_ts, "source": "rss",
                                "note": "claimed publication time (unverified)"})
        # independent server-side signal: earliest Wayback capture
        verified_ts = None
        if self.verify_online and url:
            verified_ts = self._wayback_earliest(url)
            if verified_ts is not None:
                rec.verified_pubdate_ts = verified_ts
                rec.signals.append({"signal": "wayback_earliest_capture", "ts": verified_ts,
                                    "source": "archive.org", "note": "server-side snapshot time"})
        return self._classify(rec, claimed_ts, verified_ts)

    def _classify(self, rec: TemporalRecord, claimed_ts, verified_ts) -> TemporalRecord:
        as_of, m = rec.as_of, self.margin_s
        # a verified server-side capture is the strongest signal
        if verified_ts is not None:
            if verified_ts <= as_of:
                rec.status, rec.confidence = "verified_pre_asof", 0.97
                return rec
            if claimed_ts is None or claimed_ts > as_of:
                rec.status, rec.confidence = "verified_post_asof", 0.9
                return rec
            # verified capture after as_of but claimed before → conflict
            rec.status, rec.confidence = "uncertain", 0.5
            return rec
        # claimed-only
        if claimed_ts is None:
            rec.status, rec.confidence = "undated", 0.0
            return rec
        if claimed_ts > as_of + m:
            rec.status, rec.confidence = "likely_post_asof", 0.8
        elif claimed_ts < as_of - m:
            rec.status, rec.confidence = "likely_pre_asof", 0.75
        else:                                                # within the grey zone around as_of
            rec.status, rec.confidence = "uncertain", 0.5
        return rec

    def _wayback_earliest(self, url: str) -> float | None:
        """Earliest archive.org snapshot time for the URL (server-side proof of prior existence), or None.
        Uses the availability API with timestamp=19900101 to bias toward the oldest capture."""
        try:
            q = _WAYBACK + "?" + urllib.parse.urlencode({"url": url, "timestamp": "19900101"})
            req = urllib.request.Request(q, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8", "ignore"))
            snap = (((data or {}).get("archived_snapshots") or {}).get("closest") or {})
            if snap.get("available") and snap.get("timestamp"):
                return _time.mktime(_time.strptime(snap["timestamp"][:14], "%Y%m%d%H%M%S"))
        except Exception:  # noqa: BLE001 — a failed verification is not a leak; the item stays claimed-only
            return None
        return None
