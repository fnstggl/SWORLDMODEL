"""Production evidence source connectors — Phase 2 (retrieval).

Real request construction, real response retrieval, raw-response persistence (content-addressed),
normalization, source metadata, bounded retries, timeout, explicit status, and a retrieval-trace record for
EVERY invocation (including failures). A connector distinguishes a *technical* failure (non-200, timeout,
exception, parse error) from *zero valid results* — the two must never be confused.

The Google News RSS connector is the historical-discovery workhorse. For as-of discovery it uses BOTH
`after:` and `before:` operators in the SAME query — never `before:` alone (an audited leak vector). Google's
date operators are a DISCOVERY mechanism only; every discovered item still passes independent temporal
verification downstream (see evidence_temporal.py). The connector persists the unencoded logical query, the
encoded wire URL, both dates, retrieval time, status, headers, redirects, the raw RSS bytes (by content
hash), the feed-parser version, and each item's feed rank + pubDate + links.

Dependency-free: stdlib urllib + xml.etree + email.utils + hashlib only.
"""
from __future__ import annotations

import hashlib
import re
import time as _time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from email.utils import parsedate_to_datetime
from pathlib import Path

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
FEED_PARSER_VERSION = "gnews-rss-parser-1.0"
CONNECTOR_VERSION = "1.0"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (swm-evidence/1.0)"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

CONNECTOR_STATUSES = ("ok", "zero_results", "http_error", "timeout", "network_error",
                      "parse_error", "invalid_query")


class PairedDateError(ValueError):
    """A historical RSS query was issued without BOTH after: and before: — refused, loudly.
    The production historical-discovery arm must never use before: alone (leak vector)."""


def paired_dates_ok(after_date: str, before_date: str) -> bool:
    """True iff both dates are present and ISO YYYY-MM-DD. Used by the connector AND by the invariant test
    that fails if a historical query is missing either operator."""
    return bool(after_date and before_date and _DATE_RE.match(after_date) and _DATE_RE.match(before_date))


# --------------------------------------------------------------------------- content-addressed raw store
class RawContentStore:
    """Immutable content-addressed store for raw connector responses. The same bytes are stored ONCE
    (keyed by sha256), so twenty syndicated copies of one feed cost one blob. Audit-grade: raw bytes are
    never mutated; downstream claims cite the hash."""

    def __init__(self, root: str = "data/evidence_raw"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> str:
        h = hashlib.sha256(data).hexdigest()
        p = self._path(h)
        if not p.exists():                                   # write-once
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return h

    def get(self, content_hash: str) -> bytes | None:
        p = self._path(content_hash)
        return p.read_bytes() if p.exists() else None

    def _path(self, h: str) -> Path:
        return self.root / h[:2] / f"{h}.bin"


# --------------------------------------------------------------------------- typed records
@dataclass
class DiscoveredItem:
    """One item returned by a discovery connector — raw, pre-verification. `feed_pubdate_ts` is the feed's
    CLAIMED publication time (unverified); temporal verification decides its as-of validity separately."""
    connector_id: str
    requirement_id: str
    logical_query: str
    wire_url: str
    rank: int                                                # position in the feed (1-based)
    title: str = ""
    link: str = ""                                           # article URL (Google redirect for gnews)
    google_redirect_url: str = ""
    source_name: str = ""
    feed_pubdate: str = ""                                   # raw RFC-822 pubDate string
    feed_pubdate_ts: float | None = None                     # parsed unix ts (CLAIMED, unverified)
    description: str = ""
    raw_content_hash: str = ""                               # hash of the whole feed the item came from
    retrieved_at: float = 0.0

    def item_hash(self) -> str:
        return hashlib.sha256(f"{self.link}|{self.title}".encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["item_hash"] = self.item_hash()
        return d


@dataclass
class RetrievalTrace:
    """Persisted for EVERY connector invocation — success, zero-results, or failure. This is the audit
    record that proves what was queried, when, and what came back."""
    connector_id: str
    connector_version: str
    requirement_id: str
    logical_query: str
    wire_url: str
    after_date: str = ""
    before_date: str = ""
    retrieved_at: float = 0.0
    status_code: int | None = None
    redirects: list = field(default_factory=list)
    response_headers: dict = field(default_factory=dict)
    raw_content_hash: str = ""
    n_bytes: int = 0
    feed_parser_version: str = FEED_PARSER_VERSION
    n_items: int = 0
    connector_status: str = "ok"                             # CONNECTOR_STATUSES
    error: str = ""
    latency_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class _RedirectRecorder(urllib.request.HTTPRedirectHandler):
    def __init__(self):
        self.chain = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.chain.append({"code": code, "to": newurl})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class GoogleNewsRSSConnector:
    """Historical news discovery via Google News RSS with paired after:/before:. Discovery only — items
    are temporally verified downstream. Never silently returns [] on a technical failure: the trace's
    `connector_status` separates `zero_results` from `http_error`/`timeout`/`network_error`/`parse_error`."""
    connector_id = "google_news_rss"

    def __init__(self, store: RawContentStore | None = None, *, hl="en-US", gl="US", ceid="US:en"):
        self.store = store or RawContentStore()
        self.hl, self.gl, self.ceid = hl, gl, ceid

    def _wire_url(self, logical_query: str) -> str:
        return GOOGLE_NEWS_RSS + "?" + urllib.parse.urlencode(
            {"q": logical_query, "hl": self.hl, "gl": self.gl, "ceid": self.ceid})

    def search_historical(self, query_terms: str, *, after_date: str, before_date: str,
                          requirement_id: str = "", k: int = 20, timeout: int = 20,
                          retries: int = 2) -> tuple[list, RetrievalTrace]:
        """Issue ONE paired after:/before: query. Both dates are REQUIRED (ISO YYYY-MM-DD): a historical
        query without both is refused (PairedDateError) — the production invariant. Returns (items, trace)."""
        if not paired_dates_ok(after_date, before_date):
            raise PairedDateError(
                f"historical Google News RSS query requires BOTH after: and before: (ISO YYYY-MM-DD); "
                f"got after={after_date!r} before={before_date!r}")
        logical = f"{query_terms.strip()} after:{after_date} before:{before_date}"
        wire = self._wire_url(logical)
        trace = RetrievalTrace(connector_id=self.connector_id, connector_version=CONNECTOR_VERSION,
                               requirement_id=requirement_id, logical_query=logical, wire_url=wire,
                               after_date=after_date, before_date=before_date, retrieved_at=_time.time())
        raw, backoff = None, 1.0
        for attempt in range(retries + 1):
            rec = _RedirectRecorder()
            opener = urllib.request.build_opener(rec)
            req = urllib.request.Request(wire, headers={"User-Agent": _UA,
                                                        "Accept": "application/rss+xml, application/xml"})
            t0 = _time.time()
            try:
                resp = opener.open(req, timeout=timeout)
                raw = resp.read()
                trace.status_code = resp.status
                trace.redirects = rec.chain
                trace.response_headers = {k: v for k, v in list(resp.headers.items())[:20]}
                trace.latency_s = round(_time.time() - t0, 3)
                break
            except urllib.error.HTTPError as e:
                trace.status_code = e.code
                trace.connector_status, trace.error = "http_error", f"HTTP {e.code}: {str(e)[:120]}"
            except (TimeoutError, urllib.error.URLError) as e:
                is_to = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                trace.connector_status = "timeout" if is_to else "network_error"
                trace.error = f"{type(e).__name__}: {str(e)[:120]}"
            except Exception as e:  # noqa: BLE001
                trace.connector_status, trace.error = "network_error", f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < retries:
                _time.sleep(min(8.0, backoff)); backoff *= 2
        if raw is None:
            trace.latency_s = trace.latency_s or 0.0
            return [], trace                                 # technical failure — status set, NOT zero_results

        trace.raw_content_hash = self.store.put(raw)
        trace.n_bytes = len(raw)
        try:
            items = self._parse(raw, logical, wire, requirement_id, trace.raw_content_hash, k)
        except Exception as e:  # noqa: BLE001
            trace.connector_status, trace.error = "parse_error", f"{type(e).__name__}: {str(e)[:120]}"
            return [], trace
        trace.n_items = len(items)
        trace.connector_status = "ok" if items else "zero_results"   # genuine empty result, NOT a failure
        return items, trace

    def _parse(self, raw: bytes, logical: str, wire: str, req_id: str, raw_hash: str, k: int) -> list:
        root = ET.fromstring(raw)
        out, now = [], _time.time()
        for rank, item in enumerate(root.iter("item"), 1):
            if rank > k:
                break
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = _strip_html(item.findtext("description") or "")
            pub = (item.findtext("pubDate") or "").strip()
            src_el = item.find("source")
            source = (src_el.text or "").strip() if src_el is not None else ""
            out.append(DiscoveredItem(
                connector_id=self.connector_id, requirement_id=req_id, logical_query=logical, wire_url=wire,
                rank=rank, title=title, link=link, google_redirect_url=link, source_name=source,
                feed_pubdate=pub, feed_pubdate_ts=_rfc822_ts(pub), description=desc[:400],
                raw_content_hash=raw_hash, retrieved_at=now))
        return out


def _strip_html(s: str) -> str:
    import html as _html
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def _rfc822_ts(s: str) -> float | None:
    try:
        return parsedate_to_datetime(s).timestamp() if s else None
    except (TypeError, ValueError):
        return None
