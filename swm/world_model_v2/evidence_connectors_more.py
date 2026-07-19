"""Additional production evidence connectors — Phase 2.

Uniform interface with the Google News RSS connector: each `.fetch(...)` returns (list[DiscoveredItem],
RetrievalTrace) and persists raw responses by content hash. Categories implemented and live-testable here:

  WebPageConnector          general web-page retrieval (raw HTML persisted, text normalized)
  WikipediaConnector        entity/background via the MediaWiki API — the revision timestamp is a
                            SERVER-SIDE verified time (a real independent temporal signal)
  UserDocumentConnector     user-provided text/documents (no network; visibility can be private)
  LocalDatasetConnector     structured local dataset rows (JSON/JSONL) matched by query terms
  PriorArtifactConnector    prior WorldState / evidence-bundle artifacts on disk

Every invocation records a RetrievalTrace with an explicit connector_status (never a silent []).
"""
from __future__ import annotations

import html as _html
import json
import re
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from swm.world_model_v2.evidence_connectors import (CONNECTOR_VERSION, DiscoveredItem, RawContentStore,
                                                    RetrievalTrace, _rfc822_ts)

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (swm-evidence/1.0)"


def _strip(s: str) -> str:
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def _fetch(url: str, timeout: int, store: RawContentStore, trace: RetrievalTrace, retries: int = 2):
    backoff = 1.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            t0 = _time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                trace.status_code = r.status
                trace.response_headers = {k: v for k, v in list(r.headers.items())[:15]}
                trace.latency_s = round(_time.time() - t0, 3)
            trace.raw_content_hash = store.put(raw)
            trace.n_bytes = len(raw)
            return raw
        except urllib.error.HTTPError as e:
            trace.status_code, trace.connector_status, trace.error = e.code, "http_error", f"HTTP {e.code}"
        except (TimeoutError, urllib.error.URLError) as e:
            to = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
            trace.connector_status = "timeout" if to else "network_error"
            trace.error = f"{type(e).__name__}: {str(e)[:100]}"
        except Exception as e:  # noqa: BLE001
            trace.connector_status, trace.error = "network_error", f"{type(e).__name__}: {str(e)[:100]}"
        if attempt < retries:
            _time.sleep(min(6.0, backoff)); backoff *= 2
    return None


class WebPageConnector:
    connector_id = "web_page"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def fetch(self, url: str, *, requirement_id: str = "", timeout: int = 15):
        tr = RetrievalTrace(connector_id=self.connector_id, connector_version=CONNECTOR_VERSION,
                            requirement_id=requirement_id, logical_query=url, wire_url=url,
                            retrieved_at=_time.time())
        raw = _fetch(url, timeout, self.store, tr)
        if raw is None:
            return [], tr
        text = _strip(raw.decode("utf-8", "ignore"))
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", raw.decode("utf-8", "ignore"), re.S | re.I)
        if m:
            title = _strip(m.group(1))
        item = DiscoveredItem(connector_id=self.connector_id, requirement_id=requirement_id,
                              logical_query=url, wire_url=url, rank=1, title=title, link=url,
                              source_name=urllib.parse.urlparse(url).netloc, description=text[:1200],
                              raw_content_hash=tr.raw_content_hash, retrieved_at=_time.time())
        tr.n_items, tr.connector_status = 1, "ok" if text else "zero_results"
        return [item], tr


class WikipediaConnector:
    """MediaWiki API. The revision timestamp is a SERVER-SIDE verified time — used as an independent
    temporal signal (published_verified). Discovery + background, not a news source."""
    connector_id = "wikipedia_revision"
    _API = "https://en.wikipedia.org/w/api.php"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def fetch(self, topic: str, *, requirement_id: str = "", timeout: int = 15, as_of_iso: str = ""):
        params = {"action": "query", "prop": "extracts|revisions", "titles": topic, "explaintext": 1,
                  "exintro": 1, "rvprop": "timestamp|ids", "format": "json", "redirects": 1}
        if as_of_iso:                                    # ask for the revision AS OF the question date
            params["rvstart"] = f"{as_of_iso}T23:59:59Z"
            params["rvlimit"] = 1
        url = self._API + "?" + urllib.parse.urlencode(params)
        tr = RetrievalTrace(connector_id=self.connector_id, connector_version=CONNECTOR_VERSION,
                            requirement_id=requirement_id, logical_query=f"wiki:{topic}", wire_url=url,
                            after_date="", before_date=as_of_iso, retrieved_at=_time.time())
        raw = _fetch(url, timeout, self.store, tr)
        if raw is None:
            return [], tr
        try:
            data = json.loads(raw.decode("utf-8", "ignore"))
            pages = list((data.get("query") or {}).get("pages", {}).values())
        except Exception as e:  # noqa: BLE001
            tr.connector_status, tr.error = "parse_error", str(e)[:100]
            return [], tr
        out = []
        for p in pages:
            if "missing" in p:
                continue
            extract = (p.get("extract") or "").strip()
            revs = p.get("revisions") or [{}]
            ts_iso = revs[0].get("timestamp", "")
            ts = _iso_ts(ts_iso)
            out.append(DiscoveredItem(
                connector_id=self.connector_id, requirement_id=requirement_id, logical_query=f"wiki:{topic}",
                wire_url=url, rank=len(out) + 1, title=p.get("title", topic), link=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(p.get('title', topic))}",
                source_name="Wikipedia", feed_pubdate=ts_iso, feed_pubdate_ts=ts,
                description=extract[:1500], raw_content_hash=tr.raw_content_hash, retrieved_at=_time.time()))
        tr.n_items, tr.connector_status = len(out), "ok" if out else "zero_results"
        return out, tr


class UserDocumentConnector:
    """Wrap user-provided text/documents as evidence items. No network. Visibility defaults to private to the
    supplied actor scope (the orchestrator applies the visibility hint)."""
    connector_id = "user_provided"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def fetch(self, documents: list, *, requirement_id: str = ""):
        tr = RetrievalTrace(connector_id=self.connector_id, connector_version=CONNECTOR_VERSION,
                            requirement_id=requirement_id, logical_query="user_documents",
                            wire_url="", retrieved_at=_time.time(), connector_status="ok")
        out = []
        for i, doc in enumerate(documents or [], 1):
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            h = self.store.put(text.encode("utf-8"))
            out.append(DiscoveredItem(
                connector_id=self.connector_id, requirement_id=requirement_id, logical_query="user_documents",
                wire_url="", rank=i, title=(doc.get("title", "") if isinstance(doc, dict) else "")[:120],
                source_name=(doc.get("source", "user") if isinstance(doc, dict) else "user"),
                feed_pubdate_ts=(doc.get("published_at") if isinstance(doc, dict) else None),
                description=text[:1500], raw_content_hash=h, retrieved_at=_time.time()))
        tr.n_items = len(out)
        tr.connector_status = "ok" if out else "zero_results"
        return out, tr


class LocalDatasetConnector:
    """Structured local dataset rows (JSON list or JSONL) matched by query terms. Real file IO; the vintage
    of the dataset file is its temporal basis."""
    connector_id = "dataset"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def fetch(self, path: str, query_terms: str, *, requirement_id: str = "", k: int = 8,
              text_field: str = "text", ts_field: str = "published_at"):
        tr = RetrievalTrace(connector_id=self.connector_id, connector_version=CONNECTOR_VERSION,
                            requirement_id=requirement_id, logical_query=f"{path}?q={query_terms}",
                            wire_url=path, retrieved_at=_time.time())
        p = Path(path)
        if not p.exists():
            tr.connector_status, tr.error = "network_error", "dataset file not found"
            return [], tr
        raw = p.read_bytes()
        tr.raw_content_hash, tr.n_bytes, tr.status_code = self.store.put(raw), len(raw), 200
        try:
            rows = json.loads(raw.decode("utf-8", "ignore")) if p.suffix == ".json" else \
                [json.loads(l) for l in raw.decode("utf-8", "ignore").splitlines() if l.strip()]
        except Exception as e:  # noqa: BLE001
            tr.connector_status, tr.error = "parse_error", str(e)[:100]
            return [], tr
        terms = [t for t in re.findall(r"[a-z0-9]+", query_terms.lower()) if len(t) > 2]
        out = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            text = str(row.get(text_field, ""))
            if terms and not any(t in text.lower() for t in terms):
                continue
            out.append(DiscoveredItem(
                connector_id=self.connector_id, requirement_id=requirement_id,
                logical_query=f"{path}?q={query_terms}", wire_url=path, rank=len(out) + 1,
                title=str(row.get("title", ""))[:120], source_name=str(row.get("source", p.name)),
                feed_pubdate_ts=row.get(ts_field), description=text[:1200],
                raw_content_hash=tr.raw_content_hash, retrieved_at=_time.time()))
            if len(out) >= k:
                break
        tr.n_items = len(out)
        tr.connector_status = "ok" if out else "zero_results"
        return out, tr


class PriorArtifactConnector:
    """Prior WorldState / evidence-bundle artifacts on disk — reuse of already-verified evidence."""
    connector_id = "prior_world_state"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def fetch(self, path: str, *, requirement_id: str = ""):
        tr = RetrievalTrace(connector_id=self.connector_id, connector_version=CONNECTOR_VERSION,
                            requirement_id=requirement_id, logical_query=f"prior:{path}", wire_url=path,
                            retrieved_at=_time.time())
        p = Path(path)
        if not p.exists():
            tr.connector_status, tr.error = "network_error", "prior artifact not found"
            return [], tr
        raw = p.read_bytes()
        tr.raw_content_hash, tr.n_bytes, tr.status_code = self.store.put(raw), len(raw), 200
        try:
            doc = json.loads(raw.decode("utf-8", "ignore"))
        except Exception as e:  # noqa: BLE001
            tr.connector_status, tr.error = "parse_error", str(e)[:100]
            return [], tr
        prior_claims = doc.get("included_claim_ids") or doc.get("claims") or []
        item = DiscoveredItem(
            connector_id=self.connector_id, requirement_id=requirement_id, logical_query=f"prior:{path}",
            wire_url=path, rank=1, title=f"prior evidence bundle {doc.get('bundle_id', '')}",
            source_name="prior_world_state", description=f"{len(prior_claims)} prior claims",
            raw_content_hash=tr.raw_content_hash, retrieved_at=_time.time())
        tr.n_items, tr.connector_status = 1, "ok"
        return [item], tr


def _iso_ts(s: str):
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _time.mktime(_time.strptime(s[:19], fmt))
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------- GDELT DOC 2.0 (free, keyless)
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_MIN_INTERVAL_S = 6.5                       # published rate limit: one request per 5 seconds
                                                  # (headroom — enforcement is sliding per IP)
_gdelt_last_call = [0.0]


class GdeltDocConnector:
    """Historical full-text news discovery via the GDELT 2.0 DOC API — the breadth layer the free
    stack was missing. Free, keyless, updated every 15 minutes since 2017, and date-scoped
    PRECISELY (startdatetime/enddatetime), unlike Google News RSS whose after:/before: operators
    cap at ~20 items and are sometimes ignored server-side. One requirement's facet query (an
    actor's statements, a quantity's measurements, the scheduled calendar) fans out to dozens of
    dated articles across thousands of outlets — battlefield reporting, aid packages, domestic
    politics, alliance commitments all arrive through the SAME requirement-driven queries.

    Same contract as GoogleNewsRSSConnector: paired dates REQUIRED (the production as-of
    invariant); every invocation produces a RetrievalTrace (failures included). Respects the
    published rate limit (>=5s between requests, module-wide) with one retry."""
    connector_id = "gdelt_doc"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    @staticmethod
    def _dt(date_iso: str, end: bool = False) -> str:
        return date_iso.replace("-", "") + ("235959" if end else "000000")

    def search_historical(self, query_terms: str, *, after_date: str, before_date: str,
                          requirement_id: str = "", k: int = 20, timeout: int = 25,
                          retries: int = 2) -> tuple:
        from swm.world_model_v2.evidence_connectors import PairedDateError, paired_dates_ok
        if not paired_dates_ok(after_date, before_date):
            raise PairedDateError(
                f"historical GDELT query requires BOTH after and before (ISO YYYY-MM-DD); "
                f"got after={after_date!r} before={before_date!r}")
        q = " ".join(str(query_terms).split()[:8])            # GDELT dislikes very long queries
        params = urllib.parse.urlencode({
            "query": f"{q} sourcelang:english", "mode": "artlist", "maxrecords": int(k),
            "format": "json", "sort": "hybridrel",
            "startdatetime": self._dt(after_date), "enddatetime": self._dt(before_date, end=True)})
        wire = f"{GDELT_DOC}?{params}"
        tr = RetrievalTrace(connector_id=self.connector_id, connector_version=CONNECTOR_VERSION,
                            requirement_id=requirement_id, logical_query=f"{q} [{after_date}..{before_date}]",
                            wire_url=wire, after_date=after_date, before_date=before_date,
                            retrieved_at=_time.time())
        raw = None
        for attempt in range(retries + 1):
            wait = _GDELT_MIN_INTERVAL_S - (_time.time() - _gdelt_last_call[0])
            if wait > 0:
                _time.sleep(wait)                             # module-wide pacing (published limit)
            _gdelt_last_call[0] = _time.time()
            t0 = _time.time()
            try:
                req = urllib.request.Request(wire, headers={"User-Agent": "swm-evidence/1.0 (research)",
                                                            "Accept": "application/json"})
                resp = urllib.request.urlopen(req, timeout=timeout)
                raw = resp.read()
                tr.status_code = resp.status
                tr.latency_s = round(_time.time() - t0, 3)
                if raw[:1] not in (b"{", b"["):               # rate-limit / advisory plain text
                    tr.connector_status, tr.error = "http_error", raw[:120].decode("utf-8", "ignore")
                    raw = None
                else:
                    break
            except urllib.error.HTTPError as e:
                tr.status_code = e.code
                tr.connector_status, tr.error = "http_error", f"HTTP {e.code}: {str(e)[:100]}"
            except (TimeoutError, urllib.error.URLError) as e:
                is_to = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                tr.connector_status = "timeout" if is_to else "network_error"
                tr.error = f"{type(e).__name__}: {str(e)[:100]}"
            except Exception as e:  # noqa: BLE001
                tr.connector_status, tr.error = "network_error", f"{type(e).__name__}: {str(e)[:100]}"
        if raw is None:
            tr.latency_s = tr.latency_s or 0.0
            return [], tr                                     # technical failure — status set, NOT zero
        tr.raw_content_hash = self.store.put(raw)
        tr.n_bytes = len(raw)
        try:
            items = self.parse_articles(raw, tr.logical_query, wire, requirement_id,
                                        tr.raw_content_hash, k)
        except Exception as e:  # noqa: BLE001
            tr.connector_status, tr.error = "parse_error", f"{type(e).__name__}: {str(e)[:100]}"
            return [], tr
        tr.n_items = len(items)
        tr.connector_status = "ok" if items else "zero_results"
        return items, tr

    def parse_articles(self, raw: bytes, logical: str, wire: str, req_id: str,
                       raw_hash: str, k: int) -> list:
        doc = json.loads(raw.decode("utf-8", "ignore"))
        out, now = [], _time.time()
        for rank, a in enumerate((doc.get("articles") or [])[:k], 1):
            seen = str(a.get("seendate", ""))                 # YYYYMMDDTHHMMSSZ
            ts = None
            try:
                ts = _time.mktime(_time.strptime(seen, "%Y%m%dT%H%M%SZ")) - _time.timezone
            except (ValueError, TypeError):
                pass
            out.append(DiscoveredItem(
                connector_id=self.connector_id, requirement_id=req_id, logical_query=logical,
                wire_url=wire, rank=rank, title=str(a.get("title", ""))[:300],
                link=str(a.get("url", "")), google_redirect_url="",
                source_name=str(a.get("domain", "")), feed_pubdate=seen, feed_pubdate_ts=ts,
                description="", raw_content_hash=raw_hash, retrieved_at=now))
        return out
