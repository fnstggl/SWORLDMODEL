"""FREE evidence connectors + requirement router — breadth without paid APIs.

"Thirteen articles for a world war" was the measured thinness. More Google News RSS is NOT the
answer: RSS only widens ONE modality (recent news prose). A world model needs the evidence
categories the requirements already declare — battlefield events, weapons/aid schedules, economic
constraints, domestic politics, alliance commitments, military capacity, public opinion,
leadership incentives, negotiation history — and most of those live in STRUCTURED free sources,
not in news feeds:

  * GDELT DOC 2.0        — date-bounded global news search (news breadth + the fitting corpus's
                           archived paired-date needs); no key.
  * Wikipedia search     — the structured-page classes: "Opinion polling for …" (public opinion),
                           "Order of battle …" / "List of military aid to …" (capacity, weapons/aid),
                           "Timeline of …" (negotiation history); no key.
  * Wikidata SPARQL      — machine-readable ground facts: who holds which office since when
                           (leadership incentives), treaty/alliance memberships (commitments),
                           scheduled elections (calendars); no key.
  * World Bank API       — macro constraints (GDP growth, inflation, military expenditure,
                           unemployment, reserves) per country; no key.
  * ReliefWeb v2         — conflict/humanitarian situation reports; no key (appname string only).
  * UCDP GED             — georeferenced battle events; free ACADEMIC token (UCDP_ACCESS_TOKEN);
                           degrades to absent when unset — never a paid dependency.
  * Curated institutional RSS — ISW daily assessments, UN News, NATO, defence ministries, IMF/ECB:
                           primary-institution statements news aggregators dilute; no key.

Every connector returns (items, RetrievalTrace) exactly like GoogleNewsRSSConnector, enforces the
paired-date / as-of hygiene ITS source supports (date-bounded query params where the API has them,
claimed-pubdate filtering where only the feed speaks), and DEGRADES to a recorded failure trace —
a blocked host must never abort evidence gathering (this container blocks GDELT and ISW today;
the traces say so honestly).

The FreeSourceRouter maps each EvidenceRequirement to the connectors that can actually answer its
CATEGORY (statements vs quantities vs calendars vs capabilities vs background), lexically and
universally — no scenario-specific lists.
"""
from __future__ import annotations

import json
import re
import time as _time
import urllib.parse
import urllib.request

from swm.world_model_v2.evidence_connectors import (DiscoveredItem, RawContentStore,
                                                    RetrievalTrace)

USER_AGENT = "SWORLDMODEL-research/1.0 (evidence pipeline; contact: repo)"
_WDQ = "https://query.wikidata.org/sparql"
_WD_API = "https://www.wikidata.org/w/api.php"


def _get(url: str, *, timeout: int = 15, headers: dict | None = None) -> tuple:
    """(bytes, status_code, error) — never raises."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), int(getattr(r, "status", 200) or 200), ""
    except Exception as e:  # noqa: BLE001 — connectors degrade, never abort gathering
        return b"", None, f"{type(e).__name__}: {e}"[:200]


def _trace(cid: str, req_id: str, logical: str, wire: str, **kw) -> RetrievalTrace:
    return RetrievalTrace(connector_id=cid, connector_version="1.0", requirement_id=req_id,
                          logical_query=logical, wire_url=wire, retrieved_at=_time.time(), **kw)


def _finish(tr: RetrievalTrace, raw: bytes, status, err: str, n_items: int, store, t0: float,
            parse_error: str = "") -> RetrievalTrace:
    tr.status_code = status
    tr.n_bytes = len(raw)
    tr.n_items = n_items
    tr.latency_s = round(_time.time() - t0, 3)
    if raw and store is not None:
        tr.raw_content_hash = store.put(raw)
    if err:
        tr.connector_status, tr.error = "network_error", err
    elif parse_error:
        tr.connector_status, tr.error = "parse_error", parse_error
    elif status is not None and status >= 400:
        tr.connector_status, tr.error = "http_error", f"HTTP {status}"
    elif n_items == 0:
        tr.connector_status = "zero_results"
    return tr


def _iso_compact(date_iso: str, end_of_day: bool = False) -> str:
    d = str(date_iso)[:10].replace("-", "")
    return f"{d}235959" if end_of_day else f"{d}000000"


class GdeltDocConnector:
    """GDELT DOC 2.0 artlist — date-bounded global news search, no key. The SAME paired-date
    semantics as the news connector: enddatetime is the as-of ceiling, leak-proof by the API's own
    window, which also makes it the right archived-corpus source for the intention-HR fitter."""
    connector_id = "gdelt_doc_v2"
    source_type = "news"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def search_historical(self, query_terms: str, *, after_date: str, before_date: str,
                          requirement_id: str = "", k: int = 10) -> tuple:
        t0 = _time.time()
        q = urllib.parse.quote(query_terms[:120])
        wire = (f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist"
                f"&maxrecords={int(k)}&format=json&sort=hybridrel"
                f"&startdatetime={_iso_compact(after_date)}"
                f"&enddatetime={_iso_compact(before_date, end_of_day=True)}")
        tr = _trace(self.connector_id, requirement_id, query_terms, wire,
                    after_date=after_date, before_date=before_date)
        raw, status, err = _get(wire, timeout=20)
        items, perr = [], ""
        if raw and not err:
            try:
                arts = (json.loads(raw.decode("utf-8", "replace")) or {}).get("articles") or []
                for i, a in enumerate(arts[:k], start=1):
                    ts = None
                    sd = str(a.get("seendate", ""))            # 20260715T120000Z
                    m = re.match(r"(\d{4})(\d{2})(\d{2})", sd)
                    if m:
                        ts = _time.mktime(_time.strptime("".join(m.groups()), "%Y%m%d"))
                    items.append(DiscoveredItem(
                        connector_id=self.connector_id, requirement_id=requirement_id,
                        logical_query=query_terms, wire_url=wire, rank=i,
                        title=str(a.get("title", ""))[:300], link=str(a.get("url", "")),
                        source_name=str(a.get("domain", "")), feed_pubdate=sd,
                        feed_pubdate_ts=ts, description=str(a.get("sourcecountry", "")),
                        retrieved_at=_time.time()))
            except (ValueError, KeyError) as e:
                perr = f"{type(e).__name__}: {e}"[:150]
        return items, _finish(tr, raw, status, err, len(items), self.store, t0, perr)


class WikipediaSearchConnector:
    """Wikipedia REST search — reaches the STRUCTURED page classes a topic fetch misses: opinion
    polling pages, orders of battle, military-aid lists, negotiation timelines. The router builds
    the structured-class query; this connector only searches and returns page items (revision
    hygiene is the temporal verifier's job, same as the existing topic connector)."""
    connector_id = "wikipedia_search"
    source_type = "wikipedia_revision"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def search(self, query_terms: str, *, requirement_id: str = "", k: int = 5) -> tuple:
        t0 = _time.time()
        wire = ("https://en.wikipedia.org/w/rest.php/v1/search/page?q="
                f"{urllib.parse.quote(query_terms[:200])}&limit={int(k)}")
        tr = _trace(self.connector_id, requirement_id, query_terms, wire)
        raw, status, err = _get(wire)
        items, perr = [], ""
        if raw and not err:
            try:
                pages = (json.loads(raw.decode("utf-8", "replace")) or {}).get("pages") or []
                for i, p in enumerate(pages[:k], start=1):
                    key = str(p.get("key", ""))
                    items.append(DiscoveredItem(
                        connector_id=self.connector_id, requirement_id=requirement_id,
                        logical_query=query_terms, wire_url=wire, rank=i,
                        title=str(p.get("title", ""))[:300],
                        link=f"https://en.wikipedia.org/wiki/{key}",
                        source_name="wikipedia",
                        description=re.sub(r"<[^>]+>", "", str(p.get("excerpt", "")))[:400],
                        retrieved_at=_time.time()))
            except (ValueError, KeyError) as e:
                perr = f"{type(e).__name__}: {e}"[:150]
        return items, _finish(tr, raw, status, err, len(items), self.store, t0, perr)


#: Wikidata properties that answer requirement categories with GROUND FACTS (universal, not
#: scenario lists): office-holders + start dates, memberships, applies-to-jurisdiction dates.
_WD_PROPS = {"P35": "head of state", "P6": "head of government", "P463": "member of",
             "P571": "inception", "P1906": "office held by head of state"}


class WikidataFactsConnector:
    """Entity ground facts from Wikidata: leadership (with start dates → term incentives),
    alliance/organization memberships (commitments), inception dates. Facts carry the statement's
    own time qualifiers; statements starting after as_of are dropped here (paired-date hygiene at
    the fact level)."""
    connector_id = "wikidata_facts"
    source_type = "structured_fact"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def _resolve(self, label: str) -> str:
        wire = (f"{_WD_API}?action=wbsearchentities&search={urllib.parse.quote(label[:80])}"
                f"&language=en&format=json&limit=1")
        raw, status, err = _get(wire)
        if err or not raw:
            return ""
        try:
            hits = (json.loads(raw.decode("utf-8", "replace")) or {}).get("search") or []
            return str(hits[0]["id"]) if hits else ""
        except (ValueError, KeyError, IndexError):
            return ""

    def facts(self, entity_label: str, *, requirement_id: str = "", as_of_iso: str = "",
              k: int = 12) -> tuple:
        t0 = _time.time()
        label = str(entity_label).replace("_", " ").strip()
        qid = self._resolve(label)
        props = " ".join(f"wdt:{p}" for p in _WD_PROPS)
        sparql = (f"SELECT ?propLabel ?valLabel ?start WHERE {{ VALUES ?prop {{ {props} }} "
                  f"wd:{qid or 'Q0'} ?p ?stmt . ?prop wikibase:claim ?p . "
                  f"?stmt ?ps ?val . ?prop wikibase:statementProperty ?ps . "
                  f"OPTIONAL {{ ?stmt pq:P580 ?start }} "
                  f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }} }} LIMIT {k}')
        wire = f"{_WDQ}?query={urllib.parse.quote(sparql)}&format=json"
        tr = _trace(self.connector_id, requirement_id, f"facts:{label}", wire)
        if not qid:
            return [], _finish(tr, b"", None, "entity not resolved on wikidata", 0, self.store, t0)
        raw, status, err = _get(wire, timeout=20)
        items, perr = [], ""
        if raw and not err:
            try:
                rows = ((json.loads(raw.decode("utf-8", "replace")) or {})
                        .get("results", {}).get("bindings")) or []
                asof = str(as_of_iso)[:10]
                rank = 0
                for b in rows:
                    start = str((b.get("start") or {}).get("value", ""))[:10]
                    if asof and start and start > asof:
                        continue                               # fact begins after as_of → not usable
                    rank += 1
                    prop = str((b.get("propLabel") or {}).get("value", ""))
                    val = str((b.get("valLabel") or {}).get("value", ""))
                    items.append(DiscoveredItem(
                        connector_id=self.connector_id, requirement_id=requirement_id,
                        logical_query=f"facts:{label}", wire_url=wire, rank=rank,
                        title=f"{label} — {prop}: {val}"[:300],
                        link=f"https://www.wikidata.org/wiki/{qid}", source_name="wikidata",
                        feed_pubdate=start,
                        description=f"{prop}={val}" + (f" since {start}" if start else ""),
                        retrieved_at=_time.time()))
                    if rank >= k:
                        break
            except (ValueError, KeyError) as e:
                perr = f"{type(e).__name__}: {e}"[:150]
        return items, _finish(tr, raw, status, err, len(items), self.store, t0, perr)


#: keyword → World Bank indicator code (universal macro-constraint set)
_WB_INDICATORS = (("gdp", "NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)"),
                  ("inflation", "FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
                  ("military", "MS.MIL.XPND.GD.ZS", "Military expenditure (% of GDP)"),
                  ("defense", "MS.MIL.XPND.GD.ZS", "Military expenditure (% of GDP)"),
                  ("unemployment", "SL.UEM.TOTL.ZS", "Unemployment (% of labor force)"),
                  ("reserves", "FI.RES.TOTL.MO", "Total reserves in months of imports"),
                  ("debt", "GC.DOD.TOTL.GD.ZS", "Central government debt (% of GDP)"))


class WorldBankConnector:
    """Country macro constraints, no key. Country ids resolved from the API's own country list
    (cached in-process) — no hand country tables. Only observations dated <= the as_of year are
    returned (annual data; year-level as-of hygiene, verifier sees the year stamp)."""
    connector_id = "worldbank_indicators"
    source_type = "dataset_observation"
    _countries = None

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def _country_id(self, name: str) -> str:
        if WorldBankConnector._countries is None:
            raw, status, err = _get("https://api.worldbank.org/v2/country?format=json&per_page=400",
                                    timeout=20)
            table = {}
            if raw and not err:
                try:
                    for c in (json.loads(raw.decode("utf-8", "replace")) or [None, []])[1]:
                        table[str(c.get("name", "")).lower()] = str(c.get("iso2Code", ""))
                except (ValueError, KeyError, IndexError, TypeError):
                    table = {}
            WorldBankConnector._countries = table
        low = str(name).replace("_", " ").lower().strip()
        table = WorldBankConnector._countries
        if low in table:
            return table[low]
        for full, code in table.items():
            if low and (low in full or full in low):
                return code
        return ""

    def series(self, country_name: str, indicator_terms: str, *, requirement_id: str = "",
               as_of_iso: str = "", years: int = 6) -> tuple:
        t0 = _time.time()
        terms = str(indicator_terms).lower()
        matches = [(code, label) for kw, code, label in _WB_INDICATORS if kw in terms] \
            or [(_WB_INDICATORS[0][1], _WB_INDICATORS[0][2])]
        cid = self._country_id(country_name)
        asof_year = int(str(as_of_iso)[:4]) if str(as_of_iso)[:4].isdigit() else None
        code, label = matches[0]
        y1 = asof_year or 2026
        wire = (f"https://api.worldbank.org/v2/country/{cid or 'XX'}/indicator/{code}"
                f"?format=json&per_page={years}&date={y1 - years + 1}:{y1}")
        tr = _trace(self.connector_id, requirement_id, f"{country_name} {label}", wire)
        if not cid:
            return [], _finish(tr, b"", None, f"country not resolved: {country_name}", 0,
                               self.store, t0)
        raw, status, err = _get(wire, timeout=20)
        items, perr = [], ""
        if raw and not err:
            try:
                body = json.loads(raw.decode("utf-8", "replace"))
                obs = body[1] if isinstance(body, list) and len(body) > 1 and body[1] else []
                rank = 0
                for o in obs:
                    yr, val = str(o.get("date", "")), o.get("value")
                    if val is None or (asof_year and yr.isdigit() and int(yr) > asof_year):
                        continue
                    rank += 1
                    items.append(DiscoveredItem(
                        connector_id=self.connector_id, requirement_id=requirement_id,
                        logical_query=f"{country_name} {label}", wire_url=wire, rank=rank,
                        title=f"{country_name} {label} {yr}: {round(float(val), 2)}"[:300],
                        link=wire, source_name="worldbank", feed_pubdate=yr,
                        description=f"{label}={val} ({yr})", retrieved_at=_time.time()))
            except (ValueError, KeyError, TypeError) as e:
                perr = f"{type(e).__name__}: {e}"[:150]
        return items, _finish(tr, raw, status, err, len(items), self.store, t0, perr)


class ReliefWebConnector:
    """Conflict/humanitarian situation reports (v2 API; appname string, no key). Date-bounded by
    the API's own filter — paired-date semantics like the news connectors."""
    connector_id = "reliefweb_v2"
    source_type = "official_report"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def search_historical(self, query_terms: str, *, after_date: str, before_date: str,
                          requirement_id: str = "", k: int = 8) -> tuple:
        t0 = _time.time()
        q = urllib.parse.quote(query_terms[:120])
        wire = (f"https://api.reliefweb.int/v2/reports?appname=sworldmodel&query[value]={q}"
                f"&limit={int(k)}&fields[include][]=title&fields[include][]=date"
                f"&fields[include][]=url&fields[include][]=source"
                f"&filter[field]=date.created&filter[value][from]={after_date}T00:00:00%2B00:00"
                f"&filter[value][to]={before_date}T23:59:59%2B00:00")
        tr = _trace(self.connector_id, requirement_id, query_terms, wire,
                    after_date=after_date, before_date=before_date)
        raw, status, err = _get(wire, timeout=20)
        items, perr = [], ""
        if raw and not err:
            try:
                data = (json.loads(raw.decode("utf-8", "replace")) or {}).get("data") or []
                for i, d in enumerate(data[:k], start=1):
                    f = d.get("fields") or {}
                    created = str(((f.get("date") or {}).get("created")) or "")
                    ts = None
                    if created[:10]:
                        try:
                            ts = _time.mktime(_time.strptime(created[:10], "%Y-%m-%d"))
                        except ValueError:
                            ts = None
                    src = ", ".join(s.get("shortname") or s.get("name", "")
                                    for s in (f.get("source") or [])[:2])
                    items.append(DiscoveredItem(
                        connector_id=self.connector_id, requirement_id=requirement_id,
                        logical_query=query_terms, wire_url=wire, rank=i,
                        title=str(f.get("title", ""))[:300], link=str(f.get("url", "")),
                        source_name=src or "reliefweb", feed_pubdate=created,
                        feed_pubdate_ts=ts, retrieved_at=_time.time()))
            except (ValueError, KeyError, TypeError) as e:
                perr = f"{type(e).__name__}: {e}"[:150]
        return items, _finish(tr, raw, status, err, len(items), self.store, t0, perr)


class UcdpGedConnector:
    """UCDP georeferenced battle events. The GED API now requires a token that is FREE for
    research use — read from UCDP_ACCESS_TOKEN; when unset the connector reports auth_required and
    returns nothing (a free registration is allowed, a paid dependency is not)."""
    connector_id = "ucdp_ged"
    source_type = "dataset_observation"

    def __init__(self, store: RawContentStore | None = None):
        self.store = store or RawContentStore()

    def events(self, country_or_query: str, *, after_date: str, before_date: str,
               requirement_id: str = "", k: int = 10) -> tuple:
        import os
        t0 = _time.time()
        wire = (f"https://ucdpapi.pcr.uu.se/api/gedevents/25.1?pagesize={int(k)}"
                f"&StartDate={after_date}&EndDate={before_date}")
        tr = _trace(self.connector_id, requirement_id, country_or_query, wire,
                    after_date=after_date, before_date=before_date)
        token = os.environ.get("UCDP_ACCESS_TOKEN", "").strip()
        if not token:
            tr.connector_status = "auth_required"
            tr.error = "UCDP_ACCESS_TOKEN unset (free research token: ucdp.uu.se) — source skipped"
            tr.latency_s = round(_time.time() - t0, 3)
            return [], tr
        raw, status, err = _get(wire, headers={"x-ucdp-access-token": token}, timeout=20)
        items, perr = [], ""
        if raw and not err:
            try:
                rows = (json.loads(raw.decode("utf-8", "replace")) or {}).get("Result") or []
                low = str(country_or_query).replace("_", " ").lower()
                rank = 0
                for r in rows:
                    if low and low not in str(r.get("country", "")).lower() \
                            and low not in str(r.get("conflict_name", "")).lower():
                        continue
                    rank += 1
                    items.append(DiscoveredItem(
                        connector_id=self.connector_id, requirement_id=requirement_id,
                        logical_query=country_or_query, wire_url=wire, rank=rank,
                        title=(f"{r.get('conflict_name', '')} {r.get('date_start', '')[:10]}: "
                               f"{r.get('best', '?')} deaths")[:300],
                        link="https://ucdp.uu.se", source_name="ucdp_ged",
                        feed_pubdate=str(r.get("date_start", ""))[:10],
                        description=str(r.get("where_description", ""))[:300],
                        retrieved_at=_time.time()))
                    if rank >= k:
                        break
            except (ValueError, KeyError, TypeError) as e:
                perr = f"{type(e).__name__}: {e}"[:150]
        return items, _finish(tr, raw, status, err, len(items), self.store, t0, perr)


#: institutional primary sources by DOMAIN — feeds, not scenario lists; extend freely
CURATED_FEEDS = (
    {"url": "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
     "source": "un_news", "domains": ("conflict", "diplomacy", "humanitarian")},
    {"url": "https://www.understandingwar.org/feeds.xml",
     "source": "isw", "domains": ("conflict", "military")},
    {"url": "https://www.nato.int/cps/en/natohq/news.htm?format=rss",
     "source": "nato", "domains": ("military", "diplomacy")},
    {"url": "https://www.imf.org/en/News/RSS?Language=ENG",
     "source": "imf", "domains": ("economy",)},
    {"url": "https://www.ecb.europa.eu/rss/press.html",
     "source": "ecb", "domains": ("economy",)},
    {"url": "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=1&Site=945",
     "source": "us_dod", "domains": ("military",)},
)


class CuratedRssConnector:
    """Primary-institution feeds (ISW, UN, NATO, IMF, ECB, defence ministries). Items are filtered
    to claimed pubdate <= before_date AND keyword overlap with the query — a curated feed is
    domain-scoped, not query-scoped, so relevance filtering happens here. Individual feeds 403/404
    in some environments; each failure is its own recorded trace."""
    connector_id = "curated_rss"
    source_type = "institutional_feed"

    def __init__(self, store: RawContentStore | None = None, feeds: tuple = CURATED_FEEDS):
        self.store = store or RawContentStore()
        self.feeds = feeds

    def search_historical(self, query_terms: str, *, after_date: str, before_date: str,
                          requirement_id: str = "", domains: tuple = (), k: int = 8) -> tuple:
        from swm.world_model_v2.evidence_connectors import _rfc822_ts, _strip_html
        t0 = _time.time()
        toks = {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", query_terms)}
        feeds = [f for f in self.feeds if not domains or set(f["domains"]) & set(domains)]
        items, statuses = [], []
        wire_all = ";".join(f["url"] for f in feeds)
        tr = _trace(self.connector_id, requirement_id, query_terms, wire_all[:500],
                    after_date=after_date, before_date=before_date)
        lo = _time.mktime(_time.strptime(after_date, "%Y-%m-%d"))
        hi = _time.mktime(_time.strptime(before_date, "%Y-%m-%d")) + 86399.0
        for f in feeds:
            raw, status, err = _get(f["url"], timeout=12)
            statuses.append(f"{f['source']}:{status if not err else 'ERR'}")
            if err or not raw or (status or 0) >= 400:
                continue
            text = raw.decode("utf-8", "replace")
            for m in re.finditer(r"<item>(.*?)</item>", text, re.S)or []:
                block = m.group(1)
                title = _strip_html(re.search(r"<title>(.*?)</title>", block, re.S).group(1)
                                    if re.search(r"<title>(.*?)</title>", block, re.S) else "")
                link_m = re.search(r"<link>(.*?)</link>", block, re.S)
                pub_m = re.search(r"<pubDate>(.*?)</pubDate>", block, re.S)
                ts = _rfc822_ts(pub_m.group(1).strip()) if pub_m else None
                if ts is not None and not (lo <= ts <= hi):
                    continue                                   # outside the paired-date window
                ttl = {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", title)}
                if toks and len(toks & ttl) == 0:
                    continue                                   # no keyword overlap → not relevant
                items.append(DiscoveredItem(
                    connector_id=self.connector_id, requirement_id=requirement_id,
                    logical_query=query_terms, wire_url=f["url"], rank=len(items) + 1,
                    title=title[:300], link=link_m.group(1).strip() if link_m else f["url"],
                    source_name=f["source"], feed_pubdate=pub_m.group(1).strip() if pub_m else "",
                    feed_pubdate_ts=ts, retrieved_at=_time.time()))
                if len(items) >= k:
                    break
            if len(items) >= k:
                break
        tr.response_headers = {"feed_statuses": ",".join(statuses)[:400]}
        tr.n_items = len(items)
        tr.latency_s = round(_time.time() - t0, 3)
        tr.connector_status = "ok" if items else "zero_results"
        return items, tr


# ---------------------------------------------------------------------------- the router
_DOMAIN_LEX = (("conflict", ("war", "conflict", "military", "invasion", "ceasefire", "troops",
                             "offensive", "strike", "attack", "shipping", "missile", "battle")),
               ("economy", ("inflation", "gdp", "economy", "economic", "rate", "price", "market",
                            "unemployment", "recession", "tariff", "sanction")),
               ("diplomacy", ("treaty", "alliance", "summit", "negotiation", "talks", "agreement",
                              "deal", "accord", "diplomatic")),
               ("politics", ("election", "vote", "poll", "parliament", "congress", "senate",
                             "president", "minister", "coalition", "impeach")))

#: requirement-category detection over claim_or_quantity + preferred_source_types (universal).
#: ORDER MATTERS: the sharper categories come first — "approval poll numbers" is an opinion
#: requirement even though "numbers" also smells like a quantity.
_CATEGORY_LEX = (("opinion", ("opinion", "poll", "approval", "public support", "sentiment")),
                 ("calendar", ("schedule", "scheduled", "date", "calendar", "election", "vote on",
                               "session", "meeting", "summit", "deadline")),
                 ("capability", ("capability", "order of battle", "arsenal", "equipment", "aid",
                                 "weapons", "military capacity", "forces")),
                 ("quantity", ("count", "number", "level", "rate", "gdp", "inflation", "index",
                               "expenditure", "capacity", "strength", "size", "measurement",
                               "quantity", "how many", "troops", "budget")),
                 ("statement", ("statement", "stance", "intention", "said", "position",
                                "commitment", "declared")))


def domains_for(text: str) -> tuple:
    low = " " + str(text).lower()
    return tuple(d for d, kws in _DOMAIN_LEX if any(f" {k}" in low or f"{k} " in low for k in kws))


def category_for(req) -> str:
    low = f"{getattr(req, 'claim_or_quantity', '')} {' '.join(getattr(req, 'preferred_source_types', []))}".lower()
    for cat, kws in _CATEGORY_LEX:
        if any(k in low for k in kws):
            return cat
    return "statement"


class FreeSourceRouter:
    """requirement → the free connectors that can ANSWER its category. Returns
    [(connector_name, logical_query, kwargs)] — pure and unit-testable; the orchestrator executes.
    Universal: category and domain are lexical over the requirement's own text, never a scenario
    list. GDELT rides along for every statement/news-shaped requirement (breadth); structured
    sources activate for the categories news prose cannot answer."""

    def route(self, req, question: str, *, k_per_connector: int = 6) -> list:
        cat = category_for(req)
        doms = domains_for(f"{question} {req.claim_or_quantity}")
        ent = str((req.entity_scope or [""])[0]).replace("_", " ")
        need = str(req.claim_or_quantity)[:80]
        # Three tiers, interleaved breadth-first-then-structured: the orchestrator CAPS routes per
        # requirement, so the category's STRUCTURED source (the one answering what news prose
        # cannot) must never be starved by breadth routes.
        breadth, structured, domain_routes = [], [], []
        if cat in ("statement", "opinion"):
            breadth.append(("gdelt", f"{ent} {need}".strip(), {"k": k_per_connector}))
            breadth.append(("curated_rss", f"{ent} {need}".strip(),
                            {"domains": doms, "k": k_per_connector}))
        if cat == "opinion" and ent:
            structured.append(("wikipedia_search", f"opinion polling {ent}", {"k": 3}))
        if cat == "quantity":
            if ent:
                structured.append(("worldbank", ent, {"indicator_terms": need}))
            structured.append(("wikipedia_search", f"{ent} {need}".strip(), {"k": 3}))
        if cat == "capability" and ent:
            structured.append(("wikipedia_search", f"order of battle {ent}", {"k": 3}))
            structured.append(("wikipedia_search", f"list of military aid to {ent}", {"k": 3}))
        if cat == "calendar" and ent:
            structured.append(("wikidata", ent, {}))
            structured.append(("wikipedia_search", f"{ent} {need}".strip(), {"k": 3}))
        if ent and cat in ("statement", "capability"):
            structured.append(("wikidata", ent, {}))           # leadership/membership ground facts
        if "conflict" in doms:
            domain_routes.append(("ucdp", ent or need, {}))
            domain_routes.append(("reliefweb", f"{ent} {need}".strip(), {"k": 4}))
        routes = breadth[:1] + structured + breadth[1:] + domain_routes
        # dedupe by (connector, query), preserve order
        seen, out = set(), []
        for r in routes:
            key = (r[0], r[1])
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out
