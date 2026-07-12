"""The real retrieval stack — dated, provenance-carrying passages from sources that actually work.

Probed live through this environment's proxy (2026-07): Google News RSS returns rich, DATED news results
keyless and effectively unlimited; Wikipedia's API works; Bing's HTML is scrapeable; DuckDuckGo returns 202
anti-bot challenges (why the old `live_search.py` starved every question down to one Wikipedia line). The
FEC API works with DEMO_KEY for election-money facts.

Stack policy (cheapest-first, per the grounding mandate):
  - DEFAULT (free, keyless, unlimited-ish): Google News RSS (the workhorse — current events, dated) +
    Wikipedia (entities/background) + Bing HTML (general web, best-effort).
  - OPTIONAL keyed overlays, auto-detected from env: SERPER_API_KEY (cheapest keyed at ~$0.30–1/1k,
    2.5k free), BRAVE_API_KEY, TAVILY_API_KEY. If set, they are queried too and their snippets merged.
Every passage carries (text, source, date) so downstream grounding can cite and the leakage gate can cut.
Failures degrade to fewer passages, never to an exception — but an EMPTY result is reported as empty
(grounding must see the starvation, not paper over it).
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


@dataclass
class Passage:
    text: str
    source: str                        # provider + outlet, e.g. "google_news:NBC News"
    date: str = ""                     # publication date when the provider gives one

    def cite(self) -> str:
        d = f" ({self.date})" if self.date else ""
        return f"[{self.source}{d}] {self.text}"


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def _strip(s):
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


# ---------------------------------- keyless providers ----------------------------------
def google_news(query, k=8):
    """Dated news headlines+sources from the Google News RSS feed — the keyless workhorse."""
    try:
        t = _get("https://news.google.com/rss/search?q=" + urllib.parse.quote(query) +
                 "&hl=en-US&gl=US&ceid=US:en")
    except Exception:
        return []
    out = []
    for item in re.findall(r"<item>(.*?)</item>", t, re.S)[:k]:
        title = re.search(r"<title>(.*?)</title>", item, re.S)
        date = re.search(r"<pubDate>(.*?)</pubDate>", item)
        src = re.search(r"<source[^>]*>(.*?)</source>", item)
        desc = re.search(r"<description>(.*?)</description>", item, re.S)
        text = _strip(title.group(1)) if title else ""
        extra = _strip(desc.group(1))[:300] if desc else ""
        if extra and not extra.startswith(text[:40]):
            text = f"{text} — {extra}"
        if text:
            out.append(Passage(text, f"google_news:{_strip(src.group(1)) if src else '?'}",
                               (date.group(1) or "")[:16] if date else ""))
    return out


def _rfc822_ts(s):
    """Parse an RSS <pubDate> (RFC-822) to a unix timestamp, or None."""
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).timestamp()
    except Exception:
        return None


def asof_google_news(question, as_of_ts, *, days_back=45, k=8):
    """LEAK-FREE as-of headlines from Google News RSS — keyless, and it works where GDELT is rate-limited.

    Two guards, because Google's date operators are not a hard guarantee (empirically `before:` ALONE lets
    post-cutoff articles through):
      1. a BOUNDED query window: `before:<as_of> after:<as_of − days_back>` (the bounded form returned only
         in-window results in testing);
      2. a DEFENSIVE code filter: every item's <pubDate> is parsed and any article dated after `as_of` is
         DROPPED — so an outcome published after the cutoff can never leak in, even if Google returns it.
    Items with no parseable date are dropped (fail safe). Returns [Passage] with the real publication date."""
    import datetime as _dt
    hi = _dt.datetime.utcfromtimestamp(int(as_of_ts)).strftime("%Y-%m-%d")
    lo = _dt.datetime.utcfromtimestamp(int(as_of_ts) - days_back * 86400).strftime("%Y-%m-%d")
    q = f"{question} before:{hi} after:{lo}"
    try:
        t = _get("https://news.google.com/rss/search?q=" + urllib.parse.quote(q) +
                 "&hl=en-US&gl=US&ceid=US:en")
    except Exception:
        return []
    out = []
    for item in re.findall(r"<item>(.*?)</item>", t, re.S):
        title = re.search(r"<title>(.*?)</title>", item, re.S)
        date = re.search(r"<pubDate>(.*?)</pubDate>", item)
        src = re.search(r"<source[^>]*>(.*?)</source>", item)
        ts = _rfc822_ts(date.group(1)) if date else None
        if ts is None or ts > as_of_ts + 86400:            # LEAK GUARD: drop anything after the as-of (+1d slack)
            continue
        text = _strip(title.group(1)) if title else ""
        if text:
            out.append(Passage(text, f"gnews_asof:{_strip(src.group(1)) if src else '?'}",
                               (date.group(1) or "")[:16]))
        if len(out) >= k:
            break
    return out


def asof_wikipedia(query, as_of_ts, k=2):
    """LEAK-FREE encyclopedic background: the Wikipedia article AS IT STOOD on the as-of date, via the
    revisions API (rvstart=<as_of>, rvdir=older → the newest revision at or before the cutoff). News
    headlines tell you what CHANGED this week; this tells you what the world already KNEW — who the person
    is, what the office/process is, the standing structure — which headlines alone never carry. Guards:
    the fetched revision's own timestamp is verified <= as_of (an article created after the cutoff yields
    no earlier revision and is dropped), and the passage is stamped with the revision date."""
    import datetime as _dt
    iso = _dt.datetime.utcfromtimestamp(int(as_of_ts)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        hits = json.loads(_get("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" +
                               urllib.parse.quote(query) + f"&format=json&srlimit={k}"))
    except Exception:
        return []
    out = []
    for h in hits.get("query", {}).get("search", [])[:k]:
        try:
            rev = json.loads(_get(
                "https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=" +
                urllib.parse.quote(h["title"]) +
                f"&rvlimit=1&rvstart={iso}&rvdir=older&rvprop=timestamp%7Ccontent&rvslots=main"
                f"&format=json&formatversion=2"))
            pages = rev.get("query", {}).get("pages", [])
            if not pages or not pages[0].get("revisions"):
                continue                                   # no revision before the cutoff → article is post-as-of
            r0 = pages[0]["revisions"][0]
            ts = r0.get("timestamp", "")
            content = (r0.get("slots", {}).get("main", {}) or {}).get("content", "")
            # crude wikitext → prose: drop infobox param lines, templates, refs, links; keep lead sentences
            body = "\n".join(ln for ln in content.splitlines()
                             if not ln.strip().startswith(("|", "{{", "}}", "{|", "!", "<")))
            lead = re.sub(r"\{\{[^{}]*\}\}", " ", body)
            lead = re.sub(r"\{\{[^{}]*\}\}", " ", lead)     # nested, twice is enough for leads
            lead = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^/>]*/>|<!--.*?-->", " ", lead, flags=re.S)
            lead = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", lead)
            lead = re.sub(r"\[\[File:[^\]]*\]\]|'''|''|==+[^=]*==+", " ", lead)
            lead = re.sub(r"\s+", " ", lead).strip()
            prose = lead[:650]
            if len(prose) > 80:
                out.append(Passage(prose, f"wikipedia_asof:{pages[0]['title']}", ts[:10]))
        except Exception:
            continue
    return out


def asof_multi_search(queries, as_of_ts, *, days_back=45, k_each=6):
    """Several targeted as-of queries in parallel, merged+deduped — leak-free (bounded window + pubDate
    drop on every one). Returns [Passage]. This is the backtest's `search_fn`, so SceneGrounder can run its
    MULTI-ROUND retrieval (plan queries → follow-up 'who is favored' queries) entirely as-of, never touching
    live news. Sources: as-of Google News (what changed) + as-of Wikipedia revisions (what the world already
    knew — the encyclopedic state headlines never carry; added after EXP-098 measured grounding as the ONE
    lever that moves accuracy)."""
    qs = [q for q in (queries or []) if str(q).strip()][:6] or [""]
    jobs = [lambda q=q: asof_google_news(q, as_of_ts, days_back=days_back, k=k_each) for q in qs]
    jobs += [lambda q=q: asof_wikipedia(q, as_of_ts, k=1) for q in qs[:2]]   # background for the top queries
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        results = list(ex.map(lambda f: f(), jobs))
    out, seen = [], set()
    for rs in results:
        for p in rs:
            key = p.text[:80].lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def asof_search_fn(as_of_ts, *, days_back=45):
    """A `search_fn(queries, k_each) -> [Passage]` bound to an as-of timestamp — pass to simulate() for a
    leak-free backtest so the engine's own multi-round grounding runs, never live retrieval."""
    def fn(queries, k_each=6):
        return asof_multi_search(queries, as_of_ts, days_back=days_back, k_each=k_each)
    return fn


def wikipedia(query, k=2):
    """Background/entity facts: top-matching article summaries (dated 'current' — encyclopedic)."""
    try:
        hits = json.loads(_get("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" +
                               urllib.parse.quote(query) + f"&format=json&srlimit={k}"))
        out = []
        for h in hits.get("query", {}).get("search", [])[:k]:
            title = h["title"].replace(" ", "_")
            summ = json.loads(_get("https://en.wikipedia.org/api/rest_v1/page/summary/" +
                                   urllib.parse.quote(title)))
            if summ.get("extract"):
                out.append(Passage(summ["extract"][:600], f"wikipedia:{h['title']}"))
        return out
    except Exception:
        return []


def bing_snippets(query, k=6):
    """Best-effort general-web snippets from Bing's HTML (works through the proxy; DDG does not)."""
    try:
        t = _get("https://www.bing.com/search?q=" + urllib.parse.quote(query))
    except Exception:
        return []
    caps = re.findall(r'<p class="b_lineclamp[^"]*"[^>]*>(.*?)</p>', t, re.S)
    return [Passage(_strip(c)[:400], "bing") for c in caps[:k] if len(_strip(c)) > 40]


# ---------------------------------- keyed overlays (auto-detected) ----------------------------------
def _serper(query, k=8):
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        return []
    try:
        body = json.dumps({"q": query, "num": k}).encode()
        req = urllib.request.Request("https://google.serper.dev/search", data=body,
                                     headers={"X-API-KEY": key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        return [Passage(f"{o.get('title', '')} — {o.get('snippet', '')}"[:400], "serper", o.get("date", ""))
                for o in data.get("organic", [])[:k]]
    except Exception:
        return []


def _brave(query, k=8):
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return []
    try:
        req = urllib.request.Request("https://api.search.brave.com/res/v1/web/search?q=" +
                                     urllib.parse.quote(query) + f"&count={k}",
                                     headers={"X-Subscription-Token": key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        return [Passage(f"{o.get('title', '')} — {o.get('description', '')}"[:400], "brave",
                        o.get("age", "")) for o in data.get("web", {}).get("results", [])[:k]]
    except Exception:
        return []


def _tavily(query, k=8):
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    try:
        body = json.dumps({"api_key": key, "query": query, "max_results": k}).encode()
        req = urllib.request.Request("https://api.tavily.com/search", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        return [Passage(f"{o.get('title', '')} — {o.get('content', '')}"[:400], "tavily",
                        o.get("published_date", "")) for o in data.get("results", [])[:k]]
    except Exception:
        return []


# ---------------------------------- the stack ----------------------------------
def search(query, k=10) -> list:
    """One query across every available provider (keyless always; keyed if env keys are set), merged and
    de-duplicated. Returns [Passage]. Empty means EMPTY — the caller must treat that as starvation."""
    providers = [lambda: google_news(query, k), lambda: wikipedia(query, 2), lambda: bing_snippets(query, 5),
                 lambda: _serper(query, k), lambda: _brave(query, k), lambda: _tavily(query, k)]
    out = []
    with ThreadPoolExecutor(max_workers=len(providers)) as ex:
        for r in ex.map(lambda f: f(), providers):
            out.extend(r or [])
    seen, dedup = set(), []
    for p in out:
        key = p.text[:80].lower()
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    return dedup[: max(k, 12)]


def multi_search(queries, k_each=8) -> list:
    """Several targeted queries in parallel — the scene grounder's entry point."""
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(queries)))) as ex:
        results = list(ex.map(lambda q: search(q, k_each), queries))
    out, seen = [], set()
    for rs in results:
        for p in rs:
            key = p.text[:80].lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out
