"""Live, KEYLESS web retrieval — evidence for the universal RetrievalGrounder.

The retrieval tier needs real as-of evidence to feed the LLM extractor. Two keyless backends work through the
proxy: DuckDuckGo Lite (search snippets — best-effort; the snippets are thin, so a keyed search API such as
Tavily/Brave/Serper is the production upgrade) and the Wikipedia REST summary (clean text for entities/facts).
`web_search_fn` combines them into the `search_fn(query, as_of)` that `WebRetriever` expects. Every call is
wrapped — a failure yields no passages, never an exception, so grounding degrades to the LLM's own knowledge
(with a wider, calibrated CI) rather than crashing.

Note on as-of: these keyless endpoints return CURRENT results, which is exactly right for grounding the PRESENT
(the actual product use — "what is the world now"). Historical as-of grounding for backtests needs a
time-scoped provider; that is a separate, keyed concern.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request

_DDG_LITE = "https://lite.duckduckgo.com/lite/?q={q}"
_WIKI = "https://en.wikipedia.org/api/rest_v1/page/summary/{t}"
_WIKI_SEARCH = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit=1"


def _get_text(url, timeout=15, agent="Mozilla/5.0"):
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def _strip(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def ddg_snippets(query, k=5):
    """Best-effort keyless search snippets from DuckDuckGo Lite."""
    try:
        t = _get_text(_DDG_LITE.format(q=urllib.parse.quote(query)))
    except Exception:
        return []
    snips = [_strip(s) for s in re.findall(r'class="result-snippet"[^>]*>(.*?)</td>', t, flags=re.S)]
    return [s for s in snips if len(s) > 20][:k]


def wikipedia_extract(query):
    """Clean summary text for the best-matching Wikipedia article (entity/definition facts)."""
    try:
        import json as _j
        hit = _j.loads(_get_text(_WIKI_SEARCH.format(q=urllib.parse.quote(query))))
        results = hit.get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"].replace(" ", "_")
        summ = _j.loads(_get_text(_WIKI.format(t=urllib.parse.quote(title))))
        return summ.get("extract")
    except Exception:
        return None


def web_search_fn(k=5):
    """A `search_fn(query, as_of) -> [passage, ...]` for `WebRetriever`, combining DDG Lite + Wikipedia."""
    def search(query, as_of=None):
        passages = list(ddg_snippets(query, k=k))
        wiki = wikipedia_extract(query)
        if wiki:
            passages.append(f"[Wikipedia] {wiki[:600]}")
        return passages
    return search
