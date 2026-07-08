"""GDELT as-of news loader — the information-intake pipe for beating markets (spec: market intake).

The honest diagnosis (EXP-006): we lose to markets on *information staleness*, not reasoning. The
only fix is feeding the predictor real news up to the forecast horizon T but never past it. This
loader turns GDELT's free, timestamped global news index into `AsOfStore` items so a forecast at T
can retrieve exactly what was publishable by T — leakage-proof by construction (`reject_untimestamped`
refuses any article without a `seendate`, and the store's as-of gate drops anything after T).

GDELT DOC 2.0 (`api.gdeltproject.org`) is rate-limited and occasionally 429s; `fetch` retries and
degrades gracefully. This module is the pipe; the market re-run that USES it (as-of retrieval arm vs
market@T on thin markets) is the next experiment and is documented in
experiments/exp015_information_intake_and_markets.md.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from swm.retrieval.asof_store import AsOfStore, ContextItem
from swm.retrieval.news_context import reject_untimestamped

_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def _parse_seendate(s: str) -> float | None:
    """GDELT seendate 'YYYYMMDDTHHMMSSZ' or 'YYYYMMDDHHMMSS' -> unix ts (UTC)."""
    from datetime import datetime, timezone
    s = s.replace("T", "").replace("Z", "")
    try:
        return datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


@dataclass
class GDELTLoader:
    store: AsOfStore
    max_records: int = 75

    def fetch(self, query: str, *, start: float, end: float, retries: int = 4) -> int:
        """Fetch articles for `query` published in [start, end] and backfill the store. Returns the
        number added. `end` should be <= the forecast horizon T so nothing future is ingested."""
        from datetime import datetime, timezone
        fmt = lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        params = {"query": query, "mode": "artlist", "format": "json",
                  "maxrecords": str(self.max_records), "sort": "datedesc",
                  "startdatetime": fmt(start), "enddatetime": fmt(end)}
        url = f"{_API}?{urllib.parse.urlencode(params)}"
        for a in range(retries):
            try:
                raw = urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "swm-gdelt"}), timeout=30).read()
                data = json.loads(raw) if raw.strip().startswith(b"{") else {"articles": []}
                break
            except Exception:
                if a == retries - 1:
                    return 0
                time.sleep(1.5 * (2 ** a))
        arts = data.get("articles", []) or []
        rows = []
        for i, art in enumerate(arts):
            ts = _parse_seendate(art.get("seendate", "") or "")
            if ts is None or ts > end:            # hard gate: never ingest past the horizon
                continue
            rows.append({"id": f"gdelt-{art.get('url', i)}", "timestamp": ts,
                         "text": art.get("title", ""), "source": art.get("domain", ""),
                         "entities": tuple(w for w in query.lower().split() if len(w) > 2)})
        items = reject_untimestamped(rows)        # refuses anything without a publish time
        self.store.add_many(items)
        return len(items)

    def as_of_context(self, query: str, as_of: float, *, lookback_days: float = 14.0, k: int = 8):
        """Retrieve the news publishable by `as_of` for a query (fetch if not already cached)."""
        start = as_of - lookback_days * 86400
        if not self.store.query(as_of=as_of, kind="news",
                                entities=tuple(w for w in query.lower().split() if len(w) > 2)):
            self.fetch(query, start=start, end=as_of)
        items = self.store.query(as_of=as_of, kind="news", k=k)
        self.store.assert_no_leak(as_of, items)   # belt-and-suspenders
        return items
