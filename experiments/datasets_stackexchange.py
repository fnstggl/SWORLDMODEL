"""StackExchange question-answer dataset for the response backtest (fetched via the public API).

entity = asker (user), segment = primary tag, outcome = question got answered. Complements GitHub:
askers are mostly cold (one-off), tags are strong repeat segments — so this stresses the multilevel
(entity <- segment <- global) pooling. As-of correct: asker/tag answer-rate state is built only from
earlier questions in the time-ordered stream.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

CACHE = "data/so_questions.json"
MFN = ["title_len", "n_tags", "has_code_kw", "is_howto", "body_present"]
_API = "https://api.stackexchange.com/2.3/questions"


def _get(url, retries=5):
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "swm"}),
                                        timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(1.0 * (2 ** a))


def fetch(site="stackoverflow", pages=45, fromdate=1672531200, todate=1680307200):
    """Fetch questions in [fromdate,todate] (default Jan-Apr 2023). ~45 pages x 100 = 4500."""
    out = []
    for p in range(1, pages + 1):
        url = (f"{_API}?site={site}&page={p}&pagesize=100&order=asc&sort=creation"
               f"&fromdate={fromdate}&todate={todate}")   # default filter already has the fields we need
        d = _get(url)
        if not d or not d.get("items"):
            break
        out.extend(d["items"])
        if d.get("backoff"):
            time.sleep(d["backoff"] + 0.5)
        if not d.get("has_more"):
            break
        if p % 10 == 0:
            print(f"  fetched {len(out)} questions ({p} pages)")
    Path("data").mkdir(exist_ok=True)
    Path(CACHE).write_text(json.dumps(out))
    print(f"wrote {len(out)} questions -> {CACHE}")
    return out


def load_samples():
    if not Path(CACHE).exists():
        fetch()
    qs = json.loads(Path(CACHE).read_text())
    qs = [q for q in qs if q.get("owner", {}).get("user_id") and q.get("tags")]
    qs.sort(key=lambda q: q["creation_date"])
    samples = []
    for q in qs:
        title = q.get("title", "")
        t = title.lower()
        mf = {
            "title_len": min(1.0, len(title) / 80),
            "n_tags": min(1.0, len(q["tags"]) / 5.0),
            "has_code_kw": 1.0 if any(k in t for k in ("error", "how to", "python", "java", "sql",
                                                       "function", "class", "api")) else 0.0,
            "is_howto": 1.0 if t.startswith("how") else 0.0,
            "body_present": 1.0,
        }
        entity = f"u{q['owner']['user_id']}"          # asker
        segment = q["tags"][0]                          # primary tag
        outcome = 1 if q.get("is_answered") else 0
        samples.append((entity, segment, mf, outcome))
    return samples, MFN


if __name__ == "__main__":
    fetch()
