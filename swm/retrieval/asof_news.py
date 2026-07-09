"""As-of news retrieval — information parity, leakage-free.

The crowd beats our simulation because it had the NEWS available at the question's date; our model had only the
LLM's training-time general knowledge. This closes that gap: GDELT's news archive returns the headlines
published in a DATE WINDOW ending at the question's as-of moment — exactly what an informed forecaster knew
then, and nothing after. Feeding these to the simulation's input step gives the model the same information the
market had; if the architecture is sound, computing the outcome from that information (instead of guessing
from instinct + bias) should let it match — and then beat — the crowd.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import time
import urllib.parse
import urllib.request

_DOC = ("https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist&maxrecords={k}&format=json"
        "&sort=hybridrel&startdatetime={s}&enddatetime={e}")
_STOP = {"will", "the", "a", "an", "be", "is", "are", "to", "of", "in", "on", "by", "at", "for", "and", "or",
         "any", "have", "has", "get", "gets", "before", "after", "than", "this", "that", "there", "their",
         "would", "could", "should", "do", "does", "did", "than", "with", "from", "resolve", "market", "yes", "no"}


def _fmt(ts):
    return _dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y%m%d%H%M%S")


def _query(question):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'&.-]+", question) if w.lower() not in _STOP]
    return " ".join(words[:8]) or question[:60]


def asof_headlines(question, as_of_ts, *, days_back=30, k=6, retries=0, timeout=12):
    """Headlines published in [as_of − days_back, as_of] matching the question — as-of, leakage-free."""
    url = _DOC.format(q=urllib.parse.quote(_query(question)), k=k,
                      s=_fmt(as_of_ts - days_back * 86400), e=_fmt(as_of_ts))
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "swm-asofnews/1.0"})
            data = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            out, seen = [], set()
            for a in data.get("articles", []):
                t = (a.get("title") or "").strip()
                if t and t.isascii() and t.lower() not in seen and len(t) > 15:   # de-dup, drop non-latin noise
                    seen.add(t.lower())
                    out.append(t)
            return out[:k]
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return []
