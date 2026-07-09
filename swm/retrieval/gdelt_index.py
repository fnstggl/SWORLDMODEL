"""GDELT bulk index — as-of social-state grounding, no rate limit.

The GDELT API throttles hard, but the underlying data is free bulk: one CAMEO-coded EVENT file per day (global,
1979→present, ~150k events/day). This downloads a day's events, pre-aggregates them PER COUNTRY into a tiny
daily summary (mean tone, mean Goldstein conflict-cooperation scale, event volume, and CAMEO-root counts —
protests, assaults, diplomacy, fights), and caches it. `asof_social_state(country, date, window)` then reads
the cached summaries in [date−window, date] and returns the measured social state AS OF that date —
leakage-free, structured, and covering the whole world's political/social/economic event stream.

This is the "measure the current state of the social world" grounding for the social/geopolitical slice: a
question about a country's stability, conflict, protest risk, or sentiment can be simulated forward from its
REAL as-of trajectory (tone falling, conflict rising, protest volume spiking) instead of an LLM guess.

GDELT 1.0 export columns used: [1] Day, [7] Actor1CountryCode, [17] Actor2CountryCode, [28] EventRootCode,
[30] GoldsteinScale, [31] NumMentions, [34] AvgTone, [51] ActionGeo_CountryCode.
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

CACHE = Path("data/gdelt_cache")
_EXPORT = "http://data.gdeltproject.org/events/{d}.export.CSV.zip"
# CAMEO root codes -> human buckets (the quad-class-ish social signals)
_ROOTS = {"14": "protest", "17": "coerce", "18": "assault", "19": "fight", "20": "mass_violence",
          "13": "threaten", "10": "demand", "04": "consult", "05": "cooperate_diplo", "06": "cooperate_material",
          "03": "express_intent_coop", "11": "disapprove", "12": "reject"}


def _fmt(ts):
    return _dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y%m%d") if isinstance(ts, (int, float)) else str(ts)


def index_day(date_str, *, timeout=60):
    """Download one day's global events and pre-aggregate per country. Cached (tiny JSON). Returns the summary
    {country: {n, tone, goldstein, protest, assault, fight, ...}} or None on failure."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{date_str}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    try:
        req = urllib.request.Request(_EXPORT.format(d=date_str), headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=timeout).read()
        text = zipfile.ZipFile(io.BytesIO(raw)).read(zipfile.ZipFile(io.BytesIO(raw)).namelist()[0]).decode("utf-8", "ignore")
    except Exception:
        return None
    agg = defaultdict(lambda: {"n": 0, "tone_sum": 0.0, "gold_sum": 0.0, "mentions": 0,
                               **{v: 0 for v in set(_ROOTS.values())}})
    for line in text.split("\n"):
        c = line.split("\t")
        if len(c) < 52:
            continue
        try:
            country = c[51] or c[7]                        # where it happened, else actor1's country
            if not country:
                continue
            root, gold, ment, tone = c[28], float(c[30] or 0), int(c[31] or 0), float(c[34] or 0)
            a = agg[country]
            a["n"] += 1
            a["tone_sum"] += tone
            a["gold_sum"] += gold
            a["mentions"] += ment
            if root in _ROOTS:
                a[_ROOTS[root]] += 1
        except (ValueError, IndexError):
            continue
    summary = {country: {"n": a["n"], "tone": round(a["tone_sum"] / a["n"], 3),
                         "goldstein": round(a["gold_sum"] / a["n"], 3), "mentions": a["mentions"],
                         **{k: a[k] for k in set(_ROOTS.values())}}
               for country, a in agg.items() if a["n"] >= 3}
    cp.write_text(json.dumps(summary))
    return summary


def asof_social_state(country, as_of_ts, *, window_days=30):
    """The measured social state of `country` AS OF the date, aggregated over the prior window (leakage-free).
    Returns mean tone, mean Goldstein (conflict<0 / cooperation>0), event volume, and normalized protest /
    violence / diplomacy rates — plus their TREND vs the earlier half of the window (rising conflict, etc.)."""
    end = _dt.datetime.utcfromtimestamp(int(as_of_ts))
    days = [(end - _dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(window_days)]
    recent, earlier, tot = _acc(), _acc(), 0
    for i, d in enumerate(days):
        summ = index_day(d)
        if not summ or country not in summ:
            continue
        s = summ[country]
        bucket = recent if i < window_days // 2 else earlier
        for k in ("n", "tone_sum", "gold_sum", "protest", "assault", "fight", "mass_violence",
                  "consult", "cooperate_diplo"):
            if k == "tone_sum":
                bucket["tone_sum"] += s["tone"] * s["n"]
            elif k == "gold_sum":
                bucket["gold_sum"] += s["goldstein"] * s["n"]
            else:
                bucket[k] += s.get(k, 0)
        tot += s["n"]
    if tot < 10:
        return None
    r = _summ(recent)
    e = _summ(earlier)
    return {"country": country, "as_of": end.strftime("%Y-%m-%d"), "n_events": recent["n"] + earlier["n"],
            "tone": r["tone"], "goldstein": r["goldstein"], "protest_rate": r["protest_rate"],
            "violence_rate": r["violence_rate"], "diplomacy_rate": r["diplomacy_rate"],
            "tone_trend": round(r["tone"] - e["tone"], 3), "conflict_trend": round(e["goldstein"] - r["goldstein"], 3),
            "protest_trend": round(r["protest_rate"] - e["protest_rate"], 4)}


def _acc():
    return {"n": 0, "tone_sum": 0.0, "gold_sum": 0.0, "protest": 0, "assault": 0, "fight": 0,
            "mass_violence": 0, "consult": 0, "cooperate_diplo": 0}


def _summ(b):
    n = max(1, b["n"])
    return {"tone": round(b["tone_sum"] / n, 3), "goldstein": round(b["gold_sum"] / n, 3),
            "protest_rate": round(b["protest"] / n, 4),
            "violence_rate": round((b["assault"] + b["fight"] + b["mass_violence"]) / n, 4),
            "diplomacy_rate": round((b["consult"] + b["cooperate_diplo"]) / n, 4)}
