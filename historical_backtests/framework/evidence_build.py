"""Frozen archived-bytes evidence capsules per (case, cutoff) — retrieval-leakage-hardened.

Rules (frozen):
 - Query generation is DETERMINISTIC: proper nouns from the raw question + resolution-criterion
   nouns + the cutoff-anchored date window. No frontier model anywhere; no period model needed.
 - GDELT DOC + Google News RSS are DISCOVERY ONLY (candidate URLs). Present-day rankings,
   snippets, and cached text are never evidence. Every news item must resolve to a Wayback
   capture with server-verified timestamp <= cutoff; the ARCHIVED BYTES are the evidence, hashed.
   Items with no qualifying archived version are rejected (recorded).
 - Wikipedia evidence is the exact revision at/before the cutoff (revid + timestamp proof).
 - Contamination scrub on every item: post-cutoff dates and resolution/outcome language are
   rejected, with counts recorded per capsule.
 - `first_proven_available_at <= cutoff` is re-enforced at ACCESS time by EvidenceCapsule.
 - Capsules are frozen (sha256) BEFORE simulation; simulate_world receives only the frozen
   ReplayBundle (live retrieval disabled by construction).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swm.replay.archive_evidence import (EvidenceCapsule, ReplayBundle, _ts,
                                         wayback_snapshot, wiki_revision_asof)
from historical_backtests.framework.scales import proper_nouns

ROOT = Path(__file__).resolve().parents[1]
LOOKBACK_DAYS = 90
MAX_WAYBACK_LOOKUPS = 14
MAX_NEWS_ITEMS = 8
MAX_WIKI = 4

_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")
_OUTCOME_TOKENS = (" resolved ", " won the ", " has won", " was signed", " were signed",
                   " signed the ", " passed away", " died ", " was confirmed", " took effect",
                   " came into force", " ultimately ", " would go on to", " in the end ",
                   " finally ", " outcome was", " result was", " it happened", " has ended",
                   " officially ended", " succeeded in", " failed to pass", " was rejected",
                   " was approved", " was struck down", " stepped down on", " resigned on")


def _post_cutoff_dates(text: str, cutoff_ts: float) -> int:
    """Count explicit dates strictly after the cutoff (ISO + 'Month DD, YYYY' + bare years)."""
    n = 0
    for m in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text):
        try:
            if time.mktime(time.strptime(m.group(0), "%Y-%m-%d")) > cutoff_ts + 86400:
                n += 1
        except ValueError:
            pass
    for m in re.finditer(r"\b(" + "|".join(_MONTHS) + r")\s+\d{1,2},?\s+(20\d{2})\b",
                         text, re.I):
        try:
            mon = _MONTHS.index(m.group(1).lower()) + 1
            if time.mktime(time.strptime(f"{m.group(2)}-{mon:02d}-01", "%Y-%m-%d")) \
                    > cutoff_ts + 32 * 86400:
                n += 1
        except ValueError:
            pass
    cutoff_year = int(time.strftime("%Y", time.gmtime(cutoff_ts)))
    n += sum(1 for m in re.finditer(r"\b(20\d{2})\b", text) if int(m.group(1)) > cutoff_year)
    return n


def contamination_scrub(item: dict, cutoff_ts: float, criterion_nouns: list) -> str | None:
    """Reject reason or None. Applied to EVERY capsule item (answer-adjacency + future dates)."""
    text = " " + str(item.get("text", "")).lower() + " "
    nd = _post_cutoff_dates(text, cutoff_ts)
    if nd >= 2:
        return f"post_cutoff_dates:{nd}"
    if any(t in text for t in _OUTCOME_TOKENS):
        near = any(n.lower() in text for n in criterion_nouns[:4])
        if near:
            return "resolution_language_near_criterion_nouns"
    return None


def deterministic_queries(raw_question: str, criterion: str) -> list:
    nouns = proper_nouns(raw_question, k=6)
    crit_nouns = proper_nouns(criterion or "", k=4)
    qs = []
    if nouns:
        qs.append(" ".join(nouns[:3]))
    for n in nouns[:3]:
        qs.append(n)
    for n in crit_nouns[:2]:
        if n not in qs:
            qs.append(n)
    return qs[:4] or [raw_question[:60]]


def _discover_urls(queries: list, after: str, before: str) -> list:
    """Candidate URLs from GDELT (precise windows) + Google News RSS (recency) — discovery only."""
    urls, seen = [], set()
    try:
        from swm.world_model_v2.evidence_connectors_more import GdeltDocConnector
        g = GdeltDocConnector()
        for q in queries[:3]:
            try:
                items, _tr = g.search_historical(q, after_date=after, before_date=before, k=10)
            except Exception:  # noqa: BLE001
                items = []
            for it in items:
                u = str(getattr(it, "url", None) or (it.get("url") if isinstance(it, dict) else ""))
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
    except Exception:  # noqa: BLE001
        pass
    try:
        from swm.world_model_v2.evidence_connectors import GoogleNewsRSSConnector
        conn = GoogleNewsRSSConnector()
        for q in queries[:2]:
            try:
                got, _tr = conn.search_historical(q, after_date=after, before_date=before, k=8)
            except Exception:  # noqa: BLE001
                got = []
            for it in got:
                u = str(getattr(it, "url", None) or (it.get("url") if isinstance(it, dict) else "")
                        or (it.get("link") if isinstance(it, dict) else ""))
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
    except Exception:  # noqa: BLE001
        pass
    return urls


def build_capsule_for(case: dict, cutoff_iso: str, *, out_dir: Path) -> dict:
    """Build + freeze one capsule. Returns summary (never raises — a thin capsule is recorded,
    not hidden)."""
    cid, q = case["case_id"], case["raw_question"]
    cutoff_ts = _ts(cutoff_iso)
    after = time.strftime("%Y-%m-%d", time.gmtime(cutoff_ts - LOOKBACK_DAYS * 86400))
    before = cutoff_iso[:10]
    crit = case.get("resolution_criterion", "")
    nouns = proper_nouns(q, k=6)
    queries = deterministic_queries(q, crit)
    items, rejected = [], {"no_archived_version": 0, "contaminated": 0}
    for title in nouns[:MAX_WIKI]:                            # gold tier: as-of wiki revisions
        it = wiki_revision_asof(title, before)
        if it:
            reason = contamination_scrub(it, cutoff_ts, nouns)
            if reason:
                rejected["contaminated"] += 1
            else:
                items.append(it)
        time.sleep(0.3)
    urls = _discover_urls(queries, after, before)
    n_way = 0
    for u in urls:                                            # silver tier: wayback archived bytes
        if n_way >= MAX_WAYBACK_LOOKUPS or sum(1 for i in items
                                               if i["source"] == "wayback_snapshot") >= MAX_NEWS_ITEMS:
            break
        n_way += 1
        try:
            it = wayback_snapshot(u, before)
        except Exception:  # noqa: BLE001
            it = None
        if it is None:
            rejected["no_archived_version"] += 1
            continue
        reason = contamination_scrub(it, cutoff_ts, nouns)
        if reason:
            rejected["contaminated"] += 1
            continue
        items.append(it)
        time.sleep(0.4)
    cap = EvidenceCapsule(f"{cid}@{cutoff_iso[:10]}", before, items).as_dict()
    cap["queries_deterministic"] = queries
    cap["discovery_urls_considered"] = len(urls)
    cap["rejected"] = rejected
    cap["built_at"] = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cid}__{cutoff_iso[:10]}.json"
    path.write_text(json.dumps(cap, indent=1, default=str))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json.seal").write_text(json.dumps({"sha256": digest}))
    return {"case_id": cid, "cutoff": cutoff_iso, "n_items": len(items),
            "n_wiki": sum(1 for i in items if i["source"] == "wikipedia_revision"),
            "n_wayback": sum(1 for i in items if i["source"] == "wayback_snapshot"),
            "rejected": rejected, "sha256": digest[:16], "path": str(path)}


def load_bundle(case: dict, cutoff_iso: str, *, out_dir: Path) -> ReplayBundle:
    """Load the FROZEN capsule (seal-verified) into the runtime-compatible bundle. Raises when
    the capsule is missing — a row must fail closed rather than retrieve live."""
    path = out_dir / f"{case['case_id']}__{cutoff_iso[:10]}.json"
    seal = json.loads(path.with_suffix(".json.seal").read_text())["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != seal:
        raise RuntimeError(f"capsule tampered: {path.name}")
    cap = json.loads(path.read_text())
    return ReplayBundle(cap, case["raw_question"])
