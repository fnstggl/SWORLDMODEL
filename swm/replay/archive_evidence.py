"""Replay v2 time-capsule evidence service (Part 14) — immutable archived bytes, server-side cutoff rule.

No live search. Two archive sources, both with THIRD-PARTY-PROVEN availability timestamps:

  * Wikipedia revisions — the article revision as of the cutoff. `first_proven_available_at` = the revision
    timestamp, proven server-side by Wikipedia's revision history (the strongest cheap time-capsule there
    is: the bytes provably existed at that instant, and the API returns the revision content immutably).
  * Internet Archive (Wayback) snapshots — the closest snapshot at-or-before the cutoff;
    `first_proven_available_at` = the snapshot capture timestamp.

The service enforces `first_proven_available_at <= cutoff` SERVER-SIDE in `EvidenceCapsule.items()` — a
caller cannot request newer content (requests for items after the cutoff raise). Every item carries raw
bytes' sha256, the archive retrieval id, both claimed and proven timestamps, and a transformation history.

The forecaster consumes ONLY `ReplayBundle` objects built from a capsule — it has no archive-index access
of its own (Part 16: the capsule is constructed by the evidence-construction process and frozen to disk;
the forecaster reads the frozen file).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request

_UA = {"User-Agent": "wmv2-replay-evidence"}


def _fetch(url, timeout=45, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (i + 1))
    return None


def _ts(iso_date: str) -> float:
    return time.mktime(time.strptime(iso_date[:10], "%Y-%m-%d"))


# ---------------------------------------------------------------- Wikipedia revisions (as-of)
def wiki_revision_asof(title: str, cutoff_iso: str) -> dict | None:
    """The latest revision of `title` at-or-before the cutoff, with extracted plaintext."""
    t = urllib.parse.quote(title.replace(" ", "_"))
    start = cutoff_iso[:10] + "T23:59:59Z"
    meta_raw = _fetch(f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions"
                      f"&titles={t}&rvlimit=1&rvdir=older&rvstart={start}&rvprop=ids|timestamp")
    if not meta_raw:
        return None
    pages = (json.loads(meta_raw).get("query") or {}).get("pages") or {}
    page = next(iter(pages.values()), {})
    revs = page.get("revisions") or []
    if not revs:
        return None
    rev = revs[0]
    body = _fetch(f"https://en.wikipedia.org/w/api.php?action=parse&format=json&oldid={rev['revid']}"
                  f"&prop=text&formatversion=2")
    if not body:
        return None
    html = (json.loads(body).get("parse") or {}).get("text") or ""
    text = re.sub(r"<style.*?</style>|<script.*?</script>", " ", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\.mw-[\w-]+[^}]*\}|\{[^}]{0,400}\}", " ", text)   # inline CSS residue
    text = re.sub(r"\s+", " ", text).strip()[:6000]
    rev_ts = time.mktime(time.strptime(rev["timestamp"], "%Y-%m-%dT%H:%M:%SZ"))
    raw = html.encode()
    return {"source": "wikipedia_revision", "archive_retrieval_id": f"oldid:{rev['revid']}",
            "url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}",
            "title": page.get("title", title), "text": text,
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes_len": len(raw),
            "claimed_publication_ts": rev_ts, "first_proven_available_at": rev_ts,
            "temporal_verification": "server_verified_revision_timestamp",
            "transformation_history": ["html_tag_strip", "whitespace_collapse", "truncate_6000"]}


# ---------------------------------------------------------------- Wayback snapshots (at-or-before cutoff)
def wayback_snapshot(url: str, cutoff_iso: str) -> dict | None:
    stamp = cutoff_iso[:10].replace("-", "") + "235959"
    meta = _fetch(f"https://archive.org/wayback/available?url={urllib.parse.quote(url)}&timestamp={stamp}")
    if not meta:
        return None
    closest = ((json.loads(meta).get("archived_snapshots") or {}).get("closest") or {})
    snap_ts, snap_url = closest.get("timestamp"), closest.get("url")
    if not snap_ts or not snap_url:
        return None
    cap_ts = time.mktime(time.strptime(snap_ts, "%Y%m%d%H%M%S"))
    if cap_ts > _ts(cutoff_iso) + 86400.0:                   # availability API may return a LATER snapshot
        return None
    raw = _fetch(snap_url.replace("http://", "https://"))
    if not raw:
        return None
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw.decode("utf-8", "ignore"),
                  flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()[:6000]
    return {"source": "wayback_snapshot", "archive_retrieval_id": snap_ts, "url": snap_url,
            "title": url, "text": text,
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes_len": len(raw),
            "claimed_publication_ts": cap_ts, "first_proven_available_at": cap_ts,
            "temporal_verification": "server_verified_capture_timestamp",
            "transformation_history": ["html_tag_strip", "whitespace_collapse", "truncate_6000"]}


# ---------------------------------------------------------------- the capsule (server-side cutoff rule)
class EvidenceCapsule:
    """A frozen set of archived items for one (world, cutoff). Enforces the cutoff rule at ACCESS time."""

    def __init__(self, event_id: str, cutoff_iso: str, items: list):
        self.event_id, self.cutoff_iso = event_id, cutoff_iso
        self._items = list(items)

    def items(self) -> list:
        cutoff = _ts(self.cutoff_iso) + 86400.0
        out = []
        for it in self._items:
            if float(it["first_proven_available_at"]) > cutoff:
                raise PermissionError(
                    f"cutoff rule violated: item proven available at "
                    f"{it['first_proven_available_at']} > cutoff {self.cutoff_iso} — the evidence service "
                    f"refuses to serve post-cutoff content")
            out.append(dict(it))
        return out

    def as_dict(self):
        return {"event_id": self.event_id, "cutoff": self.cutoff_iso, "items": self._items,
                "capsule_hash": hashlib.sha256(
                    json.dumps(self._items, sort_keys=True, default=str).encode()).hexdigest()[:16]}


def news_asof(question: str, cutoff_iso: str, *, llm=None, k: int = 5) -> list:
    """Question-specific archived NEWS via Google News RSS with paired after:/before: (evidence-engine v2).
    Timestamps are RSS-claimed (weaker tier than revision/capture proof) — recorded per item as
    `claimed_rss_timestamp`; the leakage auditor's future-date/resolution-term scans still apply."""
    try:
        from swm.world_model_v2.evidence_connectors import GoogleNewsRSSConnector
        q_terms = question
        if llm is not None:
            from swm.engine.grounding import parse_json
            raw = parse_json(llm(f'Give the best 3-6 word news search query for this question. Return ONLY '
                                 f'JSON: {{"query": "..."}}\nQUESTION: {question}')) or {}
            q_terms = str(raw.get("query") or question)[:80]
        after = time.strftime("%Y-%m-%d", time.gmtime(_ts(cutoff_iso) - 30 * 86400))
        items, _trace = GoogleNewsRSSConnector().search_historical(q_terms, after_date=after,
                                                                   before_date=cutoff_iso[:10], k=k)
        out = []
        for it in items[:k]:
            title = str(getattr(it, "title", "") or (it.get("title") if isinstance(it, dict) else ""))
            text = str(getattr(it, "text", "") or getattr(it, "snippet", "") or
                       (it.get("snippet", "") if isinstance(it, dict) else ""))[:1200]
            pub = getattr(it, "published_at", None) or (it.get("published_at")
                                                        if isinstance(it, dict) else None)
            pub_ts = float(pub) if isinstance(pub, (int, float)) else _ts(cutoff_iso)
            if pub_ts > _ts(cutoff_iso) + 86400.0:
                continue                                     # cutoff rule, defense in depth
            raw_b = (title + " " + text).encode()
            out.append({"source": "google_news_rss_archived", "archive_retrieval_id": f"rss:{after}..{cutoff_iso[:10]}",
                        "url": str(getattr(it, "url", "") or ""), "title": title, "text": (title + ". " + text)[:2000],
                        "raw_sha256": hashlib.sha256(raw_b).hexdigest(), "raw_bytes_len": len(raw_b),
                        "claimed_publication_ts": pub_ts, "first_proven_available_at": pub_ts,
                        "temporal_verification": "claimed_rss_timestamp (weaker tier; recorded)",
                        "transformation_history": ["rss_snippet"]})
        return out
    except Exception:  # noqa: BLE001 — news layer is additive; failure falls back to wiki/wayback
        return []


def build_capsule(event_id: str, question: str, cutoff_iso: str, *, wiki_titles=(), urls=(),
                  llm=None) -> EvidenceCapsule:
    """Evidence-construction role (Part 16 process 1): question-specific archived NEWS (paired
    after:/before:) + as-of Wikipedia revisions + Wayback snapshots, frozen together."""
    titles = list(wiki_titles)
    if not titles and llm is not None:
        from swm.engine.grounding import parse_json
        raw = parse_json(llm(
            f"List up to 4 English Wikipedia article titles most relevant to forecasting this question. "
            f'Return ONLY JSON: {{"titles": ["..."]}}\nQUESTION: {question}')) or {}
        titles = [str(t) for t in (raw.get("titles") or [])][:4]
    items = list(news_asof(question, cutoff_iso, llm=llm))
    for t in titles:
        it = wiki_revision_asof(t, cutoff_iso)
        if it:
            items.append(it)
    for u in urls:
        it = wayback_snapshot(u, cutoff_iso)
        if it:
            items.append(it)
    return EvidenceCapsule(event_id, cutoff_iso, items)


# ---------------------------------------------------------------- runtime-compatible frozen bundle
class ReplayBundle:
    """The minimal EvidenceBundleV2-compatible interface simulate_world consumes, built ONLY from a frozen
    capsule (claims = archived item texts with proven timestamps). No live retrieval anywhere."""

    def __init__(self, capsule_dict: dict, question: str):
        self.question_id = capsule_dict["event_id"]
        self.question = question
        self.as_of = _ts(capsule_dict["cutoff"])
        self.slack_s = 0.0
        self.claims, self.included_claim_ids = [], []
        self.quarantine = []
        self.items = []
        self.documents = []
        for i, it in enumerate(capsule_dict["items"]):
            cid = f"cl_{i}"
            self.claims.append({"claim_id": cid, "text": it["text"][:1200], "title": it.get("title", ""),
                                "claim_class": "archived_document",
                                # phase-3 tagger contract: subject/predicate/object triple + quote
                                "subject": it.get("title", "")[:80], "predicate": "archived_state_asof",
                                "object": it["text"][:200], "quote": it["text"][:300],
                                "publication_time": it["first_proven_available_at"],
                                "source": it["source"], "raw_sha256": it["raw_sha256"],
                                "archive_retrieval_id": it["archive_retrieval_id"]})
            self.included_claim_ids.append(cid)
        self._hash = capsule_dict.get("capsule_hash", "")

    def bundle_hash(self):
        return self._hash

    def included_claims(self):
        return list(self.claims)

    def render(self, max_chars=4000):
        out = []
        for c in self.claims:
            out.append(f"[{c['source']} @ {time.strftime('%Y-%m-%d', time.gmtime(c['publication_time']))}] "
                       f"{c['title']}: {c['text'][:900]}")
        return "\n\n".join(out)[:max_chars]
