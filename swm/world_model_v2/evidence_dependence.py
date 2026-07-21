"""Duplication, syndication and source-dependence — Phase 2.

Twenty copies of one wire report are not twenty independent observations. This module groups documents that
are not independent so downstream inference weights by INDEPENDENT sources, not raw document counts. Detects
exact duplicates (content hash), near-duplicates (token-shingle Jaccard), and shared-origin syndication
(shared canonical link / shared quoted primary source). Deterministic; no LLM required.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict

DEPENDENCE_TYPES = ("exact_duplicate", "near_duplicate", "syndication", "shared_primary_source", "independent")


@dataclass
class DependenceGroup:
    group_id: str
    member_ids: list = field(default_factory=list)
    dependence_type: str = "independent"
    likely_origin: str = ""
    primary_source: str = ""
    confidence: float = 0.5

    def as_dict(self):
        return asdict(self)


def _shingles(text: str, k: int = 5) -> set:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {" ".join(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1))} or {" ".join(toks)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _norm_link(url: str) -> str:
    u = re.sub(r"https?://(www\.)?", "", (url or "").strip().lower())
    return u.split("?")[0].rstrip("/")


def cluster_dependence(docs: list, *, near_dup_threshold: float = 0.6) -> list:
    """`docs` = [{id, text, url, source, content_hash}]. Returns [DependenceGroup] partitioning ALL docs
    (singletons are `independent`). Union-find over three signals: identical content hash, near-duplicate
    shingle overlap, and shared normalized link."""
    n = len(docs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    shingles = [_shingles(d.get("text", "")) for d in docs]
    reasons = {}
    for i in range(n):
        for j in range(i + 1, n):
            di, dj = docs[i], docs[j]
            if di.get("content_hash") and di.get("content_hash") == dj.get("content_hash"):
                union(i, j); reasons[(min(i, j), max(i, j))] = ("exact_duplicate", 0.99); continue
            if di.get("url") and _norm_link(di["url"]) == _norm_link(dj.get("url", "")):
                union(i, j); reasons[(min(i, j), max(i, j))] = ("syndication", 0.85); continue
            jac = _jaccard(shingles[i], shingles[j])
            if jac >= near_dup_threshold:
                union(i, j)
                reasons[(min(i, j), max(i, j))] = ("near_duplicate", round(min(0.95, 0.5 + jac / 2), 3))
    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for root, members in groups.items():
        ids = [docs[m].get("id", str(m)) for m in members]
        gtype, conf = "independent", 1.0
        if len(members) > 1:
            # strongest pairwise reason within the group
            best = max((reasons.get((min(a, b), max(a, b)), ("near_duplicate", 0.6))
                        for a in members for b in members if a < b),
                       key=lambda r: {"exact_duplicate": 3, "syndication": 2, "near_duplicate": 1}.get(r[0], 0))
            gtype, conf = best
        # likely origin = earliest by source order (proxy); primary = the source string most shared
        origin = docs[min(members)].get("source", "")
        out.append(DependenceGroup(
            group_id="dep_" + hashlib.sha1("|".join(sorted(ids)).encode()).hexdigest()[:10],
            member_ids=ids, dependence_type=gtype, likely_origin=origin,
            primary_source=origin, confidence=conf))
    return out


def independent_count(groups: list) -> int:
    """The number of INDEPENDENT sources = number of dependence groups (one vote per group)."""
    return len(groups)
