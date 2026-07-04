"""As-of context retrieval (audit's section 6). No post-as_of leakage — enforced by timestamp.

The simulation must consume RETRIEVED context, not only LLM priors. For HN: an author's past
submissions, the domain's past posts, similar/topic history — all strictly before as_of. This is
what builds the EntityState/ContextState the transition model conditions on.

Everything here is a pure function of records already timestamped, filtered by `as_of`. If a caller
passes a record with timestamp >= as_of it is dropped, not used — the retrieval cannot see the
future even if handed it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PostRecord:
    entity_id: str
    timestamp: float
    score: float
    title: str
    domain: str


def as_of(records: list[PostRecord], t: float) -> list[PostRecord]:
    """The leakage gate: only records strictly before t are reachable."""
    return [r for r in records if r.timestamp < t]


def author_context(records: list[PostRecord], author: str, t: float) -> dict:
    past = [r for r in as_of(records, t) if r.entity_id == author]
    scores = sorted(r.score for r in past)
    return {
        "n_past": len(scores),
        "median_past": scores[len(scores) // 2] if scores else None,
        "max_past": scores[-1] if scores else None,
        "frac_ge10": sum(s >= 10 for s in scores) / len(scores) if scores else None,
        "last_ts": max((r.timestamp for r in past), default=None),
    }


def domain_context(records: list[PostRecord], domain: str, t: float) -> dict:
    past = [r for r in as_of(records, t) if r.domain == domain and domain]
    scores = [r.score for r in past]
    return {"n_domain": len(scores),
            "domain_mean_logscore": (sum(math.log1p(s) for s in scores) / len(scores))
            if scores else None}


def similar_posts(records: list[PostRecord], title_tokens: set[str], domain: str, t: float,
                  k: int = 20) -> list[PostRecord]:
    """Nearest historical posts by crude token/domain overlap, before as_of. Feeds a topic prior."""
    cands = as_of(records, t)
    def sim(r: PostRecord) -> float:
        toks = set(r.title.lower().split())
        jac = len(toks & title_tokens) / max(1, len(toks | title_tokens))
        return jac + (0.25 if r.domain == domain and domain else 0.0)
    return sorted(cands, key=sim, reverse=True)[:k]


def topic_context(similar: list[PostRecord]) -> dict:
    if not similar:
        return {"topic_hitrate": None, "topic_n": 0}
    return {"topic_hitrate": sum(r.score >= 10 for r in similar) / len(similar),
            "topic_n": len(similar)}
