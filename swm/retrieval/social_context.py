"""As-of social context: the population/community signals a post lands into (spec Phase 3/6).

Generalizes swm/retrieval/context.py (HN author/domain/topic history) into a reusable social-context
builder over any timestamped post stream. Everything is gated at `as_of` — a post's context is built
only from posts strictly before it — so features are leakage-free by construction.

Two consumers:
- the AGGREGATE world: topic salience, domain reputation, and recent-competition come from here.
- as-of retrieval for prediction: "what was the community doing around this topic just before T".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from swm.retrieval.context import PostRecord, as_of


@dataclass
class SocialContext:
    """As-of community context over a list of timestamped PostRecords."""
    records: list[PostRecord]

    def topic_state(self, title_tokens: set[str], domain: str, t: float, *, k: int = 30) -> dict:
        """Recent hit-rate and volume for posts similar to this one, strictly before t."""
        past = as_of(self.records, t)
        def sim(r: PostRecord) -> float:
            toks = set(r.title.lower().split())
            jac = len(toks & title_tokens) / max(1, len(toks | title_tokens))
            return jac + (0.25 if r.domain == domain and domain else 0.0)
        near = sorted(past, key=sim, reverse=True)[:k]
        if not near:
            return {"topic_hitrate": None, "topic_n": 0, "topic_mean_logscore": None}
        return {
            "topic_hitrate": sum(r.score >= 10 for r in near) / len(near),
            "topic_n": len(near),
            "topic_mean_logscore": sum(math.log1p(r.score) for r in near) / len(near),
        }

    def domain_state(self, domain: str, t: float) -> dict:
        past = [r for r in as_of(self.records, t) if r.domain == domain and domain]
        if not past:
            return {"domain_n": 0, "domain_mean_logscore": None, "domain_hitrate": None}
        return {
            "domain_n": len(past),
            "domain_mean_logscore": sum(math.log1p(r.score) for r in past) / len(past),
            "domain_hitrate": sum(r.score >= 40 for r in past) / len(past),
        }

    def competition(self, t: float, window_hours: float = 6.0) -> dict:
        """How many posts landed in the `window_hours` before t — the attention-competition proxy."""
        lo = t - window_hours * 3600.0
        recent = [r for r in self.records if lo <= r.timestamp < t]
        return {"competition_n": len(recent),
                "competition_rate_per_hr": len(recent) / max(1e-9, window_hours)}
