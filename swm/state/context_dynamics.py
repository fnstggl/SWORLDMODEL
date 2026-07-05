"""Context dynamics: topic/domain/attention state that EVOLVES over time (Phase 4).

The simulation and aggregate models condition on context that is itself moving: how hot a topic is,
how a domain's reputation trends, how much attention/competition is on the platform, and how novelty
fatigues as a theme saturates. This module tracks those as as-of, online time series so the state
fed to the simulator reflects "the world right now", not a static prior.

All updates are backward-looking (as-of): observe an outcome only after it happens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.transition.nonstationarity import DriftTracker


@dataclass
class _EMA:
    halflife: float
    value: float | None = None
    n: int = 0

    def observe(self, x: float) -> None:
        a = 1.0 - math.exp(-math.log(2) / self.halflife)
        self.value = x if self.value is None else self.value + a * (x - self.value)
        self.n += 1

    def get(self, default: float = 0.0) -> float:
        return self.value if self.value is not None else default


@dataclass
class ContextDynamics:
    """Evolving platform/topic/domain context, as-of."""
    topic_salience: dict[str, _EMA] = field(default_factory=dict)      # fast hit-rate EMA per topic
    domain_reputation: dict[str, _EMA] = field(default_factory=dict)   # slow quality EMA per domain
    topic_novelty: dict[str, float] = field(default_factory=dict)      # decays with repetition (fatigue)
    attention: _EMA = field(default_factory=lambda: _EMA(halflife=200))
    competition: _EMA = field(default_factory=lambda: _EMA(halflife=50))
    drift: DriftTracker = field(default_factory=DriftTracker)
    _recent_ts: list = field(default_factory=list)

    # ---- reads (as-of) ----
    def salience(self, topic: str) -> float:
        return self.topic_salience.get(topic, _EMA(30)).get(0.1)

    def reputation(self, domain: str) -> float:
        return self.domain_reputation.get(domain, _EMA(300)).get(0.0)

    def novelty(self, topic: str) -> float:
        return self.topic_novelty.get(topic, 1.0)

    def competition_now(self, ts: float, window_s: float = 6 * 3600) -> float:
        return sum(1 for t in self._recent_ts if ts - window_s <= t < ts)

    # ---- update (after an outcome) ----
    def observe(self, *, ts: float, topic: str, domain: str, magnitude: float,
                hit_threshold: float = 40.0) -> None:
        hit = 1.0 if magnitude >= hit_threshold else 0.0
        self.topic_salience.setdefault(topic, _EMA(30)).observe(hit)
        if domain:
            self.domain_reputation.setdefault(domain, _EMA(300)).observe(math.log1p(magnitude))
        # novelty fatigue: each post on a topic lowers its novelty a touch; recovers slowly toward 1
        cur = self.topic_novelty.get(topic, 1.0)
        self.topic_novelty[topic] = max(0.2, cur * 0.9)
        for k in list(self.topic_novelty):
            if k != topic:
                self.topic_novelty[k] = min(1.0, self.topic_novelty[k] + 0.01)
        self._recent_ts.append(ts)
        if len(self._recent_ts) > 5000:
            self._recent_ts = self._recent_ts[-3000:]
        self.attention.observe(1.0)
        self.drift.observe(hit)

    def state_summary(self, topic: str, domain: str, ts: float) -> dict[str, float]:
        return {"salience": round(self.salience(topic), 4), "reputation": round(self.reputation(domain), 4),
                "novelty": round(self.novelty(topic), 4), "competition": self.competition_now(ts),
                "drift": round(self.drift.indicator(), 4)}
