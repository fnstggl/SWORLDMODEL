"""Rolling entity state from the full historical event sequence (spec Phase 2).

The failure audit's second charge: entity histories were shallow (a couple of EMA scalars) and the
benchmark never conditioned on how MUCH history an entity had. This module builds a rich, as-of
rolling state for an entity from its full past-event sequence, and — crucially — a **state
sufficiency score** and cold/warm flags so the system (and the gate) can tell when it actually knows
the entity vs. when it is guessing from the segment prior.

Everything is as-of: an entity's state at time t is built only from its events strictly before t.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EntityHistory:
    """Rolling, as-of state for one entity (e.g. an HN author) from its event sequence.

    `events` are (timestamp, outcome_magnitude) sorted ascending. All features are computed from
    events strictly before the query time by construction (feed only past events)."""
    entity_id: str
    events: list[tuple[float, float]] = field(default_factory=list)  # (ts, magnitude)

    def observe(self, ts: float, magnitude: float) -> None:
        self.events.append((ts, magnitude))

    # ---- depth / sufficiency ----
    @property
    def depth(self) -> int:
        return len(self.events)

    def is_cold(self, k: int = 1) -> bool:
        return self.depth < k

    def tier(self) -> str:
        d = self.depth
        return "cold" if d < 1 else "shallow" if d < 3 else "medium" if d < 8 else "deep"

    def sufficiency(self) -> float:
        """State-sufficiency score in [0,1]: how much we can trust an individual estimate. Grows
        with count (evidence) and shrinks with the entity's own outcome variance (noisy history is
        worth less). Saturating in depth."""
        d = self.depth
        if d == 0:
            return 0.0
        count_term = 1.0 - math.exp(-d / 6.0)             # ~0 at 0, ~0.63 at 6, ~0.86 at 12
        mags = [math.log1p(m) for _, m in self.events]
        if d >= 2:
            mean = sum(mags) / d
            var = sum((m - mean) ** 2 for m in mags) / d
            consistency = 1.0 / (1.0 + var)               # low variance -> trustworthy
        else:
            consistency = 0.5
        return round(count_term * (0.5 + 0.5 * consistency), 4)

    # ---- rich rolling features (as-of) ----
    def features(self, now: float | None = None, hit_threshold: float = 40.0) -> dict[str, float]:
        d = self.depth
        if d == 0:
            return {"eh_depth": 0.0, "eh_log_depth": 0.0, "eh_mean_logscore": 0.0,
                    "eh_max_logscore": 0.0, "eh_frac_hit": 0.0, "eh_recency_log": 12.0,
                    "eh_trend": 0.0, "eh_consistency": 0.0, "eh_sufficiency": 0.0,
                    "eh_ewma_logscore": 0.0}
        mags = [m for _, m in self.events]
        logs = [math.log1p(m) for m in mags]
        last_ts = self.events[-1][0]
        # EWMA over the sequence (recency-weighted quality)
        ewma, alpha = logs[0], 0.4
        for v in logs[1:]:
            ewma = alpha * v + (1 - alpha) * ewma
        # linear trend of logscore over its own index (are they improving?)
        n = d
        xs = list(range(n))
        mx, my = sum(xs) / n, sum(logs) / n
        denom = sum((x - mx) ** 2 for x in xs) or 1.0
        trend = sum((xs[i] - mx) * (logs[i] - my) for i in range(n)) / denom
        mean = sum(logs) / n
        var = sum((v - mean) ** 2 for v in logs) / n
        recency = math.log1p(max(0.0, (now - last_ts))) if now is not None else 0.0
        return {
            "eh_depth": float(d),
            "eh_log_depth": math.log1p(d),
            "eh_mean_logscore": mean,
            "eh_max_logscore": max(logs),
            "eh_frac_hit": sum(1 for m in mags if m >= hit_threshold) / d,
            "eh_recency_log": recency,
            "eh_trend": trend,
            "eh_consistency": 1.0 / (1.0 + var),
            "eh_sufficiency": self.sufficiency(),
            "eh_ewma_logscore": ewma,
        }


ENTITY_FEATURE_NAMES = ["eh_depth", "eh_log_depth", "eh_mean_logscore", "eh_max_logscore",
                        "eh_frac_hit", "eh_recency_log", "eh_trend", "eh_consistency",
                        "eh_sufficiency", "eh_ewma_logscore"]


class EntityHistoryStore:
    """Holds rolling histories for many entities; streaming, as-of correct (observe AFTER predict)."""
    def __init__(self) -> None:
        self._h: dict[str, EntityHistory] = {}

    def get(self, entity_id: str) -> EntityHistory:
        h = self._h.get(entity_id)
        if h is None:
            h = EntityHistory(entity_id)
            self._h[entity_id] = h
        return h

    def features(self, entity_id: str, now: float | None = None) -> dict[str, float]:
        return self.get(entity_id).features(now)

    def observe(self, entity_id: str, ts: float, magnitude: float) -> None:
        self.get(entity_id).observe(ts, magnitude)

    def repeat_actor_fraction(self, min_depth: int = 3) -> float:
        if not self._h:
            return 0.0
        return sum(1 for h in self._h.values() if h.depth >= min_depth) / len(self._h)
