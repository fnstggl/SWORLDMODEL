"""As-of entity context: an individual's history strictly before T (spec Phase 4).

The individual model conditions on the entity's own track record. This builds that record from any
timestamped event stream, gated at `as_of`, so the persona/posterior is always as-of correct. It is
the generic version of the HN author-prior and the email persona history — decoupled from channel.

Works off either raw `AsOfStore` entity-event items or a plain list of (timestamp, outcome) pairs,
so it serves both the synthetic individual harness and a real event store.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.retrieval.asof_store import AsOfStore


@dataclass
class EntityContext:
    """As-of history features for one entity."""
    store: AsOfStore | None = None

    def from_store(self, entity_id: str, as_of: float) -> dict:
        if self.store is None:
            raise ValueError("EntityContext.from_store needs an AsOfStore")
        items = self.store.query(as_of=as_of, kind="entity_event", entities=(entity_id,))
        self.store.assert_no_leak(as_of, items)
        return self._features([(it.timestamp, it.score) for it in items], as_of)

    def from_pairs(self, pairs: list[tuple[float, float]], as_of: float) -> dict:
        """pairs: (timestamp, outcome). Only those strictly before as_of are used."""
        past = [(ts, v) for ts, v in pairs if ts < as_of]
        return self._features(past, as_of)

    @staticmethod
    def _features(past: list[tuple[float, float]], as_of: float) -> dict:
        if not past:
            return {"n_past": 0, "mean_past": None, "max_past": None, "frac_hit_past": None,
                    "recency_days": None, "last_ts": None}
        vals = [v for _, v in past]
        last_ts = max(ts for ts, _ in past)
        return {
            "n_past": len(vals),
            "mean_past": sum(vals) / len(vals),
            "max_past": max(vals),
            "frac_hit_past": sum(1 for v in vals if v >= 1.0) / len(vals),
            "recency_days": round((as_of - last_ts) / 86400.0, 2),
            "last_ts": last_ts,
        }
