"""As-of news/event context (spec Phase 6: the lever that closes the market information gap).

EXP-006 diagnosed the market loss as *information staleness*: the model's Jan-2026 cutoff cannot
see mid-2026 news the market has priced. The only fix is retrieval that feeds context up to the
forecast horizon T but never past it. This module is that interface — and it is built so it
*physically cannot* leak:

- All news comes from an `AsOfStore`; every item is timestamped; a query is gated at `as_of`.
- A `LiveNewsAdapter` protocol exists for a real timestamped news API, but the default guard
  `reject_untimestamped()` refuses any item without a publish time, so a naive live web search
  (which would surface the post-resolution outcome) cannot be wired in by accident.

Honest status: we have no licensed timestamped news corpus in this environment, so the *content*
side is BLOCKED-ON-CORPUS. What is real and tested here is the leakage-proof plumbing + a
market-derived "information signal" (the market's own as-of price is a timestamped, legitimately
pre-T datum) that the market-comparison harness can use to measure how much an as-of signal closes
the gap. No fake news is injected.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.retrieval.asof_store import AsOfStore, ContextItem, LeakageError


@dataclass
class NewsContext:
    """As-of news retrieval over an AsOfStore. Returns a compact context bundle for a forecast at T."""
    store: AsOfStore

    def retrieve(self, *, as_of: float, entities: tuple[str, ...] = (), query: str = "",
                 k: int = 8) -> dict:
        items = self.store.query(as_of=as_of, kind="news",
                                 entities=entities or None,
                                 text_contains=query or None, k=k)
        self.store.assert_no_leak(as_of, items)   # belt-and-suspenders
        return {
            "as_of": as_of,
            "n_items": len(items),
            "items": [{"ts": it.timestamp, "text": it.text, "source": it.source,
                       "score": it.score} for it in items],
            "latest_ts": max((it.timestamp for it in items), default=None),
            "staleness_days": (round((as_of - max((it.timestamp for it in items), default=as_of))
                                     / 86400.0, 2)),
        }


def reject_untimestamped(items: list[dict]) -> list[ContextItem]:
    """Guard for any external/live news source: convert raw items to ContextItems, REFUSING any
    without a publish timestamp. This is what stops a live web search from injecting future facts."""
    out = []
    for i, raw in enumerate(items):
        ts = raw.get("timestamp") or raw.get("published_ts")
        if ts is None:
            raise LeakageError(
                f"news item {raw.get('id', i)!r} has no publish timestamp; refused. Live search "
                "without timestamps can leak post-resolution outcomes and is not allowed.")
        out.append(ContextItem(
            item_id=str(raw.get("id", i)), timestamp=float(ts), kind="news",
            text=raw.get("text", raw.get("title", "")), source=raw.get("source", ""),
            entities=tuple(raw.get("entities", ())), score=float(raw.get("score", 0.0))))
    return out


class LiveNewsAdapter:
    """Protocol placeholder for a real timestamped news API. Intentionally NOT implemented against
    a live web search — it must return items with publish timestamps, run through
    `reject_untimestamped`, and be back-fillable into the AsOfStore before any forecast uses it.

    BLOCKED-ON-CORPUS: wire a licensed, timestamped news source here (e.g. GDELT, a news API with
    publish times, or a snapshotted archive). Until then the world model runs 'no retrieval'."""

    IMPLEMENTED = False

    def fetch(self, query: str, as_of: float) -> list[ContextItem]:  # pragma: no cover
        raise NotImplementedError(
            "LiveNewsAdapter is blocked on a timestamped news corpus. Backfill an AsOfStore with "
            "timestamped items and use NewsContext instead; do not wire a live untimestamped search "
            "(it would leak the future).")
