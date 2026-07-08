"""General as-of retrieval store — the physical no-leakage guarantee (audit E.4, spec Phase 6).

Every retrieved item MUST carry a timestamp, and a query with `as_of=T` can only ever return items
whose timestamp is <= T. This is enforced at the store boundary: an item without a timestamp is
rejected at insert; a query without an as_of is rejected; and the filter is applied in code so the
retrieval "cannot see the future even if handed it" — the same principle as
`swm/ingestion/store.history_asof`, generalized to arbitrary typed context items (news, social,
entity events).

The store is the substrate the news/social/entity adapters build on, and the object the leakage
tests exercise. It is in-memory + JSONL-persistable (dependency-free); a production build swaps the
backend but keeps this exact contract.

Design choice: as-of is `timestamp <= as_of` (inclusive), because context "as of T" legitimately
includes information published at T. The prediction target's own resolution is always strictly
after T by construction, so this does not leak the label. (Contrast the training/label split in
`ingestion/store`, which uses strict `<` to exclude the outcome itself.)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


class LeakageError(ValueError):
    """Raised when an operation would allow future information to reach a prediction."""


@dataclass(frozen=True)
class ContextItem:
    """A timestamped, typed piece of retrievable context."""
    item_id: str
    timestamp: float                 # unix seconds; REQUIRED
    kind: str                        # "news" | "social" | "entity_event" | ...
    text: str = ""
    entities: tuple[str, ...] = ()   # entities/tickers/topics this item is about
    source: str = ""
    score: float = 0.0               # optional numeric payload (a market price, a metric, ...)
    meta: dict = field(default_factory=dict)


class AsOfStore:
    """Append-only, timestamped context store with a hard as-of read gate."""

    def __init__(self) -> None:
        self._items: list[ContextItem] = []

    # ---- writes ----
    def add(self, item: ContextItem) -> None:
        if item.timestamp is None:
            raise LeakageError(f"item {item.item_id!r} has no timestamp; refused (would be "
                               "un-gateable and could leak future info)")
        self._items.append(item)

    def add_many(self, items: list[ContextItem]) -> None:
        for it in items:
            self.add(it)

    # ---- as-of reads (the guarantee) ----
    def query(self, *, as_of: float, kind: str | None = None,
              entities: tuple[str, ...] | None = None, text_contains: str | None = None,
              k: int | None = None) -> list[ContextItem]:
        """Return items with timestamp <= as_of, most recent first. `as_of` is REQUIRED."""
        if as_of is None:
            raise LeakageError("query requires an explicit as_of; a query without it could leak the "
                               "future")
        ents = set(entities) if entities else None
        want = text_contains.lower() if text_contains else None
        out = []
        for it in self._items:
            if it.timestamp > as_of:                          # THE GATE
                continue
            if kind is not None and it.kind != kind:
                continue
            if ents is not None and not (ents & set(it.entities)):
                continue
            if want is not None and want not in it.text.lower():
                continue
            out.append(it)
        out.sort(key=lambda x: x.timestamp, reverse=True)
        return out[:k] if k else out

    def latest_before(self, *, as_of: float, kind: str | None = None,
                      entities: tuple[str, ...] | None = None) -> ContextItem | None:
        res = self.query(as_of=as_of, kind=kind, entities=entities, k=1)
        return res[0] if res else None

    def assert_no_leak(self, as_of: float, returned: list[ContextItem]) -> None:
        """Post-condition check callers/tests can run: every returned item is <= as_of."""
        bad = [it.item_id for it in returned if it.timestamp > as_of]
        if bad:
            raise LeakageError(f"LEAK: items after as_of={as_of} returned: {bad}")

    def __len__(self) -> int:
        return len(self._items)

    # ---- persistence ----
    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for it in self._items:
                f.write(json.dumps(asdict(it)) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AsOfStore":
        store = cls()
        p = Path(path)
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    d["entities"] = tuple(d.get("entities", ()))
                    store.add(ContextItem(**d))
        return store
