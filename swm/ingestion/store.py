"""Append-only, timestamped event store with as-of reads (audit C.1).

Backs every backtest: reads must be restrictable to events with timestamp < T so a split
reconstructs "what was knowable at time T". Implement over Postgres/Parquet; the interface
is: append(Event) and read_asof(entity_id, before_ts) -> list[Event].

Stub — see docs/social-world-model-audit.md for the design. Not yet implemented."""

#: build-order and design are in docs/social-world-model-audit.md
IMPLEMENTED = False  # flip to True as this module lands; see the audit for its spec
