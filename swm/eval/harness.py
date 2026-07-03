"""Backtesting harness: temporal + entity-disjoint splits (audit C.10, E.1).

Train on events < T, predict outcomes in [T, T+Δ), score vs realized. Reports BOTH "seen-entity,
future-time" and "new-entity, future-time". The gold standard is PROSPECTIVE: register predictions
before outcomes are known (ForecastBench/Metaculus mechanics), resolve later.

Stub — see docs/social-world-model-audit.md for the design. Not yet implemented."""

#: build-order and design are in docs/social-world-model-audit.md
IMPLEMENTED = False  # flip to True as this module lands; see the audit for its spec
