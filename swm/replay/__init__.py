"""WMV2 Temporal Replay Laboratory — sealed historical backtesting.

The central principle: at forecast time T, no component of the forecasting process may access anything
created after T — and where that cannot be PROVEN (a current LLM's weights), the event is causally blinded
and actively probed for leakage, and the row is classified accordingly. See docs/WMV2_TEMPORAL_REPLAY_LAB.md.
"""
from swm.replay.vault import public_events, sealed_resolutions, HistoricalEvent  # noqa: F401
