"""Discrete simulation events + a time-ordered queue (Phase 2).

A trajectory is a sequence of `SimulationEvent`s: a submission, exposure waves, reaction batches,
front-page transitions. The engine pops events in time order, each producing a `state_delta` that
updates the trajectory state and may enqueue follow-on events (second-order reactions). This is what
makes it a stepwise simulation rather than a one-shot scoring function.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from itertools import count


@dataclass
class SimulationEvent:
    timestamp: float
    event_type: str                       # "submit" | "expose" | "react" | "frontpage" | "settle"
    actor_id: str = ""                    # segment or entity that acts (or "" for world events)
    target_ids: tuple[str, ...] = ()
    content_ref: str = ""                 # action/message id
    state_delta: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)


class EventQueue:
    """Min-heap by timestamp; stable via an insertion counter."""
    def __init__(self) -> None:
        self._h: list = []
        self._c = count()

    def push(self, ev: SimulationEvent) -> None:
        heapq.heappush(self._h, (ev.timestamp, next(self._c), ev))

    def pop(self) -> SimulationEvent | None:
        return heapq.heappop(self._h)[2] if self._h else None

    def __len__(self) -> int:
        return len(self._h)

    def __bool__(self) -> bool:
        return bool(self._h)
