"""Typed event/action schema — the append-only, timestamped substrate (audit C.1, C.7).

Everything the system knows is a timestamped, typed record. Keeping ingestion append-only
and timestamped is the single most important discipline for honest backtesting: any
train/eval split must be reconstructable as "what was knowable at time T".

We deliberately do NOT model "all variables". An Event is a small, closed schema; content
is carried as an opaque embedding reference + a content hash (for dedup / leakage tracking).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    MESSAGE = "message"        # something was sent to an entity
    OPEN = "open"
    CLICK = "click"
    REPLY = "reply"
    POST = "post"
    REACT = "react"
    PURCHASE = "purchase"
    POLL_RESPONSE = "poll_response"
    UNSUBSCRIBE = "unsubscribe"


@dataclass(frozen=True)
class Event:
    """A partial, noisy observation of the world. Never assume it is complete."""
    actor_id: str
    timestamp: float               # unix seconds; the ONLY thing splits are keyed on
    type: EventType
    channel: str                   # e.g. "email", "reddit", "web"
    content_ref: str | None = None  # pointer to an embedding row (see entities/embeddings)
    content_hash: str | None = None  # for dedup + train/test leakage tracking
    target_ids: tuple[str, ...] = ()
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """A candidate intervention WE could take. Must be encodable so the transition model
    can condition on it — including novel actions never seen in the log (audit C.7)."""
    type: str                      # e.g. "send"
    channel: str
    content_ref: str | None = None
    audience_selector: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)   # {dow, hour, tz}
    dosage: float = 1.0


@dataclass(frozen=True)
class WorldEvent:
    """An exogenous thing that happens (news, competitor launch, season). Shifts the shared
    context c_t; has salience that decays. Start with a hand-curated feed per wedge."""
    timestamp: float
    type: str
    embedding_ref: str | None = None
    salience: float = 1.0
    decay: float = 0.0


@dataclass(frozen=True)
class Prediction:
    """Every prediction is a distribution + interval + calibration grade + report type.
    report_type distinguishes calibrated PREDICTION from believable-but-uncalibrated INSIGHT
    (audit E.5 / J contract rules) so the UI can never blur the line."""
    outcome_name: str
    p_mean: float
    p_interval: tuple[float, float]
    calibration_grade: str | None = None
    report_type: str = "prediction"   # "prediction" | "insight"
    as_of: float | None = None        # echoed so leakage is impossible to hide
    drivers: tuple[tuple[str, float], ...] = ()
