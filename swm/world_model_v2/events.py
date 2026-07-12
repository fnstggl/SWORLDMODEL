"""Event-driven real time — Phase 3. The simulation advances event→event on the real calendar.

No "round 1/round 2": every event has an exact timestamp (or a hazard that samples one). The queue interleaves
scheduled events (a board vote, a debate, a delivery) with hazard-sampled stochastic events (illness,
distraction, a surprise story) and background dynamics applied over the ELAPSED interval (attention drift,
memory decay, hazard accumulation). Different processes carry different clocks naturally — diffusion events
land minutes apart, polling weekly, votes on their scheduled day. A 1-day and a 30-day rollout differ because
the queue holds different events in the window, not because someone set T=1 vs T=30.

Event types are a registry, not a hardcoded messaging/election/social-media list: a compiler-proposed type
declares participants/preconditions/scheduling/visibility/read-set/possible deltas before it can be queued.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

_EVENT_TYPES: dict = {}


def register_event_type(name: str, *, participants: str = "", preconditions: str = "",
                        scheduling: str = "scheduled", visibility: str = "public",
                        reads: tuple = (), deltas: tuple = (), duration_s: float = 0.0,
                        parameter_source: str = "", validated: bool = False):
    """Universal event schema registry. `scheduling`: scheduled | hazard | endogenous."""
    _EVENT_TYPES[name] = {"participants": participants, "preconditions": preconditions,
                          "scheduling": scheduling, "visibility": visibility, "reads": tuple(reads),
                          "deltas": tuple(deltas), "duration_s": duration_s,
                          "parameter_source": parameter_source, "validated": validated}
    return name


def event_type_registered(name: str) -> bool:
    return name in _EVENT_TYPES


# foundational cross-domain types (examples — domains register their own through the same door)
for _n, _s in (("message_delivered", "scheduled"), ("inbox_checked", "hazard"),
               ("information_published", "scheduled"), ("exposure", "endogenous"),
               ("decision_opportunity", "scheduled"), ("collective_vote", "scheduled"),
               ("measurement", "scheduled"), ("deadline", "scheduled"),
               ("distraction", "hazard"), ("illness", "hazard"), ("external_shock", "hazard"),
               ("follow_up", "scheduled"), ("background_tick", "scheduled"),
               ("resolve_outcome", "scheduled")):   # terminal-outcome resolution (generic fallback + others)
    register_event_type(_n, scheduling=_s, validated=True)


@dataclass(order=True)
class Event:
    ts: float                             # exact unix timestamp — the heap key
    seq: int = 0                          # tiebreaker
    etype: str = field(default="", compare=False)
    participants: list = field(default_factory=list, compare=False)
    payload: dict = field(default_factory=dict, compare=False)
    visibility: str = field(default="public", compare=False)
    source: str = field(default="", compare=False)       # provenance: scheduled|hazard:<name>|endogenous:<op>
    preconditions: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.etype and self.etype not in _EVENT_TYPES:
            raise KeyError(f"unregistered event type {self.etype!r} — register_event_type() first")


@dataclass
class ScheduledEvent:
    event: Event


@dataclass
class StochasticHazard:
    """A rate process that SAMPLES event times: exponential inter-arrival at rate/day (optionally
    time-varying via `rate_fn(world) -> per-day`). Priors are recorded as priors (`prov_note`)."""
    etype: str
    rate_per_day: float = 0.0
    participants: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    rate_fn: object = None
    prov_note: str = "broad prior"

    def sample_next(self, now: float, rng, world=None) -> float:
        r = self.rate_fn(world) if (self.rate_fn and world is not None) else self.rate_per_day
        if r <= 0:
            return math.inf
        return now + rng.expovariate(r) * 86400.0


@dataclass
class EventQueue:
    events: list = field(default_factory=list)    # heap of Event
    hazards: list = field(default_factory=list)   # [StochasticHazard]
    horizon_ts: float = math.inf
    _seq: int = 0

    def schedule(self, event: Event):
        self._seq += 1
        event.seq = self._seq
        heapq.heappush(self.events, event)
        return event

    def add_hazard(self, hazard: StochasticHazard, *, now: float, rng, world=None):
        """Sample the hazard's first arrival and queue it (resampled on each firing)."""
        self.hazards.append(hazard)
        ts = hazard.sample_next(now, rng, world)
        if ts <= self.horizon_ts:
            self.schedule(Event(ts=ts, etype=hazard.etype, participants=list(hazard.participants),
                                payload=dict(hazard.payload), source=f"hazard:{hazard.etype}"))

    def next_event(self, *, rng=None, world=None):
        """Pop the next event <= horizon; re-arm its hazard if it came from one."""
        while self.events:
            ev = heapq.heappop(self.events)
            if ev.ts > self.horizon_ts:
                return None
            if ev.source.startswith("hazard:") and rng is not None:
                hz = next((h for h in self.hazards if h.etype == ev.etype), None)
                if hz is not None:
                    nxt = hz.sample_next(ev.ts, rng, world)
                    if nxt <= self.horizon_ts:
                        self.schedule(Event(ts=nxt, etype=hz.etype, participants=list(hz.participants),
                                            payload=dict(hz.payload), source=ev.source))
            return ev
        return None

    def empty(self) -> bool:
        return not self.events or all(e.ts > self.horizon_ts for e in self.events)
