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
               ("resolve_outcome", "scheduled"),    # terminal-outcome resolution (generic fallback + others)
               # ---- event-driven temporal architecture (§6/§9/§13/§15/§17) ----
               ("ctrl_attention", "scheduled"),          # an actor's real attention opportunity
               ("stance_relevant_change", "endogenous"), # material state change → stance review
               ("first_passage", "scheduled"),           # cumulative-hazard threshold crossing
               ("institutional_stage_complete", "scheduled"),
               ("conditional_trigger", "endogenous")):   # a watched condition became true
    register_event_type(_n, scheduling=_s, validated=True)


@dataclass(order=True)
class Event:
    ts: float                             # exact unix timestamp — the heap key
    seq: int = 0                          # HEAP STABILITY ONLY — never semantic order: same-
    #                                       timestamp events are popped as one batch and layered
    #                                       by causal dependency + canonical content order
    #                                       (temporal_runtime); insertion order must not decide
    #                                       reality (§19)
    etype: str = field(default="", compare=False)
    participants: list = field(default_factory=list, compare=False)
    payload: dict = field(default_factory=dict, compare=False)
    visibility: str = field(default="public", compare=False)
    source: str = field(default="", compare=False)       # provenance: scheduled|hazard:<name>|endogenous:<op>
    preconditions: dict = field(default_factory=dict, compare=False)
    # ---- causal metadata (§19): generic, optional, carried by the batch architecture ----
    event_id: str = field(default="", compare=False)
    parent_ids: list = field(default_factory=list, compare=False)     # causal parents
    dependency_ids: list = field(default_factory=list, compare=False) # must-run-before deps
    microstep: int = field(default=0, compare=False)      # same-timestamp causal layer
    read_set: tuple = field(default=(), compare=False)    # declared state paths read
    write_set: tuple = field(default=(), compare=False)   # declared state paths written
    trigger: dict = field(default_factory=dict, compare=False)  # DecisionTrigger.as_dict() for
    #                                                             decision events (§6)

    def __post_init__(self):
        if self.etype and self.etype not in _EVENT_TYPES:
            raise KeyError(f"unregistered event type {self.etype!r} — register_event_type() first")
        if not self.event_id:
            import hashlib as _h
            key = f"{self.ts}|{self.etype}|{sorted(map(str, self.participants))}|" \
                  f"{sorted((str(k), str(v)[:80]) for k, v in (self.payload or {}).items())}"
            self.event_id = "ev_" + _h.sha256(key.encode()).hexdigest()[:14]

    def content_key(self) -> str:
        """Deterministic content-derived ordering key — identical regardless of queue insertion
        order, so independent same-timestamp events evaluate in an insertion-order-invariant
        canonical order (§19, invariant 32)."""
        return f"{self.event_id}|{self.etype}"


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

    def pop_batch(self, *, rng=None, world=None):
        """Pop ALL events sharing the earliest timestamp <= horizon (§19). Returns a list (empty
        at exhaustion). Hazard re-arming matches next_event. The batch is the same-time unit the
        temporal runtime layers into causal microsteps — the heap's insertion order inside the
        batch carries NO semantics."""
        first = self.next_event(rng=rng, world=world)
        if first is None:
            return []
        batch = [first]
        while self.events and self.events[0].ts == first.ts \
                and self.events[0].ts <= self.horizon_ts:
            nxt = self.next_event(rng=rng, world=world)
            if nxt is None:
                break
            batch.append(nxt)
        return batch

    def peek_pending(self, limit: int = 50) -> list:
        """Non-destructive view of pending in-horizon events (truncation/horizon reporting)."""
        out = []
        for ev in sorted(self.events)[:limit]:
            if ev.ts <= self.horizon_ts:
                out.append({"ts": ev.ts, "etype": ev.etype,
                            "participants": list(ev.participants)[:4],
                            "event_id": ev.event_id})
        return out

    def empty(self) -> bool:
        return not self.events or all(e.ts > self.horizon_ts for e in self.events)
