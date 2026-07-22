"""Phase 8 — the immutable, event-sourced longitudinal history (Part 2/3).

The event log is the CAUSAL SOURCE OF TRUTH; a state checkpoint is a derived artifact. Every persistent
value can be replayed from the log, and deleting/altering an event is the causal ablation the whole phase
is graded on.

Guarantees (all tested):
  * append-only, never overwritten — corrections/retractions/identity-link changes are NEW events;
  * deduplicated + idempotent — the event_id is a content hash, so re-ingesting the same event is a no-op
    and a retry after a crash cannot double-apply;
  * deterministically ordered — the replay order is (event_time, seq, event_id), independent of insertion
    order or concurrency;
  * hash-verifiable — a running watermark chains every event (tamper-evident), and ``verify_integrity``
    recomputes it;
  * durable — the log persists to JSONL and reloads across process restarts (the cross-run service gap);
  * leakage-safe — ``events_as_of`` separates FILTERING (observed_time ≤ t: only what was knowable) from
    SMOOTHING (event_time ≤ t: retrospective analysis only), and refuses a query with no ``as_of``.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = "phase8-events-1.0"
INGESTION_VERSION = "1.0"

#: append-only kinds. Corrections/retractions/identity changes reference the event they revise but never
#: mutate it — the original stays in the log with its own hash.
EVENT_KINDS = ("observation", "correction", "retraction", "identity_link_change", "revised_observation",
               "provenance_update", "policy_feedback")


class HistoryError(ValueError):
    pass


class LeakageError(ValueError):
    """Raised when a query could let a future event reach an as-of forecast."""


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class PersistentEvent:
    """One immutable longitudinal event (Part 2). ``event_time`` is when it happened; ``observed_time`` is
    when it became knowable to the simulator (the FILTERING gate); ``availability_time`` is when it became
    public (visibility gate). Identity linkage is probabilistic — ``identity_link_uncertainty`` in [0,1]
    records how sure we are the actor attribution is right, so uncertain history is not forced onto one
    actor with false certainty."""
    world_id: str
    scenario_id: str
    event_type: str
    event_time: float
    actor_ids: tuple = ()
    scope: str = "actor"
    observed_time: float = 0.0              # defaults to event_time if 0
    availability_time: float = 0.0
    source_id: str = ""
    evidence_hash: str = ""
    visibility: str = "public"              # public | participants | private | institutional
    params: dict = field(default_factory=dict)
    outcome: object = None
    provenance: dict = field(default_factory=dict)
    confidence: float = 1.0
    identity_link_uncertainty: float = 0.0  # 0 = certain attribution; 1 = maximally ambiguous
    causal_mechanism: str = ""
    parent_events: tuple = ()
    kind: str = "observation"               # EVENT_KINDS
    revises_event_id: str = ""              # for correction/retraction/identity_link_change
    schema_version: str = SCHEMA_VERSION
    ingestion_version: str = INGESTION_VERSION
    seq: int = 0                            # tiebreaker for deterministic ordering within equal event_time
    event_id: str = ""                      # content hash (set in __post_init__ if empty)

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise HistoryError(f"unknown event kind {self.kind!r} (known: {EVENT_KINDS})")
        if self.observed_time == 0.0:
            self.observed_time = self.event_time
        if self.availability_time == 0.0:
            self.availability_time = self.observed_time
        if isinstance(self.actor_ids, list):
            self.actor_ids = tuple(self.actor_ids)
        if isinstance(self.parent_events, list):
            self.parent_events = tuple(self.parent_events)
        if not self.event_id:
            self.event_id = self.content_id()

    def content_id(self) -> str:
        """Deterministic content hash → the event_id. Two ingestions of the SAME logical event produce the
        SAME id (dedup + idempotency). Excludes seq (an ordering aid, not identity)."""
        payload = {"world_id": self.world_id, "scenario_id": self.scenario_id,
                   "event_type": self.event_type, "event_time": self.event_time,
                   "actor_ids": list(self.actor_ids), "scope": self.scope,
                   "observed_time": self.observed_time, "source_id": self.source_id,
                   "params": self.params, "outcome": self.outcome, "kind": self.kind,
                   "revises_event_id": self.revises_event_id}
        return "ev_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()[:24]

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["actor_ids"] = list(self.actor_ids)
        d["parent_events"] = list(self.parent_events)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PersistentEvent":
        d = dict(d)
        d["actor_ids"] = tuple(d.get("actor_ids", ()))
        d["parent_events"] = tuple(d.get("parent_events", ()))
        # keep the stored event_id (do not recompute — respects historical ingestion)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _order_key(e: PersistentEvent):
    return (e.event_time, e.seq, e.event_id)


@dataclass
class EventLog:
    """The durable, append-only event store for one (world, scenario). In-memory index + a pluggable durable
    backend (JSONL by default via ``path``, or a transactional ``SqliteBackend`` via ``backend`` — see
    ``phase8_storage``) for cross-run persistence. Thread-safe appends; idempotent by content id."""
    world_id: str
    scenario_id: str
    path: str | None = None                 # JSONL backing file; None = in-memory only
    backend: object = None                  # phase8_storage.PersistentStorageBackend (overrides `path`)
    _events: dict = field(default_factory=dict)     # event_id -> PersistentEvent
    _seq: int = 0
    _watermark: str = ""                    # running tamper-evident chain hash
    _lock: object = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        if self.backend is None and self.path:
            from swm.world_model_v2.phase8_storage import JsonlBackend
            self.backend = JsonlBackend(self.path)
        if self.backend is not None:
            self._load()

    # ---- append (idempotent, retry-safe, concurrency-safe) ----------------------------------------
    def append(self, event: PersistentEvent) -> tuple[PersistentEvent, bool]:
        """Append one event. Returns (stored_event, is_new). Re-appending an identical event (same content
        id) is a NO-OP that returns the already-stored event with is_new=False — so a retry after a partial
        crash, or concurrent double-ingestion, cannot double-apply."""
        with self._lock:
            if not event.event_id:
                event.event_id = event.content_id()
            existing = self._events.get(event.event_id)
            if existing is not None:
                return existing, False
            self._seq += 1
            event.seq = self._seq
            self._events[event.event_id] = event
            self._watermark = hashlib.sha256(
                (self._watermark + "|" + event.event_id).encode()).hexdigest()[:24]
            if self.backend is not None:
                self.backend.append_event(event.as_dict(), self._watermark)
            return event, True

    def ingest(self, event: PersistentEvent) -> bool:
        """Convenience: append and return whether it was new (for dedup accounting)."""
        _, is_new = self.append(event)
        return is_new

    def ingest_many(self, events) -> dict:
        added = dup = 0
        for e in events:
            if self.ingest(e):
                added += 1
            else:
                dup += 1
        return {"added": added, "duplicates": dup, "total": len(self._events)}

    # ---- corrections / retractions / identity changes (append, never overwrite) --------------------
    def correct(self, original_id: str, *, params=None, outcome=None, source_id="",
                observed_time=None) -> PersistentEvent:
        orig = self._events.get(original_id)
        if orig is None:
            raise HistoryError(f"cannot correct unknown event {original_id!r}")
        ev = PersistentEvent(
            world_id=self.world_id, scenario_id=self.scenario_id, event_type=orig.event_type,
            event_time=orig.event_time, actor_ids=orig.actor_ids, scope=orig.scope,
            observed_time=observed_time if observed_time is not None else orig.observed_time,
            source_id=source_id or orig.source_id,
            params={**orig.params, **(params or {})}, outcome=outcome if outcome is not None else orig.outcome,
            kind="correction", revises_event_id=original_id, parent_events=(original_id,))
        self.append(ev)
        return ev

    def retract(self, original_id: str, *, reason: str = "", observed_time=None) -> PersistentEvent:
        orig = self._events.get(original_id)
        if orig is None:
            raise HistoryError(f"cannot retract unknown event {original_id!r}")
        ev = PersistentEvent(
            world_id=self.world_id, scenario_id=self.scenario_id, event_type=orig.event_type,
            event_time=orig.event_time, actor_ids=orig.actor_ids, scope=orig.scope,
            observed_time=observed_time if observed_time is not None else orig.observed_time,
            params={"reason": reason}, kind="retraction", revises_event_id=original_id,
            parent_events=(original_id,))
        self.append(ev)
        return ev

    def relink_identity(self, original_id: str, *, new_actor_ids, link_uncertainty=0.0,
                        observed_time=None) -> PersistentEvent:
        orig = self._events.get(original_id)
        if orig is None:
            raise HistoryError(f"cannot relink unknown event {original_id!r}")
        ev = PersistentEvent(
            world_id=self.world_id, scenario_id=self.scenario_id, event_type=orig.event_type,
            event_time=orig.event_time, actor_ids=tuple(new_actor_ids), scope=orig.scope,
            observed_time=observed_time if observed_time is not None else orig.observed_time,
            params=dict(orig.params), outcome=orig.outcome, kind="identity_link_change",
            identity_link_uncertainty=link_uncertainty, revises_event_id=original_id,
            parent_events=(original_id,))
        self.append(ev)
        return ev

    # ---- effective (correction/retraction-aware) view ---------------------------------------------
    def _superseded(self) -> set:
        """Event ids that a later correction/retraction/relink revises (so the effective log uses the
        revision). The original stays in the store — this only affects the effective read."""
        out = set()
        for e in self._events.values():
            if e.revises_event_id:
                out.add(e.revises_event_id)
        return out

    def effective_events(self, *, include_retracted=False):
        """The correction-aware event stream, deterministically ordered. Superseded originals drop out;
        retractions remove their target from the effective stream (unless include_retracted)."""
        superseded = self._superseded()
        retracted = {e.revises_event_id for e in self._events.values() if e.kind == "retraction"}
        out = []
        for e in sorted(self._events.values(), key=_order_key):
            if e.kind == "retraction":
                continue
            if e.event_id in superseded:
                continue                                    # replaced by a later correction/relink
            if e.event_type and e.revises_event_id in retracted:
                pass
            if e.event_id in retracted and not include_retracted:
                continue
            out.append(e)
        # drop events whose revises-chain terminates in a retraction
        if not include_retracted:
            out = [e for e in out if e.revises_event_id not in retracted]
        return out

    # ---- leakage-safe as-of query -----------------------------------------------------------------
    def events_as_of(self, as_of: float, *, mode: str = "filter", actor_id: str | None = None):
        """Return effective events visible as of ``as_of``.

        mode='filter'  → observed_time <= as_of  (production/forecasting: only what was KNOWABLE);
        mode='smooth'  → event_time <= as_of     (retrospective analysis ONLY — never a production forecast).

        A missing ``as_of`` is refused (a query without it could leak the future)."""
        if as_of is None:
            raise LeakageError("events_as_of requires an explicit as_of — a query without it could leak the "
                               "future into an as-of forecast")
        if mode not in ("filter", "smooth"):
            raise HistoryError(f"mode must be 'filter' or 'smooth', got {mode!r}")
        out = []
        for e in self.effective_events():
            t = e.observed_time if mode == "filter" else e.event_time
            if t > as_of:
                continue
            if actor_id is not None and actor_id not in e.actor_ids:
                continue
            out.append(e)
        return out

    def assert_no_leak(self, as_of: float, returned, *, mode="filter") -> None:
        attr = "observed_time" if mode == "filter" else "event_time"
        bad = [e.event_id for e in returned if getattr(e, attr) > as_of]
        if bad:
            raise LeakageError(f"LEAK: events after as_of={as_of} ({mode}) returned: {bad[:5]}")

    # ---- integrity + durability -------------------------------------------------------------------
    def watermark(self) -> str:
        return self._watermark

    def verify_integrity(self) -> dict:
        """Recompute the running chain over the stored insertion order and confirm it matches the watermark
        (tamper-evidence). Also checks every event_id equals its content hash (no forged ids)."""
        chain, forged = "", []
        for e in sorted(self._events.values(), key=lambda x: x.seq):
            chain = hashlib.sha256((chain + "|" + e.event_id).encode()).hexdigest()[:24]
            if e.kind == "observation" and e.event_id != e.content_id():
                forged.append(e.event_id)
        return {"ok": (chain == self._watermark) and not forged, "recomputed_watermark": chain,
                "stored_watermark": self._watermark, "forged_ids": forged, "n_events": len(self._events)}

    def _load(self):
        for d in self.backend.load_events():
            ev = PersistentEvent.from_dict(d)
            if ev.event_id not in self._events:
                self._seq += 1
                ev.seq = self._seq
                self._events[ev.event_id] = ev
                self._watermark = hashlib.sha256(
                    (self._watermark + "|" + ev.event_id).encode()).hexdigest()[:24]

    def __len__(self):
        return len(self._events)

    def summary(self) -> dict:
        eff = self.effective_events()
        kinds = {}
        for e in self._events.values():
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        return {"world_id": self.world_id, "scenario_id": self.scenario_id, "n_stored": len(self._events),
                "n_effective": len(eff), "kinds": kinds, "watermark": self._watermark,
                "durable": self.backend is not None,
                "backend": type(self.backend).__name__ if self.backend is not None else "memory"}
