"""Typed context-conditioning schema — Phase 7, Part 2.

A nonlinear mechanism's parameters or structure may vary by CONTEXT: actor, population segment, community,
network position, platform, institution, time-of-day, cumulative exposure, source diversity, prior belief,
resource/queue state, regime, … This module gives context variables the same typed-provenance discipline the
rest of the world state has, and — critically — enforces two anti-leakage rules the spec is emphatic about:

  * NO FUTURE CONTEXT: a context value whose validity window starts after the as-of cutoff cannot enter.
  * NO OUTCOME LEAKAGE: a variable flagged `derived_from_outcome` is refused as a conditioning input.

The LLM may NOMINATE candidate context variables and interactions (Part 4H / the mid-run directive), but it
may not decide they are active, set their effect size, or choose the form — those are fixed by fitted data and
held-out validation downstream. A ContextVariable therefore records where it came from and how it is allowed
to enter, never a magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

CONTEXT_SOURCES = ("observed", "inferred_phase3", "documented_event", "derived_from_history",
                   "population", "network", "institution", "assumed", "llm_nominated")
SCALES = ("binary", "categorical", "ordinal", "count", "continuous", "rate", "fraction")


class ContextError(ValueError):
    pass


@dataclass
class ContextVariable:
    """One typed context dimension a mechanism may condition on."""
    name: str
    definition: str
    source: str                                   # CONTEXT_SOURCES
    scale: str = "continuous"
    observed_or_inferred: str = "observed"        # observed | inferred
    phase3_posterior_ref: str = ""                # latent id in the Phase-3 posterior, if inferred
    state_path: str = ""                          # where the runtime reads it from WorldState
    lo: float | None = None
    hi: float | None = None
    temporal_validity: str = "as_of"              # as_of | interval | event_time
    missingness: str = "explicit_default"         # explicit_default | drop | impute_pooled
    default: object = None
    allowable_interaction: bool = True            # may this enter interaction terms?
    transport_risk: str = "medium"                # low | medium | high
    leakage_risk: str = "none"                     # none | temporal | outcome
    derived_from_outcome: bool = False            # HARD BLOCK if True

    def __post_init__(self):
        if self.source not in CONTEXT_SOURCES:
            raise ContextError(f"{self.name}: bad source {self.source!r}")
        if self.scale not in SCALES:
            raise ContextError(f"{self.name}: bad scale {self.scale!r}")

    def as_dict(self):
        return asdict(self)


@dataclass
class ContextSchema:
    """The set of context variables a mechanism instance is allowed to see, with a validated read step."""
    mechanism_family: str
    variables: list = field(default_factory=list)     # [ContextVariable]

    def by_name(self):
        return {v.name: v for v in self.variables}

    def leakage_audit(self, *, as_of: float | None = None, available: dict | None = None) -> dict:
        """Return the leakage verdict BEFORE any value is used. `available` maps name → {value, valid_from}.
        Refuses outcome-derived variables and any value whose validity starts after as_of."""
        blocked, allowed = [], []
        for v in self.variables:
            if v.derived_from_outcome or v.leakage_risk == "outcome":
                blocked.append({"name": v.name, "reason": "derived_from_outcome"})
                continue
            info = (available or {}).get(v.name)
            if info and as_of is not None and info.get("valid_from") is not None:
                if float(info["valid_from"]) > float(as_of):
                    blocked.append({"name": v.name, "reason": "future_context",
                                    "valid_from": info["valid_from"], "as_of": as_of})
                    continue
            allowed.append(v.name)
        return {"allowed": allowed, "blocked": blocked, "leakage_free": not blocked}

    def read(self, world, *, actor_id: str | None = None, extra: dict | None = None) -> dict:
        """Materialize the context vector from WorldState (+ optional precomputed `extra`), applying typed
        defaults and NEVER pulling an outcome-derived field. Read paths: 'actor.<field>', 'quantity.<name>',
        'clock.hour'/'clock.dow', 'population.<id>.<seg>'. Unknown/missing → typed default (recorded)."""
        out, missing = {}, []
        extra = extra or {}
        for v in self.variables:
            if v.derived_from_outcome:
                continue
            if v.name in extra:
                out[v.name] = extra[v.name]
                continue
            val = self._read_path(world, v, actor_id)
            if val is None:
                val = v.default
                missing.append(v.name)
            out[v.name] = val
        out["_missing_context"] = missing
        return out

    def _read_path(self, world, v: ContextVariable, actor_id):
        path = v.state_path
        if not path:
            return None
        try:
            if path.startswith("actor.") and actor_id and actor_id in (world.entities or {}):
                fname = path.split(".", 1)[1]
                return world.entity(actor_id).value(fname)
            if path.startswith("quantity."):
                q = (world.quantities or {}).get(path.split(".", 1)[1])
                return q.value if q is not None else None
            if path == "clock.hour":
                import time as _t
                return _t.gmtime(world.clock.now).tm_hour
            if path == "clock.dow":
                import time as _t
                return _t.gmtime(world.clock.now).tm_wday
        except Exception:
            return None
        return None

    def as_dict(self):
        return {"mechanism_family": self.mechanism_family,
                "variables": [v.as_dict() for v in self.variables]}


# ---------------------------------------------------------------- a small library of common context vars
def actor_prior_belief(default=0.5):
    return ContextVariable(name="prior_belief", definition="actor's prior on the claim before exposure",
                           source="inferred_phase3", scale="fraction", observed_or_inferred="inferred",
                           phase3_posterior_ref="belief", state_path="actor.beliefs", lo=0.0, hi=1.0,
                           default=default, transport_risk="high")


def cumulative_exposure(default=0.0):
    return ContextVariable(name="cum_exposure", definition="count of prior exposures to the item/source",
                           source="derived_from_history", scale="count", state_path="", lo=0.0,
                           default=default, temporal_validity="event_time", transport_risk="medium")


def source_diversity(default=1.0):
    return ContextVariable(name="source_diversity", definition="# independent sources seen so far",
                           source="derived_from_history", scale="count", lo=1.0, default=default,
                           temporal_validity="event_time", transport_risk="medium")


def network_degree(default=0.0):
    return ContextVariable(name="degree", definition="actor's network out-degree (reach)",
                           source="network", scale="count", state_path="actor.degree", lo=0.0,
                           default=default, transport_risk="high")


def time_of_day():
    return ContextVariable(name="hour", definition="hour of day (circadian activity)",
                           source="observed", scale="ordinal", state_path="clock.hour", lo=0, hi=23,
                           default=12, temporal_validity="event_time", transport_risk="low")
