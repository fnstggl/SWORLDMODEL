"""Explicit world-state objects (audit C; the spec's section 1).

A WorldState is a snapshot at a timestamp. It is deliberately POMDP: every latent quantity is a
Posterior (value + evidence weight), never a bare number, so uncertainty flows through the
transition. Nothing here assumes perfect information.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Posterior:
    """A scalar latent with an evidence weight. mean is the estimate; n is how much evidence
    backs it (uncertainty shrinks as ~1/sqrt(n)). Supports EMA-style and Bayesian updates."""
    mean: float
    n: float = 1.0

    @property
    def uncertainty(self) -> float:
        return 1.0 / math.sqrt(max(1e-9, self.n))

    def observe(self, value: float, weight: float = 1.0) -> "Posterior":
        self.mean = (self.mean * self.n + value * weight) / (self.n + weight)
        self.n += weight
        return self


@dataclass
class EntityState:
    """Latent state of one entity (a person, an author, an account). Factors are the mapped
    variables; which ones survive is decided by ablation, not by fiat."""
    entity_id: str
    stable_traits: dict[str, Posterior] = field(default_factory=dict)     # slow: quality, ceiling
    response_style: dict[str, Posterior] = field(default_factory=dict)    # slow: verbosity, topic mix
    relationship_stance: dict[str, Posterior] = field(default_factory=dict)  # standing w/ community
    current_attention: dict[str, Posterior] = field(default_factory=dict)   # fast: recency, streak
    history_features: dict[str, float] = field(default_factory=dict)      # counts, maxima
    uncertainty: float = 1.0

    def get(self, key: str, default: float = 0.0) -> float:
        for d in (self.stable_traits, self.response_style, self.relationship_stance,
                  self.current_attention):
            if key in d:
                return d[key].mean
        return self.history_features.get(key, default)


@dataclass
class ContextState:
    """Exogenous, shared state: what is salient right now, channel conditions, time, drift."""
    topic_salience: dict[str, Posterior] = field(default_factory=dict)   # per-topic hit-rate EMA
    domain_reputation: dict[str, Posterior] = field(default_factory=dict)  # per-domain quality EMA
    recent_events: list[dict] = field(default_factory=list)
    channel_conditions: dict[str, float] = field(default_factory=dict)  # e.g. competition level
    time_features: dict[str, float] = field(default_factory=dict)
    drift_indicators: dict[str, float] = field(default_factory=dict)     # base-rate drift


@dataclass
class WorldState:
    timestamp: float
    population_state: dict[str, float] = field(default_factory=dict)     # global base rates
    entity_states: dict[str, EntityState] = field(default_factory=dict)
    context_state: ContextState = field(default_factory=ContextState)
    graph_state: object | None = None                                   # optional diffusion graph
    uncertainty: dict[str, float] = field(default_factory=dict)

    def entity(self, entity_id: str) -> EntityState:
        return self.entity_states.setdefault(entity_id, EntityState(entity_id))


@dataclass(frozen=True)
class Action:
    """A candidate intervention that can be conditioned on — including novel ones (audit C.7)."""
    action_id: str
    actor_id: str
    content_features: dict[str, float] = field(default_factory=dict)   # length, is_show, topic, ...
    target_ids: tuple[str, ...] = ()
    segment: str | None = None
    timing: dict[str, float] = field(default_factory=dict)            # hour, weekday
    channel: str = "hn"
    dosage: float = 1.0
    meta: dict = field(default_factory=dict)                          # title, domain (for retrieval)


@dataclass(frozen=True)
class OutcomeEvent:
    timestamp: float
    entity_id: str
    action_id: str
    observed: float          # the band index or binary
    magnitude: float         # raw score / count
    raw: dict = field(default_factory=dict)
