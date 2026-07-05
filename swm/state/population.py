"""Population / aggregate state (Phase-3 spec: the state an aggregate query conditions on).

An aggregate prediction — "how will HN / this subreddit / this market / this electorate respond" —
conditions on a *population* state, not one entity. This module is that state object. It holds the
fields the spec enumerates:

- population priors        : global base rate(s) for the outcome
- subgroup priors          : per-segment base rates (topic, source-tier, format, ...)
- topic salience           : how "hot" each topic is right now (fast EMA)
- domain/source reputation : per-source quality prior (slow EMA)
- attention / competition  : how much attention is available vs. how much is being competed for
- recent event context     : exogenous events in the window (retrieved, as-of)
- incentives / stakes       : `IncentiveState` (see incentives.py)
- diffusion / network state : optional graph handle (see graph.py)
- nonstationarity / drift  : `DriftTracker` indicators (see transition/nonstationarity.py)
- uncertainty              : evidence weight per field

It is deliberately a data container with cheap sufficient-statistics updates; the *dynamics* live
in `swm/transition/aggregate_transition.py`. Every quantity is a Posterior so uncertainty flows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.state.state import Posterior


@dataclass
class SubgroupPrior:
    """A segment's outcome propensity, as a rate posterior (successes/total, smoothed)."""
    rate: Posterior
    n: float = 0.0

    def observe(self, hit: float, weight: float = 1.0) -> None:
        self.rate.observe(hit, weight)
        self.n += weight


@dataclass
class PopulationState:
    """Aggregate/community latent state at a timestamp.

    `base_rate` is the pooled outcome propensity. `subgroups[key]` are conditional propensities
    (e.g. topic='ai', tier='top-domain', format='show'). `topic_salience` / `domain_reputation`
    are the fast/slow context EMAs. `attention` and `competition` model the finite-attention
    channel mechanics (a post competes with everything else live). `drift` holds nonstationarity
    indicators populated by the DriftTracker.
    """
    timestamp: float
    base_rate: Posterior = field(default_factory=lambda: Posterior(0.1, 1.0))
    subgroups: dict[str, SubgroupPrior] = field(default_factory=dict)
    topic_salience: dict[str, Posterior] = field(default_factory=dict)
    domain_reputation: dict[str, Posterior] = field(default_factory=dict)
    attention: Posterior = field(default_factory=lambda: Posterior(1.0, 1.0))     # available attention (rel.)
    competition: Posterior = field(default_factory=lambda: Posterior(1.0, 1.0))   # concurrent load (rel.)
    recent_events: list[dict] = field(default_factory=list)
    incentive_state: object | None = None       # swm.state.incentives.IncentiveState
    graph_state: object | None = None           # swm.state.graph.Graph
    drift: dict[str, float] = field(default_factory=dict)
    n_observed: int = 0

    # ---- read helpers ----
    def subgroup_rate(self, key: str) -> float:
        sg = self.subgroups.get(key)
        return sg.rate.mean if sg is not None else self.base_rate.mean

    def salience(self, topic: str) -> float:
        p = self.topic_salience.get(topic)
        return p.mean if p is not None else self.base_rate.mean

    def reputation(self, domain: str, default: float = 0.0) -> float:
        p = self.domain_reputation.get(domain)
        return p.mean if p is not None else default

    # ---- write helpers (sufficient statistics; the transition module calls these) ----
    def observe_outcome(self, hit: float, *, subgroup_keys: tuple[str, ...] = (),
                        topic: str | None = None, domain: str | None = None,
                        salience_weight: float = 0.35, reputation_weight: float = 0.7,
                        domain_value: float | None = None) -> None:
        self.base_rate.observe(hit)
        self.n_observed += 1
        for k in subgroup_keys:
            sg = self.subgroups.get(k)
            if sg is None:
                sg = SubgroupPrior(rate=Posterior(self.base_rate.mean, 1.0))
                self.subgroups[k] = sg
            sg.observe(hit)
        if topic is not None:
            self.topic_salience.setdefault(topic, Posterior(self.base_rate.mean, 1.0)).observe(
                hit, salience_weight)
        if domain:
            dv = hit if domain_value is None else domain_value
            self.domain_reputation.setdefault(domain, Posterior(0.0, 1.0)).observe(
                dv, reputation_weight)

    # ---- serialization (for persisting a fitted world) ----
    def to_dict(self) -> dict:
        def pd(p: Posterior) -> list[float]:
            return [p.mean, p.n]
        return {
            "timestamp": self.timestamp,
            "base_rate": pd(self.base_rate),
            "subgroups": {k: [sg.rate.mean, sg.rate.n, sg.n] for k, sg in self.subgroups.items()},
            "topic_salience": {k: pd(p) for k, p in self.topic_salience.items()},
            "domain_reputation": {k: pd(p) for k, p in self.domain_reputation.items()},
            "competition": pd(self.competition),
            "drift": self.drift,
            "n_observed": self.n_observed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PopulationState":
        s = cls(timestamp=d["timestamp"])
        s.base_rate = Posterior(*d["base_rate"])
        s.subgroups = {k: SubgroupPrior(rate=Posterior(m, n), n=cnt)
                       for k, (m, n, cnt) in d["subgroups"].items()}
        s.topic_salience = {k: Posterior(*v) for k, v in d["topic_salience"].items()}
        s.domain_reputation = {k: Posterior(*v) for k, v in d["domain_reputation"].items()}
        s.competition = Posterior(*d["competition"])
        s.drift = d.get("drift", {})
        s.n_observed = d.get("n_observed", 0)
        return s

    def uncertainty_summary(self) -> dict[str, float]:
        return {
            "base_rate": self.base_rate.uncertainty,
            "n_observed": float(self.n_observed),
            "n_subgroups": float(len(self.subgroups)),
            "n_topics": float(len(self.topic_salience)),
            "n_domains": float(len(self.domain_reputation)),
            **{f"drift.{k}": v for k, v in self.drift.items()},
        }
