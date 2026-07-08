"""Diffusion / cascade dynamics over a Graph (audit C.5, C.8).

Three standard, correct mechanics — kept minimal and testable:

- independent_cascade: each newly-active node gets one chance to activate each out-neighbor with
  prob = edge weight. Classic Kempe-Kleinberg-Tardos IC. Returns activation trajectory.
- linear_threshold: a node activates when the weighted fraction of active in-neighbors exceeds its
  threshold. (Granovetter.)
- hawkes_intensity: self-exciting temporal intensity λ(t) = μ + Σ α e^{-β (t - t_i)} — the standard
  model for reply/retweet bursts over time (no network needed).

These are the *aggregate temporal/networked* transition when an outcome unfolds over time rather
than resolving in one step. They produce trajectories you can score against observed cascade sizes;
until a wedge has such data the honesty gate labels their output `unvalidated` (they are exposed
via simulation, never as a calibrated prediction on an un-backtested domain).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from swm.state.graph import Graph


def independent_cascade(graph: Graph, seeds: list[str], *, steps: int = 10,
                        rng: random.Random | None = None) -> dict:
    """Run IC from `seeds`. Edge weight is the activation probability. Returns per-step newly-active
    counts and the final active set."""
    rng = rng or random.Random(0)
    active: set[str] = set(seeds)
    frontier = set(seeds)
    per_step = [len(active)]
    for _ in range(steps):
        new_active: set[str] = set()
        for u in frontier:
            for v, w, _etype in graph.neighbors(u):
                if v not in active and v not in new_active and rng.random() < max(0.0, min(1.0, w)):
                    new_active.add(v)
        if not new_active:
            break
        active |= new_active
        frontier = new_active
        per_step.append(len(active))
    return {"final_active": len(active), "active_set": active, "per_step_total": per_step}


def linear_threshold(graph: Graph, seeds: list[str], thresholds: dict[str, float], *,
                     steps: int = 20) -> dict:
    """Run LT. A node activates when the summed weight of its active in-neighbors >= its threshold.
    `thresholds[node]` in (0,1]; missing defaults to 0.5. Deterministic given thresholds."""
    # build in-edges
    in_edges: dict[str, list[tuple[str, float]]] = {n: [] for n in graph.nodes}
    for u, outs in graph.out_edges.items():
        for v, w, _ in outs:
            in_edges.setdefault(v, []).append((u, w))
    active: set[str] = set(seeds)
    per_step = [len(active)]
    for _ in range(steps):
        new_active: set[str] = set()
        for v in graph.nodes:
            if v in active:
                continue
            influence = sum(w for u, w in in_edges.get(v, []) if u in active)
            if influence >= thresholds.get(v, 0.5):
                new_active.add(v)
        if not new_active:
            break
        active |= new_active
        per_step.append(len(active))
    return {"final_active": len(active), "active_set": active, "per_step_total": per_step}


@dataclass
class HawkesProcess:
    """Univariate self-exciting point process. λ(t) = mu + sum_{t_i < t} alpha * exp(-beta (t-t_i)).

    Used to model the temporal shape of a response cascade (replies, retweets) — a burst that
    excites more of itself and decays. `expected_count` integrates the branching structure; for a
    stable process (alpha < beta) the expected total triggered per event is alpha/beta.
    """
    mu: float = 0.1        # background rate
    alpha: float = 0.5     # excitation
    beta: float = 1.0      # decay (alpha < beta for stability)

    def intensity(self, t: float, history: list[float]) -> float:
        return self.mu + sum(self.alpha * math.exp(-self.beta * (t - ti))
                             for ti in history if ti < t)

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    def expected_total(self, base_events: float, horizon: float) -> float:
        """Expected total events over `horizon` given `base_events` exogenous arrivals, via the
        branching-ratio geometric sum (stable only if branching_ratio < 1)."""
        n = self.branching_ratio
        immigrants = self.mu * horizon + base_events
        return immigrants / (1 - n) if n < 1 else immigrants * (1 + horizon)

    def simulate(self, horizon: float, *, rng: random.Random | None = None,
                 max_events: int = 10000) -> list[float]:
        """Ogata thinning simulation of event times on [0, horizon]."""
        rng = rng or random.Random(0)
        t = 0.0
        events: list[float] = []
        while t < horizon and len(events) < max_events:
            lam_bar = self.intensity(t, events) + self.alpha  # upper bound just after now
            if lam_bar <= 0:
                break
            t += -math.log(rng.random()) / lam_bar
            if t >= horizon:
                break
            if rng.random() <= self.intensity(t, events) / lam_bar:
                events.append(t)
        return events
