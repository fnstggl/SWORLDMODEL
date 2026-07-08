"""LAYER 1 — strategy optimization in variable space (the piece that removes the LLM's bias).

We do NOT ask an LLM for emails and rank them. We search the low-dimensional space of message-controllable
VARIABLES (personalization, pushiness, credential_signaling, contrarian_pitch, …) for the vector that
maximizes the world model's reply probability for THIS recipient — text-free. The output is a `StrategySpec`:
the optimal *strategy*, not an email. No draft anchors it, so it finds the global optimum of the objective
instead of the best of a few human-written guesses.

Method: gradient-free coordinate ascent with random restarts. The space is ~8-dimensional and bounded, and
the scorer is cheap, so we can evaluate THOUSANDS of strategy vectors (the whole point — a human can compare
four). The objective is the scorer's pessimistic LOWER BOUND (a low percentile across the elasticity
ensemble), not the mean — so the search can't win by exploiting one high-variance weight (anti-Goodhart).

The returned spec carries, per variable, the chosen value and its driver contribution, plus the top
recipient-conditioned interactions — so the recommendation ("credential_signaling → 0 because this
recipient's status_orientation is high") is auditable, not asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.decision.strategy_scorer import MESSAGE_VARS, StrategyScorer
from swm.variables.schema import spec


def _bounds(name: str) -> tuple[float, float]:
    return (-1.0, 1.0) if spec(name).signed else (0.0, 1.0)


@dataclass
class StrategySpec:
    """The optimal message strategy for a recipient — values in variable space, plus why."""
    strategy: dict
    mean: float
    lower_bound: float
    drivers: list = field(default_factory=list)
    q: float = 0.2

    def describe(self) -> list[str]:
        """Human-readable spec lines, ordered by leverage."""
        lines = []
        for d in self.drivers:
            sign = "+" if d["contribution"] >= 0 else "−"
            lines.append(f"{sign} {d['term']}: {d['contribution']:+.3f}")
        return lines

    def summary(self) -> dict:
        return {"strategy": {k: round(v, 3) for k, v in self.strategy.items()},
                "predicted_reply_mean": round(self.mean, 4),
                "predicted_reply_lower_bound_q%d" % int(self.q * 100): round(self.lower_bound, 4),
                "top_drivers": self.drivers[:10]}


def optimize_strategy(scorer: StrategyScorer, *, q: float = 0.2, restarts: int = 12,
                      grid: int = 11, sweeps: int = 3, seed: int = 0) -> StrategySpec:
    """Coordinate ascent + random restarts, maximizing scorer.lower_bound(strategy, q).

    grid    — candidate values tried per coordinate per sweep (finer grid = more thorough).
    sweeps  — passes over all coordinates per restart (re-optimizing each given the others).
    restarts— independent starts (neutral + random) to escape local optima.
    """
    import random
    rng = random.Random(seed)
    vars_ = MESSAGE_VARS

    def objective(strat):
        return scorer.lower_bound(strat, q=q)

    def neutral_start():
        return {v: (spec(v).default if not spec(v).signed else 0.0) for v in vars_}

    def random_start():
        out = {}
        for v in vars_:
            lo, hi = _bounds(v)
            out[v] = lo + rng.random() * (hi - lo)
        return out

    best_strat, best_val = None, -1.0
    starts = [neutral_start()] + [random_start() for _ in range(restarts)]
    for start in starts:
        strat = dict(start)
        val = objective(strat)
        for _ in range(sweeps):
            improved = False
            for v in vars_:
                lo, hi = _bounds(v)
                cur = strat[v]
                best_local_v, best_local_val = cur, val
                for g in range(grid):
                    cand = lo + (hi - lo) * g / (grid - 1)
                    strat[v] = cand
                    cv = objective(strat)
                    if cv > best_local_val + 1e-6:
                        best_local_val, best_local_v = cv, cand
                strat[v] = best_local_v
                if best_local_val > val + 1e-9:
                    val, improved = best_local_val, True
            if not improved:
                break
        if val > best_val:
            best_val, best_strat = val, dict(strat)

    dist = scorer.score_dist(best_strat)
    return StrategySpec(strategy=best_strat, mean=dist.mean, lower_bound=dist.lower_bound(q),
                        drivers=dist.drivers, q=q)
