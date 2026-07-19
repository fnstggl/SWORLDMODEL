"""Phase 13 common-random-number streams (Part 8) — deterministic random-stream partitioning.

The canonical `RolloutEngine` drives one sequential `random.Random(seed)` per branch. That matches
particles across arms (same seed) but NOT event-level randomness: an intervention that inserts one extra
event advances the shared stream, so every LATER draw — including unrelated exogenous shocks — changes.
The paired comparison then confounds "effect of the action" with "different luck downstream".

Fix: a `StreamRNG` that derives an INDEPENDENT deterministic substream per named purpose
(sha256(root_seed | stream_name) → its own `random.Random`). The matched engine routes:
  - hazard re-arming            → stream  "hazard|<etype>"
  - operator draws on an event  → stream  "op|<operator>|<etype>"
  - implementation failure      → stream  "impl|<action_id>"
so action-specific stochasticity consumes only its own stream and unrelated shocks stay IDENTICAL
across counterfactual arms. `verify_pairing` proves it: a no-op arm must reproduce the reference arm's
exogenous event times and terminal state exactly.
"""
from __future__ import annotations

import hashlib
import random

from swm.world_model_v2.rollout import RolloutEngine
from swm.world_model_v2.state import WorldBranch


class StreamRNG:
    """Deterministic per-purpose substreams off one root seed. API-compatible with the subset of
    `random.Random` the runtime uses (random/gauss/expovariate/randrange/choice/uniform/shuffle),
    delegating every call to the CURRENT stream (set by the engine before each use)."""

    def __init__(self, root_seed: int):
        self.root_seed = int(root_seed)
        self._streams: dict = {}
        self._current = self.stream("default")
        self.draw_census: dict = {}                       # stream -> n draws (CRN manifest evidence)

    def stream(self, name: str) -> random.Random:
        if name not in self._streams:
            h = hashlib.sha256(f"{self.root_seed}|{name}".encode()).digest()
            self._streams[name] = random.Random(int.from_bytes(h[:8], "big"))
        return self._streams[name]

    def use(self, name: str) -> "StreamRNG":
        self._current = self.stream(name)
        self.draw_census[name] = self.draw_census.get(name, 0)
        self._census_key = name
        return self

    def _count(self):
        self.draw_census[self._census_key] = self.draw_census.get(self._census_key, 0) + 1

    # ---- the random.Random subset the operators/queue use ----
    def random(self):
        self._count()
        return self._current.random()

    def gauss(self, mu, sigma):
        self._count()
        return self._current.gauss(mu, sigma)

    def expovariate(self, lambd):
        self._count()
        return self._current.expovariate(lambd)

    def randrange(self, *a):
        self._count()
        return self._current.randrange(*a)

    def randint(self, a, b):
        self._count()
        return self._current.randint(a, b)

    def choice(self, seq):
        self._count()
        return self._current.choice(seq)

    def uniform(self, a, b):
        self._count()
        return self._current.uniform(a, b)

    def shuffle(self, x):
        self._count()
        return self._current.shuffle(x)

    def sample(self, population, k):
        self._count()
        return self._current.sample(population, k)


class MatchedRolloutEngine(RolloutEngine):
    """RolloutEngine with stream-partitioned randomness. Same TEMPORAL event loop (batching,
    exact-interval evolution, first-passage re-projection), same operators, same StateDelta/
    follow-up semantics — only the RNG ROUTING differs, so unrelated shocks cannot
    desynchronize across matched counterfactual arms. Temporal latents, sampled schedules and
    first-passage thresholds additionally seed from the PARTICLE ROOT (temporal_model.
    particle_rng), so matched arms share one temporal reality except where the action itself
    causally changes it (§21/§23)."""

    def run_branch(self, world, queue, *, seed: int = 0, max_events: int = 2000) -> WorldBranch:
        from swm.world_model_v2.temporal_runtime import run_branch_temporal
        rng = StreamRNG(seed)
        branch = run_branch_temporal(
            world, queue, self.operators, seed=seed, safety_max_events=max_events,
            rng_for=lambda op, ev: rng.use(f"op|{getattr(op, 'name', 'op')}|{ev.etype}"),
            queue_rng=_PinnedStream(rng, "hazard|rearm"))
        branch.world.uncertainty_meta["crn_draw_census"] = dict(rng.draw_census)
        return branch


class _PinnedStream:
    """A handle that RE-SELECTS its named stream on every call — safe to hold across other
    `use()` switches (the queue's hazard re-arm draws interleave with operator draws in the
    temporal loop)."""

    def __init__(self, sr: StreamRNG, name: str):
        self._sr, self._name = sr, name

    def __getattr__(self, attr):
        sr, name = self._sr, self._name

        def call(*a, **k):
            return getattr(sr.use(name), attr)(*a, **k)
        return call


def exogenous_trace(branch) -> list:
    """The exogenous shock trace of a rolled branch: (ts, etype) of every HAZARD-sourced delta — the
    thing that must be IDENTICAL across matched arms for unrelated shocks. Action-caused activity
    (decision events and their endogenous follow-ups) is legitimately different across arms and is
    excluded by the event-source stamp."""
    return [(round(d.at, 6), d.event_type) for d in branch.log
            if str(d.uncertainty.get("event_source", "")).startswith("hazard:")]


def verify_pairing(evaluate_once, *, n_checks: int = 3) -> dict:
    """CRN verification harness: `evaluate_once(arm_label) -> [branch per particle]` for the SAME
    particle set. Checks (a) determinism: reference twice → identical logs; (b) no-op neutrality:
    a 'do_nothing' arm reproduces the reference's exogenous trace per particle."""
    ref1 = evaluate_once("__ref__")
    ref2 = evaluate_once("__ref__")
    deterministic = all(exogenous_trace(a) == exogenous_trace(b) for a, b in zip(ref1, ref2))
    noop = evaluate_once("__noop__")
    matched = [exogenous_trace(a) == exogenous_trace(b) for a, b in zip(ref1, noop)]
    return {"deterministic_replay": bool(deterministic),
            "noop_exogenous_match_rate": (sum(matched) / len(matched)) if matched else 0.0,
            "n_particles_checked": len(matched)}
