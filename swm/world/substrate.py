"""World substrate — a persistent, coupled world on one clock (the digital-twin architecture).

Every `simulate(question)` before this built a small, single-mechanism, throwaway model. This is the other
thing: ONE persistent world where entities at different scales coexist and AFFECT EACH OTHER, evolving on a
single shared clock. It is the honest core of "simulate the whole relevant slice of the world":

  - ENTITIES are the nodes — a person, an institution, a population segment, the environment (economy,
    media). Each carries STATE and is advanced by its OWN mechanism (`step_fn`) and read by its own
    `readout_fn`. The mechanism library (single-agent / committee / electorate / SCM) supplies these.
  - COUPLINGS are the edges — the OUTPUT of one entity wired into the INPUT of another (individuals' mood
    into an institution's pressure; the institution's decision into the environment; the environment back
    into individuals). This is what makes the world non-separable ACROSS SCALES.
  - ONE CLOCK: `advance(dt)` steps every entity by the same elapsed time (calibrated diffusion downstream),
    so a person, a committee, and an economy move together — not on private horizons.

A question becomes a QUERY against a forward-simulation of this shared world: advance to the horizon, then
`query` the entity whose outcome you asked about (Monte-Carlo for a distribution).

The discipline the brief demands: **couplings must EARN their place.** `without_couplings()` gives the same
entities with the edges cut (each scale simulated separately); scoring coupled-vs-separate against real
outcomes is the test that decides whether the shared world beats independent models — before scaling up.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Entity:
    """A node in the world. `step_fn(state, inputs, dt, rng) -> new_state` advances it one tick using its
    mechanism; `readout_fn(state, inputs, rng) -> outcome` reads its current outcome without mutating."""
    entity_id: str
    kind: str                                    # 'individual' | 'committee' | 'population' | 'environment'
    state: dict = field(default_factory=dict)
    step_fn: object = None
    readout_fn: object = None


@dataclass
class Coupling:
    """A directed edge: `wire(src_state) -> dict` becomes part of the destination entity's inputs."""
    src: str
    dst: str
    wire: object


@dataclass
class World:
    entities: dict = field(default_factory=dict)
    couplings: list = field(default_factory=list)
    clock: float = 0.0

    def add(self, entity: Entity) -> "World":
        self.entities[entity.entity_id] = entity
        return self

    def couple(self, src: str, dst: str, wire) -> "World":
        self.couplings.append(Coupling(src, dst, wire))
        return self

    def _inputs(self, external: dict = None) -> dict:
        inp = {eid: dict((external or {}).get(eid, {})) for eid in self.entities}
        for c in self.couplings:                             # wire each source's output into its target
            for k, v in (c.wire(self.entities[c.src].state) or {}).items():
                inp[c.dst][k] = v
        return inp

    def advance(self, dt: float = 1.0, external: dict = None, rng=None) -> "World":
        """Step every entity one tick on the shared clock. Inputs are gathered from couplings + externals
        BEFORE any entity steps, so within a tick entities see each other's PREVIOUS state (well-defined)."""
        inp = self._inputs(external)
        for eid, e in self.entities.items():
            if e.step_fn is not None:
                e.state = e.step_fn(e.state, inp[eid], dt, rng)
        self.clock += dt
        return self

    def query(self, entity_id: str, rng=None, external: dict = None):
        inp = self._inputs(external)
        e = self.entities[entity_id]
        return e.readout_fn(e.state, inp[entity_id], rng) if e.readout_fn is not None else dict(e.state)

    def without_couplings(self) -> "World":
        """The ablation: same entities, edges cut. Each scale evolves separately — the baseline the coupled
        world must beat to justify itself."""
        return World(entities={k: Entity(v.entity_id, v.kind, dict(v.state), v.step_fn, v.readout_fn)
                               for k, v in self.entities.items()}, couplings=[])

    def snapshot(self) -> dict:
        return {"clock": self.clock, "entities": {k: dict(v.state) for k, v in self.entities.items()}}


def rollout(world: World, entity_id: str, horizon: float, dt: float = 1.0, *, external_fn=None, rng=None):
    """Advance the world to `horizon` then query `entity_id`. `external_fn(clock) -> {eid: {input}}` injects
    exogenous, dated events (a shock timeline). Returns the queried outcome."""
    rng = rng or random.Random(0)
    steps = max(1, int(round(horizon / dt)))
    for _ in range(steps):
        world.advance(dt, external=(external_fn(world.clock) if external_fn else None), rng=rng)
    return world.query(entity_id, rng=rng)


def montecarlo_world(build_world, entity_id: str, horizon: float, dt: float = 1.0, *, n: int = 2000,
                     external_fn=None, seed: int = 0) -> dict:
    """Monte-Carlo a FRESH world per draw (build_world() -> World) to `horizon`, collecting the queried
    entity's outcome. Numeric -> mean/quantiles; label -> distribution. This is a 'question as a query
    against forward-simulations of the shared world'."""
    from swm.simulation.structural import montecarlo
    rng = random.Random(seed)

    def once(r):
        w = build_world()
        return rollout(w, entity_id, horizon, dt, external_fn=external_fn, rng=r)
    return montecarlo(lambda r: once(r), n=n, seed=seed)
