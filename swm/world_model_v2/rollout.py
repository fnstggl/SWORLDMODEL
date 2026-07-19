"""Shared-state event-driven rollout + terminal readout + matched counterfactuals — Phases 5(exec)/6/7.

One world, many mechanisms: the loop pops the next event off the REAL-TIME queue, advances the clock by the
actual elapsed duration, applies background dynamics over that interval, then runs every applicable registered
operator — each producing a machine-readable StateDelta appended to the branch log. At the horizon, the
outcome contract PROJECTS the answer from terminal states (frequencies over worlds) — no LLM is asked for a
number after simulation.

Counterfactuals (Phase 7): sample the initial latent worlds ONCE; for each intervention, CLONE every sampled
world and apply the intervention to the clone; exogenous randomness is seeded per (particle, event-stream) so
matched clones face the same shocks; compare terminal utilities per-particle → expected utility, downside,
P(best), expected regret. This isolates intervention effects from world luck.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from swm.world_model_v2.events import Event, EventQueue
from swm.world_model_v2.state import WorldBranch


@dataclass
class RolloutEngine:
    operators: list                        # instantiated TransitionOperator objects (registry-vetted)
    background_every_days: float = 1.0     # apply background dynamics at most this often (elapsed-driven)

    def run_branch(self, world, queue: EventQueue, *, seed: int = 0, max_events: int = 500) -> WorldBranch:
        """Advance one world event→event to the horizon. Returns the branch with its full delta log."""
        rng = random.Random(seed)
        branch = WorldBranch(branch_id=world.branch_id, world=world)
        last_bg = world.clock.now
        for _ in range(max_events):
            ev = queue.next_event(rng=rng, world=world)
            if ev is None:
                break
            elapsed = ev.ts - world.clock.now
            if elapsed < 0:
                continue                                     # stale event behind the clock — skip
            world.clock.advance_to(ev.ts)
            # background dynamics over the elapsed interval (attention drift, memory decay)
            if (ev.ts - last_bg) >= self.background_every_days * 86400.0:
                bg = Event(ts=ev.ts, etype="background_tick",
                           payload={"elapsed_days": (ev.ts - last_bg) / 86400.0})
                for op in self.operators:
                    if op.applicable(world, bg):
                        delta, _ = op.run(world, bg, rng)
                        if delta is not None:
                            branch.log.append(delta)
                last_bg = ev.ts
            for op in self.operators:
                if op.applicable(world, ev):
                    delta, vr = op.run(world, ev, rng)
                    if delta is not None:
                        branch.log.append(delta)
                        # A4: endogenous action→event chains. Follow-ups are validated against the
                        # event-type registry, must not travel back in time, and are horizon-capped by
                        # the queue itself; invalid proposals are logged, never silently queued.
                        for fu in delta.follow_up_events:
                            try:
                                fev = Event(ts=max(float(fu.get("ts", world.clock.now)), world.clock.now),
                                            etype=str(fu["etype"]),
                                            participants=list(fu.get("participants") or []),
                                            payload=dict(fu.get("payload") or {}),
                                            source=f"endogenous:{op.name}")
                            except (KeyError, TypeError, ValueError) as e:
                                branch.log.append(_rejection_delta(
                                    world, ev, op,
                                    type("VR", (), {"ok": False,
                                                    "reasons": [f"invalid follow-up event: {e}"]})()))
                                continue
                            queue.schedule(fev)
                    elif not vr.ok:
                        branch.log.append(_rejection_delta(world, ev, op, vr))
        branch.terminal = True
        return branch


def _rejection_delta(world, ev, op, vr):
    from swm.world_model_v2.transitions import StateDelta
    d = StateDelta(at=world.clock.now, event_type=ev.etype, operator=op.name,
                   reason_codes=["action_rejected"] + vr.reasons[:3])
    return d


@dataclass
class WorldModelV2Run:
    """The full forecast: initial-state model + queue builder + operators + contract → native distribution."""
    initial: object                        # init_state.InitialStateModel
    queue_builder: object                  # callable(world) -> EventQueue (fresh queue per branch)
    operators: list
    contract: object                       # contracts.OutcomeContract (validated)
    n_particles: int = 30

    def run(self, *, seed: int = 0) -> dict:
        self.contract.validate()
        engine = RolloutEngine(operators=self.operators)
        worlds = self.initial.sample_particles(self.n_particles, seed=seed)
        # independent particle worlds roll out serially or thread-parallel (SWM_BRANCH_THREADS);
        # submission-order collection keeps serial/parallel results identical
        from swm.world_model_v2.materialize import run_branches
        jobs = [(w, self.queue_builder(w), seed * 7919 + i) for i, w in enumerate(worlds)]
        branches = run_branches(engine, jobs)
        result = self.contract.project(branches)
        result["n_deltas"] = sum(len(b.log) for b in branches)
        result["readout"] = "terminal_states"                # provenance: numbers came from worlds, not an LLM
        return result, branches

    # ---------------- Phase 7: matched counterfactuals ----------------
    def evaluate_interventions(self, action_space, utility, *, seed: int = 0) -> dict:
        """Clone matched initial worlds per intervention; same exogenous seeds; compare per-particle."""
        self.contract.validate()
        engine = RolloutEngine(operators=self.operators)
        base_worlds = self.initial.sample_particles(self.n_particles, seed=seed)
        per_arm = {}                                        # id -> [terminal branch per particle]
        for iv in action_space.interventions:
            iv.validate()
            branches = []
            for i, w0 in enumerate(base_worlds):
                w = w0.clone(branch_id=f"{w0.branch_id}:{iv.intervention_id}")
                q = self.queue_builder(w)
                if iv.apply is not None:
                    iv.apply(w, q)
                branches.append(engine.run_branch(w, q, seed=seed * 7919 + i))   # MATCHED seed per particle
            per_arm[iv.intervention_id] = branches
        # per-particle paired comparison
        utils = {a: [utility.fn(b.world) for b in bs] for a, bs in per_arm.items()}
        n = self.n_particles
        p_best, regret = {a: 0.0 for a in utils}, {a: 0.0 for a in utils}
        for i in range(n):
            row = {a: utils[a][i] for a in utils}
            best = max(row.values())
            winners = [a for a, v in row.items() if v == best]
            for a in winners:
                p_best[a] += 1.0 / len(winners) / n
            for a, v in row.items():
                regret[a] += (best - v) / n
        report = {a: {**utility.score(per_arm[a]),
                      "p_best": round(p_best[a], 3), "expected_regret": round(regret[a], 4)}
                  for a in utils}
        ranked = sorted(report.items(), key=lambda kv: -kv[1]["expected_utility"])
        return {"ranking": [{"intervention": a, **r} for a, r in ranked],
                "best": ranked[0][0], "n_matched_worlds": n, "readout": "terminal_states"}
