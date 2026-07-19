"""Shared-state EVENT-DRIVEN rollout + terminal readout + matched counterfactuals.

The loop is the temporal runtime (temporal_runtime.run_branch_temporal): pop the full batch of
events at the earliest real timestamp, advance every continuous process over the EXACT elapsed
interval (no daily background tick — a 10-day gap is one exact 10-day update), evaluate
same-time events in causal microsteps with insertion-order-invariant canonical ordering, run
every applicable registered operator (each producing a machine-readable StateDelta), let
first-passage hazards re-project when written state touches their read sets, and continue to
causal quiescence or the real horizon. `max_events` is a SAFETY budget: exhausting it marks the
branch temporally truncated (recorded, surfaced) — it is never treated as natural completion.

At the horizon, the outcome contract PROJECTS the answer from terminal states (frequencies over
worlds) — no LLM is asked for a number after simulation.

Counterfactuals (Phase 7): sample the initial latent worlds ONCE; for each intervention, CLONE
every sampled world and apply the intervention to the clone; exogenous randomness, sampled
temporal latents, and first-passage thresholds are seeded per PARTICLE (particle-root streams),
so matched clones face the same shocks and the same temporal reality except where the action
itself causally changes them; compare terminal utilities per-particle → expected utility,
downside, P(best), expected regret."""
from __future__ import annotations

from dataclasses import dataclass

from swm.world_model_v2.events import EventQueue


@dataclass
class RolloutEngine:
    operators: list                        # instantiated TransitionOperator objects (registry-vetted)

    def run_branch(self, world, queue: EventQueue, *, seed: int = 0, max_events: int = 2000):
        """Advance one world event→event to the horizon through the temporal runtime. Returns
        the branch with its full delta log and `branch.temporal_stats` (§27)."""
        from swm.world_model_v2.temporal_runtime import run_branch_temporal
        return run_branch_temporal(world, queue, self.operators, seed=seed,
                                   safety_max_events=max_events)


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
        branches = []
        for i, w in enumerate(worlds):
            q = self.queue_builder(w)
            branches.append(engine.run_branch(w, q, seed=seed * 7919 + i))
        result = self.contract.project(branches)
        result["n_deltas"] = sum(len(b.log) for b in branches)
        result["readout"] = "terminal_states"                # provenance: numbers came from worlds, not an LLM
        from swm.world_model_v2.temporal_runtime import aggregate_temporal_stats
        result["temporal_runtime"] = aggregate_temporal_stats(branches)
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
