"""World rollout over the SimulationEngine: stepwise trajectories, uncertainty by horizon, and
scenario summaries over multiple action plans (Phase 2/6).

Wraps `HNSimulationEngine` to expose what a world-model rollout should: not one number, but the
distribution of simulated futures with per-timestep uncertainty, and a comparison across candidate
actions. The probability still comes from the trajectory distribution — this module only records and
aggregates the trajectories the engine already simulates.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from swm.simulation.engine import HNSimulationEngine


@dataclass
class WorldRollout:
    engine: HNSimulationEngine

    def rollout(self, feats: dict, *, author_rep: float = 0.0, ctx: dict | None = None,
                n_samples: int = 200, seed: int = 0) -> dict:
        """Simulate n trajectories; return per-timestep score distribution (uncertainty by horizon),
        the final outcome distribution, and the calibrated P(hit)."""
        ctx = ctx or {}
        rng = random.Random(seed)
        n_steps = self.engine.params.n_steps
        per_step_scores = [[] for _ in range(n_steps)]
        finals, frontpage_step = [], []
        for i in range(n_samples):
            score, st = self.engine._trajectory(feats, author_rep, ctx, rng)
            finals.append(score)
            for j, (_, _, delta) in enumerate(st.accumulated_outcomes[:n_steps]):
                cum = sum(d for _, _, d in st.accumulated_outcomes[:j + 1])
                per_step_scores[j].append(cum)
        pred = self.engine.predict(feats, author_rep=author_rep, ctx=ctx, n_samples=n_samples, seed=seed)
        per_step = []
        for j in range(n_steps):
            xs = sorted(per_step_scores[j])
            if not xs:
                continue
            m = len(xs)
            per_step.append({"t": j + 1, "median": xs[m // 2],
                             "interval80": [xs[int(0.1 * m)], xs[min(m - 1, int(0.9 * m))]],
                             "mean": round(sum(xs) / m, 2)})
        finals.sort()
        return {
            "report_type": "simulation",
            "per_step": per_step,                       # uncertainty by horizon
            "final_interval80": [finals[int(0.1 * len(finals))], finals[min(len(finals) - 1, int(0.9 * len(finals)))]],
            "final_median": finals[len(finals) // 2],
            "p_hit": round(pred["p_hit"], 4), "p_hit_raw": round(pred["p_hit_raw"], 4),
            "band_probs": [round(b, 4) for b in pred["band_probs"]],
        }

    def scenario_tree(self, candidates: list[tuple[str, dict, float, dict]], *,
                      n_samples: int = 150, seed: int = 0) -> dict:
        """candidates: list of (label, feats, author_rep, ctx). Simulate each and rank by P(hit)."""
        branches = []
        for label, feats, rep, ctx in candidates:
            r = self.rollout(feats, author_rep=rep, ctx=ctx, n_samples=n_samples, seed=seed)
            branches.append({"label": label, "p_hit": r["p_hit"], "final_median": r["final_median"],
                            "final_interval80": r["final_interval80"]})
        branches.sort(key=lambda b: b["p_hit"], reverse=True)
        return {"report_type": "simulation", "ranked": branches,
                "recommended": branches[0]["label"] if branches else None}
