"""Reusable SENSITIVITY HARNESS — measure how much the forecast depends on the documented
structural choices (stance hazard ratios, coupling constants), on any pinned world.

Method: assemble the pinned world ONCE (the offline demo's production-path assembly), then re-roll
the SAME world under forced overrides — differences are attributable to the overridden constant
alone. Arms:
  * production (sampled priors — the default),
  * agreement-HR / victory-HR point overrides (0.2 … 1.6),
  * coupling point overrides (pathway_step, endogenous_stance_split, own_pathway_weight,
    contested_suppression, persistence survival) — forced by pinning COUPLING_PRIORS to a
    degenerate (v, v, v) triple for the arm.

Reads: P(absorbed by horizon), the mode×time marginal, and the CDF quartiles per arm. If the
headline moves materially across an arm's range, the forecast is assumption-driven at that
constant — that is the honest number this harness exists to surface.

Run:  PYTHONPATH=. python experiments/replay_v3/sensitivity_harness.py [--particles 120]
Artifact: experiments/results/replay_v3/sensitivity_sweep.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import swm.world_model_v2.event_time as ET
import swm.world_model_v2.world_dynamics as WD
from swm.world_model_v2.materialize import run_from_plan

OUT = Path("experiments/results/replay_v3/sensitivity_sweep.json")


def _run_pinned_world(n_particles: int, seed: int = 0) -> dict:
    """One assembly + rollout of the pinned Ukraine world (production path, pinned elicitations)."""
    from experiments.replay_v3.offline_event_time_demo import (assemble, build_ukraine_plan,
                                                               _fake_llm)
    plan = build_ukraine_plan(n_particles)
    assemble(plan, _fake_llm)
    plan.compute_plan["n_particles"] = n_particles             # keep sweeps affordable
    result, _branches = run_from_plan(plan, llm=None, seed=seed)
    ev = result["event_time"]
    return {"p_absorbed": result["distribution"].get("absorbed_by_horizon"),
            "mode_distribution": ev["mode_distribution"],
            "cdf_last": ev["cdf"][-1], "q10": ev["first_passage_quantiles_ts"].get("0.1")}


def sweep(n_particles: int, seed: int = 0) -> dict:
    arms = {}
    arms["production_sampled"] = _run_pinned_world(n_particles, seed)

    for v in (0.2, 0.5, 0.8, 1.0, 1.6):                        # stance HR point overrides
        ET.AGREEMENT_HR_OVERRIDE = v
        try:
            arms[f"agreement_hr={v}"] = _run_pinned_world(n_particles, seed)
        finally:
            ET.AGREEMENT_HR_OVERRIDE = None
    for v in (0.5, 1.0, 1.5):
        ET.VICTORY_HR_OVERRIDE = v
        try:
            arms[f"victory_hr={v}"] = _run_pinned_world(n_particles, seed)
        finally:
            ET.VICTORY_HR_OVERRIDE = None

    coupling_grid = {"pathway_step": (0.01, 0.04, 0.10),
                     "endogenous_stance_split": (0.3, 0.6, 1.0),
                     "own_pathway_weight": (0.5, 1.0, 1.6),
                     "contested_suppression": (0.2, 0.5, 0.9),
                     "persistence_survival_shared": (0.5, 0.75, 0.95)}
    saved = dict(WD.COUPLING_PRIORS)
    for name, values in coupling_grid.items():
        for v in values:
            WD.COUPLING_PRIORS[name] = (v, v, v)               # degenerate → point override
            try:
                arms[f"{name}={v}"] = _run_pinned_world(n_particles, seed)
            finally:
                WD.COUPLING_PRIORS.update(saved)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arms = sweep(args.particles, args.seed)
    ps = [a["p_absorbed"] for a in arms.values() if isinstance(a.get("p_absorbed"), (int, float))]
    summary = {"n_arms": len(arms), "p_min": min(ps), "p_max": max(ps),
               "p_production": arms["production_sampled"]["p_absorbed"],
               "assumption_span": round(max(ps) - min(ps), 4)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "arms": arms}, indent=1))
    print(json.dumps(summary, indent=1))
    for name, a in arms.items():
        print(f"  {name:<36} P={a['p_absorbed']:.3f}  modes={a['mode_distribution']}")
    print(f"artifact → {OUT}")


if __name__ == "__main__":
    main()
