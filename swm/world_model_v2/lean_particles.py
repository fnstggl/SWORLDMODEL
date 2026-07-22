"""Adaptive progressive particles — deterministic prefixes, explicit stopping conditions.

The full budget stays the ceiling; lean rolls index-keyed batches through the SAME prepared run
(`run_persistence_slice` — pilot/extension slices equal a direct roll by construction) and stops
early ONLY when every §17 condition holds simultaneously:

  * prediction drift across successive batches is small;
  * the sampling interval is acceptably narrow AND does not cross 0.5 while the side of 0.5 is
    being relied on;
  * unresolved mass and truncation are below their ceilings;
  * structural models do not materially disagree and no reversal-capable hypothesis is
    outstanding (flags supplied by the runtime);
  * the action ranking is stable when actions were requested.

The tolerances are COMPUTE-CONTROL settings — explicit, configurable, recorded on the stopping
manifest, tested, and never part of causal state. When stability is not reached, the full budget
runs; nothing already rolled is ever discarded."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

LEAN_PARTICLES_VERSION = "lean.particles.v1"


@dataclass
class LeanParticleTolerances:
    """Compute-control settings only (§17): nothing here asserts anything about social reality."""
    batch_size: int = 8                      # == full-fidelity pilot floor
    min_particles: int = 8
    drift_tolerance: float = 0.04            # |p_k - p_{k-1}| across successive batch checkpoints
    ci_halfwidth_max: float = 0.13           # binomial SE-based interval half-width ceiling
    near_half_no_stop_margin: float = 0.0    # extra margin: interval crossing 0.5 blocks stopping
    unresolved_share_max: float = 0.35
    truncation_share_max: float = 0.5
    require_batches_stable: int = 2          # consecutive stable checkpoints required

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParticleStoppingRecord:
    model_id: str
    n_full_budget: int
    n_executed: int = 0
    stopped_early: bool = False
    stop_reason: str = ""
    checkpoints: list = field(default_factory=list)
    tolerances: dict = field(default_factory=dict)
    forced_full_reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"version": LEAN_PARTICLES_VERSION, **asdict(self),
                "particles_avoided": max(0, self.n_full_budget - self.n_executed)}


def _leading(dist: dict) -> tuple:
    if not dist:
        return "", 0.0
    k = max(dist, key=lambda o: dist[o])
    return k, float(dist[k])


def _halfwidth(p: float, n: int) -> float:
    if n <= 0:
        return 1.0
    return 1.96 * math.sqrt(max(p * (1.0 - p), 1e-6) / n)


def _action_ranking(projection: dict) -> tuple:
    acts = projection.get("action_distribution") or projection.get("actions") or {}
    if isinstance(acts, dict) and acts:
        return tuple(sorted(acts, key=lambda a: -float(acts[a]))[:3])
    return ()


def run_progressive_particles(handle: dict, *, seed: int, tolerances: LeanParticleTolerances,
                              particle_scope=None, model_id: str = "",
                              structural_disagreement: bool = False,
                              reversal_outstanding: bool = False,
                              outcome_pathway_settled: bool = True,
                              actions_requested: bool = False) -> tuple:
    """Roll batches [0,b), [b,2b), … through the prepared run; evaluate the stopping conditions
    at every checkpoint; return (branches, ParticleStoppingRecord). The first N particles of a
    stopped run are byte-identical to the first N of a full run (index-keyed sampling)."""
    from swm.world_model_v2.phase8_pipeline import run_persistence_slice
    n_full = int(handle["n_particles"])
    tol = tolerances
    rec = ParticleStoppingRecord(model_id=model_id, n_full_budget=n_full,
                                 tolerances=tol.as_dict())
    if structural_disagreement:
        rec.forced_full_reasons.append("material_structural_disagreement")
    if reversal_outstanding:
        rec.forced_full_reasons.append("reversal_capable_hypothesis_outstanding")
    if not outcome_pathway_settled:
        rec.forced_full_reasons.append("outcome_pathway_repairs_unsettled")
    branches: list = []
    prev_p, stable_streak, prev_ranking = None, 0, None
    while len(branches) < n_full:
        start = len(branches)
        stop = min(n_full, max(start + tol.batch_size, tol.min_particles))
        new = run_persistence_slice(handle, seed=seed, n_total=n_full, start=start, stop=stop,
                                    particle_scope=particle_scope)
        branches.extend(new)
        n = len(branches)
        projection = handle["run"].project(list(branches))
        dist = dict(projection.get("distribution") or {})
        unresolved = float(projection.get("unresolved_share") or 0.0)
        truncated = float(projection.get("truncated_share")
                          or projection.get("truncation_share") or 0.0)
        lead, p = _leading(dist)
        hw = _halfwidth(p, n)
        drift = None if prev_p is None else abs(p - prev_p)
        ranking = _action_ranking(projection)
        ranking_stable = (prev_ranking is None) or (ranking == prev_ranking)
        interval_crosses_half = (p - hw) < 0.5 < (p + hw) if lead else True
        conditions = {
            "drift_small": drift is not None and drift <= tol.drift_tolerance,
            "interval_narrow": hw <= tol.ci_halfwidth_max,
            "side_of_half_stable": not interval_crosses_half,
            "unresolved_below_ceiling": unresolved <= tol.unresolved_share_max,
            "truncation_below_ceiling": truncated <= tol.truncation_share_max,
            "no_structural_disagreement": not structural_disagreement,
            "no_outstanding_reversal": not reversal_outstanding,
            "outcome_pathway_settled": outcome_pathway_settled,
            "action_ranking_stable": (not actions_requested) or ranking_stable,
        }
        stable_now = all(conditions.values())
        stable_streak = (stable_streak + 1) if stable_now else 0
        rec.checkpoints.append({
            "n": n, "leading": lead, "p": round(p, 4), "halfwidth": round(hw, 4),
            "drift": None if drift is None else round(drift, 4),
            "unresolved_share": round(unresolved, 4), "truncated_share": round(truncated, 4),
            "action_ranking": list(ranking), "conditions": conditions,
            "stable_streak": stable_streak})
        prev_p, prev_ranking = p, ranking
        if n >= n_full:
            break
        if stable_streak >= tol.require_batches_stable and not rec.forced_full_reasons:
            rec.stopped_early = True
            rec.stop_reason = (f"all stopping conditions held for {stable_streak} consecutive "
                               f"checkpoints at n={n} (full budget {n_full})")
            break
    rec.n_executed = len(branches)
    if not rec.stopped_early:
        rec.stop_reason = rec.stop_reason or (
            "full budget executed: " + ("; ".join(rec.forced_full_reasons)
                                        if rec.forced_full_reasons
                                        else "stability conditions never held long enough"))
    return branches, rec
