"""Stability-aware repeated runs — escalate only on internal instability signals, never by default.

Mean-of-K stays opt-in (`simulate_world_stable`). The lean profile runs ONE simulation and adds a
SECOND independent pass only when the run's own signals say the result may be unstable, capped at
one automatic escalation. Two instability kinds stay separate (§18):

  EXECUTION instability — the same compiled world produces unstable rollout outcomes. Tested by
  re-rolling the SAME prepared run (same evidence, structural models, temporal model, cohorts,
  shared artifacts, decision caches intact) at behavioral replicate index 1: every decision
  context re-keys, so equivalent situations get ONE fresh independent draw while every compiled
  artifact is reused.

  COMPILATION instability — independent compilation builds materially different worlds. Only the
  full-fidelity ensemble measures that; the lean run reports the signal and offers escalation
  instead of quietly regenerating everything to average a problem away.

Replicate results are REPORTED (per-replicate forecasts + spread), never silently averaged."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

LEAN_STABILITY_VERSION = "lean.stability.v1"


@dataclass
class StabilitySignals:
    primary_challenger_disagree: bool = False
    critic_unresolved_reversal_risk: bool = False
    evidence_insufficient: bool = False
    major_compilation_repair: bool = False
    outcome_pathway_repaired: bool = False
    progressive_never_stabilized: bool = False
    terminal_prior_divergence: bool = False
    details: list = field(default_factory=list)

    def escalate(self) -> bool:
        return any((self.primary_challenger_disagree, self.critic_unresolved_reversal_risk,
                    self.evidence_insufficient, self.major_compilation_repair,
                    self.outcome_pathway_repaired, self.progressive_never_stabilized,
                    self.terminal_prior_divergence))

    def as_dict(self) -> dict:
        return {"version": LEAN_STABILITY_VERSION, **asdict(self),
                "escalation_recommended": self.escalate()}


def detect_signals(*, ens, stopping_records: list, evidence_sufficiency: dict = None,
                   outcome_pathways: dict = None, model_results: dict = None,
                   posterior_means: dict = None) -> StabilitySignals:
    s = StabilitySignals()
    dists = {m: dict(getattr(r, "raw_distribution", None) or {})
             for m, r in (model_results or {}).items()}
    if len(dists) > 1:
        opts = set().union(*[set(d) for d in dists.values()]) if dists else set()
        vals = [[d.get(o, 0.0) for d in dists.values()] for o in opts]
        spread = max((max(v) - min(v) for v in vals), default=0.0)
        if spread > 0.1:
            s.primary_challenger_disagree = True
            s.details.append(f"primary/challenger max spread {spread:.3f} > 0.1")
    if getattr(ens, "structurally_underidentified", False):
        s.critic_unresolved_reversal_risk = True
        s.details.append("structurally underidentified (critic named unresolved alternatives)")
    if (evidence_sufficiency or {}).get("starved"):
        s.evidence_insufficient = True
        s.details.append("evidence-starved run (0 effective observations)")
    for m, op in (outcome_pathways or {}).items():
        if (op or {}).get("repaired"):
            s.outcome_pathway_repaired = True
            s.details.append(f"outcome pathway repaired on {m}: {op.get('repairs')}")
    for rec in stopping_records or []:
        if not rec.stopped_early and "stability conditions never held" in (rec.stop_reason or ""):
            s.progressive_never_stabilized = True
            s.details.append(f"{rec.model_id}: progressive particles never stabilized")
    for m, r in (model_results or {}).items():
        p = getattr(r, "raw_probability", None)
        prior = (posterior_means or {}).get(m)
        if p is not None and prior is not None and abs(float(p) - float(prior)) > 0.45:
            s.terminal_prior_divergence = True
            s.details.append(f"{m}: terminal {p:.3f} vs evidence-updated prior {prior:.3f} "
                             f"diverge sharply")
    return s


@dataclass
class ReplicateRun:
    replicate_index: int
    forecast: float = None
    distribution: dict = field(default_factory=dict)
    n_particles: int = 0
    status: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def execution_replicate(handle: dict, *, controller, seed: int, n_particles: int,
                        particle_scope=None) -> ReplicateRun:
    """The capped execution-instability probe: re-roll the SAME prepared run at behavioral
    replicate 1. Compiled artifacts, cohorts and caches are all reused; only decision draws
    re-key (replicate index enters every context signature)."""
    from swm.world_model_v2.phase8_pipeline import run_persistence_slice
    prev = controller.config.behavioral_replicates_per_decision_context
    controller.config.behavioral_replicates_per_decision_context = 2
    controller._replicate_override = 1
    orig = controller._replicate_for
    controller._replicate_for = lambda world: 1               # every particle → replicate 1
    try:
        branches = run_persistence_slice(handle, seed=seed, n_total=n_particles, start=0,
                                         stop=n_particles, particle_scope=particle_scope)
        projection = handle["run"].project(list(branches))
        dist = dict(projection.get("distribution") or {})
        lead_p = max(dist.values()) if dist else None
        return ReplicateRun(replicate_index=1, forecast=lead_p, distribution=dist,
                            n_particles=len(branches), status="completed")
    except Exception as e:  # noqa: BLE001 — the probe must never break the primary result
        return ReplicateRun(replicate_index=1, status=f"failed:{type(e).__name__}")
    finally:
        controller.config.behavioral_replicates_per_decision_context = prev
        controller._replicate_for = orig
        controller._replicate_override = None
