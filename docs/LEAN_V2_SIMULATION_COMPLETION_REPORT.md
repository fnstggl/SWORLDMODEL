# Lean V2 Simulation-Completion — implementation & five-question measurement

## The failure this fixed

The prior Lean V2 simulation resolved almost no world mass: ~100% of every forecast came
from the historical prior and ~0% from resolved simulation. The root cause was
architectural — the system treated **"we do not know the actor's true private state"** as
**"this simulated world cannot produce an answer."** Uncertainty about a private state is the
*reason* to simulate multiple weighted worlds, not a reason to stop.

This work rebuilds the simulation-completion layer at its source. No prior-versus-simulation
combiner is calibrated and no BTF-3 outcome is trained on — per the directive, the goal is
first to make the simulation *capable* (resolve mass, produce a valid simulation-only
probability), and to *measure* its accuracy separately.

## What changed (canonical `simulate_world(..., execution_profile="lean_v2")`)

1. **Actor-state completeness is a hard invariant** (`state_completeness.py`). Every
   consequential actor must exit with a non-empty weighted state set via a 4-attempt recovery
   ladder (deterministic alias/parse repair → targeted regeneration → targeted actor-local
   evidence → grounded decision-spanning fallback basis). An empty set can never reach
   rollout; a provider failure mid-ladder is recorded and the ladder continues.
2. **Unknown-state worlds are gone** (`engine.py`). Represented states carry the full branch
   mass (weights normalize to 1). Omitted-state doubt is a small **bounded per-actor
   residual** widening the interval `1 − ∏(1−r_a)` — never a branch, never multiplied across
   actors. A consequential actor with zero represented states fails **loudly** before rollout.
3. **A readiness gate proves the world can answer before rollout** (`readiness.py`). One pure
   terminal law + a synthetic YES→1 / NO→0 round-trip through the *same* evaluator + recovery
   path the live run uses + terminal-writer canonicalization. A round-trip that stays broken
   is a **hard stop** — never a silent fall-back to the prior.
4. **Missing mechanisms are diagnosed and repaired** (`mechanisms.py`). A numeric-threshold
   terminal with no bridge climbs a 5-attempt ladder (reuse → deterministic threshold parse →
   leakage-checked verbatim observation extraction → regime mapping → bounded grounded
   approximation). Only a *proven* impossibility leaves `missing_mechanism` — with its proof.
5. **Deadline-forced completion + completion audit.** Waiting actors are reopened at the
   deadline; a required participant who keeps waiting past a hard deadline (abstention not
   permitted) is **forced to a substantive vote drawn from their simulated state's own
   action** — resolving the world instead of leaving it as dead mass. A `SimulationCompletionAudit`
   tracks resolved-vs-unresolved-by-cause mass against explicit acceptance targets.
6. **Terminal-law correctness.** Vote thresholds evaluate absolute "at least N" counts and
   fractional majorities correctly, option matching is case-normalized, and a **mis-scaled
   absolute threshold** (26 votes for a 50-seat parliament modeled as 5 actors; 5 of 9 modeled
   as 5) is translated to a *majority of the modeled votes* — the semantic the `majority` rule
   actually means. This is what made Wale simulable and BoJ non-trivial.
7. **Prior and simulation stay fully separate.** Every run reports `prior_forecast`,
   `simulation_forecast`, `resolved_simulation_mass`, `unresolved_mass_by_cause`,
   `simulation_probability_bounds`, `headline_forecast`, `headline_source`. The engine's
   distribution always emits both binary keys, so an all-NO simulation (P=0) is a valid mapped
   forecast that never silently reverts to the prior.

25 focused tests (`tests/test_lean_v2_completion.py`) plus the updated existing suites — 108
tests green.

## Five-question measurement (EXP-113, same frozen BTF-3 rows, no leakage, 12 min / 100 call guard)

| # | Question | Outcome | Prior p (n) | Sim-cond p | Resolved | Unresolved by cause | Final | Final−prior | Sim−prior | Sim right dir? | Prior Brier | Sim Brier | Final Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Banxico | 1 | 0.833 (2) | 0.099 | 1.00 | none | 0.099 | −0.735 | −0.735 | no | 0.028 | 0.812 | 0.812 |
| 2 | BoJ | 1 | 0.875 (3) | 0.036 | 1.00 | none | 0.036 | −0.839 | −0.839 | no | 0.016 | 0.929 | 0.929 |
| 3 | visionOS | 1 | 0.833 (2) | 1.00 | 0.75 | missing_mechanism 0.25 | 0.958 | +0.125 | +0.167 | **YES** | 0.028 | 0.000 | 0.002 |
| 4 | Wale | 1 | 0.167 (2) | 0.130 | 1.00 | none | 0.130 | −0.036 | −0.036 | no | 0.694 | 0.757 | 0.757 |
| 5 | Hormuz | 0 | 0.500 (2) | 1.00 | 0.667 | missing_mechanism 0.33 | 0.833 | +0.333 | +0.500 | no | 0.250 | 1.000 | 0.694 |

Across all five: mean prior Brier **0.203**, mean simulation-only Brier **0.700**, mean final
Brier **0.639**, mean resolved mass **0.883**, simulation moved toward the outcome **1/5**,
terminal `unknown_state` mass **0/5**, guard passed **5/5** (143 calls / 757 s / $0.12 total).

## What this shows — read honestly

**The completion architecture succeeded.** Every §21 acceptance criterion for the *completion*
fix is met:
- no required actor has an empty state set;
- terminal `unknown_state` mass is **zero on all five** (the central failure — eliminated);
- terminal mapping failures are zero (every round-trip passes; the visionOS discard bug is dead);
- `missing_mechanism` remains only where **proven unavoidable** (visionOS 0.25, Hormuz 0.33 —
  the resolution criterion carries no parseable numeric threshold, proof recorded);
- provider/parser failures never became forecast mass;
- mandatory decisions complete (Banxico/BoJ/Wale all resolve 100%);
- resolved simulation mass is substantial (mean 0.883);
- simulation-only probabilities are produced and **scoreable on all five**;
- no prior silently replaces a resolved simulation (the fall-back bug is fixed — headline now
  tracks the simulation, e.g. BoJ 0.036 not 0.875);
- no outcome leakage.

**The simulation-only accuracy is currently poor, and that is the point of measuring it.** The
raw simulation is right on 1/5 (visionOS, where it beat both the prior *and* full-fidelity) and
confidently wrong on the other four. The pattern is a systematic bias toward *disagreement /
change*: it models central-bank boards and parliaments as splitting when they in fact reached
consensus (Banxico unified after a 3-2 split; BoJ raised; Wale won an upset) and models a
Hormuz disruption that did not occur. This is a defensible-but-wrong simulation, not a
mechanical failure — exactly the signal the "measure simulation-only accuracy before
calibrating" directive was meant to surface.

The exp112 mean Brier of 0.074 was *the prior scoring well on five questions*, not the
simulation working. Now the simulation genuinely runs to completion and we can see it is not
yet more accurate than the prior. The prior remains visible and separate; the system does not
claim to have passed on accuracy. The next phase — deferred here — is to reduce the simulation's
disagreement bias (correlation/convergence in deliberation, stronger evidence-grounded state
weighting) and only then learn a leakage-audited prior↔simulation combiner.

Full under-the-hood traces (exact prompts and responses per call, six manifests per question)
are under `experiments/results/lean_v2_accuracy/<qid>-completion/`.
