# World Model V2 — Activation Completion v2 (the continuation, built)

This run builds the continuation manifest from the integration-completion run, inside the ONE canonical
runtime (no new pipeline): **P10 execution → P9 causal consumers + relevance gating → P7/P6 emission →
P4 strategic-actor gating → P11 shock validation**, then re-measures activation at the **execution** level
(StateDeltas + matched causal ablations) against the independent relevance labels.

## What was built

**Relevance gate** (`swm/world_model_v2/activation_synthesis.py::phase_requirements`): a deterministic
per-phase requirement judgment from the compiler's own causal analysis (required causal processes, the
question/interpretation wording, declared structural sections). Fixed *before* measurement; never reads
benchmark question IDs (regression-tested).

**Spec synthesis** (`synthesize_activation`, default-on in `unified_runtime` after rule normalization):
completes the missing execution linkage for REQUIRED phases from **already-declared** components, and gates
off ornamental execution for NOT-required ones. Never invents structure (tested: institutional process
named but no institution declared → nothing synthesized).

**New causal consumers** (`swm/world_model_v2/phase_consumers.py` — registered TransitionOperators, each
StateDelta-producing, broad-prior-labeled):
- **P10 `institutional_decision`** — member yes-propensity drawn from the *evidence-updated posterior* (the
  same base rate the terminal resolver uses — the institution *transforms* the rate, never invents one);
  votes ~ Binomial(n); the declared threshold/quorum rule decides; writes the canonical outcome quantity
  *before* `resolve_outcome` (the generic safety net already yields to a domain resolution). Tested: a 67/100
  supermajority passes far less often than 51/100 at the same propensity — the rule is causally real.
- **P9 `population_aggregation`** — declared segment weights × heterogeneity priors → per-particle aggregate.
- **P9 `network_diffusion`** — declared relations classified into semantic layers (communication/exposure/
  trust/influence/authority/alliance), per-particle independent-cascade percolation with layer-specific
  transmissibility priors → reach fraction.
- **P4 `actor_action_aggregation`** — chosen typed actions' lexical polarity (compiler negation lexicon +
  curated affirmative set) → affirmative share; nonpolar actions are skipped, never guessed.

**Terminal consumption channel**: `resolve_outcome` now carries bounded `rate_modulation` entries
(Σw ≤ 0.45, enforced at synthesis and again at apply); the generic resolver blends consumer quantities into
the base rate — so declared structure genuinely moves the terminal and its removal genuinely changes it.

**P6**: per-process registry selections (previously recorded in provenance and *dropped*) are promoted into
executable `behavioral_mechanism` pack events for dispatchable families with published packs; families
without a runnable pack are recorded as an honest omission.

**Sanitation**: a compiler decision event naming an undeclared actor used to crash the rollout; it is now
dropped with a recorded omission.

## Execution-level measurement (n=45 scored, 1 error; `experiments/results/integration/activation2.json`)

Recall = ≥1 StateDelta of the phase when the label says required; false execution = StateDeltas when not
required (post-gate); causal effect = a matched ablation (same plan, same seed, only the phase's synthesis
forced off) shifts the terminal by ≥0.02.

| Phase | exec recall (was, emission-era execution: ~0) | false execution (was) | causal-effect rate |
|---|---|---|---|
| p4  | **0.667** | **0.367** (0.742) | 0.10 |
| p6  | **0.000** | 0.095 | — |
| p7  | **0.889** (0) | 0.083 | 0.625 |
| p9pop | **0.583** (0 consumers) | **0.212** (1.000) | 0.429 |
| p9net | **0.222** (0 consumers) | **0.000** (0.946) | 0.50 |
| p10 | **0.833** (0 executable) | **0.212** (1.000) | **0.80** |

**Honest gate verdicts (13–19, not lowered):** the ≥0.95 recall / ≤0.10 false gates are still **NOT met**
for most phases — but every phase except p6 moved from zero (or from no-discrimination) to genuine gated
execution with measured causal effects. Failures are structural and named per row: p9net misses are mostly
questions where the compiler declared no relations to consume; p6's three required cases select no
dispatchable registry family; p4's causal-effect rate is limited by nonpolar (act/wait) action proposals.
Continuation: compiler-side emission for relations on contagion questions; more dispatchable pack families;
polar candidate-action proposals.

**Gate 20 (P11) — PASSES, fully** (`experiments/results/integration/phase11_shock_validation.json`):
through the real `RecompilationController`, 8 injected structural shocks (new actor/institution, dated rule
change, authority/coalition change, exogenous shock, outcome-space change, network restructuring) →
**trigger recall 1.0**; 4 stable + 6 adversarial near-miss controls (alias of a known actor, future-dated
rule, transient outage, causally-irrelevant actor, unsourced rule, known institution) → **false activation
0.0**; **migration integrity 1.0** (32/32 gate checks: no time reversal, no duplicate/lost events, non-empty
normalized ensembles).

## Phase-12 recalibration path

The runtime fingerprint now includes `activation-synthesis-1.0` + `phase-consumers-1.0`, so every earlier
corpus/calibrator remains **diagnostic_only / INCOMPATIBLE**; `experiments/phase12_refit.py --regen` (after
gates pass) is the product-eligible path, stamped with the current fingerprint.
