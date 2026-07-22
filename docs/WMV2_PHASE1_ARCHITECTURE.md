# WMv2 Phase 1 — The Universal World Compiler (no-abstention, production)

*One entry point takes an arbitrary natural-language social question (+ as-of + optional intervention +
evidence + prior state + budget) and returns a terminal-state forecast for it. There is **no** scenario
branch, no keyword router, no domain hard-coding — a single generic pipeline compiles a causally-sufficient,
uncertainty-aware world, materializes it, runs a valid simulation, and reads the answer off the terminal
states. Epistemic weakness never blocks the forecast; it lowers the support grade.*

Entry: `swm.world_model_v2.pipeline.simulate(question, *, llm, evidence, as_of, horizon, intervention,
n_particles, seed, calibrator, cal_key) -> SimulationResult`. The runtime facade routes
`architecture="world_model_v2"` here (`swm/facade.py`); nothing downstream imports engines directly.

## Pipeline (question → SimulationResult)

```
NL question ─▶ compile_world ─▶ WorldExecutionPlan ─▶ build_world ─▶ WorldState
                    │                    │                                │
             LLM decomposition    causal-sufficiency               materialize entities/
             (qualitative only)   fidelity plan +                  institutions/latents,
                    │             fallback hierarchy               provenance-typed fields
             parse + salvage +          │                                │
             coherence check      guaranteed executable            InitialStateModel
                    │             resolver + bound readout          (particles, correlated latents,
             outcome contract           │                           structural hypotheses)
             (repaired, never                                              │
              refused)                                              RolloutEngine (event-driven,
                                                                    real calendar time, endogenous
                                                                    action→event chains)
                                                                          │
                                                                    OutcomeContract.project
                                                                    over TERMINAL states
                                                                          │
                                                                    result_from_run ─▶ SimulationResult
```

### B1–B3 · Compilation target and inputs
`compile_world` builds a `WorldExecutionPlan`: outcome contract, entities/populations/institutions/
relations, quantities, latents (always distributions), scheduled + stochastic events, accepted /
rejected / candidate-experimental mechanisms, mechanism tier choices, fallbacks used, structural
hypotheses, interpretations, omissions, fidelity + uncertainty + compute plans, support grade, degraded
flag, provenance, and a `plan_hash()`. The LLM proposes only **qualitative structure** — actors,
relationships, mechanism *names* from the registry, a directional `outcome_lean`, per-hypothesis leans,
and sensitivities. It may **not** mint final probabilities, latent values, edge strengths, population
weights, mechanism coefficients, institutional rules, intervention effects, or terminal outcome
probabilities (enforced: probabilities come from typed mechanisms / broad priors, never from the LLM
number).

### B4 · Iterative, self-checking compilation
The decomposition prompt (`_DECOMPOSE_PROMPT`) asks for causal sufficiency, competing interpretations,
outcome lean, structural hypotheses (with leans), required causal processes, and explicitly-listed
omissions (each with sensitivity + reason). Parsing is bounded-retry: a plain parse failure retries with a
"STRICT JSON only" nudge; a **truncated** reply is salvaged (`_salvage_json`) by recovering the parsed
prefix — the outcome contract, listed first, survives a `max_tokens` cut so a coherent question still
compiles rather than failing. A genuinely incoherent question (`coherent:false` and no outcome) →
`ClarificationRequired`.

### Causal sufficiency (the fidelity plan)
`_fidelity_plan` follows the **causal-sufficiency** mandate, not minimality: it includes every
actor/population/institution/relationship/latent/mechanism/exogenous factor that could materially change
the outcome distribution. HIGH-sensitivity components (incl. mechanisms, via the top-level `sensitivity`
map) get explicit representation; LOW-sensitivity ones are **marginalized *with uncertainty*** — kept and
represented, never dropped. Uncertain-relevance components are kept. Particle count scales with latents and
structural hypotheses (competing structures need coverage). High-sensitivity unknowns that are hard to
infer are **not** omitted — they become broad-prior latents, competing structural hypotheses, or explicit
sensitivity contributors.

### B5 · Structural hypotheses as competing particles
When the LLM proposes ≥2 structural hypotheses (each with a prior + a directional lean),
`materialize._run_with_hypotheses` stratifies particles across them by prior and overrides the terminal
resolver's lean per hypothesis, so competing *structures* produce genuinely different terminal outcomes.
The report carries `structural_disagreement` (= normalized priors materialized as particle mass; Phase 1
does not yet evidence-reweight — that is Phase 3). This is the mechanism-disagreement tier (7,
`highly_speculative`).

### B6–B8 · Fidelity planning, mechanism selection, guaranteed executability
Mechanisms named by the LLM are registry-vetted: accepted only if they resolve to an executable operator
and are not experimental; the rest are recorded as `rejected` (never fabricated) or
`candidate_experimental` (handled by the fallback hierarchy, **not executed** as validated). The
**fallback hierarchy** (see `WMV2_NO_ABSTENTION_MIGRATION.md` §2) then guarantees ≥1 executable mechanism
and a canonical outcome resolver: the `generic_outcome_prior` operator is **always** attached as the
terminal safety net. It writes the canonical readout quantity **only if unset at the horizon**, so a domain
mechanism that genuinely resolves the outcome takes precedence, and a plan whose domain mechanisms leave it
unset still produces a broad-prior forecast instead of an unresolved no-op.

### B9 · Mandatory terminal readout + repair
`_repair_readout` guarantees the terminal readout binds AND is written. A readout pointing at a declared
quantity the mechanism chain can populate is kept; anything else (an entity.field nothing writes on the
general path) is routed through the canonical `outcome` quantity the resolver writes. Binary option order
is normalized so the **affirmative** outcome is `options[0]` (the lean + projection convention), preventing
a silent polarity inversion when the LLM lists the negative option first. `check_readout_binding` +
`run_from_plan` raise `CompilerExecutionError` (not an abstention) if a readout is *technically*
unbindable — an engineering failure, taxonomy'd.

### B10 · Storage / reproducibility
Every compilation is persisted (`_persist_compilation` → `experiments/results/compiler_attempts/`, a
re-derivable runtime cache) keyed by `plan_hash`, which digests the question, contract, mechanisms,
structural hypotheses, and compiler version. `plan_hash` + the provenance block (prompt hash, evidence
bundle hash, compiler version, seed) make a run reproducible. The forward ledger
(`forward_ledger_v2.py`) locks forward forecasts with full provenance for fair later scoring.

## The result contract
`result_from_run` builds the `SimulationResult`: `simulation_status` / `support_grade` /
`recommendation_status`, the raw + calibrated distribution and binary projection, uncertainty
decomposition, structural + mechanism disagreement, evidence quality, limitations, fallbacks used
(each naming its tier), mechanism tiers, omitted high-sensitivity variables, sensitivity contributors,
interpretation hypotheses, plan hash, provenance, cost and latency. A forecast is present whenever the
simulation ran.

## What is NOT domain-hardcoded (B14)
`swm/world_model_v2/` contains no `if election / if email / if viral / …` scenario branch — enforced by a
static gate (`test_no_scenario_branches_in_v2_source` and the harness `_no_keyword_router` check). Every
domain flows through the identical compile→materialize→rollout→readout path; only the LLM's qualitative
decomposition and the registry's typed mechanisms differ per question.
