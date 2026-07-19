# Structural-Model Uncertainty — the Default World Model V2 Runtime

**The exact claim this architecture makes:** the system now simulates uncertainty both *within*
plausible causal models and *across* materially different causal models.

**What it does not claim:** calibration; predictive superiority; real-world accuracy; optimal model
coverage; consequential-recommendation readiness. Each of those requires separate held-out evidence that
does not yet exist. The live forensic runs under `artifacts/structural_ensemble/forensics/` are
architecture probes — proof of the execution path and its cost, never of accuracy.

---

## 1. Structural uncertainty versus particle uncertainty

The runtime separates three uncertainty levels and never collapses them:

| level | uncertainty about | representation |
|---|---|---|
| **A. structural-model** | which actors, institutions, constraints, mechanisms, boundaries and information routes determine the outcome at all | its own independently generated, independently executable `WorldExecutionPlan` per model |
| **B. within-model world** | hidden facts, private states, exogenous events, parameters, initial conditions inside ONE structure | particles / coherent world hypotheses inside that model's plan (`structural_hypotheses`, latents, posterior particles) |
| **C. behavioral** | what actors do after seeing their own information | qualitative actor decisions across coherent particles |

A structural model is **not** a random seed, a parameter draw, an entry in
`plan.structural_hypotheses`, an outcome mode, or a narrative label on a shared schema. It is a distinct
executable causal representation that can contain different actors, institutions, boundaries,
relationships, mechanisms, constraints, information routes, action-response pathways, scheduled events
and terminal dependencies. Two candidates whose *prose* differs but whose executable structure is
equivalent are conservatively merged (with a recorded certificate); two candidates that differ on any
result-relevant structural element are never merged.

**Why one valid schema is insufficient:** a perfectly executed simulation of the wrong causal model is
still wrong. Level-B machinery (more particles, wider priors) explores hidden states *inside* one
explanation of how the situation works; it cannot represent the possibility that the explanation itself
is wrong — that the decisive actor, institution or constraint is missing from the schema entirely.

## 2. The default pipeline

```
question, context, intervention, as-of evidence
→ Stage A: independent causal reconnaissance — SEPARATE actual LLM calls (target 4, min 3),
  each blind to the others, each through a different general causal perspective
→ adversarial critics: structural-omission, per-candidate causal, cross-model contrast
  (+ deterministic executability check); adaptive expansion up to a soft ceiling (~8; higher in
  maximum-capacity mode) — the ceiling is never proof of completeness
→ shared evidence gathered ONCE (union of recon requirements, one as-of boundary, immutable bundle)
→ Stage B: each surviving candidate compiled SEPARATELY into its own executable plan
  (same question/intervention/cutoff/horizon/bundle + its OWN causal thesis as structural directive)
→ conservative dedup: deterministic structural comparison first; a blind-label LLM equivalence
  judge only for unresolved near-matches; merge only on high-confidence equivalence
→ per-model conditioning: evidence recompile, ITS OWN posterior, fidelity, event-time, Phase 11
→ REAL pilot per plausible model (canonical funnel, full causal depth, ~20% of that model's own
  full particle budget, absolute floor 8)
→ uncertainty-aware conservative promotion
→ full per-model budgets: every promoted model ≥ its complete single-model particle count,
  pilot particles reused as a deterministic prefix; extra particles when disagreement is material
→ per-model distributions + labeled equal-weight mixture + robust range + sensitivity
  classification + reversal conditions + structural value-of-information
```

Implementation: `swm/world_model_v2/structural_contracts.py` (typed contracts),
`ensemble_compiler.py` (Stage A/B + critics + dedup), `structural_runtime.py` (pilots, promotion,
budgets, aggregation), `llm_call_cache.py` (metering + content-addressed caching),
`phase13/ensemble.py` (cross-model decisions).

## 3. Default public entrypoints

`swm.facade.forecast(..., architecture="world_model_v2")` → `simulate_world(...)` runs the ensemble by
default for **every** route: binary, categorical, event-time (`when` questions), personal/individual
reactions (frame ensembles through the qualitative-actor runtime), population/institutional/intervention
simulations, historical replays with frozen bundles, and Phase 13 (`recommend_action` /
`evaluate_actions` / `optimize_policy` refuse a bare single plan without the explicit ablation flag).
The ordinary caller enables nothing.

**Explicit single-model ablation** (the only sanctioned single-plan path):
`execution_policy={"structural_mode": "single_structural_model"}` — for scientific ablations, frozen
historical artifact compatibility and isolated compiler tests. AST + call-spy tests
(`tests/test_structural_ensemble_enforcement.py`) fail if the production runtime invokes the single-plan
compiler outside the ensemble's Stage B.

## 4. Independent generation and the adaptive model count

Stage A makes separate actual LLM calls — never "one call returning four alternatives" — each blind to
every other candidate, through general reasoning perspectives (actor/relationship, institutional/
procedural, resource/constraint, information/distribution, exogenous/external, adversarial
alternative). These shape attention, not scope: every output is a complete causal model. Expansion
triggers (critic-identified missing structures, shared decisive assumptions, mostly-duplicates,
missing axes) add candidates; convergence, critic exhaustion, or the ceiling stop generation. Hitting
the ceiling with open critic findings marks the run **structurally_underidentified** — surfaced in the
result, never hidden in metadata.

## 5. Model support without minted probabilities

LLM critics assign only qualitative support classes (`strongly_supported / plausible /
weak_but_possible / contradicted / unresolved`) grounded in evidence fit; numeric fields a critic
returns anyway are stripped (`_strip_minted_probabilities`, tested). With no defensible weights, the
aggregate is an explicitly labeled **equal-weight uncalibrated structural average** plus per-model
distributions, min/max robust ranges, and a between-model/within-model variance decomposition. Material
disagreement lowers the support grade and is explained, never averaged away.

## 6. Pilots, promotion and full budgets

Pilots run through the **canonical runtime** — the model's actual conditioned plan, its own posterior,
qualitative actors, institutions, real event queue, real horizon; only the particle count is reduced
(default 20% of that model's own budget, floor 8 — chosen against the production minimum full budget of
12 so pilots resolve a binary distribution but stay below the smallest full run). Promotion is
conservative and uncertainty-aware: a model is excluded from full simulation only on hard grounds
(invalid, nonexecutable after bounded repair, evidence-contradicted with cited claims, genuinely
equivalent, or pilot-indistinguishable from a structurally similar *stronger* model with enough pilot
particles that the equality is not noise). Never for a low probability, an inconvenient result, a bad
action, critic preference, or pilot-mean rank; a noisy pilot always promotes.

Every promoted model then receives **at least the complete single-model production budget** — pilot
particles are a deterministic prefix (index-keyed worlds and seeds; progressive N-small → N-full equals
direct N-full, tested), so pilot computation is reused, never discarded. Budgets are never divided
across models (`three models × N/3` raises `EnsembleIntegrityError`). Material structural disagreement
adds particles (+25% per promoted model), never removes them.

## 7. Cost controls that cannot reduce accuracy

- **shared evidence**: gathered once per run under one as-of boundary (union of recon requirements);
- **content-addressed LLM cache**: byte-identical (backend-fingerprint, prompt) inputs only — merely
  similar prompts never share; identical conditioning calls across models (e.g. resolution-criterion
  parsing) hit for free;
- **cross-model actor-decision sharing**: an actor with the byte-identical view/prompt at the same
  particle index (same CRN seed law across models) reuses the decision — cheaper *and* better
  controlled; any differing causal input forces a new call; within-model behavioral variance untouched;
- **common random numbers**: every model rolls particle *i* with the same exogenous seed law, so
  cross-model differences are structural, not sampling luck;
- **pilot-prefix reuse** and **conservative dedup** (screening removes only genuinely redundant work);
- **no weak fallback**: a missing backend fails loudly (`unavailable_service`); numeric actor fallback
  is surfaced as degradation and caps the support grade; nothing silently downgrades models, horizons,
  operators or particle counts.

Observed multipliers live in `artifacts/structural_ensemble/cost_benchmark.json` (scripted-backend
exactness) and `artifacts/structural_ensemble/forensics/` (live counts). They depend on how many models
survive and how many actor calls they need; no fixed multiplier is promised.

## 8. Decisions, reversal conditions, structural value-of-information

Phase 13 evaluates every action inside every surviving model (full canonical per-model pipeline, same
seed → CRN-aligned) and reports winner-by-model, per-model rankings and feasibility, worst-model
downside, minimax regret across models, a labeled equal-mixture view, and a stability classification.
Split winners produce a conditional strategy + information-gathering recommendation (or a robust set) —
never one average utility hiding the disagreement. Forecast results carry reversal conditions (which
concrete model's assumption flips the answer, with its falsifiers) and structural VOI (observations that
distinguish surviving models, derived from the models' own falsifiers/evidence requirements and actual
predicted differences — no minted EVSI numbers).

## 9. Known limitations

- Support classes are qualitative critic judgments over evidence fit; no fitted model-selection prior
  exists, so the mixture is explicitly uncalibrated.
- Sensitivity thresholds (`SENSITIVITY_*` in `structural_contracts.py`,
  `DECISION_REGRET_MATERIAL_SHARE` in `phase13/ensemble.py`) are exposed and tested for monotone
  behavior but not yet validated against outcome data.
- Candidate-specific supplemental evidence beyond the shared bundle is recorded as an open requirement
  rather than fetched live.
- The equivalence judge sees blind labels but remains an LLM judgment; merges therefore additionally
  require deterministic structural near-equality, and false-merge risk is biased toward retaining
  duplicates.
- Live forensic runs prove the execution path and cost, not accuracy; the Phase-12 calibrator remains
  incompatible until refit on post-ensemble output.
