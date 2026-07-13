# WMv2 Phase 3 — Audit & Architecture (Production Posterior World-State Inference)

*Phase 3 turns the Phase-2 evidence bundle into a **numeric, likelihood-updated posterior** over scenario-
specific hidden state AND competing causal structures, materialized as weighted WorldState particles that
mechanisms consume and that propagate posterior uncertainty into the terminal distribution. It replaces the
Phase-2 heuristic (`hypothesis_prior × 1.5/0.6` reweight; prior-only particle stratification) with real
observation-model likelihoods. The Phase-1 no-abstention contract is preserved: weak/contradictory/
unidentifiable evidence widens the posterior and lowers the support grade — it never blocks the forecast.*

---

## Part 0 — Ruthless audit of the current inference path

### Runtime call path traced (evidence → particle)
`evidence_pipeline.simulate_with_evidence` → `compile_world` → `requirements_from_plan` → `gather_evidence`
(EvidenceBundleV2) → `recompile_with_evidence` → `attach_evidence_observations` → `run_from_plan` →
`materialize._run_with_hypotheses` → `OutcomeContract.project`.

**The break:** at no point are Phase-2 CLAIMS converted into `Observation` objects, and at no point is a
likelihood computed. `_run_with_hypotheses` stratifies particles by the **prior** hypothesis weight and
reports `structural_posterior = normalized priors` — explicitly *"Phase-1 does not assimilate evidence to
reweight"*. `recompile_with_evidence` changes structure and applies a **heuristic** `hypothesis_reweight`
(`prior *= 1.5` up / `0.6` down) — not a likelihood. So evidence changes the qualitative plan (Phase 2), but
**no numeric posterior over hidden state exists on the production path.**

### Capability classification (file:symbol → status)

| capability | file:symbol | status | why insufficient |
|---|---|---|---|
| Weighted particles + likelihood reweight | `posterior.py:ParticlePosterior.assimilate` | **production-executable but ORPHANED** | never called on the evidence path; only `inference_layer`/tests use it |
| Effective sample size, systematic resampling, rejuvenation | `posterior.py` | executable, unvalidated on real evidence | provenance-aware jitter is real; never exercised by production |
| Observation likelihoods | `observation.py:GaussianMeasurement,BernoulliDetection` | executable | only **2** models; spec Part C needs ~20 claim-class-specific ones |
| Observation registry | `observation.py:register_observation_model` | executable | nothing registers models from Phase-2 claims |
| Hierarchical shrinkage (beta-binomial EB, normal) | `inference_layer.py:hierarchical_rates,shrunk_rate,shrunk_mean` | **production-executable but ORPHANED** | real partial pooling; never fed real evidence on the general path |
| Filtered rollout + structural posterior | `inference_layer.py:run_filtered` | **executable, synthetic-validated, ORPHANED** | computes a real likelihood-updated `structural_posterior`; not wired to `simulate_with_evidence` |
| Structural hypotheses as particle strata | `inference_layer.py:HypothesisSet.assign`; `materialize._run_with_hypotheses` | executable but **prior-only** on production | production uses priors; likelihood update lives only in `run_filtered` |
| Latent records, correlated joint sampling, coherence | `init_state.py:LatentVariableRecord,InitialStateModel,CorrelationRule,CoherenceRule` | production-executable | no typed spec (observation model, consumers, identifiability); no claim→latent mapping |
| Evidence-conditioned latent from counts | `inference_layer.py:latent_from_rate_evidence` | executable | needs real count evidence; not driven by Phase-2 claims |
| Claim → latent-variable specification | — | **absent** | the missing semantic mapping layer |
| Claim → observation-model selection | — | **absent** | the missing evidence→likelihood layer |
| Dependence-corrected likelihood | — | **absent** | `assimilate` multiplies each obs independently; Phase-2 dependence groups unused |
| Prior provenance + transport-risk inflation | partial (`init_state` methods labeled) | **partially implemented** | no versioned transport model, no prior-source registry |
| Posterior consumed by mechanisms on general path | — | **ornamental risk** | `_run_with_hypotheses` lean is consumed; latent posteriors are not systematically consumed |
| Real-data posterior validation on the general path | — | **absent** | `test_inference_layer.py` is synthetic recovery only |

### The 5 planes (spec) — current state
1. **Code plane**: strong (particle filter, shrinkage, observation likelihoods, filtered rollout) — reuse.
2. **Evidence plane**: Phase-2 bundle is real (claims, dependence, contradictions, temporal, visibility).
3. **Posterior plane**: exists in `run_filtered`/`ParticlePosterior` but **disconnected from real evidence**.
4. **World-state plane**: `InitialStateModel.sample_particles` materializes latents; posterior weights not applied.
5. **Execution plane**: mechanisms consume the lean; latent posteriors **not systematically consumed** — ornamental risk.

**Conclusion.** Phase 3's job is NOT to invent a particle filter — it exists. It is to build the **evidence→
posterior bridge** (claim → latent spec → observation model → dependence-corrected likelihood → assimilated
`ParticlePosterior` over state + structure), **wire it into the production general path**, make mechanisms
**consume** the posterior, and **validate on real data** with ablations proving lift over the Phase-2
qualitative reweight.

---

## Part 1 — Target architecture (built this run)

```
EvidenceBundleV2 (Phase 2)                          WorldExecutionPlan (Phase 1)
        │                                                    │
        └────────────► phase3_latent_spec ◄──────────────────┘   LatentVariableSpec (typed, per-variable
                        (LLM proposes latents from claims;         observation model, prior, consumers,
                         validated: measurable, support,           identifiability, sensitivity)
                         observation model, consumer)
                              │
                       phase3_priors  ──►  prior + provenance + transport-risk inflation (versioned)
                              │
                       phase3_observation  ──►  claim → Observation (of_path, value, reliability from
                              │                  source+dependence, reported_at from temporal, visibility);
                              │                  +claim-class-specific models (strategic/sincere statement,
                              │                  poll, official record, absence, financial commitment, …)
                              │
                       phase3_posterior  ──►  ParticlePosterior.assimilate with DEPENDENCE-CORRECTED
                              │                likelihood (one effective obs per Phase-2 dependence group);
                              │                joint posterior over latent state + structural hypotheses;
                              │                ESS / resample / rejuvenate; assimilation ledger
                              │
                       weighted WorldState particles (posterior draws; provenance = claim ids)
                              │
                       mechanisms consume particle-specific latent values  (consumption map)
                              │
                       StateDelta trajectories differ per particle
                              │
                       terminal distribution = posterior-weighted projection (not mean-world)
                              │
                       SimulationResult (+ posterior uncertainty decomposition, support grade)
```

Entry: `phase3_pipeline.simulate_with_posterior(question, …) -> (SimulationResult, artifacts)` — extends
`simulate_with_evidence`; the Phase-1 `simulate()` and Phase-2 `simulate_with_evidence()` remain unchanged.

(Design of each subsystem — latent spec, priors, observation models, dependence correction, structural
posterior, particle posterior, assimilation, WorldState/mechanism consumption — is documented section-by-
section below as each is implemented; validation and gate results are in `WMV2_PHASE3_VALIDATION.md`.)

---

## Part 2 — As-built (what this run actually implemented, honestly reconciled)

### Modules delivered
| module | role | plane |
|---|---|---|
| `phase3_latent_spec.py` | `LatentVariableSpec` (typed, support, observation model, **consumer**, identifiability, transport risk); `ClaimTag`; `tag_claims` (LLM qualitative tags → fixed reliability table) | CODE→EVIDENCE |
| `phase3_representation.py` | representation-choice: candidate KINDs (LLM-proposed) + executable fitters (scalar/continuous/discrete/mixture/hybrid) + held-out calibration selection + **anti-ornamental guard** | CODE |
| `phase3_priors.py` | reference-class prior + provenance + **transport-risk variance inflation**; generic-lean fallback | POSTERIOR |
| `phase3_observation.py` | `DirectionalRateModel` (P(claim dir \| rate r)); `StructuralDetectionModel` (P(claim \| h true)); `collapse_by_dependence` (Part D) | EVIDENCE→POSTERIOR |
| `phase3_posterior.py` | `infer_posterior`: prior → dependence-collapsed likelihood → particle posterior over rate + structural posterior; ESS/resample/rejuvenate; assimilation ledger; warnings | POSTERIOR |
| `phase3_pipeline.py` | `simulate_with_posterior`: compile→evidence→tag→**prior**→infer→materialize→consume→run; five-plane trace | all |
| `fallback.py` (edit) | `GenericOutcomeOperator` draws each Bernoulli rate from the **posterior** particles when present (`rate_source`), else the lean-Beta prior | EXECUTION |
| `materialize.py` (edit) | `_inject_posterior_rate` puts posterior particles on the `resolve_outcome` event (shared by both exec paths); `_run_with_hypotheses` weights strata by the **structural posterior** when present | WORLD-STATE→EXECUTION |
| `result.py` (edit) | additive `posterior_inference` field (prior→posterior deltas, ESS, ledger, latent specs) | — |

### The five planes — AS WIRED (with the crossing point named)
1. **CODE** — the modules above.
2. **EVIDENCE** — Phase-2 `EvidenceBundleV2.included_claims()` + `documents` + `dependence_group`. **Crossing:** `tag_claims` reads verified claims; `collapse_by_dependence` reads Phase-2 dependence groups.
3. **POSTERIOR** — `PosteriorResult`: numeric, likelihood-updated, reproducible, LLM-free numbers. **Crossing:** `infer_posterior` multiplies `DirectionalRateModel.likelihood` / `StructuralDetectionModel.loglik` into particle log-weights.
4. **WORLD-STATE** — `plan.posterior_rate_particles` + `plan.structural_posterior`. **Crossing:** `_inject_posterior_rate` copies particles onto the `resolve_outcome` payload; `_weight(h)` reads the structural posterior.
5. **EXECUTION** — `GenericOutcomeOperator.apply` draws the per-particle rate from the posterior (`rate_source=="posterior"` in the emitted `StateDelta.uncertainty`); strata allocated by posterior mass. **Proof of consumption:** the terminal `StateDelta` records `rate_source`, read back by the harness and asserted in `test_posterior_moves_the_terminal_distribution` and the live `rate_source` field.

> A posterior that stops at plane 3 is **scaffolding**; one stored at plane 4 but unread is **ornamental**. Every number here crosses into plane 5 and is read back from the terminal delta — see `WMV2_PHASE3_FORENSIC_TRACES.md`.

### LLM inference contract — as enforced
The LLM is the **semantic-mapping layer only**. It emits: qualitative claim tags (direction ∈ {supports_yes, supports_no, neutral}, supported hypotheses, strength bucket, is_strategic), candidate representation KINDs, a reference-class descriptor + qualitative transport-risk level. Every NUMBER — likelihood, sensitivity/specificity, reliability, dependence discount, prior α/β, transport inflation, structural weight, posterior mean, terminal probability — comes from a **fixed registered table or an explicit prior×likelihood update**, never the LLM. Enforcement points: `tag_claims` overwrites reliability from `_SOURCE_RELIABILITY` and only accepts enum tags; `DirectionalRateModel`/`StructuralDetectionModel` keep sens/spec/detect in module constants; `phase3_priors` takes base rates from DATA only; `test_llm_probability_minting_is_ignored` (Phase-1) still holds on this path.

### Representation-choice principle — as implemented
Hidden concepts are **not** auto-scalarized. Each is a typed representation chosen for causal adequacy + identifiability + **held-out calibration**, not intuition:
- `outcome_rate` → **continuous_probabilistic** (bounded [0,1] particle set) — the outcome is a probability; evidence is directional votes; the Bernoulli resolver consumes it.
- `structural_hypothesis` → **discrete_structural** — qualitatively distinct causal structures, kept distinct from their numeric posterior weights (which come only from likelihoods).
- `phase3_representation.choose_representation` empirically compares scalar/continuous/discrete/mixture/hybrid by held-out log-loss; `assert_not_ornamental` refuses any representation that is neither evidence-linked nor causally consumed (the `trust=0.7` anti-pattern). The empirical ablation (`experiments/phase3_representation_ablation.py`) shows the scalar baseline is dominated in every structured family and the winner is family- and identifiability-dependent.

### Honest scope deltas vs the Part-1 target
- **Observation models**: 2 production models (`DirectionalRateModel`, `StructuralDetectionModel`) drive the two consumed posteriors, not ~20. The registered-model *registry* and the reliability/strategic/dependence machinery exist; adding claim-class-specific models (poll, official record, absence, financial commitment) is a documented extension in `WMV2_PHASE3_LIMITATIONS_AND_DEPENDENCIES.md`, not silently skipped.
- **Sequential assimilation** is single-pass over dependence-collapsed observations (not a time-ordered filter over reported_at windows); `inference_layer.run_filtered` provides the windowed variant and remains available.
- **Correlated multi-latent joint state** (`init_state.CorrelationRule`) is not yet driven by claims on the general path; the current consumed latents are `outcome_rate` and `structural_hypothesis`.
- **Learned latent representation** is a declared candidate KIND without a fitter (dependency: a trained encoder) — named, not faked.

Status axes (never collapsed to "complete") — see the four-status table in the final report and `WMV2_PHASE3_VALIDATION.md`.
