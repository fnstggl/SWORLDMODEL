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
