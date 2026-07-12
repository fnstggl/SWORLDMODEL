# WMv2 Posterior World-State Inference (Phase 3)

Code: `swm/world_model_v2/inference_layer.py`, `init_state.py`, `posterior.py`, `observation.py`.
Recovery tests: `tests/test_inference_layer.py` (5, all pass).

## What exists (the audit found NO inference — only prior sampling)

1. **Hierarchical shrinkage estimators** (`hierarchical_rates`, `shrunk_rate`, `shrunk_mean`): beta-binomial
   and normal partial pooling (person←segment←population) with empirical-Bayes method-of-moments precision.
   Every posterior carries a distribution + uncertainty, never a point.
2. **Evidence-conditioned latents** (`latent_from_rate_evidence`, `latent_from_llm_with_floor`): build
   `LatentVariableRecord`s from real count/value evidence with provenance, evidence dependencies, and
   confidence from posterior concentration. LLM-proposed latents get an enforced sd FLOOR (no unsupported
   precision).
3. **Structural hypotheses** (`StructuralHypothesis`, `HypothesisSet`): competing world structures carried
   as per-particle mechanism/parameter/world-patch assignments with prior weights; stratified assignment
   guarantees every hypothesis with prior ≥ 1/N gets a particle.
4. **Filtered rollout** (`run_filtered`): the assimilation loop the audit found missing — roll all branches
   to each observation's reported_at, reweight through the observation model, resample on ESS collapse,
   continue; **branch weights genuinely change**; model (hypothesis) posterior reported separately from
   within-world randomness.
5. **Correlated latents** (`CorrelationRule`, range-corrected): couplings (workload↔attention,
   responsiveness↔attention) sample jointly; the range fix removed a silent bias for non-[0,1] latents.

## Recovery validation (semi-synthetic, ground truth known)

| test | result |
|---|---|
| hierarchical shrinkage vs no-pooling AND full-pooling | shrinkage MSE < both, on 40 persons × 8 obs; posterior sd shrinks with n |
| normal shrinkage moves toward data with n | prior-only → 2-obs → 20-obs monotone toward truth; sd shrinks |
| filtered rollout recovers hidden drift | truth drift 0.8; posterior recovers within 0.15 of truth (prior mean 0.5); filtered terminal median beats prior-only rollout |
| filtered rollout moves weights | max branch weight or an ESS-triggered resample fires |
| structural posterior concentrates on generating mechanism | fast-drift generated data → structural posterior fast > 0.85 (prior 0.5) |

## Downstream causal effect (not ornamental)

The filtered-rollout test proves inferred state changes terminal prediction: the assimilated posterior's
terminal median is closer to truth than the unfiltered prior rollout's. In the structural test the
hypothesis posterior (fast vs slow) drives which operator set produces the terminal outcome. Particles are
causally distinct worlds whose latents mechanisms read.

## Four-status

- **software-implemented**: YES (estimators, filtering, structural hypotheses, observation models).
- **executes-end-to-end**: YES (filtered rollout integrated with RolloutEngine + contract projection).
- **empirically-validated**: recovery on semi-synthetic ground truth (posteriors beat priors; structural
  weight concentrates on the truth). Real-data latent-recovery (person-disjoint) is demonstrated indirectly
  via the persistence result (hierarchical user-rate shrinkage carries the persistence win, transfer
  Δ−0.027). Full multi-domain SBC is the remaining validation.
- **production-eligible**: hierarchical shrinkage YES (load-bearing in the persistence win); the full
  filtered-assimilation loop is validated on synthetic + one real stream, not yet across domains.

## Limitations (stated)

- Inference engines implemented: conjugate updates, empirical Bayes, particle filtering. NOT yet: full MCMC
  / variational (offline-fit hooks exist in `ingestion.py`).
- Contradictory-evidence handling is at the observation-model level (reliability-flattened likelihood);
  a full claim-level contradiction graph feeding the posterior is built in `evidence.py`/`leakage_audit.py`
  but not yet wired into `run_filtered`.
- Resampling rebuilds queues from world state (documented approximation: in-flight queue state not derivable
  from the world does not survive resampling).
