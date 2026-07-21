# WMv2 Phase 3 — Accuracy Architecture
*The three accuracy improvements added in this run — a larger real backtest, FITTED hierarchical observation models, and scenario-specific causal latents — plus the production selector. Every number is copied from a frozen artifact under `experiments/results/phase3acc/`. Prior positive, null and negative results are preserved.*

## Gap 1 — adequately powered real backtest
A NEW 93-question resolved corpus (`experiments/phase3acc_corpus.py`), event-family- AND temporally-disjoint from BOTH frozen prior sets (23-question diagnostic, 34-question Phase-3B locked). 9 domains, multiple horizons, all-new families. The prior 23/34 sets stay frozen as dev/prior-validation artifacts and are reused ONLY for fitting/selection, never as the new test.

## Gap 2 — fitted hierarchical observation models
`swm/world_model_v2/phase3_fitted_obs.py` learns, by penalized logistic regression with partial pooling, a per-evidence-class discrimination weight `w[class]` (shrunk toward a global weight) from training-question outcomes. Its per-observation likelihood ratio feeds both the generic rate and the causal-latent inference. Fit on TRAIN only.

Validation log-loss (lower better): fitted_generic **0.6016** vs phase3_raw **0.7096**, phase2 **0.7384**, prior_only **0.6931** — the fitted model is the best evidence arm on validation.

## Gap 3 — scenario-specific causal latents
`swm/world_model_v2/phase3_causal_latents.py`: for each question the LLM proposes (qualitatively only) a small set of TYPED latents (intent, capability, authority, feasibility, coalition, resources, readiness, hazard, regime) with operational definitions and a combination structure (necessary-conjunction / sufficient-disjunction / single-driver / weighted-mean); claims are mapped to latents. The NUMERIC inference is offline & deterministic: each latent has a registered type-prior, a (fitted or hand-set) observation model, a Beta posterior, and a registered combination mechanism producing the rate. Every number is registered/fitted; the LLM mints none.

Honest calibration finding: the raw necessary-conjunction mechanism is systematically pessimistic (products of ~0.5 latent means), so the causal rate is **Platt-recalibrated** on training (A=0.5560, B=0.1091). A small B indicates the raw causal signal carries **little discriminative power** after recalibration — a preserved negative for the causal approach as implemented.

## Production selector (Part 4)
Frozen policy: **fitted_gated** (min_effective=3). It uses ONLY pre-outcome features (non-neutral effective observation count) and **safely returns Phase-2** when Phase-3 lacks demonstrated support. Selected on validation among {phase2, repaired, fitted_gated, ensemble_gated, causal_gated}.

Dev split: train **12** / validation **11** (event-family disjoint). All params frozen before the locked test opened.
