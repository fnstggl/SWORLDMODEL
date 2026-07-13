# WMv2 Phase 3 — Accuracy Limitations and Dependencies

*What this accuracy-hardening run does and does NOT establish. Read alongside
`WMV2_PHASE3_ACCURACY_ARCHITECTURE.md` and `WMV2_PHASE3_ACCURACY_VALIDATION.md`. All prior positive, null and
negative results are preserved; nothing was weakened to make a number move.*

## Power

- The new locked test targets **≥75** questions (the corpus holds **93**). Actual scored power is whatever the
  live capture completes; a handful of questions can fail the compiler (`KeyError: 'plan'` degenerate
  early-return) and are preserved, not hidden. If the scored count lands below 75, the run is **not** claimed
  as adequately powered and **not** production-eligible — the verdict downgrades to "improves but underpowered"
  and Phase-2 stays the default. The `powered` flag in `locked_results.json` records this.
- Even at ~75-90 questions, per-domain cells are small (2-12 each); domain breakdowns are **directional only**.

## Training size for the fitted observation models

- The fitted hierarchical observation model is fit on the **frozen 23-question dev set** (12 train / 11 val by
  event family). Partial pooling makes this usable, but 12 training questions is thin for a per-evidence-class
  fit; the class-level deltas are small and the global weight dominates. A larger labeled claim→outcome corpus
  (the mandate's Part 2 ideal) would support deeper pooling (source × horizon). This is a documented
  dependency, not done at scale here. The validation result (fitted_generic beats the baselines on val) is
  real but on 11 val questions.

## Causal latents — honest scope

- The causal-latent path is a **focused, real** implementation: typed latents with operational definitions,
  registered type-priors, a (fitted or hand-set) observation model, Beta posteriors, and registered
  combination mechanisms producing a rate that is scored against the outcome. The LLM proposes only
  qualitative structure; every number is registered/fitted.
- **Honest negative:** the raw necessary-conjunction mechanism is systematically pessimistic (products of
  ~0.5 latent means), and after the training-fit Platt recalibration the causal signal's slope (B≈0.11) is
  small — i.e. the causal arm, as implemented, carries **little discriminative power** beyond a near-constant.
  It is retained as an arm and a selector option, not asserted as the winner. Full per-latent WorldState
  materialization into the multi-mechanism rollout (rather than a combination mechanism producing the rate)
  remains a dependency; here the latent posteriors are consumed by a registered combination consumer and the
  terminal effect is measured, but they are not yet each injected as separate WorldState particles through the
  full rollout.

## What is genuinely new and validated (subject to the locked verdict)

- A **fitted** observation model that beats the hand-set tables / global gamma / raw posterior on validation.
- A **production selector** that uses only pre-outcome features and **safely falls back to Phase-2** — so the
  system can never do worse than Phase-2 by construction when it lacks support.
- A **much larger, family-disjoint locked test** for real power.

## Leakage discipline

- The 93 locked families are disjoint from BOTH frozen prior sets and from each other; as_of is strictly before
  resolution; retrieval uses the unchanged strict as-of machinery (paired `after:`/`before:`, per-document
  temporal verification, claim-level leakage audit). Residual risk is the same as prior runs (RSS date
  imprecision; widely-reported outcomes alluded to pre-as-of). The causal-latent proposal is made from the
  question + horizon only (no outcome), and claim mapping uses only the as-of claims.

## Reproducibility

- The numeric posterior + all offline arms are deterministic given the frozen capture + frozen params. Live
  retrieval drifts across wall-clock, so the capture records its retrieval date; the locked test is scored on
  its own frozen forecasts. Fitting never touches the locked corpus.

## What later phases MUST NOT assume

- Do not treat the causal-latent arm as a validated winner — it is, on this data, low-discrimination.
- Do not remove the Phase-2 safe fallback; the selector's guarantee depends on it.
- Do not cite validation-set improvement as the acceptance result — only the locked test with its power flag
  counts.

## Open dependencies (for a future run)

1. A large labeled claim→outcome corpus for deep hierarchical observation-model pooling (source × horizon).
2. Full per-latent WorldState particle materialization through the multi-mechanism rollout (not just a
   combination mechanism).
3. ≥150 total / ≥75 locked with balanced outcomes actually *scored* (not just authored) for a powered verdict.
4. Genuine as-of market/crowd baselines where licensing permits.
