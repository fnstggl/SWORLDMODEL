# WMv2 Best-Action & Counterfactual Evaluation (Phase 13)

Code: `swm/world_model_v2/rollout.py` (`evaluate_interventions`), `swm/world_model_v2/contracts.py`
(ActionSpace/Intervention/UtilityFunction). Artifact: `experiments/results/wmv2_best_action.json`.

## Two arms

### (A) Matched-counterfactual MECHANICS — validated

A controlled scenario (customer renewal, discount interventions) proves the machinery does the right thing:
- CLONED worlds per intervention, sampled ONCE then cloned (identical latent worlds across arms);
- COMMON RANDOM NUMBERS per particle (matched exogenous shocks);
- an intervention (`discount_boost`) that genuinely shifts a mechanism's transition;
- P(best) and expected-regret by PAIRED per-particle comparison (not unrelated random runs).

Result: the machinery correctly identifies `big_discount` as the best action (`correct_best=True`), with the
higher-boost intervention winning P(best) and carrying ~0 regret, `no_discount` the highest regret. The
matched design isolates intervention effect from world luck.

### (B) REAL randomized-intervention benchmark — Upworthy Research Archive

Each `clickability_test_id` randomly assigned headline variants to readers, so **CTR differences are
causal** — this is genuine randomized-experiment data, not predictive accuracy relabeled. Decision:
"which headline to publish." We score realized-CTR REGRET vs the oracle (best variant) and vs a random pick.

**Benchmark-integrity fix:** the loader pre-sorts variants by CTR (winner first), which leaks the answer
into position. Variants are shuffled deterministically before any policy sees them, so `first_listed` is a
genuine baseline.

| policy | mean realized CTR | mean regret | P(improve vs random) | lift vs random |
|---|---|---|---|---|
| random | 0.01603 | 0.00407 | 0.492 | 0.0 |
| first_listed | 0.01585 | 0.00425 | 0.495 | −0.00018 |
| longest | 0.01625 | 0.00385 | 0.524 | +0.00022 |
| **surface_model** (population world readout) | 0.01619 | 0.00390 | 0.516 | **+0.00017** |
| **oracle** | 0.02010 | 0.00000 | 1.000 | +0.00407 |

## The honest verdict

**Decision-lift mechanics work; decision lift on real intervention data is negligible.** The surface
population model recovers only ~4% of the oracle regret gap (+0.00017 of a possible 0.00407) — essentially
no lift over random. Headline CTR is nearly unpredictable from these features on this slice; the LLM
interpretation channel (quarantined) HURT in the prior round. So the counterfactual engine is correct, but
the underlying predictive signal is too weak to yield deployable decision lift here.

## What is and isn't demonstrated

- ✅ Typed interventions modify the shared world (not a post-hoc score); matched CRN; P(best)/regret/
  downside/expected-utility from terminal states.
- ✅ Evaluated on REAL randomized-experiment data (not synthetic-only).
- ❌ No meaningful realized decision lift on the available randomized-intervention benchmark.
- ❌ Sequential/adaptive policies, robust optimization across structural hypotheses, and off-policy
  estimation with propensities are implemented as interfaces (`decision/best_action.py` racing, policy
  search) but not validated on multi-step intervention data — a real gap, not claimed.

## Four-status

- **software-implemented**: YES (matched CF, action spaces, utility, P(best)/regret).
- **executes-end-to-end**: YES (interventions execute through the shared world).
- **empirically-validated**: mechanics YES (synthetic ground truth); real decision lift NEGATIVE (Upworthy).
- **production-eligible**: NO — no demonstrated decision lift on real intervention data; honestly reported.

## Answer to "does it produce real best-action lift?"

**No, not on the available randomized-intervention benchmark.** The counterfactual machinery is correct and
would produce lift if the underlying mechanisms were predictive enough; on Upworthy headline choice they are
not. This is a preserved negative, consistent with the round's pattern (structure is disciplined; lift
requires a mechanism with real held-out signal, which diffusion and persistence have and headline-CTR does
not).
