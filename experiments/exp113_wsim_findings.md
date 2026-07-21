# WSim — the simulation's own probability becomes the answer

Two coupled fixes so the general-purpose simulation genuinely starts AND its own weighted probability is
served (not silently replaced by the outside-view fallback).

## 1. Compiler-collapse fix (the simulation now starts)
EXP-112 showed 4/5 frozen questions collapsed at ensemble compilation: every well-formed candidate
(2–5 named actors, 2–4 mechanisms) was rejected because the compile-time `_executability_check` instantiated
operators with `llm=None`, and the default-on qualitative actor policy's §19 refusal aborted instantiation
before the terminal outcome writer. Fix (`ensemble_compiler._executability_check` + `materialize.
operators_from_plan(defer_backend_operators=…)`): the check DEFERS runtime-backend operators (present at
rollout, not a plan defect, not an outcome writer) and REPAIRS a missing outcome writer via
`ensure_outcome_pathway` instead of rejecting. Rollout path unchanged (still refuses, no silent numeric).
Result: **BoJ compiles 5 models and runs 1,348 actor decisions + institutional voting + evidence
assimilation + 2,280 nonlinear steps** — the world model actually executes.

## 2. Grounded-weighted simulation forecast (WSim)
The runtime was discarding completed simulations when worlds disagreed (§NAP suppressed the headline; the
fallback took over). New `swm/world_model_v2/model_weighting.py` + ensemble aggregation:

- **curate** — drop malformed / entirely-unresolved / fallback-derived worlds (recorded reasons);
- **merge duplicates** — same causal-story signature collapses so a repeated view can't dominate by count;
- **grounded, cited weights** — `weight ∝ objective_quality (support × evidence assimilation × resolved
  share × critic findings) × a CITED LLM plausibility`. The LLM must cite evidence / base rates / actor
  state / institutional constraints, name unsupported assumptions, judge setup AND simulated-behaviour
  realism, and must NOT reward agreement with the baseline (anti-circular). Uncited confidence is capped;
  no LLM ⇒ objective anchor (never equal weights);
- **weighted forecast** — `Σ (world weight × P(YES) inside that world)`; each world's P(YES) is the
  frequency across its own many particle rollouts (Bayesian model averaging: 0.6·0.2+0.3·0.7+0.1·0.9 =
  0.42, not the 0.6 mean). "Which world" (weight) is separated from "what happens in it" (rollout dist);
- **three transparent numbers** — outside-view, simulation-derived, and a final-combined that shrinks the
  simulation toward the baseline ONLY in proportion to how weak the simulation is (`alpha` from support /
  evidence / rollout depth / cross-world agreement) — regularization by support, never by agreement;
- **no silent substitution** — one selection contract: valid simulated worlds ⇒ source
  `weighted_simulation`; the outside-view fallback is used ONLY when no valid world exists. A test fails if
  a fallback substitutes while valid sims ran.

### P(YES) projection fix (caught in the first rerun)
BoJ's simulated distribution `{no_raise: 0.73, raise: 0.019, unresolved: 0.25}` was being read as P=0.73 —
which is P(*no_raise*), a max-fallback grabbing the modal outcome, the OPPOSITE of "will BoJ raise"; the
served headline was 0.0 (contract options didn't key the distribution). Fix (`affirmative_p` /
`weighted_p_yes`): identify P(YES) by contract options when they match, else SEMANTICALLY via the negation
lexicon (`raise` beats `no_raise`), excluding unresolved mass from the denominator. The headline is written
in canonical 2-option form so the served number always equals the reported simulation number.

## 3. Validation — BoJ (full fidelity)
| number | value |
|---|---|
| outside-view | 0.246 |
| **simulation-derived** | **0.199** (= served headline) |
| final-combined (alpha 0.169) | 0.238 |
| source | **weighted_simulation** |

4 valid worlds, real disagreement (spread 1.0), grounded/unequal/cited weights: `m2_resource_constraint`
w=0.595 p_yes=0.0 ("behaviour consistent: zero raise aligns with historical"); `m1_institutional` w=0.197
p_yes=1.0; `m0_actor_relationship` w=0.170 p_yes=0.012 ("behaviour inconsistent" → down-weighted);
`m4_adversarial` w=0.038 p_yes=0.0 (0 citations, thesis truncated → down-weighted). Weighted = 0.199.
The actor simulation honestly predicts BoJ HOLDS (actual outcome was a raise; SOTA 0.74) — so on BoJ the
simulation under-predicts the hawkish surprise. That is a genuine, traceable simulation reading, not an
artifact — exactly the signal needed to judge whether simulation beats the baseline.

Tests: 10 weighting + updated ensemble contract tests, all green (curation, dedup, unequal/cited weights,
uncited cap, normalization, weighted aggregation, semantic YES, combined-shrinkage, no-silent-substitution).

## 4. Full frozen-5 comparison — PENDING (rerun in progress)
Reporting per question: outside-view / simulation-derived / final-combined, world weights + citations,
final source, and whether the final number genuinely came from actor simulation vs. the fallback — then
rich-simulation vs. the lean grounded baseline.

## 5. Roadmap (the deeper architecture from the user's note)
Delivered here: the weighting/aggregation half — plausibility × outcome, grounded/cited weights, dedup,
three numbers, within-world rollouts already present. Next phase (not this iteration): replace
diversity-oriented independent generation with ONE grounded base world + controlled variants around the few
decisive uncertainties; weight worlds BEFORE simulation and update AFTER; and a pastcasting-calibrated
world-weighter to replace the LLM's uncalibrated judgment with learned plausibility.
