# Qualitative LLM actors — pilot evaluation report (honest)

**Run:** 2026-07-17, DeepSeek `deepseek-v4-flash` (decisions t=0.9 max_tokens=2000; hypotheses
t=0.8; persona arm t=0.3), 9 frozen historical decision cases
(`experiments/frozen_decision_cases.json`), K=3 hypotheses × 2 samples per case for the
counted arms, candidate order shuffled per case (label-position debiased), leakage rule
asserted per case (every evidence line predates `as_of`; the label never enters any prompt).
Raw artifacts: `actor_policy_benchmark_1784311395.json`, verbatim prompt/response captures
(`verbatim_exchange_khrushchev_*.txt`), demo artifacts (`qualitative_demo_*.json`).

**Claims ladder used throughout:** implemented / mechanically verified / plausible /
backtested / calibrated / statistically better than baseline / still speculative.

## 1. Headline numbers (n = 9 cases — treat every point estimate as wide)

| Arm | Next-action acc. | Top-2 | Log loss | Brier | Conf−acc gap | Novel rate | LLM calls | Latency |
|---|---|---|---|---|---|---|---|---|
| A `numeric_policy` | 0.444* | 0.556 | 1.386* | 0.750 | 0.19 | 0 | 0 | ~0s |
| B `persona_blended_numeric_policy` | 0.111 | 0.778 | 1.580 | 0.787 | 0.30 | 0† | 9 | 59s |
| C `stateless_llm_policy` | 0.556 | 0.556 | 1.138 | 0.778 | 0.28 | 0.33 | 54 | 606s |
| **D `persistent_qualitative_llm_policy`** | **0.778** | 0.778 | **0.805** | **0.512** | **0.00** | 0.17 | 54 | 881s |
| E `hybrid_relevant_actor_policy` (same engine as D, second run) | 0.556 | **0.889** | 0.918 | 0.630 | 0.17 | 0.17 | 55 | 914s |

\* The numeric arm has no fitted pack, no stances, and no intercept data for these actors: its
posterior was uniform 0.25 over four candidates in every case (log loss = ln 4 exactly). Its
44% "accuracy" is argmax tie-breaking luck; the honest description of arm A on these cases is
**an uninformed uniform prior**. † Arm B proposed novel actions in most cases but its blend
spread mass over them; several of its top-1 picks were semantically-correct novel phrasings
(`record_defiant_video` for Zelenskyy) that exact-match scoring counts as wrong.

Per-case correctness (Y/n, with p(actual)):

```
case        A-numeric  B-persona  C-stateless  D-persistent  E-hybrid
khrushchev  n .25      n .28      Y .83        Y .83         Y 1.00
nixon       Y .25      n .29      n .00        Y .67         n .33
yeltsin     Y .25      n .28      n .00        Y .67         Y 1.00
thatcher    n .25      n .04      n .17        n .00         n .17
musk        n .25      n .06      Y .67        Y .33         Y .33
johnson     Y .25      n .31      n .00        n .00         n .00
zelenskyy   Y .25      n .25      Y 1.00       Y 1.00        Y .50
deklerk     n .25      n .39      Y .67        Y .83         n .33
disney      n .25      Y .40      Y 1.00       Y 1.00        Y 1.00
```

## 2. What the pilot genuinely supports

1. **The architecture ordering matches the hypothesis.** Persona blend (LLM rates, numbers
   blend) ≪ stateless LLM choice < persistent qualitative choice, on every proper-scoring
   metric. Letting the model *choose* beats making it *rate*; giving it a persistent hidden
   reality beats a fresh role-play.
2. **The persistence ablation is the cleanest evidence.** C and D share model, prompts,
   evidence, cases, and sampling; the only difference is the inhabited hypothesis state.
   D beat C by +22pp accuracy and −0.33 log loss. Concretely: Nixon — C's stateless actor
   held to the public "no resignation" line in all 6 branches (p(exit)=0); D's
   private-doubt hypotheses resigned in 4 of 6.
3. **Counted distributions were the best-calibrated on their face.** D's mean confidence
   equaled its accuracy (gap 0.00); the blend arm was the most overconfident relative to
   accuracy. This is *not* fitted calibration — see §4.
4. **The mechanics behaved as specified live**: per-branch independent decisions that
   genuinely diverge by hypothesis; qualitative state updates with provenance; novel actions
   compiled to ontology anchors or explicitly marked `novel_action_unmodeled`; numeric policy
   never invoked for a routed choice; zero fallbacks across 163 LLM decisions (after the
   decision-first schema + truncation salvage fix — the first attempt lost every arm-C call
   to token-cap truncation, which is itself a finding about fragility).

## 3. What the pilot cannot support (read before quoting any number)

1. **Training contamination.** These nine episodes are famous history; the model plausibly
   *knows* the outcomes even though the prompts don't contain them. The pilot therefore
   measures *whether the architecture elicits what the model knows about how these people
   decide* — not true forward prediction. The B≪C<D ordering survives this caveat (all three
   share the contamination); the A-vs-D comparison does not (A cannot use world knowledge at
   all). Genuine validation requires post-training-cutoff cases via the sealed replay vault
   protocol.
2. **Run-to-run variance is large.** D and E ran the *identical* engine for these cases
   (every case actor is Tier 1 under the selector); their 0.778 vs 0.556 gap is sampling
   noise at K=3×2 samples and t=0.9 — i.e. ±2 cases (±22pp) between runs. n=9 with this
   variance means D-vs-A is suggestive, not statistically significant.
3. **A systematic failure mode: public-posture capture.** Every LLM arm scored 0 on Johnson
   and near-0 on Thatcher — resignation cases where the actor had *publicly vowed to fight
   on*. The inhabited personas honored the public posture over the collapsing private
   position; only Nixon's private-doubt hypotheses broke through. Simulated leaders
   under-predict capitulation. Hypothesis diversity is the designed lever (it worked for
   Nixon) but the hypothesizer must be pushed to include harder private-collapse variants.
4. **Exact-match scoring undercounts semantic accuracy.** The verbatim Khrushchev capture
   shows a chosen `direct_response_to_kennedy` whose stated intent is precisely `accept`;
   cluster-1.0 (deterministic, auditable) does not merge such phrasings onto candidates. A
   versioned LLM-assisted cluster→candidate mapping is the highest-leverage missing
   component for measurement.
5. **Single decision points, not trajectories.** Nothing here measures error compounding
   over long rollouts — the regime where persistence should matter most and where the
   literature documents degradation.
6. **One model family.** All minds share one prior; correlated error is unmeasured. The
   architecture supports per-actor backends; the pilot didn't exercise them.
7. **Calibration is absent by design and labeled.** Every distribution shipped
   `unvalidated`; 6 branches give 1/6 probability granularity. `ActorPolicyCalibrator`
   exists but has no fitted pack — fitting it needs a real historical decision corpus.
8. **Hand-fed evidence.** Five curated bullets per case, not the production evidence
   pipeline's grounded stances and information ledger; richer views could move results
   either way.

## 4. Demonstrations

**Single individual (dinner cancellation, `qualitative_demo_individual.json`).** DeepSeek
generated three distinguishable hidden Danas — `the_hurt_protector`,
`the_accommodating_pragmatist`, `the_silently_reconsidering` — from five lines of
relationship history. Each particle read the exact message, reacted internally ("the
disappointment feels like clarity now"), and chose an observable response; the same
hypothesis chose consistently across samples. Counted distribution: `reply_now` 78%,
`reply_later` 22%, labeled `unvalidated`. 9 calls, 79s.

**Multi-actor geopolitical (`qualitative_demo_geopolitical.json`).** Six branches, two
rounds, frozen ceasefire scenario; Putin and Zelenskyy as Tier-1 qualitative actors with
grounded stances; per-branch decisions executed through the standard path so pathway
quantities moved per branch; counted per-actor distributions vs the numeric arm on the same
scenario. See the artifact for the branch-by-branch decision log, hypothesis labels, state
evolution, and distribution comparison.

## 5. Cost and latency (DeepSeek `deepseek-v4-flash`, sequential)

Per qualitative decision: 1 call, ~10–18s, ~2.5–4k tokens in / ~0.7–1.5k out. Per case
(K=3×2): ~6 calls + amortized 1 hypothesizer call, ~90–120s wall. Whole 5-arm pilot: 172
calls, ~41 minutes wall. The hybrid mode's Tier-3 routing and the per-run budget are the
cost controls for production runs.

## 6. Verdict, on the ladder

- **Implemented, mechanically verified**: the full required architecture (qualitative
  particles, per-branch LLM choice, counted distributions, external calibration slot,
  causal tiering, single-individual mode, novel-action compilation) — 1212 tests green.
- **Backtested**: pilot-scale only, with the contamination caveat stated above.
- **Suggestively better than every baseline** on this pilot, with the persistence ablation
  (C→D) as the cleanest architecture signal, and honest variance bars of ±2 cases.
- **Not calibrated**; **not statistically significant**; **trajectory fidelity, model
  diversity, and novel-action semantic clustering remain open**.

## 7. What would make this a real validation

1. Post-cutoff frozen cases through the sealed replay vault (kills contamination).
2. 50–200 cases with actor-decision ground truth; paired McNemar tests between arms.
3. Versioned LLM-assisted semantic clustering of chosen actions onto candidates.
4. Fit `ActorPolicyCalibrator` on the corpus; report reliability curves.
5. K=5+ hypotheses with an adversarial "private collapse" generation lens (targets the
   public-posture-capture failure).
6. ≥2 model families per actor; measure cross-model distribution divergence.
7. Multi-event trajectory scoring on the event-time vault (CRPS), not just next actions.
