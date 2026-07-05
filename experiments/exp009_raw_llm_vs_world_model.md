# EXP-009 — Raw LLM vs. world model (the hard-rule test)

**The directive's hard rule:** test whether actual world-state simulation beats raw LLM + retrieved
context. If it does not, say so.

**Headline (said plainly):** on real HN, **explicit world-state simulation does NOT decisively beat
a raw LLM**. On identical items the state models and the raw LLM are a **statistical tie** (bootstrap
CIs straddle zero); **as-of retrieval did not help the raw LLM** (it slightly hurt, via
overconfidence); and the single **statistically significant** lever is the **calibration layer**,
not the state machinery. State *does* clearly beat a naive structured model and is the best-calibrated
non-LLM tier — but a raw LLM reasoning about a title already captures most of what the state model
provides, because its pretrained knowledge of HN dynamics is itself a substitute for explicit state.

This is the honest, uncomfortable answer the directive asked for. Details below.

## Setup (identical items, no cheating)
- 1,800 real HN stories (Feb–Apr 2026, post-Jan-2026 cutoff → contamination-free), temporal 70/30
  split. Common LLM subset = the 100 most recent test stories (base rate P(≥40)=0.08). Non-LLM tiers
  also scored on the full 540-story test set for power.
- "As-of context" for HN = the author's past-submission track record + the domain's past performance,
  computed strictly before each story (leakage-free retrieval — the one honest retrieval signal
  available without a news corpus).
- LLM tiers produced by a 5-agent swarm (the only LLM available; no ANTHROPIC_API_KEY), title-only
  and title+context, blind to outcomes, no browsing. Non-LLM tiers run live and are reproducible.

## Results — the six tiers on the common n=100 (base rate 0.08)
| tier | log loss | Brier | ECE | uplift@20 |
|---|---|---|---|---|
| 1. raw LLM (title only) | 0.2859 | 0.0773 | 0.0195 | −0.030 |
| 2. raw LLM + as-of context | 0.2996 | 0.0805 | 0.0546 | +0.020 |
| 3. structured (as-of features) | 0.3262 | 0.0902 | 0.0727 | −0.030 |
| 4. calibrated (structured + Platt) | **0.2781** | 0.0736 | 0.0098 | −0.030 |
| 5. aggregate world (state-transition) | 0.2842 | 0.0750 | 0.0244 | −0.030 |
| 6. individual world (per-author) | 0.2792 | 0.0736 | 0.0095 | +0.020 |

## The statistics that matter (paired bootstrap of per-item log-loss difference, n=100)
| comparison | Δ log loss | 95% CI | significant? |
|---|---|---|---|
| calibrated − raw LLM | −0.0078 | [−0.034, +0.019] | **no** |
| individual world − raw LLM | −0.0068 | [−0.032, +0.019] | **no** |
| aggregate world − raw LLM | −0.0018 | [−0.030, +0.023] | **no** |
| raw LLM + context − raw LLM | +0.0137 | [−0.000, +0.030] | borderline (retrieval **hurt**) |
| calibrated − structured | −0.0481 | [−0.099, −0.009] | **YES** |

Read literally: **no world-model tier significantly beats the raw LLM** (every CI crosses 0). The
only significant effect is that a **calibration layer significantly fixes the raw structured model**.

## Non-LLM tiers at higher power (full 540-story test set, base rate 0.104)
| tier | log loss | ECE | uplift@20 |
|---|---|---|---|
| structured (as-of features) | 0.3759 | 0.0806 | −0.030 |
| calibrated | 0.3410 | 0.0334 | −0.030 |
| **aggregate world (state-transition)** | **0.3362** | **0.0190** | **+0.035** |
| individual world | 0.3414 | 0.0332 | +0.017 |
| base rate (train rate → test) | 0.3442 | — | — |

At power, among non-LLM tiers the **state-transition model is best**: lowest log loss, by far the
best calibration (ECE 0.019), and the only tier with clearly positive decision lift. The **structured
model overfits** (worse than base rate). So *within the statistical family*, explicit evolving state
is the winner — consistent with EXP-005/010.

## What each headline question resolves to
- **Does state simulation beat raw LLM + context?** On HN: **not decisively — it's a tie.** The
  state model wins among statistical models but only matches the raw LLM. Point estimates favor the
  world model by ~0.5–0.8% relative log loss, but n=100/8-positives cannot make that significant.
- **Did retrieval improve predictions?** For the raw LLM on HN: **no** — as-of author/domain context
  *slightly hurt* it (ECE 0.020→0.055; the context made it overconfident). There was little
  information gap to close: the LLM already prices HN dynamics from pretraining. (This is the mirror
  image of the market case, where the gap is real but the *corpus* is missing — see market report.)
- **What actually moved the needle?** The **calibration layer** (significant), and — among
  non-LLM models — **explicit evolving state** (best calibration + only positive lift at power).

## Why this is the expected result, not a failure of the build
HN hit/miss is dominated by front-page-ranking luck (a large aleatoric floor), so *every* method
sits close to the base rate and the achievable spread is small. And a frontier LLM's pretrained
"world knowledge" of HN culture is a soft substitute for an explicit world state — which is exactly
why the explicit state model ties rather than dominates it *here*. The explicit-state advantage
should show up where the LLM has **no** pretrained prior: a specific private entity's response, a
proprietary channel, a novel action — the individual/counterfactual regime where no aggregator or
pretrained prior has priced the answer. That is the honest place to point the product, and it is
where EXP-004's decision-lift and the individual model live.

## Caveats (stated, not buried)
- n=100 for the LLM comparison is **underpowered** (8 positives). The tie is "not distinguishable,"
  not "proven equal." A larger contamination-free LLM run would tighten it; the point estimates are
  unlikely to flip to a *decisive* world-model win given the aleatoric ceiling.
- Fully reproducible: the LLM predictions + the common item set are committed under
  `experiments/results/exp009_llm/`; the structured/calibrated/state tiers rerun live from the HN
  stream via `experiments/exp009_harness.py`. (The raw HN stream stays under gitignored `data/`;
  `prep` re-fetches it.)
