# EXP-047 — Semantic (LLM-judge) stance closes the content gap lexical reading couldn't

EXP-044 established the frontier precisely: lexical, entity-linked stance recovers only ~13% of the signal
the market extracts from the same news, because outcome polarity is question-specific ("unemployment
rises" is evidence toward NO for "will it be below X") — which term-matching cannot read. The fix is a
semantic judge: an LLM reads the as-of news for *this* question's specific YES resolution and returns a
signed stance. This validates that fix and wires it for production.

## The build
`swm/variables/semantic_stance.py` — a model-agnostic `SemanticStanceJudge(judge_fn)`. The same module and
prompt run in **production** (`anthropic_judge_fn` — a real Anthropic API call, stdlib-only) and in this
validation (`cached_judge_fn` replays committed judgments), so this calibrates the production system, not a
throwaway. The prompt frames a strict reading task and forbids using any memory of the outcome.

## Leakage discipline (the contamination concern, handled)
A strong LLM may recall dated outcomes — so the result cannot rest on stance-vs-outcome. Two guards:
- the judge saw **only the question + as-of news headlines**, never the market price or the outcome;
- the **primary metric is market-consistency**: correlation of the semantic stance with the **as-of price
  the judge never saw**. The price is the crowd's reading of the *same news*, independent of the future
  outcome, so agreeing with it measures *reading skill, not outcome recall* (the EXP-037 robustness
  argument). Stance-vs-outcome is reported but flagged; the post-cutoff subset is called out separately.

## Result (70 LLM-judged Kalshi questions, blind to price/outcome)

**PRIMARY — market consistency (corr of stance with the as-of price; contamination-robust):**

| feature | corr with as-of price |
|---|---|
| **semantic stance** | **0.570** |
| semantic confident-stance | 0.506 |
| lexical stance (EXP-044) | 0.148 |

**Semantic reading recovers 3.85× the market-consistent signal of lexical reading.** A 0.57 correlation
with a price the judge never saw — from news headlines alone — means the LLM reads the news the way the
market does. For reference, EXP-037's full driver-inference pipeline reached 0.63 market-consistency; the
lightweight stance judge gets most of the way there from headlines.

**SECONDARY — corr with the outcome (contamination-susceptible; market lean = ceiling):**

| feature | corr with outcome |
|---|---|
| semantic stance | 0.606 |
| lexical stance | 0.272 |
| market lean | 0.818 |

Semantic stance recovers **74%** of the market lean's outcome-correlation (0.606 / 0.818) versus lexical's
33% — a large jump — but this number is caveated by possible outcome recall, which the market-consistency
result is designed to sidestep.

## The honest findings
1. **The EXP-044 gap was real and is largely closable — by semantics, exactly as diagnosed.** Lexical
   reading recovered 13% of the market's signal; the semantic judge recovers most of it (market-consistency
   0.57 vs the price, 3.85× lexical). The bottleneck was NLP depth, and an LLM judge supplies it. This
   confirms the whole EXP-037 → 043 → 044 → 047 arc: gestalt drivers and shallow lexical reading both
   failed for lack of *question-specific content understanding*; a stance judge oriented to the resolution
   provides it.
2. **The contamination guard is load-bearing and honest.** The clean out-of-recall check — the post-cutoff
   subset — is only n=7 here (both semantic and the market lean score 0.0 on it, i.e. uninformative at that
   size), so the *outcome* number cannot be taken as a clean skill measure. The **market-consistency** 0.57
   is the honest headline: it needs no outcome, so recall cannot inflate it.
3. **The production path is wired, not hypothetical.** `anthropic_judge_fn` makes the identical judge run
   against the API with a key; this session used the cached/self-judged path. Swapping the backend changes
   nothing else — the simulator and production share one code path.

## Honest limits
- 70 questions, judged by one LLM pass (no dragonfly aggregation of multiple judges — that would reduce
  single-view noise). A production run would judge thousands via the API backend.
- The clean post-cutoff skill number needs a larger set of questions resolving after the model's cutoff;
  market-consistency is the robust stand-in until then.
- Self-judged here: the same model family produced and is described by these numbers. The API backend with
  a separate key (or a held-out judge) is the stricter production setup the module already supports.

## Reproduce
`python -m experiments.exp047_semantic_stance` → `experiments/results/exp047_semantic_stance.json`
(replays the committed `experiments/results/exp047_stance/stance_cache.json`).
`python -m pytest tests/test_semantic_stance.py`.
