# EXP-019 — Does content richness (LLM semantic features → world model) help? Result: NO here — and why that matters

Tested the top lever hypothesized in EXP-018: feed LLM-extracted *semantic* features (clarity,
actionability, reproducibility, specificity, sentiment, effort-to-answer, feature/bug) — not the
LLM's probability — into the world model, and see if they help the cold/content-dominated regime on
GitHub issue-response. An 8-agent swarm extracted features on 1,600 issues (100% coverage);
compared shallow vs shallow+semantic under the stacked combiner, as-of temporal split.

## Result: no help (mildly negative)
| slice | shallow | + LLM semantic | Δ |
|---|---|---|---|
| overall (n=480) | 0.4061 | 0.4089 | **−0.0028** |
| cold repo (0 history, n=388) | 0.4190 | 0.4226 | −0.0036 |
| repeat repo (1–4, n=72) | 0.3891 | 0.3910 | −0.0019 |

**The semantic features did not help — not even on the cold slice, where the hypothesis said they
should.** They are correlated with the shallow structural features (is_bug ≈ is_bug_report) and add
variance, not signal.

## Why — the sharp, useful finding: the OUTCOME TYPE determines the lever
The hypothesis was half-right, and the experiment corrected it:

- **Entity-driven outcomes** — "will this repo/person *respond / reply*?" (GitHub, Enron) — are
  dominated by **WHO** (the responder's state/incentives), not by the semantic quality of the
  message. So the lever is **deep entity state**, and richer *content* adds almost nothing. That is
  exactly what we see here, and it is why the same model's accuracy jumps with entity-history depth
  (GitHub deep-repo log loss halved) but not with content features.
- **Content-driven outcomes** — "will this *post go viral / get upvoted*?" (HN) — depend on the
  artifact's properties, and there LLM semantic features *did* help (EXP-013). 

So content richness is a **secondary lever for content-driven aggregate outcomes**, and a **non-lever
for entity-driven individual outcomes**. Since the highest-value individualized predictions (will
*this* person reply / convert / object) are entity-driven, this **reranks the levers**:

1. **Deep entity-state data** is THE lever for individualized prediction — confirmed twice now
   (depth-scaling wins; content richness does not substitute for it). The moat is proprietary
   repeat-entity outcome history, not a fancier content encoder.
2. **Content richness (LLM semantic features)** helps *only* where the outcome is genuinely
   content-driven (aggregate virality/engagement), where it gives a modest lift.
3. **The learned combiner** (EXP-018) remains the general auto-adapting fusion — neutral on average,
   valuable for generality.

## Honest caveats
- This sample is cold-dominated (388 cold vs 72 repeat in test) and small (n=480 test, ~65 positives),
  so the estimates are noisy — but the direction (no help) is consistent overall and on every slice,
  and it agrees with the mechanism (entity-driven outcome).
- The features were extracted from titles + body length, not full issue bodies; richer extraction
  from full text might recover some signal, but it would have to overcome the WHO-dominance shown by
  the depth-scaling result.

## Conclusion
Measured, not assumed: for the entity-driven individual-response outcomes that matter most, **the next
highest-leverage step is deeper entity-state data, not a richer content representation.** The content
lever is real but confined to content-driven aggregate outcomes. This is the corrected answer to
"what's the highest-leverage next step," and it points squarely at data acquisition (proprietary
repeat-entity outcome logs) as the frontier — exactly where the honest version of this project lands.
