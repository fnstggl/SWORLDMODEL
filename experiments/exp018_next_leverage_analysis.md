# EXP-018 — What's the next highest-leverage step? (analysis + implementation)

The question: toward a general social world model with the most accurate aggregate AND
hyper-individualized predictions, is the next lever *improving the simulation states*, or something
else? This is the first-principles answer, with the top candidate implemented and measured honestly.

## First-principles error decomposition (from everything measured this project)
Across HN, GitHub, Enron, StackExchange, every prediction splits into two regimes:
- **STATE-RICH** (repeat entity / high context): the world model wins decisively and the win **scales
  with entity-history depth** (GitHub deep-repo log loss 0.294→0.132; Enron cold 0.367→deep 0.266).
- **COLD / STATE-POOR**: we fall back to a flat prior and the LLM's *content* understanding wins.

So accuracy is limited in exactly two places: **cold-start** (no state) and **the fusion** of
state-model vs content-prior (we did it with a hand-set depth gate). The highest-leverage candidate is
therefore *not more simulation machinery* — HN simulation is already aleatoric-ceilinged — it is
**using the state better**: a learned, evidence-aware fusion, plus a better cold-start prior.

## What was implemented: the learned evidence-aware combiner (stacking)
`swm/transition/stacked_response.py::StackedResponseModel`: a meta-logistic over base predictions
(content model, entity pooled rate, segment rate, recency EWMA) + evidence features (depth,
sufficiency) + the KEY interaction terms `entity_logit·sufficiency` and `content_logit·(1−sufficiency)`
— the *learned gate* that trusts the entity state where evidence exists and the content prior where it
doesn't, with the crossover **fit from held-out data instead of guessed**. It also ingests the raw
message features, so it can always reconstruct (and not underperform) the content model. Fit only on a
held-out tail of TRAIN; as-of throughout.

## Honest result: modest / neutral — a real diminishing-returns signal
Held-out log loss vs the best *hand-picked single* config per domain:

| domain | best single | STACKED combiner | Δ |
|---|---|---|---|
| StackExchange | 0.6855 | 0.6832 | **+0.0023** |
| GitHub | 0.3052 | 0.3047 | +0.0005 |
| Enron | 0.2749 | 0.2783 | −0.0034 |

The combiner is **approximately a wash** — a small win where the state/content boundary is nontrivial
(StackExchange, GitHub), a small loss where a simple recency-logistic already captures it (Enron,
where the two-stage fit costs more data than it gains). **This is the finding:** at 10³–10⁴ rows we
have hit diminishing returns on statistical/structural machinery. The signal that was extractable by
better *state structure* (recency + hierarchy-where-real) is mostly extracted; a fancier *fusion* adds
little on top.

**Why keep it anyway:** its value is **generality, not raw superiority**. It auto-adapts the fusion
to each domain without hand-tuning — which is exactly what a *general* world model needs, because you
cannot hand-pick the config per domain in production. It stays within ~0.003 of the best hand-tuned
config everywhere and adapts automatically. The same mechanism unifies the aggregate regime (fuse
simulation-p + LLM-p + evidence → meta), replacing the hand-set HN hybrid gate.

## So what IS the highest-leverage lever now? (ranked — and UPDATED by EXP-019)
Since more model complexity is no longer the answer, the remaining error is **aleatoric noise** +
**semantic content understanding** + **missing deep state**. I tested the content-richness lever
directly (EXP-019) and it **failed on entity-driven outcomes** — which reranks the list. The honest,
measured ranking:

1. **Deeper-state / proprietary repeat-entity data — THE lever (promoted after EXP-019).** The world
   model's edge is
   largest and grows monotonically with entity-history depth, and it beats a raw LLM given the same
   info *because it models the state precisely* (GitHub deep-repo: log loss halved; beats LLM+context
   0.258 vs 0.304). The highest-value data is a domain with **deep entity state the LLM has no prior
   over** — a company's own CRM/email/support history. There, entity-state modeling wins biggest and
   the moat is real. This is a data-acquisition step, not a modeling one.

2. **Content richness — LLM semantic features → world model — DEMOTED (EXP-019 disconfirmed it).**
   Tested directly: on GitHub issue-response, semantic features (clarity/actionability/…) did NOT help
   (Δ −0.003 overall and on the cold slice). Entity-driven outcomes ("will they respond/reply") are
   dominated by WHO, not the message's semantic quality. Content richness helps **only** where the
   outcome is genuinely content-driven (post virality/engagement — HN, where it did lift modestly). So
   it is a secondary lever for aggregate content outcomes, a non-lever for individual response outcomes.

3. **Aggregate simulation realism** (large-scale regime): fit segment affinities/weights from data
   (currently hand-set priors), add second-order dynamics (comment/controversy re-exposure), and
   apply the learned combiner to the sim+LLM fusion. Medium leverage; the HN aleatoric ceiling caps it.

4. **Better cold-start priors via entity similarity** — tested (`cold_start_prior`); ~neutral here,
   because segment + message already capture most of what's observable for a cold entity. Revisit when
   entities carry richer observable features.

## Bottom line (updated after EXP-019)
The next highest-leverage step is **not** more simulation machinery, **not** a fancier fusion, and —
tested and disconfirmed — **not** a richer content encoder for the individual-response outcomes that
matter most. All three hit diminishing returns or failed on measurement. The measured answer is
**deeper entity-state DATA**: the world model's edge is entity-state-driven and grows with history
depth, and no amount of content richness substitutes for it on entity-driven outcomes. The frontier
for hyper-individualized accuracy is proprietary repeat-entity outcome logs (CRM/email/support); for
large-scale accuracy it is aggregate-simulation realism within the aleatoric ceiling. The learned
combiner is kept as the general auto-adapting predictor. This is the honest, measured conclusion —
including the parts where my own top hypothesis was wrong.
