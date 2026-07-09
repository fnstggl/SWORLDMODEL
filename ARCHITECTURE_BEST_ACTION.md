# Architecture — the general best-action finder (typed action search)

**Goal:** `argmax_a E[U(outcome) | do(a), context]` for *any* action, trying MANY candidates (not 3–4), on
one shared spine. The message optimizer is the *generative-text* specialization of this.

## The insight: the action has a TYPE, and the search operator must match it

| Action type | Common asks | Search operator (tries thousands) | LLM's role |
|---|---|---|---|
| **Continuous** | best price, discount, bid/budget, dosage, rate, timing | grid → local-refine over the response curve (`best_continuous`) | proposes which levers exist |
| **Discrete** | best candidate, vendor, market, feature, channel, influencer | enumerate + best-arm race (`best_action`) | scores/among options |
| **Generative** | best copy, subject line, pitch, script, post | propose → score → mutate (the message architecture) | proposer + critic |
| **Structured** | best policy bundle, campaign (channel+budget+creative), product bundle, portfolio | coordinate ascent / combinatorial over the fields | generates text fields; realism critic |

The shared spine (`swm/decision/best_action.py`, already present): a calibrated **world-model objective**
and **best-arm racing** (successive elimination with confidence intervals) that finds the true argmax and
says when it's a *tie within noise* — not a fixed-N guess.

## The harness (`swm/decision/action_finder.py`)

- **Typed action spaces**: `Continuous`, `DiscreteChoice`, `GenerativeText`, `Structured`.
- **`world_model(score_fn, value_fn=None)`** — wraps a calibrated world model into the `sample_fn(action,
  rng) -> outcome` the racer needs. `score_fn(action)` returns `P(outcome)` (a point *or* an ensemble of
  samples for predictive uncertainty); default maximizes `P(outcome)`, or pass `value_fn(action, success)`
  to maximize a utility (e.g. revenue = price × sale).
- **`find_best_action(space, sample_fn, …)`** — dispatches to the operator matched to the type and returns
  a `DecisionResult` (winner + ranking + confidence + contrast vs baseline). New `Structured` search:
  coordinate ascent over the fields (every choice of each field, held against the rest, sweeping to
  convergence), then races the winner vs its strongest neighbors for a confidence statement.

`experiments/exp089` runs all four: best **price** 44.2 (revenue-optimal), best **vendor** globex, best
**campaign** referral+story+max-budget, best **copy** the punchy one.

## The unlock is the world model, not the search

Search is general and cheap; **every world model is its own data-and-calibration project** — a demand curve
for price, a persuasion model for copy, a causal/economic model for policy. The message case proved it:
the search finds the scorer's argmax instantly; the scorer's calibration (~56% on CMV) is the ceiling. So
the product is **a growing library of calibrated per-domain world models + this one typed search harness**
(the repo's "stacked calibrated wedges" thesis).

## Recipient conditioning from history (`swm/decision/recipient_history.py`)

The standing product rule: **ingest all the recipient data you can get.** For any entity, gather their
writing history → `deep_inference` builds their persona (the automated interview, leakage-safe as-of) →
`persona_to_recipient` maps it to the objective's recipient variables **and a per-recipient base rate**.
This is what de-compresses predictions off the population mean: with a constant recipient the only variance
is the action and predictions collapse to the base rate; with a real per-recipient persona, different
recipients get different baselines and elasticities, so the model can say 0.9 for a great fit and 0.1 for a
bad one.

`HistoryStore.ingest(entity, text, ts)` / `.recipient(entity, now)` is the durable primitive.

### Honest measurement (`experiments/exp088`)
- **On CMV it did NOT help pair-accuracy** — and the reason is structural: the CMV winning-args *pair task*
  compares two arguments sent to the **same OP**, so it **controls for the recipient by construction**
  (per-OP base rate cancels within a pair; across-OP success-rate std is only ~0.14). Recipient conditioning
  cannot move a metric that fixes the recipient. Earlier claim that it would "break 0.56" was wrong for this
  metric — an honest correction.
- **The method is validated on a multi-recipient task** (synthetic, recipients genuinely differ):
  history-conditioning lifts cross-recipient AUC **0.596 → 0.660** and widens the prediction range
  **[0.35,0.57] → [0.30,0.64]** — it works *and* de-compresses when recipients actually vary.

The real validation needs a **multi-recipient corpus** (a sent→replied cold-email set); the primitive is
ready for it.
