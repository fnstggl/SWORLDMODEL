# EXP-005 — From LLM predictor to a real state-transition world model

What changed, what's now real, what still isn't. No hype: no "digital twin," no "simulate society."

## What changed (architecture)
The system had one-step `p(outcome | features)`. It now has explicit **state → action → next-state**
dynamics:

- `swm/state/state.py` — `WorldState` (population, per-entity, context, optional graph, uncertainty),
  `EntityState` (stable traits / response style / relationship stance / current attention, each a
  `Posterior` with an evidence weight), `ContextState` (topic salience, domain reputation, time,
  drift), `Action`, `OutcomeEvent`.
- `swm/state/factors.py` — a `FactorRegistry`: candidates are mapped expansively; each declares
  source/timescale/leakage-risk and, for stateful factors, an **update rule**. This reconciles the
  two principles: *map every decision-relevant variable* (be expansive) but *keep only what earns
  its place* (ablation decides).
- `swm/state/transition.py` — `TransitionModel.step()` implements `p(next_state, outcome | state,
  action)`: a calibrated statistical **outcome head** predicts the band distribution; deterministic
  **factor update rules** evolve entity + context state; the LLM is upstream feature/prior only.
- `swm/state/trajectory.py` — n-sample rollouts: sample an outcome, evolve state, continue → a
  **distribution of futures**, with an honesty gate.
- `swm/state/ablation.py` — automated keep/drop by held-out temporal performance.
- `swm/retrieval/context.py` — as-of retrieval (author/domain/topic history strictly before `as_of`).
- API `/v1/rollout` — multi-step simulation, **distinct from `/predict`**, labeled unvalidated
  unless a backtest exists.

## Is it now a real state-transition world model? Yes — measurably, on HN.
`experiments/state_transition_harness.py` runs one world / many authors / global time over **2,810
posts, 70 authors**. For each post it extracts factors from the *current* state, records the sample,
then transitions the state with the actual outcome. Results (test = last 30% by time, target
P(score≥40)):

| model | log loss | Brier | ECE |
|---|---|---|---|
| base rate (no model) | 0.2155 | — | — |
| content + time only | 0.2158 | 0.0516 | 0.0355 |
| + entity state | 0.2141 | 0.0514 | 0.0316 |
| **+ entity + context state (FULL)** | **0.2128** | **0.0511** | **0.0296** |

**Explicit, evolving state improves held-out prediction and calibration** (log loss −1.4%, ECE −17%
vs content-only). Small, but real and monotonic.

## Which variables actually mattered (ablation survivors)
Removing a factor and re-fitting on the temporal split, the biggest KEEPs (removal worsens held-out
log loss most) are the **stateful** ones — which is the whole thesis:

| factor | Δlog-loss if removed | verdict |
|---|---|---|
| author_standing (updates) | +0.0026 | KEEP |
| domain_reputation (updates) | +0.0013 | KEEP |
| author_volume (updates) | +0.0008 | KEEP |
| hour_sin | +0.0005 | KEEP |
| author_ceiling (updates) | +0.0002 | KEEP |
| author_quality | −0.0014 | EXPERIMENTAL (redundant with standing/ceiling) |
| is_show / is_ask / is_text | ≤ 0 | EXPERIMENTAL |

The top two survivors — a running "community standing" and "domain reputation" that **transition
after every post** — are exactly the state-evolution variables, not static content features. That is
the empirical case that this is a state model, not a feature model. Ablation also correctly flags
`author_quality` as redundant and several content flags as not-earning-their-place.

## How multi-step degrades
Underpowered at this n: per-horizon log loss is dominated by the shifting realized hit-rate
(horizons 1–4 realized P≥40 fall 0.08→0.03 so log loss "improves"; horizon 5 spikes on a hit
cluster). **Honest verdict: the one-step transition is validated; multi-step degradation is not yet
cleanly measurable — it needs more per-author depth and a true free-running (sampled-state) rollout
eval.** The API therefore validates `/rollout` only at horizon 1 on HN and labels everything else
`unvalidated`.

## What is still NOT real
- **Multi-step / long-horizon accuracy is unvalidated.** Rollouts beyond one step are qualitative.
- **Other domains (elections, policy, business, individual email) have no state-transition backtest**
  — the registry is extensible to them, but until backtested the honesty gate labels them
  `unvalidated`. This is deliberate: no fake general simulator.
- The "incentives / attention" latent factors are proxied crudely (standing, recency); richer
  latent structure is future work, admitted as candidates and subject to the same ablation filter.
- Context is HN-shaped (domain/topic). Cross-domain context (news events, macro state) is not built.

## Did it improve held-out prediction over the previous predictor? 
Yes, modestly and honestly: adding explicit evolving state lowered held-out log loss (0.2158 →
0.2128) and ECE (0.0355 → 0.0296) vs the content-only model, and ablation shows the improvement is
carried by the *stateful* factors. It is not a large jump — at HN scale most of the signal is
content/author-prior, and state adds a real but small increment. The value of the state machinery
grows with (a) entity depth (repeat actors), (b) multi-step questions, and (c) domains where state
genuinely evolves — which is where the next backtests should go.

## Next bottleneck
Multi-step validation. One-step state-transition is proven; the distinctive claim of a *world model*
— that rolling the state forward predicts multiple steps out — is still unproven here and is the
next thing to backtest (more per-author depth; a proper free-running eval that updates state with
sampled, not actual, outcomes and measures calibration-by-horizon).
