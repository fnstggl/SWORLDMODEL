# Final report — general social world model: what is now real

This is the honest close-out for the general-SWM build. It answers the directive's final questions
directly. Companion documents: the gap audit (`exp008_general_swm_gap_audit.md`), the head-to-head
(`exp009_raw_llm_vs_world_model.md`), and the three regime reports (`aggregate_model_report.md`,
`individual_model_report.md`, `market_benchmark_report.md`).

## The architecture that now exists (real, tested, backtested)
`WorldState_t + Action_t → Outcome_t + WorldState_{t+1}` is implemented for both regimes:

- **State** — `swm/state/`: POMDP `WorldState`/`EntityState`/`ContextState` (pre-existing) plus new
  `PopulationState` (subgroups, salience, reputation, attention/competition, incentives, drift),
  `HierarchicalPosterior`/`BetaHierarchical` (partial pooling), `IncentiveState`, `Graph`.
- **Transition** — `swm/transition/`: `AggregateTransition` and `IndividualTransition` (both make
  state genuinely enter a calibrated head and both evolve the state after each outcome), `diffusion`
  (independent-cascade / linear-threshold / Hawkes), `nonstationarity` (drift tracker), shared
  `transition_head`.
- **Retrieval** — `swm/retrieval/`: `AsOfStore` that *physically* cannot return future items, plus
  news/social/entity as-of adapters and a REAL leakage gate (`swm/eval/leakage.py`, was a stub).
- **Worlds** — `swm/worlds/`: `AggregateWorld`, `IndividualWorld` (fit-stream + temporal backtest +
  persistence).
- **Simulation** — `swm/simulation/`: free-running rollout + calibration-by-horizon eval, scenario
  tree, counterfactuals.
- **Eval** — `swm/eval/`: `raw_llm_vs_world_model`, `benchmark_matrix`, `market_comparison`,
  `individual_response_eval`, `decision_lift`.

55 tests pass (36 new). Two independent agent audits (one of the pre-existing code, one of the new
code) were run; every actionable finding was fixed (rollout used a state-ignoring `PriorHead` while
claiming a grade → now loads a real fitted model; multi-step eval mixed training rows → now
held-out + warm-started; in-sample Platt → held-out; two-stage pooling weight bug; no-pooling
baseline mislabeled; entity as-of tie leak; segment-zip bug).

## 1. What is now real
- A one-step **aggregate state-transition model that beats content-only and the base rate on 1,800
  real HN stories**, calibrated **grade A** (ECE 0.019), with the lift carried by *stateful*
  features. Persisted (`models/hn_aggregate.json`) and wired into `/v1/rollout`.
- A **hierarchical individual response estimator** validated on synthetic data with known structure:
  partial pooling beats both segment and no-pooling and stays calibrated where no-pooling overfits;
  cold→warm behavior measured by evidence bucket.
- A **free-running multi-step eval** (the one the prior repo admitted it never ran): free-running
  calibration **tracks teacher-forced with no catastrophic drift** at horizons 1–4 on HN.
- **Leakage-proof as-of retrieval** with tests, and a real leakage gate.
- The **head-to-head benchmark** (EXP-009) with a statistically honest verdict.
- A **fair, no-cheat market comparison** module (segmented, price-at-fixed-horizon).
- Reproducible artifacts committed: `models/hn_aggregate.json`, `experiments/results/*`.

## 2. What is still fake / unvalidated (stated, not buried)
- **The individual model on real behavior** — BLOCKED-ON-PRIVATE-DATA. Validated as an estimator,
  unproven as a world claim. No real labeled response dataset exists here; none was faked.
- **Multi-step accuracy as a precise degradation curve** — only "no catastrophic drift over 1–4
  steps" is shown; the absolute per-horizon numbers are cohort-noise-dominated. Needs per-author
  depth.
- **As-of NEWS retrieval for markets** — plumbing built + leakage-tested, but content BLOCKED-ON-
  CORPUS (no timestamped news source). Arms 2/3 of the market test cannot run without fabricating
  news (which would be cheating).
- **Diffusion dynamics** — implemented (IC/LT/Hawkes) but not backtested against real cascade sizes
  (no networked-outcome wedge yet).
- **Belief/stance-distribution transitions** — proxied by salience/reputation, not measured directly.
- Remaining design-only stubs: `transition/mechanistic.py`, `transition/llm_rollout.py`,
  `graph/diffusion.py` (superseded), `inference/filter.py`, `entities/embeddings.py`,
  `memory/memory.py`.

## 3. Did state simulation improve predictions?
**Two answers, both honest:**
- **Yes, over a statistical baseline.** On HN (n=1,800) the evolving-state model beats content-only
  (log loss 0.3405→0.3362, ECE 0.041→0.019, grade A) and is the best-calibrated non-LLM tier with
  the only positive decision lift; the naive structured model overfits. State earns its place *within
  the statistical family*.
- **No, not decisively over a raw LLM.** On identical HN items the state models and a raw LLM are a
  **statistical tie** (EXP-009 bootstrap CIs straddle 0). A frontier LLM's pretrained knowledge of HN
  is a soft substitute for explicit state, so state ties rather than dominates *on a public domain the
  LLM already knows*. Per the hard rule: **said plainly — world-state simulation does not beat raw LLM
  on HN.** The expected place it *should* win is the private/individual/counterfactual regime with no
  pretrained prior — which is exactly the regime blocked on private data.

## 4. Did retrieval improve predictions?
- On **HN** (where leakage-free as-of retrieval is available): **no** — as-of author/domain context
  slightly *hurt* the raw LLM (overconfidence), because there was little information gap to close.
- On **markets** (where the gap is real): **unknown — blocked on a timestamped news corpus.** The
  diagnosis (EXP-006) is that retrieval is the *only* lever that can close it, and the near-parity of
  the information-symmetric subset localizes the gap as informational, not reasoning.
- General finding: **retrieval helps only when the predictor genuinely lacks the information AND a
  timestamped source of it exists.** Neither condition held on HN; the second is missing for markets.

## 5. Are we closer to beating prediction markets?
**Not on the leaderboard number, and honestly so.** At a fair 48h horizon the market still wins
(0.178 vs 0.260). What changed: the fair-comparison machinery and the leakage-proof retrieval layer
are now built and tested, so the *one experiment that could move it* (as-of news retrieval) is ready
to run the moment a corpus exists. And the strategic read is sharpened: the model is at **parity on
the information-symmetric subset** (52% head-to-head), which is the only slice a static-knowledge
model can win — the rest is an information gap a liquid market is purpose-built to own.

## 6. What bottleneck remains
Ranked:
1. **Private individual outcome data** — the single highest-leverage unlock. It converts the
   validated individual *estimator* into a validated world claim and opens the regime where explicit
   state should beat a raw LLM (no pretrained prior). Path exists (`ingestion/store`, `gmail_search`,
   `IndividualWorld`); needs consented real data.
2. **A timestamped news corpus** — unblocks market/event as-of retrieval; the only lever that closes
   the market information gap.
3. **Entity depth for multi-step** — to turn "no catastrophic drift" into a precise, graded
   degradation curve.

## Bottom line
The repo now has a genuine, backtested, honestly-graded state-transition world model for aggregate
prediction and a validated individual estimator — not a fake simulator. It measurably beats
statistical baselines and is grade-A calibrated on HN. It does **not** beat a raw LLM on HN, and it
says so; retrieval did not help there and is blocked where it would; the market is still unbeaten at
a fair horizon. The build replaced asserted capability with **measured** capability, and the two
remaining bottlenecks are **data**, not architecture — exactly where the original audit predicted the
honest version of this project would land.
