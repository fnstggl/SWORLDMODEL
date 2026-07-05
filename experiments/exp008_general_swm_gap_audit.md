# EXP-008 — General Social World Model: gap audit

**Date:** 2026-07-04. **Purpose:** before building toward the *end-state general* social world
model (not an MVP), state exactly what already exists, what is real vs. dressed-up raw LLM, and
what is missing for each target capability. This is a pre-implementation audit; every claim below
is checked against source (`file:line`). The companion build is Phases 2–8.

The end-state target is:

```
WorldState_t + Action_t  ->  Outcome_t + WorldState_{t+1}
```

for two regimes — **aggregate** (population/market/community outcomes) and **individualized**
(this entity + this action + this context → response distribution) — judged on *calibration and
decision lift*, with an explicit test of whether state simulation beats **raw LLM + retrieved
context**.

---

## 0. What exists today (inventory, real vs. stub)

| Capability | Where | Status | Notes |
|---|---|---|---|
| State objects | `swm/state/state.py` | **REAL** | `WorldState`, `EntityState`, `ContextState`, `Posterior(mean,n)`, `Action`, `OutcomeEvent`. POMDP by construction (every latent is a posterior with an evidence weight). |
| Transition model | `swm/state/transition.py` | **REAL** | `TransitionModel.step()` = calibrated `OutcomeHead` (one logistic per score-band threshold, monotone) + deterministic factor update rules. This is a genuine `p(next_state, outcome | state, action)`, teacher-forced or sampled. |
| Factor registry | `swm/state/factors.py` | **REAL** | `FactorRegistry` with `extract`/`update` per factor; HN factor set (author quality/ceiling/standing/volume/recency; domain reputation; topic salience). Stateful factors actually mutate state. |
| Trajectory sampling | `swm/state/trajectory.py` | **REAL** | `rollout()` samples N futures, evolves state each step, aggregates per-step intervals + band means. Has an honesty gate (grade = `unvalidated` unless (domain,horizon) is backtested). |
| Ablation | `swm/state/ablation.py` | **REAL** | Leave-one-factor-out refit on a temporal split; KEEP iff removal worsens held-out log-loss/uplift. Updates registry status in place. |
| Calibration | `swm/uncertainty/calibration.py`, `data/calibration.json` | **REAL** | ECE grade (A/B/C/F); a fitted per-threshold Platt layer (identity where no lift). |
| Metrics | `swm/eval/metrics.py` | **REAL** | log loss, Brier, ECE, base rate, uplift@k, CRPS. No deps. |
| Individual (email) world | `swm/worlds/world.py`, `swm/entities/persona.py`, `swm/eval/harness.py` | **REAL code, NO real data** | Hierarchical partial pooling (Beta/Normal posteriors, person←segment←population), L0–L4 ablation ladder, VOI elicitation, correct-a-guess. Backtest harness is real; there is **no labeled individual dataset** committed, so the individual claim is unproven on real behavior. |
| Action encoder | `swm/actions/encoder.py` | **REAL** | Message features + persona-interaction (style-match) features. |
| As-of retrieval (HN) | `swm/retrieval/context.py` | **REAL but narrow** | `as_of()` timestamp gate; author/domain/topic context from in-memory `PostRecord`s. HN-shaped; no persistent store, no external corpus. |
| Event store | `swm/ingestion/store.py` | **REAL** | Append-only SQLite; `history_asof()` is a strict `timestamp < T` read; reply labels derived with anti-inflation window logic. |
| /v1/rollout | `api/app.py:160` | **REAL, honest** | Multi-step simulation endpoint; uses `PriorHead` so it is `unvalidated` by construction; `_VALIDATED = {"hn": {"grade":"C","horizon":1}}` (`api/app.py:157`) is the *only* backtested cell. |
| Raw-LLM services | `swm/llm.py` | **REAL (feature/gen only)** | Trait extraction + draft generation via Claude; graceful heuristic fallback. **Never** the probability source in a backtested metric. |
| Benchmark harnesses | `experiments/hn_harness.py`, `hn_harness2.py`, `state_transition_harness.py`, `manifold_harness.py`, `decision_lift.py`, `auto_loop.py` | **REAL** | HN (per-round + pooled), Manifold (fair as-of price @ fixed lead), state-transition A/B/C, decision lift, live loop. |

**Stubs (`IMPLEMENTED = False`, design-only):**

- `swm/eval/leakage.py` — leakage/contamination gate (only a schema-level check exists in `tests/test_leakage_gate.py`).
- `swm/transition/mechanistic.py` — Hawkes/cascade/SEIR temporal dynamics.
- `swm/transition/llm_rollout.py` — generative-agent rollout.
- `swm/graph/diffusion.py` — typed graph + independent-cascade / linear-threshold / Hawkes.
- `swm/inference/filter.py` — amortized hidden-state inference `p(s_t | o_{1:t})`.
- `swm/entities/embeddings.py` — learned latent θ_i.
- `swm/memory/memory.py` — episodic+semantic memory.

**Target module layout that does NOT yet exist** (the build list): `swm/worlds/{aggregate_world,
individual_world}.py`; `swm/state/{latent,graph,incentives,population}.py`; `swm/transition/
{aggregate_transition,individual_transition,diffusion,nonstationarity}.py`; `swm/retrieval/
{asof_store,news_context,social_context,entity_context}.py`; `swm/eval/{raw_llm_vs_world_model,
benchmark_matrix,market_comparison,individual_response_eval,decision_lift}.py`; `swm/simulation/
{rollout,scenario_tree,counterfactuals}.py`.

---

## 1. What is still raw LLM prediction?

Almost nothing in the *backtested* path — this repo is unusually disciplined about it — but the
places where an LLM (or an LLM-agent swarm) is the actual probability source are:

- **The market/HN LLM experiments' predictions.** In `manifold_harness.py` and `hn_harness2.py`,
  the probabilities scored are produced by Claude/agent swarms (`data/mf_pred_agent*.json`,
  `data/hn_*_pred.json`). That is raw-LLM-as-predictor (EXP-002/003/006). It is *evaluated*
  honestly (no-cheat construction, proper scoring), but the predictor itself is the LLM, not a
  state model.
- **`swm/llm.py`** produces traits and drafts — but those feed features/insight, never the scored
  probability (`swm/transition/readout.py` produces the probability). So this is *not* raw-LLM
  prediction; it's correctly bounded.

Everything else scored in a backtest — HN state-transition (`state_transition_harness.py`), the
email ladder (`eval/harness.py`), decision lift (`decision_lift.py`) — is a **statistical head over
an explicit factor/state vector**, not an LLM. Verdict: **the state-transition path is real; the
"can an LLM predict society" claims live only in the LLM experiments and are labeled as such.**

The gap the whole task turns on: there is **no head-to-head module** that pits, on the same items
and metrics, `raw LLM` vs `raw LLM + as-of context` vs `structured statistical` vs `state
transition`. That comparison is the point of EXP-009 and does not exist yet.

## 2. What is real state-transition modeling?

Real, and measurably so on HN (EXP-005):

- `TransitionModel.step()` (`swm/state/transition.py:86`) advances an explicit `WorldState`: after
  each outcome, `FactorRegistry.apply_update()` (`swm/state/factors.py:59`) mutates the author's
  latent traits, the domain reputation, and the topic salience. The next prediction is conditioned
  on a changed world. That recurrence — not the head — is what makes it a world model.
- EXP-005 result: adding evolving state lowered held-out log loss `0.2158 → 0.2128` and ECE
  `0.0355 → 0.0296` vs content-only, and ablation shows the lift is carried by the **stateful**
  factors (`author_standing`, `domain_reputation`), not static content. Small but real and
  monotone (`experiments/exp005_state_transition_world_model.md`).

Limitation: it is **one world, one channel (HN), one-step**. The state is HN-shaped.

## 3. What is missing for aggregate prediction?

Aggregate exists *implicitly* (HN population/community response) but is not packaged or generalized:

- **No `PopulationState`/`AggregateWorld` object.** State is per-entity + a HN `ContextState`; there
  is no explicit population-prior / subgroup-prior / attention-competition / salience container that
  the task's Phase-3 spec calls for.
- **No subgroup priors, no incentives/stakes state, no network/diffusion state, no nonstationarity/
  drift indicators as first-class state.** `ContextState` has empty `drift_indicators`
  (`swm/state/state.py:58`) that nothing populates.
- **No aggregate transition** that updates topic salience / attention / belief-stance / uncertainty
  as its own module (the logic is inlined in the HN harness, not reusable).
- **One domain only.** No second public domain backtested; the registry is extensible but unproven
  beyond HN.

## 4. What is missing for individualized prediction?

The estimator is real; the evidence is not:

- **No real individual dataset.** `swm/worlds/world.py` + persona pooling is fully implemented but
  there are zero labeled sends committed (`data/` is gitignored; only `calibration.json` survives).
  The go/no-go `run_ladder()` verdict (`eval/harness.py:102`) has never run on real behavior here.
- **No individual-vs-alternatives eval module.** The task wants: individual vs segment vs raw-LLM
  vs raw-LLM+context, plus ablations by evidence source. Only the L0–L4 ladder (segment vs person)
  exists; no raw-LLM arm, no evidence-source ablation, no `individual_response_eval.py`.
- **No general `IndividualWorld`** decoupled from the email channel; `World` is email/HN-specific.
- Latent structure is proxied (responsiveness/verbosity/formality/latency); no learned θ_i
  (`embeddings.py` stub), no amortized filter (`inference/filter.py` stub).

## 5. What is missing for multi-step simulation?

- **Multi-step accuracy is UNVALIDATED and honestly labeled so.** `state_transition_harness._multistep`
  (`experiments/state_transition_harness.py:138`) admits the per-horizon result is dominated by the
  shifting realized hit-rate, not model quality; EXP-005 §"How multi-step degrades" says one-step is
  proven, multi-step is not cleanly measurable at this n.
- **No free-running eval** that rolls state forward on *sampled* (not actual) outcomes and measures
  calibration-by-horizon. The rollout code samples, but the *evaluation* of sampled rollouts is
  missing.
- **No scenario tree / counterfactual** machinery (branch over action choices, `do(action)` deltas).
- The API correctly refuses to claim multi-step calibration (`_VALIDATED` horizon = 1).

## 6. What is missing for beating prediction markets?

EXP-006 already ran the *fair* test and lost honestly: at a fixed 48h lead, market Brier **0.178**
vs swarm **0.260**; on the information-symmetric (market-uncertain) subset it's near-parity
(0.247 vs 0.234, 52% head-to-head). The diagnosis: **information staleness, not reasoning** —
the model's Jan-2026 cutoff can't see mid-2026 news the market has priced.

Missing to move it:

- **As-of information retrieval** — feed news/context up to T but never past T. This is the only
  lever that closes an information gap and it does **not exist** (`retrieval/context.py` is HN-only;
  no news/entity/social as-of adapters). `swm/retrieval/news_context.py` etc. are on the build list.
- **A reusable, segmented market-comparison module** (the logic is inlined in `manifold_harness.py`).
- **Leakage tests for retrieval** (that every retrieved item has `ts <= as_of`, and that live search
  can't inject post-resolution facts).

## 7. What is missing for no-cheat as-of retrieval?

- The **primitive** exists and is correct: `as_of(records, t)` (`swm/retrieval/context.py:26`) and
  `store.history_asof` (`swm/ingestion/store.py:107`) both hard-filter `timestamp < t`. The Manifold
  harness reconstructs price at T from bet history strictly before T (`manifold_harness.py:54`).
- **Missing:** a *general* persistent as-of store (`asof_store.py`); typed external-context adapters
  (news/social/entity) that **refuse untimestamped or live results**; and an enforced leakage gate
  (`swm/eval/leakage.py` is a stub). Today the guarantee is "the harness is careful," not "the
  retrieval layer physically cannot return the future" for external sources.

## 8. What is missing for decision-lift product use?

- Decision lift is **measured** (EXP-004: model captures 50% of >=40 winners at top-20% vs 20%
  random vs 33% author-aware baseline; lift over random large and unambiguous, lift over strong
  baseline directional but tail-data-starved).
- **Missing:** a reusable `decision_lift.py` in the library (it's an experiment script); a
  `scenario_tree`/`counterfactuals` layer to turn predictions into *action choices with expected
  value*; and the individual-channel wedge where outcomes stream in (blocked on private data).

---

## Bottom line (what this audit commits the build to)

**What is already real:** explicit state objects; a genuine one-step state-transition model that
beats a content-only predictor on HN with the lift carried by *stateful* factors; automated
ablation; calibrated heads + a Platt layer; proper scoring + decision-lift metrics; a fair,
no-cheat market backtest; a fully-built (but data-starved) hierarchical individual estimator; an
honesty gate that labels un-backtested rollouts `unvalidated`.

**What is fake/unvalidated (and admitted):** multi-step accuracy; any domain beyond HN; the
individual model on real behavior; aggregate-population state as a first-class object; external
as-of retrieval; beating a liquid market (lost, diagnosed as information not reasoning).

**The build (Phases 2–8) must therefore add, and *backtest*, not assert:**
1. `PopulationState`/`AggregateWorld` + aggregate transition (subgroups, salience, attention,
   incentives, diffusion, drift) — backtested on HN, then a 2nd public domain if feasible.
2. `IndividualWorld` + `individual_response_eval` (individual vs segment vs raw-LLM vs
   raw-LLM+context; evidence-source ablations) — validated on synthetic where the response
   function is known; **marked blocked-on-private-data** for real behavior.
3. A general as-of retrieval layer (`asof_store` + news/social/entity adapters) that **physically**
   rejects future/untimestamped items, with **leakage tests**, and a real `swm/eval/leakage.py`.
4. `simulation/{rollout,scenario_tree,counterfactuals}` + a free-running, calibration-by-horizon
   multi-step eval.
5. **EXP-009**: the head-to-head — raw LLM vs raw LLM+as-of context vs structured vs current
   calibrated vs aggregate state-transition vs individual state-transition — on identical items and
   metrics, with a clear verdict on **whether state simulation beats raw LLM + context**, and
   **whether retrieval closes the market information gap**.

**Hard-rule reminder honored throughout:** the goal is calibrated prediction + decision lift, not
architectural purity. Where the state machinery does not beat raw LLM + context, the reports say so.
