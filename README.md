# SWORLDMODEL — a social world model, built the honest way

A **social world model** predicts the *distribution* of human/social responses to a proposed
action (a message, product, policy, or event), under **partial observation**, with **calibrated
uncertainty**, and — crucially — **backtested against real, time-forward outcomes.**

This repository starts from a research audit (**[`docs/social-world-model-audit.md`](docs/social-world-model-audit.md)**)
and scaffolds the build it recommends. The audit is deliberately anti-hype: it separates what is
**established**, what is **speculative**, and what would require **original research**, and it argues
for starting with a narrow, paid, backtestable wedge instead of "ChatGPT for simulating the future."

## The thesis in five lines
1. **Believable ≠ accurate.** The category's failure mode is confident, well-narrated, *uncalibrated* output. So the **evaluator is the product**: build the harness that can embarrass the model *before* the model.
2. **Start narrow.** First wedge: **outbound-message response prediction & optimization** (B2B email reply, then marketing copy) — clean outcomes, timestamps, a baseline to beat, a paying buyer.
3. **Partial + probabilistic.** Every output is a distribution with an interval and a calibration grade. No prophecy.
4. **The moat is data + evaluation, not the model.** No public dataset pairs readable message content with observed outcomes; proprietary content→outcome logs are the scarce asset. Own the public, contamination-controlled benchmark too.
5. **Earn generality.** Generality is stacked calibrated wedges, each with its own backtest — never asserted up front.

## Repo layout (see Section I of the audit)
```
swm/         core library — modules map 1:1 to the architecture (Section C)
  eval/      FIRST-CLASS: harness, metrics, baselines, leakage gate  (built before the model)
api/         FastAPI service (Section J): /predict /compare-actions /simulate /backtest ...
benchmarks/  public-data harnesses (Upworthy, Criteo) for credibility results
experiments/ dated, reproducible result scripts
tests/       incl. the leakage gate that must pass in CI
```
## Core architecture — known + inferred variable mapping (EXP-020)
Every individual simulation flows through one state object: for a given `(entity, action, context)`,
the `VariableInferenceEngine` maps **every behavioral variable acting on the response** — the *known*
ones from data/user-context and the *inferred* ones from context clues (LLM) — into a
provenance-tracked, uncertainty-aware `VariableMap` (`swm/variables/`), which the `VariableWorld`
(`swm/worlds/variable_world.py`) turns into a calibrated, backtested prediction. The LLM infers
variables, never the outcome; all inference is as-of (leakage-safe); each variable earns its place on
held-out data. This routing matches the best entity-state model at ~zero accuracy cost while adding
auditability, uncertainty, and user-context merging (see `experiments/exp020_variable_mapping_architecture.md`).

## Status (updated EXP-008 — general world model)
Beyond the original scaffold, the repo now has a **real, backtested state-transition world model**
for both regimes:

- **Aggregate** (`swm/worlds/aggregate_world.py`): `PopulationState + Action → Outcome +
  PopulationState'` with subgroup priors, topic salience, domain reputation, attention/competition,
  incentives, and drift — state genuinely conditions a calibrated head. Backtested on HN
  (`experiments/aggregate_harness.py`).
- **Individual** (`swm/worlds/individual_world.py`): `this entity + action + context → response
  distribution` via hierarchical partial pooling (person ← segment ← population). Validated as an
  estimator on synthetic data; a fitted, *graded* readout still needs private labeled sends. But an
  unknown recipient is **never a hard block** — we **bias to inferring the variables when we're not
  told them**. With no private history, `World.predict` falls back to an inference-only prediction
  labeled **`unvalidated`** (honest about provenance, not a refusal), and for a **public figure**
  `swm/entities/public_figure.py` infers their disposition, status, and even responsiveness by
  **searching online** (pluggable web + LLM backend, transparent offline fallback) — folding that
  `web`-provenance evidence into the persona/`VariableMap`. Import labeled sends and `/fit` to upgrade
  an `unvalidated` inference into a backtested grade.
- **As-of retrieval** (`swm/retrieval/asof_store.py`, `news/social/entity_context.py`): a
  retrieval layer that *physically* cannot return future items, with a real leakage gate
  (`swm/eval/leakage.py`) and tests.
- **Simulation** (`swm/simulation/`): free-running rollout + a calibration-by-horizon multi-step
  eval, scenario trees, counterfactuals.
- **The head-to-head** (`swm/eval/raw_llm_vs_world_model.py`, `experiments/exp009_harness.py`):
  raw LLM vs raw LLM + as-of context vs structured vs calibrated vs aggregate/individual
  state-transition, on identical items and metrics.

Reports: `experiments/exp008_general_swm_gap_audit.md` (what's real vs stub),
`exp009_raw_llm_vs_world_model.md`, `individual_model_report.md`, `aggregate_model_report.md`,
`market_benchmark_report.md`. The hardest claims (does state simulation beat raw LLM + context? does
retrieval close the market gap? multi-step calibration?) are **measured, not asserted** — and where
the world model does *not* beat the baseline, the reports say so.

- **Episodic memory + reflection** (`swm/memory/`, EXP-074): the situation-conditioned recall layer for the
  individual regime — an episodic stream with recency × importance × relevance retrieval (Generative-Agents
  style), generative reflection that mints reusable abstractions, recency-decayed persona synthesis, and a
  retrieval-augmented `response_fn` (Beta-Binomial shrinkage toward the person's own rate). Beats the global
  persona on held-out history-driven behavior (+0.055 skill), leakage-safe, and self-limiting where the
  message (not the person) drives the outcome.

Remaining stubs (design-only): `swm/transition/mechanistic.py`, `swm/transition/llm_rollout.py`,
`swm/graph/diffusion.py` (superseded by `swm/transition/diffusion.py`), `swm/inference/filter.py`,
`swm/entities/embeddings.py`.

See the audit for the full literature map, data-acquisition and evaluation plans, wedge specs, API
spec, competitive analysis, moats, and a brutal critique of why this probably fails and how to
de-risk it.
