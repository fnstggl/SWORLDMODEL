# `swm/engine/` — the grounded-agent world engine (what we ARE / ARE NOT using)

This package is the **only** forecasting path for people-driven questions on the `agent-engine-on-main`
branch. It is deliberately self-contained: it imports from the rest of the repo **only** the LLM backend and
(in hybrid mode) the parametric fallback. It never touches the logistic / elasticity / opinion-dynamics-ODE
machinery that earlier versions relied on.

## ✅ What the agent engine USES

| file | role |
|---|---|
| `retrieval.py` | live + **as-of** news (Google News RSS, bounded `before:/after:` + pubDate leak-guard), Wikipedia, Bing; optional keyed overlays |
| `grounding.py` | `SceneGrounder`: checklist-planned retrieval → evidence-only distilled `SceneDossier`; loud abstention; already-resolved short-circuit |
| `casting.py` | `CastingDirector`: question → social process + REAL actors (named / weighted segments) + native answer space + real-time horizon |
| `agents.py` | per-persona grounded dossier + LLM decision (reasoning, not a scalar) |
| `society.py` | dated-round rollout; per-round LLM reasoning; **log-linear opinion pool** aggregation; branch Monte-Carlo |
| `individual.py` | one person × exact message → N latent-state reasoned runs → scenario-specific p (never a base rate) |
| `actions.py` | best-action: generate real artifacts → same audience personas → ranked |
| `outcome.py` | native-typed `Forecast` (named options / ranked texts / p(reply)) |
| `calibrate.py` | grade-or-abstain registry; **log-linear pool**; **finite-sample smoothing**; **out-of-sample temperature** recalibration |
| `router.py` | `ParadigmRouter`: **people → agents (always)**, non-human stochastic → parametric |
| `front_door.py` | `hybrid_world_model()` / `AgentWorldModel.simulate()` — the one entry point |

**External imports (the only two):** `swm.api.deepseek_backend.default_chat_fn` (LLM) and
`swm.api.world_model.general_world_model` (parametric fallback, hybrid only).

**Grading / measurement (wrapped around the engine, not part of the generative model):**
`swm/eval/grade_vs_crowd.py`, `crowd_sets.py`, `grade_agent_engine.py`, `forecastbench.py`, and main's
`forecasting_corpus.py` + `backtest_harness.py` + `event_backtest.py`.

## ❌ What the agent engine does NOT use (legacy / parametric-only — never on the people path)

- `swm/api/compiler.py` — the `calibrated_readout` **logistic over LLM-invented elasticities** (the original sin).
- `swm/api/{state_grounding,live_grounding,grounding_sources,selecting_compiler,spec_validator}.py` — the variable-grounding stack for the logistic path.
- `swm/variables/*` — `bayes_logistic`, elasticity registries, calibrated weights.
- `swm/decision/*` — message optimizer, strategy scorer, elasticity fit.
- `swm/simulation/agent_society.py` — the **scalar-position bounded-confidence ODE** (homophily/consensus_pull constants). Our `society.py` is a *reasoning* rollout; it does not use this.
- `swm/api/generative_simulator.py` — the earlier LLM-position + ODE assembly.
- `swm/api/mechanisms.py` — main's 7 parametric kernels. Used **only** by the hybrid router for genuinely non-human stochastic questions (price/rate/record/launch), **never** for a people question.

## Rules that keep it honest (see `__init__.py` constitution)

1. No variables-with-weights. 2. No grounding theater (abstain loudly). 3. Native answer types.
4. Cognition is reasoning, not a scalar. 5. Real calendar time. 6. Grade-or-abstain. 7. Never the base rate.
8. **Backtests never cheat**: as-of evidence only, via bounded `before:/after:` windows + a pubDate leak-guard;
   grade on `cutoff_clean` questions vs the crowd/market.
