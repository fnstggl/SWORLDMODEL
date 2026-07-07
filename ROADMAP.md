# Roadmap — a general social world model simulator

**The goal.** Take an arbitrary natural-language question ("Where is LeBron more likely to win a title,
Miami or Cleveland, as of today?" / "Will the Fed cut in March?" / "Will this user churn if we email
them?"), automatically construct the belief state, map every variable acting on it, roll the simulation
forward under uncertainty, and return a **calibrated distribution over outcomes** — and the **best action**
to reach a desired outcome. This document maps everything needed to get there, grounded in what we have
built and, crucially, in what the no-cheat experiments have proven about what's hard.

## The full generative loop is assembled (EXP-057)
`swm/api/generative_simulator.py` — **one `simulate(question)` call**: identify the deciding agents + map
their known/inferred variables (LLM) → instantiate `PersonaAgent`s with an LLM persona `position_fn` → run
`AgentSociety` deliberation → emergent, auditable outcome. Verified two ways: (A) with a **structured**
position_fn on the Supreme Court it **reproduces EXP-055 exactly** (margin MAE 0.168 vs composite 0.208) —
the assembly is provably the validated agent simulation; (B) an LLM-persona worked example runs end-to-end
and produces an **emergent flip** (a committee the naive count fails, deliberation passes, full audit
trail). The general agent-based social world model exists as one callable. Remaining: a leakage-free skill
number for the LLM-driven loop (post-cutoff / market-consistency controls) and real retrieval to fill each
agent's variables from accessible knowledge.

## Agent-based simulation lands, and the interventional frontier is semantic (EXP-055/056)
- **`AgentSociety`** (`swm/simulation/agent_society.py`, EXP-055) — the real architecture the thesis wanted:
  persona agents that take positions and INTERACT (influence + homophily + consensus + bounded confidence),
  producing emergent outcomes (an influential minority **flips** a vote; deliberation drives **consensus**;
  bounded confidence sustains **polarization**) — none reachable by a mean of independents. On **real
  institutional agents (the Supreme Court, 954 cases)** it beats the independent composite on the vote
  **margin** (MAE 0.168 vs 0.208, −19%): modelling justices as deliberating agents predicts coalition size
  better than independent voting. **First real-data case where interaction beats compositing** — and it
  vindicates that institutional events (Fed, courts, awards) ARE populations of modelable agents.
- **Semantic interventional model** (EXP-056): an LLM picking the causally-better headline captures **36.5%
  of achievable uplift vs lexical's 14%** (2.6×) on the randomized-A/B KPI — the interventional task was
  semantic, not hopeless, closing much of the EXP-054 gap. The pattern holds end to end: gestalt fails
  (EXP-037), lexical fails (044/054), **semantic reading of real content wins** (047/056).
- **General prior** (Q1 fix): `llm_prior.prior_from_llm` generates the value-axis prior per question via an
  LLM (wired like `semantic_stance`); the estimator is general, not hardcoded to GSS.

## Acting on the audit (EXP-053/054)
The audit's two mandated builds landed, and both gave honest, decisive results:
- **Mean-field coupling loop** (`swm/simulation/mean_field.py`, EXP-053): makes aggregation non-separable
  (`∂pᵢ/∂pⱼ ≠ 0`). It **recovers an emergent cascade S-curve a mean-of-independent-predictions cannot**
  (coupled trajectory MAE 0.015 vs the independent flat 0.60) — genuine simulation of emergence. But on
  **real GSS aggregate opinion it does NOT beat the independent mean or persistence** — for well-calibrated
  marginal prediction, compositing suffices, exactly as the audit warned. So: coupling earns the word
  "simulate" for *dynamics/emergence*, not for beating the marginal number. The honest split is now measured.
- **Interventional KPI** (Upworthy randomized A/B, EXP-054): the first KPI that tests "what happens if I do
  X." Our models capture only **9.5% of achievable headline uplift** and rank arms at **chance (CATE-sign
  0.49)** — reconstruction accuracy did NOT transfer to intervention skill. This is the honest scoreboard
  going forward (policy-regret + CATE-sign), and the frontier is again *semantic, not lexical*.

## ⚠️ Simulation audit (see SIMULATION_AUDIT.md) — read this first
An adversarial 5-lens agent-swarm audit found the shipped pipeline is **~85% compositing, ~15% genuine
dynamics**, and the flagship `GroundedSimulator.simulate_population` is **100% composite** — a literal mean
of independent per-person regressions (`∂pᵢ/∂pⱼ = 0`, no agent interaction, no state evolution). The
genuine dynamics (`UnifiedBeliefDynamics.update_person`, `MultiStepRollout`) are **disconnected** —
`update_person` has zero call sites; the agent substrate (`actors.py`, `graph/diffusion.py`
IMPLEMENTED=False) is unwired. The KPIs (MAE-on-share, market-consistency) reward marginal-recovery and
crowd-mimicry, not simulation. The mandate: **wire a mean-field coupling loop into `simulate_population`
(rolling `update_person` over the population across sampled events) and hold it to skill-vs-persistence +
interventional KPIs — or rename the claim.** EXP-051/052 (front door + full-VariableMap) are honest
assembly/estimation work but do not change this verdict.

## What the experiments already settled (these constrain the whole design)

1. **State + cross-sectional readout work** (EXP-014/016/021/023/028): mapping a person/population to a
   VariableMap and reading out a response/opinion beats strong baselines, no-cheat.
2. **The one-step event transition works** (EXP-030): an LLM-judged event impact predicts the *next*
   belief move; on genuine event steps it calls direction 85% right vs a martingale's 0%.
3. **Multi-step extrapolation of dynamics FAILS** (EXP-033): prediction-market belief is a near-martingale
   — *nothing beats "belief stays put"* over 1–10 days, and learned momentum/mean-reversion make it
   worse. **The one-step event edge is local; it does not survive rollout.** ⇒ Long-horizon forecasting
   is a **future-event-forecasting problem, not a better-dynamics problem.**
4. **Bottom-up aggregation beats top-down** (EXP-034): simulating individuals from VariableMaps and
   aggregating beats modeling the aggregate as one number (~9% overall, ~12% on distinctive groups). ⇒
   Build the population as a set of individuals, not a scalar.
5. **Grounded-variable simulation beats the crowd composite — and the frontier is estimation, not
   enumeration** (EXP-040): on a real social outcome (OpinionQA population shares), mapping each person's
   *real* variables and simulating their answer beats the aggregate composite (individual log-loss
   0.612 → 0.585; accuracy 0.656 → 0.690), and beats the top-down aggregate by **24%** on distinctive
   subgroups. **Crucially, "map more variables" only helps once you can *estimate* their joint effect** —
   naive Bayes over 11 correlated variables (party≈ideology) overfit and was *worse* than one variable
   (log-loss 0.92 vs 0.60) until regularized. ⇒ The differentiated capability (compute an outcome from its
   constituents, beating a single aggregate number) is **real**; the binding constraint is variable
   *estimation quality* (grounding, partial pooling, correlation-aware readouts), not variable count.

6. **The estimator, not the variable list, is the fix** (EXP-041): a correlation-aware, partially-pooled
   readout makes "add more variables" safe — where naive Bayes and un-pooled logistic *collapse* as
   variables grow (log-loss 0.61 → 0.74 / → 0.86), the pooled readout stays flat (0.61 → 0.606) and cuts
   data-poor-question error **29%** (0.85 → 0.60). Partial pooling toward the prior is the load-bearing
   piece. ⇒ `swm/variables/pooled_readout.py` is the deployable grounded-variable estimator.
   **Refined (EXP-048): modelling the correlation structure beats shrinking it.** Decomposing the
   correlated variables into orthogonal **latent value factors** (PCA) and estimating on those —
   double-counting impossible by construction — beats the pooled logistic (log-loss 0.596 vs 0.610, +1.3
   acc, wins data-poor), self-selects K=3 axes, and yields an interpretable value profile. ⇒
   `swm/variables/latent_factor_readout.py` is the best self-tuning estimator; the estimation frontier is
   *structure*, not regularization.
7. **The forward operator works; its structure earns mechanism, not aggregate accuracy** (EXP-042): on
   real opinion-change events (CMV), coupling a grounded actor to the event-transition operator beats
   persistence (+0.025 log-loss) and one-sided baselines, and the **gating mechanism is verified** — the
   same argument predicts change ~38% more for responsive than entrenched people (slope 0.71 vs 0.51).
   But the multiplicative coupling doesn't separably beat an additive model at this scale/noise. ⇒ The
   operator is validated per-step; the interaction is real but needs grounded (low-noise) variables and
   more data to pay in aggregate.
8. **Grounding is necessary but not sufficient — content extraction is the frontier** (EXP-043): crude
   features from the *real* as-of news (volume, result-cue, polarity) also fail to beat the base rate
   (corr 0.047), even though the market extracts the same articles into a decisive signal (corr 0.84).
   The LLM gestalt (EXP-037) and shallow real-news reading fail for the same reason from opposite ends. ⇒
   The bottleneck for question-level forecasting is **reading real content well** (entity-linked,
   resolution-aware stance detection), not listing or nominally-grounding drivers.

9. **A grounded POPULATION is not a martingale — multi-step forward simulation beats persistence**
   (EXP-045, GSS 1972–2024, 72,707 respondents, 15 items, 406 rolling-origin forecasts): where a market
   belief cannot be rolled forward past persistence (item 3), a grounded population *can* — composing the
   opinion from demographic cells at the target year's composition beats persistence (MAE 0.0264 vs
   0.0288 overall) and **the edge grows with horizon** (4–7y: 0.0288 vs 0.0419, −31%), with change
   directional accuracy 0.593 vs 0.480. A population has predictable structure (evolving composition) a
   price does not. ⇒ The "simulate forward N steps" thesis is validated on real longitudinal data — for
   outcomes that are the aggregate of a modelable, evolving population, exactly where no market exists.
10. **Resolution-aware content extraction recovers more signal but not enough — the frontier is semantic**
    (EXP-044): entity-linked, resolution-aware lexical stance recovers 1.26× the raw outcome-correlation
    of crude features (0.112 vs 0.088) but only **13%** of what the market extracts (0.84), and doesn't
    beat the base rate in calibrated prediction. ⇒ Confirms item 8: the bottleneck is semantic stance
    detection against the specific resolution criterion (embeddings / LLM judge), not lexical features.
13. **LLM-informed priors make world knowledge worth hundreds of datapoints** (EXP-049): using the LLM's
    world knowledge as a *prior on the effect coefficients* (a one-hot logistic regularized toward the
    prior, not toward zero) dominates the data-only estimator at every training size on GSS individual
    prediction, most where data is scarce — N=50 log-loss 0.619 vs 0.669 (−8%), converging by N=5000. The
    prior is worth several hundred labeled respondents; it grounds the estimate, carries data-poor cells,
    and resolves per-question polarity. Honest caveat: the zero-shot prior *alone* underperforms the
    marginal (direction right, magnitude over-confident) — the value is prior + a little data, not the
    prior alone. ⇒ `swm/variables/llm_prior.py`; the estimation frontier now has grounding, structure
    (EXP-048), and world-knowledge priors — the three pieces the bottleneck needed.
12. **The semantic judge closes most of the content gap** (EXP-047): an LLM stance judge, reading only the
    as-of news for a question's *specific* YES resolution (blind to price and outcome), reaches **0.57
    market-consistency** (corr with the as-of price it never saw) vs lexical's 0.148 — **3.85×**, most of
    the way to EXP-037's full-pipeline 0.63. The contamination guard is the market-consistency metric (no
    outcome, so recall can't inflate it); the clean post-cutoff outcome check is still too small (n=7) for
    a skill number. ⇒ The item-8 bottleneck is *semantics*, and an LLM judge supplies it. `semantic_stance`
    is wired for production (Anthropic API backend) behind the same code path used here.

11. **Opinion change decomposes into composition (near-term) + period drift (far-term)** (EXP-046, GSS):
    coupling the compositional rollout (item 9) with a forecast of the composition-removed period residual
    helps *exactly at long horizons* — 4–7y MAE 0.0251 vs compositional-only 0.0288 (−13%) and persistence
    0.0419 (−40%) — but adds noise near-term, so compositional-only stays best overall. Responsiveness-
    gating the period shock (the EXP-042 operator at the aggregate) adds nothing. ⇒ A production population
    forecaster should weight the period term **by horizon** (off near-term, on far-term); the individual
    gating mechanism is real per-person but washes out in the aggregate share.

14. **The estimation pieces unify into one self-configuring readout, and the pipeline is assembled**
    (EXP-050): `GroundedReadout` composes latent factors + LLM prior (projected exactly into factor space
    as Vᵀ·prior) + reliability weighting (attenuate inferred variables by provenance). Because the pieces
    help conditionally, `fit_auto` self-selects the winning combination on a train-internal hold-out —
    compounding (beats every single piece and plain, GSS N=150: 0.607 vs best-single 0.608 vs plain 0.618)
    without the fixed-recipe regression. `GroundedSimulator` wires it end-to-end: question + population →
    grounded variables → structured/primed estimation → bottom-up aggregation → calibrated outcome + value
    decomposition (MAE 0.0045 vs true share; "should marijuana be legal?" predicted 0.336 vs true 0.337).
    ⇒ For the opinion/behavior domain the simulate-the-event pipeline is now one callable, not a shelf of
    experiments.

Design corollary: the simulator's power comes from (a) **who** is in the population (VariableMaps), (b)
**what events** will hit and their impact, and (c) honest **uncertainty** over both — *not* from
extrapolating a belief curve. And its *edge over the crowd* comes from **well-estimated grounded
variables** simulated bottom-up, most of all where the population is heterogeneous — not from a longer
variable list, and not from predicting a market's price (which is a near-martingale — item 3). The two
regimes are now cleanly separated by evidence: a **price is a martingale** (item 3, don't roll it
forward), a **grounded population is not** (item 9, roll it forward — the edge grows with horizon). The
concrete frontiers, in order: **estimation** (done — item 6), **grounded low-noise variables** (item 7),
**real-content extraction** (items 8/10, the current bottleneck — now known to require *semantic* stance,
not lexical), and **coupling population rollout to period/event shocks** (item 9's next step, joining
EXP-045's compositional dynamics with EXP-042's event operator).

## The honest north-star boundary (corrected)
An earlier framing over-claimed that you "cannot beat a liquid market's probability." That conflated two
different claims: you cannot predict a liquid market's price *path* (item 3, a near-martingale), but a
**structural bottom-up simulation is a different information source than the price** — it computes the
outcome from its constituent decisions, which the crowd's noisy aggregate only approximates. EXP-040 is
the first direct evidence that this simulation *beats* the aggregate composite on a real outcome. On the
most liquid mega-markets professional modelers already do this, so the price impounds it and the edge is
thin; but on the vast space of **off-market / niche / individual-scale** questions no one has run the
simulation, and grounded bottom-up simulation is both the only method available and the entire value
proposition. The project's job is that simulation — not a price-prediction heuristic.

## The target architecture (pipeline), with build status

```
 NL question
   │  [A] QUESTION INTAKE  →  proposition, resolution criterion, horizon, relevant entities/segments
   ▼
 [B] STATE CONSTRUCTION  →  current belief s_0 (retrieved market/poll/prior) + the population of
   │                        VariableMaps (individuals or demographic segments) acting on it
   ▼
 [C] VARIABLE MAPPING    →  for each actor, the known+inferred variables (VariableMap + EvidenceFusion)
   ▼
 [D] EVENT MODEL         →  the distribution of FUTURE events over the horizon + each event's impact
   ▼
 [E] ROLLOUT ENGINE      →  Monte-Carlo over event paths; per step apply the transition (per actor,
   │                        modulated by their VariableMap responsiveness), aggregate bottom-up
   ▼
 [F] OUTCOME DISTRIBUTION → calibrated P(outcome), widening bands, pivotal-branch decomposition
   ▼
 [G] DECISION / ACTION   →  for each candidate action, expected utility over F; argmax = best action
```

| stage | status | what exists / what's missing |
|---|---|---|
| A. Question intake | ❌ not built | need: LLM parses question → proposition + resolution + horizon + entities. The "front door." |
| B. State construction | ◑ partial | retrieval scaffolding exists (`swm/retrieval/`); VariableMap/EvidenceFusion build actor state. Missing: auto-retrieve the current belief (market/poll) for an arbitrary proposition, and instantiate the population. |
| C. Variable mapping | ✅ built | VariableMap + inference + EvidenceFusion + deep per-person inference (EXP-020/25/29). |
| D. Event model | ◑ variance/timing built; direction open | EXP-035: forecasts the *distribution* of belief moves (heteroskedastic variance) — a calibrated predictive distribution that beats persistence 24% / a constant band 14% on CRPS. Key finding: for an efficient series you forecast event *variance/timing*, not *direction* (direction is unforecastable — EXP-033). Open: an event *calendar* (known future dates) to place variance even better; directional forecasting only where the series is inefficient. |
| E. Rollout engine | ✅ distributional MC built | one-step transition (EXP-030), unified individual/aggregate form (EXP-032), Monte-Carlo distributional rollout with heteroskedastic variance (EXP-035), bottom-up aggregation (EXP-034). Open: pivotal-branch decomposition for multimodal futures; coupling the population rollout to the event rollout. |
| F. Outcome distribution | ◑ calibrated one-var; multivar open | EXP-035: CRPS-scored, horizon-calibrated interval bands (80% coverage) — a proper predictive distribution per horizon. Open: pivotal-branch conditionals; joint distributions across coupled questions. |
| G. Decision/action | ❌ not built | counterfactual scaffolding (`swm/simulation/counterfactuals.py`); need action → outcome-distribution → expected-utility argmax. |

## The hard core: future events and branching realities

EXP-033 is unambiguous: **you cannot forecast forward without forecasting the events.** This is the
branching-realities problem. Worked example:

> **"Will the Fed cut rates at the March 2026 meeting?"** Today: 55%. Between now and March, pivotal
> events each split the future: Jan jobs report (strong→↓ / weak→↑), Feb CPI (hot→↓ / cool→↑), Feb FOMC
> (hold→↓ / cut→↑). With *n* binary pivots there are 2ⁿ reality branches — combinatorial by months out.

You do **not** enumerate 2ⁿ, and you do **not** pick one branch. Three tractable architectures, in
increasing sophistication:

1. **Monte-Carlo trajectory sampling (the default).** Sample K trajectories. In each, step T+1 by day:
   draw whether a pivotal event fires and its outcome from the **event model (D)**, apply the transition
   (E), continue to the horizon. The K terminal beliefs form the forecast; `P(cut) = fraction of
   trajectories resolving "cut."` Scales **linearly in K**, not 2ⁿ; gives a calibrated distribution;
   uncertainty widens naturally. **This is how "vast simulation space" reduces to percentages.**
2. **Particle filter / beam over realities.** Keep M weighted "live realities" (belief-state,
   probability). At each pivot, branch each by the outcomes, reweight by outcome probability, then
   resample/prune back to M — concentrating compute on plausible worlds. Better when a few branches
   dominate.
3. **Moment propagation.** Propagate only mean+variance analytically (events as calibrated noise).
   Cheapest, but **loses multimodality** — so use it only when the future is unimodal.

**Multimodality is the real trap.** When a single pivot makes the future genuinely bimodal (e.g. "if the
Feb FOMC holds → 25%; if it cuts → 85%"), the *mean* (≈55%) is a lie no one should act on. The honest
output is the **pivotal-branch decomposition**: surface the branch explicitly — "conditional on the Feb
decision: 25% / 85%, and here's P(cut in Feb)" — rather than averaging over it. The rollout engine (E)
must detect high-variance pivots and report conditionals, not just a marginal.

**Decisions never collapse the distribution.** For "best action to reach a desired outcome," run the
ensemble **once per candidate action** (the action is an intervention in the rollout), get an outcome
distribution per action, and choose `argmax_a  E[utility | action a]` (or `argmax_a P(desired | a)`).
The branching is handled by the ensemble; the decision is an expected-utility argmax over percentages —
never a single predicted reality.

## Build order (dependency-ranked, each with a no-cheat test)

1. **Event model (D) — the unlock.** Two sub-parts: (i) *event forecasting* — over a horizon, the
   distribution of pivotal events and their timing (calendar events like FOMC/elections are known;
   surprises are a base-rate hazard model); (ii) *event→impact* — reuse EXP-030's LLM channel per
   sampled event. Test: multi-step rollout **with sampled future events** vs persistence on SWM-Bench
   futures — the real version of EXP-033. This is where long-horizon accuracy is won or lost.
2. **Rollout engine (E) — the Monte-Carlo loop** composing D + the unified transition (EXP-032) + bottom-
   up aggregation (EXP-034), with pivotal-branch decomposition. Test: horizon calibration + does it beat
   persistence *given* the event model.
3. **Horizon uncertainty (F).** Conformalize the terminal distribution per horizon; bands must hit their
   coverage at each h (EXP-033's band was over-wide — recalibrate). Test: coverage vs horizon.
4. **Question intake + state construction (A/B) — the front door.** LLM parses the question; auto-retrieve
   the current belief (Kalshi/Polymarket/poll/prior) and instantiate the population. Test: end-to-end on
   held-out resolved questions (incl. a real NBA-champion market, no-cheat as-of).
5. **Individual temporal transition.** Validate the person-level rollout directly — needs a dataset of
   individual belief *before/after dated events* (**ANES panel**, **USC Understanding America Study**;
   both registration-gated, not on HF). With it: learn `responsiveness_from_map` end-to-end (today it's a
   grounded closed form, EXP-032). Test: predict a held-out person's post-event belief.
6. **Decision/action layer (G).** Action as a rollout intervention; expected-utility argmax over the
   outcome distribution. Test: on data with observed interventions (A/B tests, outreach) — does the
   recommended action raise the desired-outcome rate?

## Honest feasibility

- **Reachable now** on public data: #1–#3 on SWM-Bench (weeks-ahead), #4's retrieval for market-backed
  questions. A short-horizon, no-cheat NBA-championship-*market* backtest is doable (pull the market +
  dated news); a trustworthy months-out "who wins 2026" forecast is **not** until #1 is built and shown
  to beat persistence with a real future-event model.
- **Data-blocked**: #5 (individual temporal — gated panels), long-horizon (months/years — SWM-Bench caps
  at ~16 days; need longer trajectories).
- **The efficiency ceiling is real — but direction is still forecastable.** Where the belief source is
  efficient (liquid markets) you cannot beat its *probability* on the point (EXP-033/035). BUT the
  *direction* implied by that probability is correct at the calibration rate — EXP-036: the lean predicts
  the move direction at 0.6–0.8 (0.85 on confident beliefs), far above chance; momentum is useless. So the
  honest product is calibrated probabilities + a directional call + pivotal conditionals. For questions
  with **no market**, the whole game is *inferring the lean (P(outcome)) from the drivers* — the VariableMap
  applied to the question; the direction then follows. That inference is the highest-leverage open build,
  and it is where a general SWM is most differentiated from just reading a market.

## One-line status

State ✅ · cross-sectional Readout ✅ · one-step event Dynamics ✅ · unified individual/aggregate form ✅ ·
bottom-up aggregation ✅ · **calibrated multi-step distributional rollout ✅ (EXP-035)** — **missing: the
question-intake front door (A/B), pivotal-branch decomposition, and the decision/action layer (G) that
turn these into "ask anything, simulate forward, choose the best action."** The forecasting core now
produces calibrated distributions over horizons; what remains is the front door (parse a question →
construct state) and the back door (outcome distribution → best action).

**EXP-064 — the world-model COMPILER (Stage ②, the keystone): `simulate(question)`.** The front door is
built. `swm/api/model_spec.py` (spec IR + a whitelisted/safe structural-equation evaluator — no eval() of
LLM code), `swm/api/compiler.py` (`StructuralCompiler`: question → ModelSpec via pluggable LLM backend;
mechanism library dispatch), `swm/api/world_model.py` (`WorldModel.simulate(question)` → retrieve → compile
→ Monte-Carlo → distribution + reducible/irreducible + forecastability + spec audit). Mechanism library:
bracket / committee (Level 2) / electorate (Level 3) / single_agent (Level 1) / generic_scm — so Levels 1–3
+ the bracket are now the compiler's library, selected PER QUESTION. EXP-064: one simulate() call handles 5
question types, each dispatched to its real generative process — NBA→bracket P(OKC)0.36, FOMC→committee
0.32, referendum→electorate 0.86, incumbent→generic_scm 0.28 (flagged UNFORECASTABLE, irreducible 93%),
email→single_agent 0.75. 8 tests incl. safe-eval rejecting __import__/attribute access; full suite 252.
Remaining honest edges: (1) spec quality is the LLM's job — build a spec-quality benchmark (tests the
"inference is good enough" bet directly); (2) scored validation on forecastable questions.

**EXP-063 — the architectural pivot: structural simulation (see ARCHITECTURE_WORLDMODEL.md).** The NBA miss
was a MECHANISM bug (deliberation on a competition), not too-few-variables. First-principles reframe: accuracy
= right causal STRUCTURE + CALIBRATED TIME + honest IRREDUCIBLE uncertainty, NOT variable count (blind
overbuilding compounds estimation error — measured: bandwagon hurt GSS). Built `swm/simulation/structural.py`:
`montecarlo` (any stochastic simulate_once), `StructuralModel` (calibrated-time diffusion SCM, drift·dt +
vol·√dt Wiener scaling), `variance_decomposition` (reducible/epistemic vs irreducible/aleatoric = the
forecastability ceiling). Results: (A) NBA as a Monte-Carlo playoff bracket → favorite ~37% (42% even with
strengths KNOWN → 58% irreducible playoff variance); the composite's 52% was OVERCONFIDENT, not wrong-by-a-
little. (B) TIME IS CALIBRATABLE and now checked: real GSS per-year σ=0.031 → 80% interval covers 85% of the
realized future (nominal 80%); a 2× clock error breaks coverage (100%/53%); engine √dt diffusion matches
closed form. (C) compiled social SCM (incumbent seat) → distribution + 95% irreducible. Target architecture
= a world-model COMPILER (question → LLM emits structural model → calibrate → Monte-Carlo → distribution +
horizon); the Levels 1–3 + bracket are its mechanism library. Runtime built; the missing keystone is Stage ②
(the compiler). NEXT: build Stage ②.

**EXP-061/062 — Level 3 (large-scale demographic) + Level 2 demographic backdrop, and the RIGHT KPI.**
Built general (an election is one instance, not the target). `swm/simulation/population_simulator.py`:
real demographic cells → coupled opinion (mean-field) + participation/turnout coupling (mobilization
cascade) → pluggable aggregator (`share_aggregator` general; `winner_take_all_aggregator` = electoral
shape). `swm/eval/population_metrics.py`: the honest KPI suite — **share-RMSE, coupling skill (does
interaction beat the marginal?), interval coverage** — because log-loss scores a binary label, not a
continuous share, and can't isolate coupling value. `AgentSociety` gains a `public_field` +
`public_sensitivity` backdrop (backward compatible). Findings, measured honestly on real data:
- **Level 3 decisive result (GSS, 1,927 predictions, 15 topics)**: on full-population opinion coupling does
  NOT beat the marginal — pure conformity skill **−0.000** (identical to the poll average, mean-preserving),
  bandwagon **−0.15** (worse; most attitudes don't endogenously bandwagon). Coupling only earns its place
  when the real process has the coupling. Confirms EXP-053 at scale with the proper KPI.
- **Where it bites (mechanism, real turnout constants)**: participation-weighted mobilization moves the
  outcome 3–4 pts (can flip a close call) — the general shape of turnout surges/adoption; needs a
  participation-weighted ground-truth dataset (real election returns+turnout, or an adoption panel) to
  *score* — named next data step.
- **Level 2 backdrop (real SCOTUS + real GSS mood, 954 cases)**: neutral-to-harmful (MAE 0.168→0.168 best,
  worse at high sensitivity) — justices' own records already price public responsiveness; a centrist mood
  pull is the wrong prior for a lopsided court. Useful null: the backdrop matters where the stakeholder
  record is ABSENT, not where decades of votes exist.

**EXP-060 — Level-1 individual simulator (the person as a dynamical system).** The three-level framework
(1: individual · 2: stakeholder group · 3: large-scale demographic) made concrete for Level 1. A person is
now `IndividualAgent` = a `VariableMap` (who they are) + a mutable STATE (mood/busyness/load/reciprocity)
that evolves as they are contacted (`swm/simulation/individual_agent.py`). Response via a pluggable
`response_fn` — grounded `StructuredResponseModel` (receptivity × quality interaction + a state gate, zero
at rest) validated here; `llm_response_fn` (LLM-as-the-person) in production. Front door
`swm/api/individual_simulate.py`: `predict_response` / `best_message` / `simulate_thread`. Validated on REAL
ChangeMyView persuasion (1,200 threads, temporal split): (A) person × message interaction beats
message-only by +0.0144 log-loss; (B) **best_message on a real natural experiment — 23 mixed-outcome OPs,
model precision@1 0.739 vs 0.518 random = +22-pt causal lift** (the "best email" product working); (C) the
same ask lands at 0.57 vs 0.64 after a pushy vs kind opener (state carryover a static vector can't express).
Honest boundary: person×message coefficients are fit; state dynamics are grounded first-principles (CMV is
one-shot), validated as a mechanism, to be calibrated on threaded reply data (named gap).

**EXP-059 — no-cheat NBA-2026 backtest through the real pipeline.** Stress-tested the end-to-end system on
an untuned domain with real leakage control (2026 Finals are post-cutoff → winner unknown; as-of Jan-2026
evidence only). Honest finding: on a competitive (mutually-exclusive) outcome the social-deliberation
aggregator is the wrong tool (injects 0.18 of spurious conformity); the right aggregator is a competition
normalization. Point pick OKC 52%, pre-registered. The system's edge is SOCIAL questions, not sports.

**EXP-058 — retrieval front door + leakage-free live forecaster.** The generative loop (EXP-057) now
has a real input (`swm/api/retrieval.py`: `web_search_retriever` for prod, `asof_retriever` for
leakage-free eval) and a forward scoring log (`swm/eval/live_forecast.py`: retrieve → simulate →
forecast → log to PostMortemLog, scored on resolution). **Settles the cutoff question:** the training
cutoff limits *memorization*, not *capability* — retrieval supplies current evidence (proven: the
committed FOMC context is dated June-2026, post-cutoff, incl. new Chair Warsh). The cutoff bites only
on *measurement*, and only for tests; two clean paths handle it — FORWARD (future event, nothing to
leak) and AS-OF BACKTEST (`asof_retriever`, evidence pre-dates resolution). In production on the API,
serving a real user question, there is no leakage to worry about. Live run: P(FOMC hike July-2026) =
0.333 → leans HOLD, from post-cutoff retrieved evidence, logged for scoring on 2026-07-29.
