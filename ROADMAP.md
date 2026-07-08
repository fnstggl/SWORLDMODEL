# Roadmap — a general social world model simulator

**The goal.** Take an arbitrary natural-language question ("Where is LeBron more likely to win a title,
Miami or Cleveland, as of today?" / "Will the Fed cut in March?" / "Will this user churn if we email
them?"), automatically construct the belief state, map every variable acting on it, roll the simulation
forward under uncertainty, and return a **calibrated distribution over outcomes** — and the **best action**
to reach a desired outcome. This document maps everything needed to get there, grounded in what we have
built and, crucially, in what the no-cheat experiments have proven about what's hard.

## Real semantic embeddings + the continuous (idempotent) harvest (current)
The two flywheel follow-ups, both built:
- **Real embeddings** (`swm/variables/embeddings.py`, EXP-080): a production HuggingFace/OpenAI embedder
  (all-MiniLM-L6-v2) behind the pluggable `embed_fn` seam, with a committed cache (`prior_embeddings.json`,
  597 keys) so semantic transfer works offline. **Paraphrase probes recover the right elasticity 4/5 with real
  embeddings vs 1/5 lexical** — "consumer price growth → monetary tightening" finds inflation→rate_hike (+0.44),
  "joblessness" finds unemployment→rate_hike (−0.57), where lexical finds nothing. True meaning-based transfer
  across the 592-elasticity corpus.
- **Continuous harvest** (`experiments/exp081_continuous_harvest.py`): rebuilds the registry FROM SCRATCH over
  all 8 datasets each run — idempotent (precision-weighted combination would double-count on re-run, so a
  fresh rebuild is the correct design), refreshes embeddings for new keys, and appends to
  `harvest_manifest.json`. The entry point a scheduled Routine invokes to keep `learned_priors.json`
  compounding as data lands (registry now single-counted: party=republican n=60k, was double-counted).

## The six peak-architecture builds — all landed + validated (current)
The full ARCHITECTURE_PEAK.md list, built this session:
- **#1 Corpus harvest** (`swm/eval/harvest.py`, EXP-076): fit elasticities across **8 datasets** (GSS, OQA,
  CMV, FOMC, Upworthy, StackExchange, Telco-churn, GlobalOpinions) → **592 learned priors, 15 outcome-classes**
  committed to `learned_priors.json`, all sign-correct (inflation→hike +0.44, unemployment→hike −0.58,
  ideology=liberal→conservative −0.26). The flywheel at corpus scale.
- **#2 Regime router** (`swm/eval/regime_router.py`, EXP-078): a calibrated classifier (fit on the portfolio +
  a world-knowledge prior) that routes population/diffusion → rich_sim (0.75–0.83) and macro/election/market →
  baseline (≤0.35) per question. Never lose to a simple baseline by over-simulating.
- **#3 Adaptive fidelity** (`swm/api/adaptive_fidelity.py`): variance-triage — rank variables by their share
  of outcome variance so calibration compute goes only to the high-leverage few (makes "model everything"
  tractable).
- **#4 Embedding-keyed registry** (`swm/variables/embedding_registry.py`): cross-phrasing elasticity transfer
  with an sd widened by transfer distance; pluggable real-embedding backend (lexical default offline).
- **#5 Event model** (`swm/simulation/event_model.py`, EXP-077): calibrated event-hazard + impact places
  discrete pivotal-event variance over the horizon — on FOMC rate jumps, **82% interval coverage (nominal 90%)
  vs persistence's 3%**. The first version of the long-horizon frontier.
- **#6 Full-covariance weight posterior** (`bayes_logistic.predict_dist(full_cov=True)`): honest joint
  uncertainty for correlated variables via Cholesky sampling.

## Portfolio backtest + calibration wired into the compiler + the flywheel ON
The data-scaling program: calibration made a default of the compiler, mapped across many domains, and fed by
real data.
- **Calibration wired into the compiler**: `SpecVar` now carries a `weight`/`weight_sd` (elasticity + CI +
  provenance); the new `calibrated_readout` mechanism integrates BOTH value and weight uncertainty in the
  Monte-Carlo (an unknown weight widens, never biases). `swm/api/calibrated_compiler.py` overrides LLM weights
  with data-learned ones (`apply_registry`) and fits weights from any labeled dataset (`calibrate_from_data`).
- **The learned-prior registry** (`swm/variables/prior_registry.py`): elasticities from every dataset
  accumulate, precision-weighted, keyed by (variable, outcome-class) — more data ⇒ tighter transferable priors.
- **EXP-075 (harvest) — flywheel ON**: 15 GSS items → **59 learned elasticities** committed to
  `learned_priors.json`, all sign-correct (party=republican +0.18 conservative, ideology=liberal −0.26,
  attendance=high +0.20); the shared party→conservative elasticity tightens as items accumulate (sd 0.199 →
  0.151). The compiler consults these so demographic variables arrive pre-calibrated.
- **EXP-074 (portfolio) — WHERE fidelity wins, 6 real domains no-cheat** (`swm/eval/portfolio.py`; new
  downloaded domains: OWID adoption, Swiss referenda, MIT Senate): fidelity WINS big on modelable evolving
  populations/diffusions — **opinion +0.150, adoption +0.316 skill vs persistence** — and adds little where a
  strong simple baseline exists (senate +0.03, referenda ≈ base-rate, fomc momentum-dominated on direction),
  and can fail where change is period- not composition-driven (spending −0.04). The empirical rule: reach for
  the rich sim on modelable populations with weak baselines.
- **ARCHITECTURE_PEAK.md**: the six changes to peak — (1) run the calibration harvest at corpus scale,
  (2) a learned regime router, (3) adaptive fidelity via variance triage in the loop, (4) embedding-keyed
  registry, (5) a validated event model (the real long-horizon gap), (6) full-covariance weight posterior.

## Calibrated weights + the no-cheat event backtest — the thesis wins on real data
The disagreement ("model every variable" vs "less is more") is settled empirically, in favor of the thesis
*conditional on proper calibration*:
- **The calibration engine** (`swm/variables/bayes_logistic.py`, `calibrated_weights.py`): a weight is a
  causal elasticity carried WITH uncertainty. Four sources → a `WeightPrior` (mean + CI + provenance: fit /
  pooling / literature effect-size / LLM-elasticity); a Laplace **posterior over the weights** (per-weight
  prior precision from the CI); `predict_dist` **integrates weight uncertainty** (unknown weight → wider
  forecast); **empirical-Bayes** n-adaptive shrinkage (`fit(tune=True)`); variance **triage** and
  **active-learning** targets (high leverage × high uncertainty = measure next).
- **The event backtest** (`swm/eval/event_backtest.py`): score any forecaster vs the skeptic's free
  baselines (persistence/momentum, base rate, market) on historical "predict the future" questions, with an
  as-of leakage guard. SKILL = 1 − loss/loss_baseline.
- **EXP-072 (fidelity ladder, OpinionQA)**: adding 12 variables HURTS naive +0.111 log-loss but properly-
  tuned shrinkage makes it +0.015 (and up to 6 vars it HELPS) — the binding constraint is weight calibration,
  not count.
- **EXP-073 (GSS opinion, rolling origin @2006, 133 no-cheat forecasts) — the decisive win**: the calibrated
  11-variable forward simulation beats persistence **+0.107 skill**, and the linear trend and base rate too
  (**beats ALL baselines**). Crucially **more calibrated variables flipped a loss into a win** (2 vars
  −0.032 → 11 vars +0.107). On a modelable evolving population where simple baselines are weak, fidelity
  buys accuracy — the digital-twin bet pays. (SCOTUS/FOMC didn't clear the bar because their simple
  baselines — static ideology, policy inertia — are strong; the lesson is *where* to reach for the rich sim.)

## The action layer, the navigable object, the select loop, and the flagship demotion (current)
The two core value props are built on the compiler and scored on real data, and the compiler's keystone gap
is closed:
- **The general action layer** (`swm/decision/`, `swm/api/action_simulate.py`): `argmax_a E[U|do(a)]` as an
  inner Monte-Carlo × outer **best-arm racing** loop — typed `do`-operators on the compiled `ModelSpec`,
  risk objectives (mean/quantile/CVaR), a confident winner *or an honest "tie within noise"*, contrast vs
  do-nothing. **Re-earned on REAL data (EXP-069): CMV best-message precision@1 0.739 vs 0.518 random =
  +22pt**, with proven selection parity to the old `best_message` path; Upworthy interventional scoreboard
  runs on real randomized `do(x)` data (lexical floor 9.6%, semantic ceiling needs an LLM). **All 7 design
  components are now built** (see `ARCHITECTURE_ACTION_LAYER.md` completion map): typed parameter/structural/
  **temporal** (`inject_event`) interventions, continuous **refine** + LLM **propose→mutate** generation,
  best-arm racing, **constrained** risk, navigable + **calibration grade**, **sequential policies**
  (`best_policy`; person-as-dynamical-system + timed schedules — EXP-069 Part C reproduces the opener→ask
  state-carryover on the real fitted model), and a **provenance** label (validated vs hypothesis domain).
- **The navigable object** (`swm/report/navigable.py`): replaces a scalar with distribution +
  reducible/irreducible split + automatic **pivotal-branch** discovery ("37%, here's the fork").
- **The compiler's candidate-and-SELECT loop** (`swm/api/selecting_compiler.py`): EXP-068 added
  validate→repair for ONE spec; this proposes K candidate structures, scores each on validity × an optional
  LLM critic (a broken spec can't win on charm), selects the best, and reports cross-candidate **agreement**
  (the honest "how sure are we of the structure" signal). The verify/select keystone the compiler needed.
- **Flagship demoted, hard**: `GroundedSimulator.simulate_population` → `IndependentPopulationReadout.
  predict_share` — a calibrated bottom-up compositor and one leaf in the mechanism library, NOT a
  simulation. The word "simulate" is reserved for the compiler running the right mechanism
  (`WorldModel` / `ActionWorldModel`). Old names remain as deprecated back-compat aliases.

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

**EXP-074 — episodic memory + reflection: situation-conditioned recall beats the global persona.** The
individual regime carried a person as a global average (persona + state); this adds the Generative-Agents
recall layer (`swm/memory/`): an episodic stream with recency × importance × relevance retrieval, generative
reflection that mints reusable abstractions fed back into retrieval, recency-decayed persona synthesis
(`deep_inference`), and a retrieval-augmented `response_fn` (Beta-Binomial shrinkage toward the person's own
rate — self-limiting, calibrated by construction). Leakage-safe (strict as-of, `assert_no_leak`). Scored on
held-out next behavior: (A) history-driven regime — retrieval beats the global persona **+0.055 skill** (log-
loss 0.681→0.644, better Brier); (B) recency — a *calibrated* half-life (hl=12) beats flat and over-decay
(hl=3) hurts (the Law-2 lesson); (C) honest negative — where the MESSAGE (not the person) drives the outcome,
retrieval correctly finds no exploitable signal (self-limiting −0.036), reproducing EXP-069's "persona models
WHO, not message-driven outcomes"; (D) persona synthesis of a drifting trait tracks the recent value (0.50→
0.74) vs flat's stale average. ⇒ For the single-individual product (reply/churn/adherence) recall is a real
lift; next is re-earning it on real threaded reply data through the same harness. `swm/memory/{embeddings,
memory,retrieval_response}.py`; 19 new tests; full suite 349 (+ the optional-`api` fastapi test).

**EXP-073 — the best-message ceiling + DeepSeek estimation (answering "why not 90-95%?").** Measured on 64 real mixed-outcome CMV cases (leave-one-OP-out), DeepSeek re-scored all 138 args on richer persuasion dimensions. Findings: (1) 90-95% is NOT achievable -- even overfitting all data with rich features tops out ~0.83, so ~17% of "will THIS message flip THIS person" is genuinely irreducible; (2) real headroom exists -- the ceiling ROSE 0.75->0.83 with DeepSeek features, so better estimation exposes more reducible signal; (3) the bottleneck is DATA not the model -- richer features raised the ceiling but NOT leave-one-out (0.656), because 138 examples is too few to learn an 18-feature mapping. Immediate win: DeepSeek's holistic judgment ranked directly beats the trained pipeline (0.672 vs 0.656) with ZERO training (= the InterventionSelector via the stronger backend). Path to genuinely better: more data (full CMV corpus + more datasets) to reach the ~0.83 ceiling, and chase higher-ceiling mechanisms. Validates the user's more-data+DeepSeek plan; honest target ~0.80 on persuasion, not 0.95. swm/api/deepseek_backend.py wired as default backend.

**EXP-072 — real contagion/tipping test: the coupled dynamic FINALLY beats simple baselines.** The regime
EXP-070/071 predicted a shared-world model would win: strong endogenous feedback + weak simple baselines.
Real SSA baby-name shares (481 names, 1880-2008 — pure imitation-driven fashion cascades), forecast H=10yr
ahead, leakage-free. Models: persistence, trend, and CONTAGION (coupled bandwagon+saturation: growth dragged
down by its own level → rise, peak, reverse; 2 params fit on train names, scored on test names). Result:
**at TURNING POINTS (near peak, n=1083) contagion MAE 0.152 beats persistence 0.264 and trend 0.570 — +42%
skill** — the first real-data case where a coupled non-separable dynamic substantially and cleanly beats the
simple baselines. Overall +9%. Honestly loses on STABLE (trend better) and RISING (persistence safest) — not
a universal win, the right tool specifically in the tipping regime. Closes the through-line: SCOTUS/FOMC
coupling ties (strong simple baselines), contagion coupling WINS (weak baselines + genuine cascade) — the
complete measured answer to WHEN the shared-world machinery beats separate models. 3 tests; data cached.

**EXP-071 — environment→individuals→institution on REAL FOMC data (the substrate's next coupling, scored).**
The coupling EXP-070 pointed to, on real FRED data (FEDFUNDS/CPI/UNRATE, 1985-2026, 494 months, leakage-free,
40% holdout). Macro pressure → members' desired policy → committee vote → rate move. Three honest findings:
(1) the environment→decision coupling EARNS its place — Taylor pressure lifts direction 0.641→0.707 over
always-hold; (2) the middle MEMBER scale does NOT — routing through discrete/saturating voting members
degrades back to 0.641 (throws away the graded signal); (3) policy INERTIA dominates — a momentum baseline
crushes every macro model (MAE 0.071 vs 0.177, dir 0.92). Verdict: keep the environment coupling, DROP the
member scale for FOMC — the discipline working. Across both scored couplings (SCOTUS EXP-070, FOMC EXP-071)
the individual↔institution scale does not beat simpler baselines (inertia/static ideology); the substrate's
machinery is real (bank-run cascade) but a shared world wins only where endogenous cross-scale feedback is
strong AND simple baselines are weak — a measured finding about WHEN to reach for it. 3 tests; data cached.

**EXP-070 — the persistent World substrate (the digital-twin leap), scored.** `swm/world/substrate.py`:
`Entity` (node: person/institution/population/environment, advanced by its own mechanism step_fn, read by
readout_fn) + `Coupling` (directed edge wiring one entity's output into another's input) + `World`
(advance(dt) steps all entities on ONE shared clock; query(); without_couplings() for the ablation;
rollout/montecarlo_world = question as a query against forward-sims of the shared world). The missing single
time-axis + cross-scale wiring, vs independent per-question models. (A) Cross-scale feedback demo — a bank-run
world (rumor→depositors→bank→rumor): identical shock → COUPLED cascades to FAILED, SEPARATE stays stable;
contagion is emergent, unreachable by independent scales. (B) Real scored two-scale (justices→Court, SCDB,
leakage-free): individual-scale ideology DRIFT coupled up vs STATIC justices — margin MAE 0.182 coupled vs
0.170 separate → coupling does NOT beat separate → per the discipline, DO NOT scale up here (consistent with
EXP-062). The substrate is the machinery; whether to wire any pair of scales is now an empirical question
with a scoreboard. Next coupling to test: environment→individuals→institution (FOMC macro→members→vote) where
the feedback A demonstrates is real. 6 tests; full suite 292.

**EXP-069 — deep per-person inference (the interview-gap lever), measured.** Our scalable analog of SOTA's
2h interview = deep multi-pass inference over a person's writing history (`swm/variables/deep_inference.py`).
On 160 real CMV authors (8-25 docs each, agent-swarm persona signals): (1) DEPTH HELPS, monotonically —
predicting a held-out doc's facets from the person's prior docs, MAE drops 0.096→0.084 as history deepens
1→16 (−13%); the confidence-blend (deep shrunk toward population by 1−confidence) beats the population
baseline by ~8% and is now the default `persona_to_vars`, wired into the Level-1 response model
(`DeepPersonaStore.vars_asof`). (2) DOWNSTREAM payoff is outcome-dependent: the arguer's persona does NOT
beat base-rate at predicting whether a specific argument persuades (0.74 vs 0.66) — that outcome is driven
by the argument+OP, not the arguer's disposition. Answer to "do we need it?": YES for modeling WHO a person
is (the individual/single-agent lever), not as a universal multiplier. Improvement modest (~8%) because a
doc is a noisy trait realization — mirrors why SOTA tops out ~85%. 7 tests; full suite 264.

**EXP-068 — self-correcting front door + scored end-to-end run.** (1) `WorldModel` now wraps its compiler in
`ValidatingCompiler` by default (`validate=True`): every simulate() validates + repairs the spec before
running, validation report in the output; `validate=False` keeps the raw path. Also added a `non_numeric_
field` static check + guarded `run()`. (2) Scored end-to-end on 15 GSS opinion topics (LLM compiled the WHOLE
spec via Qwen-72B, ~12yr horizon): the validator caught 5/15 malformed specs LIVE (LLM used categorical
string labels for numeric stances — non_numeric_field, no crash) — self-correction working in the wild. On
the 10 clean specs: MAE 0.078 vs persistence 0.076 → skill −0.02 (TIES persistence, doesn't beat it),
coverage 0.50 (over-confident). Honest: decade-horizon opinion is persistence-dominated (per EXP-053/061);
the world-model's wins are on STRUCTURED questions (EXP-065), not mass opinion. Calibration miss diagnosed to
mechanism choice (electorate under-propagates variance vs a diffusion) → next build: mechanism selection
should weigh uncertainty propagation. Full suite 262.

**EXP-067 — spec validator + repair loop (closes the EXP-066 gap: buggy equations).** `swm/api/
spec_validator.py`: `validate(spec)` runs static + a simulate-and-inspect dynamic pass — equilibrium-out-of-
bounds/saturation (root-finds where drift=0; flags if outside [lo,hi] — the load-bearing check), degenerate/
trivial outcome, event-threshold-outside-support, value/volatility sanity, bad equations. `ValidatingCompiler`
= compile → validate → LLM repair → re-validate (pluggable repair_fn). Demonstrated on Qwen's REAL inflation
bug: validator flagged saturates_bound + degenerate + trivial; LIVE Qwen repair rewrote the equation to
proper mean-reversion in 1 round → sane forecast (P=0.92, interval [2.3,5.7]). No false positives (3 clean
specs → 0 errors); every check fires on targeted broken specs. Pipeline is now question → compile → VALIDATE
& REPAIR → Monte-Carlo → distribution (generated model tested before trusted). 7 tests; full suite 261.

**EXP-066 — can the LLM pick the right RATE on its own? (the open measurement, now closed).** External
model (Qwen-72B, blind) estimates the per-topic year-to-year opinion VOLATILITY for 15 GSS topics (data
truth 1.5–4.9 pp/yr, with a drift-vs-volatility trap). Findings: (1) SCALE excellent — geo-mean ratio
LLM/data 0.96, 100% within 2×; (2) per-topic DISCRIMINATION weak — Spearman 0.39 (confuses cumulative drift
with volatility, e.g. rates marijuana most-volatile); (3) DOWNSTREAM calibration fine — forecasts from LLM
rates cover 0.798 vs 0.803 from true rates ≈ nominal 0.80, so the weak ranking barely matters; (4) full
autonomous spec authoring runs but is buggy — Qwen's inflation generic_scm had right structure but a
malformed mean-reversion equation (equilibrium ~35%, saturated the bound → degenerate P=1.0). Verdict:
mechanism ✅, rate-scale ✅, rate-ranking ⚠️(harmless), full-equation-authoring ❌ (needs a spec
validator/repair loop). The core bet largely holds; NEXT concrete build = a spec validator (equilibrium-
in-bounds / no-saturation / dimensional sanity via simulate-and-check). `swm/api/hf_backend.py` reused
(token from env only). Full suite 254 (no new py-tests; empirical experiment).

**EXP-065 — spec-quality benchmark + scored validation of the compiler on REAL outcomes.** The direct test
of the thesis. PART 1 spec quality: (a) mechanism selection graded by an EXTERNAL blind model (Qwen-72B via
HF) = 15/15 correct on answered (5 hit HF credit limit, not model error) — picking the generative structure
is robust; (b) rate calibration — with the data-measured GSS σ=0.027/yr the 80% interval covers exactly
0.80, a 2× clock error breaks it (0.98/0.48) — rates are checkable and consequential. PART 2 scored
validation through the ONE compiler interface on real resolved data: committee/SCOTUS (400 cases) margin MAE
0.172 BEATS independent 0.215 (reproduces EXP-055); single_agent/CMV best-message precision@1 0.69 vs 0.51 =
+0.18 lift (reproduces EXP-060); electorate/GSS RMSE ties marginal (honest, marginal-dominated). The unified
front door reproduces every validated per-mechanism result on real outcomes. `swm/api/hf_backend.py`
(external LLM backend, token from env only), `swm/eval/world_model_bench.py` (uniform scoreboard). 2 tests;
full suite 254. FRONTIER now isolated: variable/rate ESTIMATION quality, not architecture. Open: automate +
measure the LLM's rate CHOICE (needs API budget; HF probe was credit-limited).

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
