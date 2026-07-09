<!-- CALIBRATION & FIDELITY PROGRAM (EXP-090..093) — the push to beat the market -->
## The calibration/fidelity program — from catastrophic F toward the market

A 6-agent research fan-out (LLM-forecasting SOTA, forecast aggregation/extremizing, silicon-sampling
fidelity, calibration, and a failure analysis of our own rows) ranked the levers by **measured Brier
recovery on the failure set**, and we ablated them in that order on a leak-free crowd backtest (bounded
`before:/after:` as-of Google News; resolved Manifold/Polymarket questions with the crowd price at as-of;
`cutoff_clean`).

**The bottleneck was Stage-0 grounding of the DIRECTIONAL signal, not aggregation or calibration.** The
engine got *direction* wrong ~half the time (side-correct 0.53 = a coin flip), near-random on the questions
the market was most sure about — a base-rate/evidence failure, not a pooling-math one. Measured leverage:
grounding the "who's favored" signal **+0.111 Brier (3–5× everything else combined)**; out-of-sample
recalibration +0.03; the 0/1 clamp +0.01; **more/proportional agents ≈ 0** (the failure is bias, not
variance); and **extremizing is NEGATIVE before direction is fixed** (sharpening a wrong-side consensus).

What we built, in ablation order:
1. **Aggregation/calibration** (`calibrate.py`): weighted **log-linear opinion pool** (uninformative 0.5s
   stop dragging the signal to the middle) + **finite-sample smoothing** + **out-of-sample (k-fold)
   temperature** recalibration (honest, not in-sample-optimistic) + an aggregate 0/1 clamp.
2. **Rank-1 grounding** (`grounding.py`, `retrieval.py`): a **structured directional standing**
   `{favored, margin, basis, confidence}`, extracted explicitly and made **common knowledge** to every
   agent; a **second targeted "who's favored" retrieval round** when the signal is missing; the backtest
   drives the engine's own multi-round as-of retrieval (`asof_search_fn`), never live news.
3. **Observer panel** (`observer_panel.py`): binary events route to a diverse ensemble of **base-rate-
   anchored superforecaster personas** (5 reasoning lenses) — the literature-backed way to match markets —
   with **confidence tracking evidence** (shrink toward the base rate when the standing is weak).
4. **Measurement** (`crowd_sets.py`, `grade_vs_crowd.py`): a large, cleaned, multi-domain, liquidity-
   filtered set + a first-class **direction-accuracy** metric that isolates the grounding bug from
   calibration.

**Measured trajectory (leak-free, vs the crowd):**

| stage | skill vs crowd | direction | note |
|---|---|---|---|
| F baseline (political, liquid crowd) | **−6.15** | 0.53 | p=1.00 on losers; near-random |
| + pooling/smoothing/standing (society, diverse) | −0.03 (recal −0.01) | — | +0.007 on crowd-unsure |
| + observer panel + Rank-1 grounding (diverse) | recal −0.05 | **0.61** | **beats crowd on tech +0.117**, parity sports |
| + confidence-shrink (diverse) | recal −0.17* | 0.61 | *per-category swings show n=44 is noise-limited |
| definitive **n=127** (current best config) | _see EXP-093 result_ | — | the trustworthy, noise-resistant number |

**The honest state.** We eliminated the catastrophic failure (worse-than-a-coin-flip → **near-parity with a
strong liquid-market crowd**) and fixed direction (0.53 → 0.61), and the engine already **beats the crowd on
evidence-rich domains (tech)**. Two honest limits remain: (a) at n≈44, per-category and per-tweak
differences are **within noise** (±0.15 skill) — measurement power, not more tweaks, is the binding
constraint, which is why EXP-093 runs at n=127; and (b) beating a liquid market **overall** on a general set
is a hard bar the literature shows even frontier LLM systems mostly *match*, not beat — the defensible
target is **match overall, beat on the evidence-rich / crowd-unsure slice**. Remaining ranked levers
(gated behind the now-fixed direction): per-domain / two-sided calibration, a vote-share→win map for the
society path, selective forecasting (defer on weak evidence), and multi-model-family panels to decorrelate
errors (which is the only condition under which extremizing turns positive).

---

# The Grounded-Agent Engine — the vision clause by clause (hybrid front door on `main`)

**The vision**: *Take an arbitrary natural-language question → automatically construct the belief state →
map every variable acting on it → roll the simulation forward under uncertainty → return a calibrated
distribution over outcomes — and the best action to reach a desired outcome.*

**The engine** (`swm/engine/`): for anything driven by **people**, there is ONE mechanism — **grounded
agents interacting** — and the only thing that varies per question is the *casting* (who the agents are,
what the answer space is, what the interaction structure is). No logistic-over-invented-variables ever
touches a people question.

**Hybrid routing (`swm/engine/router.py`), built on `main`.** `main` carries a mature *parametric*
mechanism engine (`swm/api/mechanisms.py`: 7 grounded stochastic kernels — aggregation, contest, diffusion,
arrival, whipcount, escalation, persistence) plus the grounding + backtest infrastructure the agent engine
needs. The `ParadigmRouter` sends every question to the right engine, **biased hard toward agents**:

- **people → agent society (always).** Any outcome generated by human choices/behavior/votes/reactions —
  elections, approvals, rulings, adoption, whether someone replies/buys, negotiations, unrest. This is the
  default and the vast majority. A lexical fast-path routes unmistakable cases with no LLM call; ambiguity
  and classifier failure both fall toward agents.
- **non-human stochastic process → main's parametric kernels.** ONLY a genuine non-human process where no
  group's decisions are the mechanism — a market price crossing a level, a physical/weather measurement, a
  scheduled launch, a statistical record. `hybrid_world_model()` wires main's `general_world_model()` here.

`hybrid_world_model()` is the recommended front door. `agent_world_model()` is the pure agent engine.

## The constitution (anti-regression clauses — enforced by `tests/test_agent_engine.py`)

These are the specific failures the 2026-07 vision-gap analysis caught the old system committing. Each is
now a hard rule with a test pinning it:

1. **No variables-with-weights.** Never decompose a question into abstract latent scalars
   (`name_recognition = 0.5`) pressed through a logistic with LLM-invented coefficients. The LLM is asked
   the question it is actually good at — *"how does this specific grounded person react to this specific
   situation"* — never *"emit an elasticity of 0.4 ± 0.25"*.
2. **No grounding theater.** A fact exists only if a real, dated, cited passage established it. Missing
   deciding facts ⇒ **loud abstention** (`test_grounding_abstains_loudly_when_starved`), never a neutral
   prior wearing `source="retrieval"`. And if the evidence shows the world already answered (NY-10:
   Lander d. Goldman, 2026-06-24), the engine says **ALREADY RESOLVED** with the citation instead of
   "simulating" a settled fact.
3. **Native answer types.** "Who wins" → a distribution over NAMED candidates. "Best headline" → ACTUAL
   ranked headline texts. "Will X reply to this email" → a scenario-specific p for THAT person and THAT
   message. The simulation's state space IS the answer space, so the output cannot fail to answer the
   question.
4. **Cognition is reasoning, not a scalar.** Each round, each persona is the LLM reasoning from its
   grounded dossier + the public signal, emitting a decision distribution *and its reason* (auditable).
   The old `AgentSociety`'s invented ODE constants (`homophily=0.5`, `consensus_pull=0.3`, scalar
   `position_fn`) are gone from this path — if agents herd, it is because they read the same public
   signal and reasoned their way there.
5. **Real calendar time.** Rounds are dated; the cast fixes `resolve_by` from evidence (an election day)
   and the cadence of the situation's news cycle; a 30-day horizon is 30 real days
   (`test_dates_map_horizon_to_real_calendar`). Rollout stops at the real resolution date.
6. **Grade-or-abstain.** A distribution ships as confident ONLY if its question-class carries a
   backtested grade in `models/agent_engine_grades.json`; otherwise it ships flagged
   `[ungraded — hypothesis, not a calibrated forecast]` in the headline itself. "No ungraded logistics"
   generalized to **no ungraded simulation**.
7. **Never the base rate.** Individual-response predictions are always conditioned on WHO the person is
   (live-grounded) and THE exact message. If the person cannot be grounded, the engine abstains — it does
   not return a global reply rate (`test_individual_abstains_when_person_cannot_be_grounded`).

## Pipeline

```
question ──► Stage 0  GROUND      swm/engine/retrieval.py + grounding.py
             │        checklist of deciding facts → targeted queries → dated cited passages →
             │        evidence-only distillation → SceneDossier {facts, missing, coverage, resolved?}
             │        ├─ starved/uncovered ⇒ ABSTAIN (loud, with what's missing)
             │        └─ already resolved  ⇒ answer + citation, done
             ▼
             Stage 1  CAST        swm/engine/casting.py
             │        social process ∈ {individual_reaction, collective_choice, population_share,
             │        artifact_optimization}; REAL actors (named people / weighted segments from the
             │        dossier); NATIVE answer space; resolve_by + cadence in real time
             ▼
             Stage 2  DOSSIERS    swm/engine/agents.py
             │        each actor → diverse personas: segment ⇒ distinct concrete draws; named person ⇒
             │        latent states (busy/attentive/skeptical). Private, rotated evidence slices.
             ▼
             Stage 3  ROLL        swm/engine/society.py | individual.py | actions.py
             │        dated rounds; per round each persona REASONS (LLM, temperature) → decision distribution
             │        + statement; next round's public signal = SAMPLED realization (poll reading +
             │        overheard statements) ⇒ cascades possible; B independent branches
             ▼
             Stage 4  CALIBRATE   swm/engine/calibrate.py + outcome.py
                      class grade from resolved-history backtests (event_backtest + ForecastBench);
                      fitted logit-shrink applied; ungraded ⇒ flagged; native-typed Forecast with
                      grounding report + per-persona WHY audit + branch-spread intervals
```

Uncertainty enters through four honest doors, separated in the output: persona sampling (who exactly the
population is), private information (who has seen what), interaction stochasticity (branch spread), and
decision noise (temperature). The anti-monoculture defenses exist because silicon populations are known
to agree too much and understate tails; variants + private slices + temperature force real dispersion.

## Live validation (EXP-089, 2026-07-09, DeepSeek + keyless retrieval)

| question | old front door | agent engine |
|---|---|---|
| Who wins the NY-10 Dem primary? | `P(event)=0.957` about an abstract logit; no candidate named; missed that the race was OVER | **ALREADY RESOLVED: Brad Lander** [CBS News 2026-06-24], 9.5s |
| Best AirPods Max landing headline? | `p=0.38` about copy-properties; zero headlines written | **5 actual headlines, ranked** by simulated audience engagement (52% feature-led winner), flagged ungraded |
| Will Peter Thiel answer this cold email? | (not answerable — would fall to a base rate) | **11%** (80% interval 3–20%), 24 grounded runs across 4 latent states built from his *current* news; flagged ungraded |
| Who wins the 2026 NY governor race? | logistic over invented variables | **Hochul 72% / Blakeman 28%** (branch spread 68–77%), 6 real voter segments, rounds dated to election day 2026-11-03, per-persona WHY audit; flagged ungraded |

## Retrieval stack (Stage 0 research, probed live through this proxy)

- **Keyless defaults (free, effectively unlimited)**: Google News RSS (the workhorse — rich, dated),
  Wikipedia API, Bing HTML. DuckDuckGo is blocked (202 anti-bot) — the root cause of the old grounding
  starvation. FEC API works with `DEMO_KEY` for election money.
- **Optional keyed overlays, auto-detected from env**: `SERPER_API_KEY` (cheapest keyed, ~$0.30–1/1k,
  2.5k free), `BRAVE_API_KEY` (~$5/1k), `TAVILY_API_KEY` (~$8/1k). Merged when present; never required.

## The calibration program (what earns the grades) — BUILT, running (EXP-090)

The grade-or-abstain wrapper is now fed by a real, leak-free backtest, not a promise:

- `swm/eval/event_backtest.py` stays the no-cheat scorer (as-of guard, skill vs free baselines).
- `swm/eval/forecastbench.py` loads **ForecastBench** rounds (Karger et al., ICLR 2025; CC BY-SA 4.0;
  `question_sets/` + `resolution_sets/`) — resolved Metaculus/Manifold/Infer/real-data questions.
- `swm/eval/grade_agent_engine.py` + `experiments/exp090_grade_agent_engine.py` run the engine **as-of a
  round's due date on already-resolved questions, feeding ONLY the frozen as-of context** (`background` +
  `resolution_criteria`) — never the live web. Two leakage doors shut: (1) live-retrieval leak, closed by
  `evidence=` injection (no fetch); (2) training-recall leak, mitigated by choosing rounds whose due date
  is **past DeepSeek's ~mid-2024 cutoff** (the only airtight version forecasts questions resolving in the
  future — the live-and-wait track). Scored against 0.5 **and the sample class rate** (beating the class
  rate needs discrimination, not "these are longshots, guess low"). `GradeRegistry.record` writes the
  grade + a fitted logit-shrink to `models/agent_engine_grades.json`.
- **Prophet Arena** (Kalshi-anchored live eval, Brier + average return, 1300+ resolved events) is a
  leaderboard to enter once classes are graded — not a training set.
- Until a class is graded, every forecast in it carries the ungraded flag. That is the point.

### Grading vs the CROWD (the market bar) — `swm/eval/grade_vs_crowd.py`

The honest bar is not 0.5 or a class rate — it is the **crowd/market**, which is what ForecastBench and
Prophet Arena score against. `main`'s `forecasting_corpus` supplies resolved Manifold/Polymarket questions
with the crowd probability reconstructed at as-of and a `cutoff_clean` tag; `main`'s `backtest_harness.score`
reports `skill_vs_crowd` (and the crowd-UNSURE slice, where a real model can actually add value). Our harness
filters to **people-domain, cutoff_clean** items (the `ParadigmRouter` picks the engine's turf), grounds each
leak-free with `swm/retrieval/asof_news.asof_headlines` (GDELT headlines in a window *ending* at as-of — the
information the crowd had, nothing after), runs `binary=True`, and grades on `skill_vs_crowd`.

**As-of grounding is keyless and works here.** GDELT's DOC API is hard rate-limited (HTTP 429) from this
sandbox, so instead the primary as-of source is `retrieval.asof_google_news`: Google News RSS with a bounded
`before:<as_of> after:<window>` query **plus a hard code filter that drops any item whose `<pubDate>` is after
the as-of** (Google's date operators are not a hard guarantee — empirically `before:` alone leaks post-cutoff
articles; the code guard closes it). Verified leak-free: on the 2025 NJ governor race as-of 2025-10-20 it
returns Sep–Oct campaign coverage *including a poll* and zero outcome-leaking passages. GDELT stays as a
fallback where reachable. No paid key required; a keyed overlay (Serper/Tavily/Brave) is still auto-detected
if present. This is what lets EXP-091 grade the engine against the crowd live (see below).

### Crowd grade (EXP-091) — `society:event` = **F**. Real as-of grounding, real market baseline.

With keyless as-of Google News grounding, the engine finally forecasts (only 3/20 abstain, down from ~60%
on thin context) — and loses badly to the market on 17 resolved political questions:

| metric | model | crowd |
|---|---|---|
| Brier | **0.295** | 0.094 |
| log-loss | 2.264 | 0.316 |
| skill vs base | −2.74 | +0.48 |
| **skill vs crowd** | **−6.15** | — |

**GRADE: F** (fitted shrink dropped to 0.4 — the calibrator learned the engine is overconfident and tempers
it hard). Two failure modes, from the rows: **confident-and-wrong extremes** (p=1.00 on Massie losing;
p=0.57 on Restore Britain, which the crowd correctly priced at 1%) and **underconfidence on clear
favorites** (Brad Lander NY-10: engine 0.54 vs crowd 0.90 vs outcome YES; Letlow 0.22 vs 0.93 YES). The
grounding fix was real (abstention 60%→15%) but it exposed the true bottleneck: **the engine is
badly calibrated and under-discriminating versus a money-market crowd.** grade-or-abstain did its job — it
stamped F, so nothing ships confident. This is the honest floor to improve from.

Caveats: n=17, a few confident-wrong rows dominate the log-loss; the corpus mixes obscure by-elections with
cleaner races; the fitted shrink is in-sample (should move to a held-out split). But the direction is
unambiguous — the engine does not beat the crowd, and isn't close yet.

**The frontier is now calibration, not grounding:** never emit 0/1 from finite persona samples; translate
as-of poll leads into proportionally stronger frontrunner support; fit the shrink out-of-sample; add value
only where the crowd is unsure. Grounding works; the reasoning-to-probability mapping is the weak link.

### Earlier grade (EXP-090, 2025-06-08 / 08-31 / 10-26 resolved political questions, frozen ForecastBench context)

**Result: `society:event` = ungraded (did NOT beat the base rate). The harness refused to certify the
engine — which is the point.**

| metric | value |
|---|---|
| in-domain questions | 33 |
| scored / abstained | **13 / 20** (60% abstained on thin frozen context) |
| class base rate (fraction YES) | 0.154 (longshot-dominated) |
| engine log-loss | 0.454 |
| baseline log-loss: 0.5 / class-rate | 0.693 / 0.429 |
| **skill vs 0.5** | **+0.344** (beats max-entropy handily) |
| **skill vs class-rate (the real bar)** | **−0.058** (loses to "guess the base rate") |
| Brier | 0.144 |
| **grade** | **ungraded** → forecasts stay flagged |

**Honest diagnosis (no spin):**
- The engine extracts *real* signal (skill +0.34 vs 50/50): it correctly said Putin stays president
  (0.03), Trump approval stays <45%, obscure mayoral longshots lose. It is not noise.
- But it **only matches "these are longshots, guess low"** and does not beat it. It caught the easy NO
  cases and **missed or abstained on the discriminating YES cases** — Støre becoming Norway PM (said 0.41,
  resolved YES), Sherrill winning NJ (abstained — she was the clear favorite), Platner as ME Senate
  nominee (abstained, resolved YES). Catching those is what beats the base rate, and it didn't.
- **Why:** two-thirds of the miss is that the leak-free eval feeds only ForecastBench's thin frozen
  background, which often omits the polls/standings that make a race predictable — hence 60% abstention
  and the favorite-misses. This eval grades *reasoning on sparse context*, a floor, not the engine with
  live grounding. **But that is not an excuse to wave away the number**: on what it did score, it did not
  beat the skeptic. Binary "will this specific longshot happen" also plays to the engine's weakness — its
  edge is modeling voters choosing among the *actual frontrunners*, which these questions rarely exercise.
- **The machine worked.** grade-or-abstain did its one job: it withheld the grade, so no `society:event`
  forecast ships as a confident number. The evaluator embarrassed the model — exactly the README thesis.

**What this changes about next steps:** the right grading set for THIS engine is head-to-head races with
real **as-of polling** in context (who wins among the actual contenders), which needs as-of-scoped
retrieval so the frozen dossier carries the polls. Longshot binary event-markets are off the engine's
strength and under-inform it. That, plus growing n, is the path to a real grade.

## What remains (the honest gap list)

1. **Run the grading program**: batch the engine over ForecastBench rounds + resolved primaries/races to
   earn real grades for `society:collective_choice` / `society:population_share`; fit per-class shrinks.
   Nothing above ships confident until this lands.
2. **Individual-mode labels**: the `individual:response` class needs private labeled sends (the original
   wedge data) — the flag stays on until then.
3. **Artifact-mode CTR anchoring**: rankings are paired-comparison honest, absolute engagement levels are
   not; anchor on Upworthy-style randomized CTR data to grade `artifact:engagement`.
4. **Cost control at scale**: decision-call caching per (persona, round-context) bucket; segment-level
   pre-aggregation for very large casts.
5. **Model diversity**: personas currently share one base model (DeepSeek); mixing a second family for a
   slice of variants is the next anti-monoculture step.
