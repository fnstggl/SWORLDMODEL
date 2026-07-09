# The Grounded-Agent Engine — one mechanism, the vision clause by clause

**The vision**: *Take an arbitrary natural-language question → automatically construct the belief state →
map every variable acting on it → roll the simulation forward under uncertainty → return a calibrated
distribution over outcomes — and the best action to reach a desired outcome.*

**The engine** (`swm/engine/`): there is ONE mechanism — **grounded agents interacting** — and the only
thing that varies per question is the *casting* (who the agents are, what the answer space is, what the
interaction structure is). The mechanism zoo (bracket/committee/electorate/`calibrated_readout`) is dead
as a front door; `general_world_model()` now returns this engine, and the legacy structural-compiler path
survives only as `engine="legacy"` for reproducing old experiments.

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

### First grade (EXP-090, 2025-06-08/08-31/10-26 resolved political questions, leak-free)

_Filled in by the run — see `experiments/results/exp090_grade_agent_engine.json` and the PR comment._
The engine **abstains** on genuinely thin-context questions (honest, ~⅓ of the set) and scores the rest;
the recorded grade for `society:event` gates whether those forecasts ship confident or flagged.

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
