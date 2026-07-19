# The true architecture for a general social world model

*A first-principles answer to: is our approach right, how far are we, and how do we build the real vision?*

> **Core architecture (default-on for every V2 simulation):** explicit dynamically expandable
> world boundaries per structural model; a residual outside-world event process; mechanically
> bounded, heterogeneous LLM cognition (attention → finite working memory → imperfect persistent
> memory → interpretation → limited action search → one choice); a hybrid runtime where LLM
> actors and real institutional/population/numerical/operational/physical mechanisms share one
> typed world, clock and `StateDelta` contract; and an absolute prohibition on silently
> replacing an LLM actor with a different numerical psychology when compute is exhausted — the
> branch truncates and is reported. Results classify honestly: `completed`,
> `completed_with_degradation`, `under_modeled` (+subtypes), `truncated`,
> `clarification_required`, `execution_failed`. See **docs/WMV2_CORE_ARCHITECTURE.md**.

---

## 1. Your thesis, stated precisely — and where it is right vs wrong

> If we model a system with the **same variables and calibrations as the real world at the same
> complexity**, roll the state forward at **real-world timescales**, we get real-world-accurate predictions.

This is the **digital-twin thesis**. It is the correct instinct, and it is exactly how the most successful
predictive systems in existence work (weather, climate, epidemics, orbital mechanics). But three laws of
prediction constrain it, and they *dictate the architecture*. Getting them wrong is why we got NBA wrong.

### Law 1 — It's the STRUCTURE, not the variable count.
Accuracy comes from matching the real **generative process** (the causal structure), not from piling on
variables. This is the big one, and it is the opposite of "overbuild the variable list":

- Every inferred variable is an **estimate with error**. Over a multi-step rollout those errors **compound
  multiplicatively**. Add 50 noisy variables and you don't get a richer truth — you get 50 error sources
  interacting. We *measured* this: adding bandwagon coupling to the GSS model made it **worse** (skill
  −0.15), because the extra mechanism was wrong for that process.
- The NBA miss was **not** too few variables. It was the **wrong mechanism**: we ran a *deliberation* on a
  *competition*. The fix (EXP-063) was not more variables — it was the **right structure**: a playoff
  bracket of best-of-7 games. Same information, right generative process → a sane answer.

**Consequence for the architecture:** the hardest, highest-value step is **discovering the right causal
structure per question** — not maintaining a giant universal variable list. Model the structure *richly and
correctly*; estimate each variable *humbly* (with uncertainty); integrate the uncertainty out. Overbuild
the **structure**, not the **point estimates**.

### Law 2 — Time must be calibrated, and it can be.
"Roll forward one day" is only meaningful if each variable moves by **one day's worth**. This is a real,
concrete, and — crucially — *checkable* requirement, and your question "is this calibrated properly?" was
sharp: **the old rollouts were NOT.** "6 rounds" of an agent society is not tied to any real duration; the
coefficients (openness 0.35, k_social 0.1) were hand-set, not calibrated to how fast the variable actually
moves.

The fix is to make dynamics a proper **diffusion**: drift scales with `dt`, stochastic change scales with
`√dt` (Wiener scaling), and **volatility is a per-unit-time quantity calibrated against real data.** EXP-063
does this and *verifies* it: using the **real** per-year volatility of public opinion measured from GSS
(σ ≈ 0.031/yr), the model's 80% forward interval covers the realized future **85%** of the time (nominal
80% — calibrated). A clock 2× too fast covers 100%; 2× too slow covers 53%. **Miscalibrated time is now a
visible, fixable defect** — exactly what you asked for.

### Law 3 — Irreducible uncertainty is the point, not a nuisance.
Social systems are **chaotic** (sensitive dependence — tiny errors explode; this is why perfect physics
still caps weather at ~2 weeks) and **partly random**. Past the predictability horizon, **more fidelity
cannot help.** So a high-fidelity simulation does **not** yield a confident point — it yields a
**distribution**, and the honest model separates:

- the **reducible** (epistemic) spread — better estimates *would* shrink it, and
- the **irreducible** (aleatoric) spread — the **forecastability ceiling** no model can beat.

EXP-063's NBA result is the clean illustration: **even with team strengths known *exactly*, the favorite
wins only 42%.** The other 58% is irreducible playoff variance. The old composite's "52%" wasn't more
precise — it was **overconfident about a genuinely uncertain event.** A world model that returns "37%, and
here's why most of the uncertainty is irreducible" is *more correct* than one that returns a confident
number, even though it looks less impressive.

**The synthesis of your vision and these laws:** *Richness in structure, humility in estimates, Monte-Carlo
over the irreducible. Model the whole relevant slice of the world's causal machinery, admit what you can't
pin down, and roll it forward at calibrated time as an ensemble.* That is the digital-twin thesis, made real.

---

## 2. The target architecture: a world-model **compiler**

The mistake so far was building **one mechanism** (an agent society) and forcing every question through it.
The real thing is a **compiler**: for each question it *builds the right model*, then simulates it. Five
stages:

```
QUESTION
   │
   ▼
① RETRIEVE the relevant slice of the world        (real evidence — we have this: swm/api/retrieval.py)
   │
   ▼
② COMPILE a STRUCTURAL CAUSAL MODEL               (the missing core: LLM proposes the generative process)
   │   • entities & variables that drive THIS outcome (rich, inferred — your bet)
   │   • the mechanism: bracket? election? negotiation? diffusion? market? deliberation?
   │   • structural equations (how variables push each other) + noise
   │   • per-variable CURRENT VALUE + EST-UNCERTAINTY + TIMESCALE/VOLATILITY
   │
   ▼
③ CALIBRATE the dynamics                          (rates from data/priors — diffusion, √dt; EXP-063 engine)
   │
   ▼
④ MONTE-CARLO the ensemble forward at real time   (the engine: swm/simulation/structural.py)
   │   → outcome DISTRIBUTION
   │
   ▼
⑤ DECOMPOSE & REPORT                              (reducible vs irreducible; honest horizon; best action)
```

- **Stage ② is the heart and the hard part.** It is where "map the entire relevant slice of the world"
  actually happens — and where the LLM + retrieval is uniquely powerful: it has read how NBA playoffs,
  elections, negotiations, product launches, and revolutions actually work, so it can *propose the
  generative structure* and the variables that structure is sensitive to. Your bet — that *inferred*
  structure and variables are good enough — is most plausible **here**, at the structural level, and it is
  the bet worth pressing hard.
- **The mechanism is chosen per question, not fixed.** A championship compiles to a bracket; an election to
  an electorate + turnout + aggregation (Level 3); a committee vote to a deliberation (Level 2); a personal
  reply to a single-agent dynamical system (Level 1); an adoption question to a diffusion S-curve. The
  Levels we built are **not competing designs — they are members of the compiler's mechanism library.**
- **The engine (④) is done and general** (`swm/simulation/structural.py`): `montecarlo` runs any stochastic
  `simulate_once`; `StructuralModel` is the calibrated-time diffusion SCM; `variance_decomposition` gives
  the reducible/irreducible split.

---

## 2b. Structural-model uncertainty is DEFAULT-ON (the ensemble compiler)

Law 1 says accuracy lives in matching the causal **structure** — which makes *uncertainty about the
structure itself* the first-class uncertainty. Compiling one causally-sufficient schema and simulating
uncertainty inside it is not enough: a perfectly executed simulation of the wrong causal model is still
wrong. The canonical runtime therefore no longer begins with one `compile_world(...)`; it begins with a
**structural ensemble** (`swm/world_model_v2/ensemble_compiler.py` + `structural_runtime.py`):

- several **independent** actual LLM generation calls (normal target four, adaptive up to a soft
  ceiling) each propose a materially different causal model — separate calls, blind to each other,
  through general causal perspectives (actors, institutions, constraints, information, exogenous
  systems, adversarial alternative);
- adversarial critics hunt for missing decisive actors/institutions/constraints/mechanisms and can
  spawn expansion candidates; invalid or evidence-contradicted candidates are rejected with cited
  claims; equivalent candidates merge conservatively (deterministic structural comparison first, a
  blind LLM judge only for near-matches);
- every surviving model compiles into its **own executable plan** against ONE shared immutable as-of
  evidence bundle, receives its **own posterior**, its own event-time conversion and Phase-11 lineage;
- every plausible model gets a **real pilot** through the full canonical funnel (reduced particle count
  only), and every promoted model then receives **at least the complete single-model particle budget**,
  with pilot particles reused as a deterministic prefix — budgets are never divided across models;
- results report per-model distributions, a labeled equal-weight compatibility mixture (no LLM-minted
  model probabilities, ever), robust ranges, a structural-sensitivity classification, the assumption
  that would reverse the answer, and the observation that would distinguish the surviving models;
- Phase 13 evaluates every action across the surviving models (winner-by-model, minimax regret,
  conditional strategies when models disagree); personal-reaction questions run several causal frames
  of the reaction through the same machinery.

Single-model compilation survives only as the explicit
`execution_policy={"structural_mode": "single_structural_model"}` ablation (plus frozen-artifact
compatibility and isolated compiler tests); enforcement tests fail if a production route reaches the
single-plan compiler any other way. Full contract, budgets, cost controls and known limitations:
`docs/WMV2_STRUCTURAL_ENSEMBLE.md`.

## 3. How far are we — honestly

| Stage | Status |
|---|---|
| ① Retrieve | **Built** (EXP-058). Works for arbitrary questions. |
| ② Compile structural model | **The gap.** We have a variable *schema* and mechanism *pieces* (Levels 1–3, bracket), but no LLM step that, per question, **selects the mechanism and emits the structural equations**. Today a human picks the mechanism. |
| ③ Calibrate time | **Newly real** (EXP-063): diffusion + √dt, volatility calibrated from data, coverage-verified. Needs to be wired as the default for every variable, with a volatility-estimation library. |
| ④ Monte-Carlo engine | **Built** (EXP-063). General, mechanism-agnostic, fast. |
| ⑤ Decompose / horizon | **Built** (EXP-063 + the earlier forecastability work EXP-038). Needs wiring into the front door. |

**So: the runtime is now in place; the missing keystone is Stage ② — the compiler that turns a question
into the right structural model.** That is the single highest-value thing to build next, and everything
else (the Levels, retrieval, the engine, calibration, decomposition) becomes its library.

We are **not** close to a general model that predicts *everything* well — and by Law 3 no one ever will be,
because much of the social world is past its predictability horizon. But we are close to something real and
honest: **a system that builds the right model per question, runs it at calibrated time, and reports a
distribution with a truthful reducible/irreducible split** — which on the *forecastable* questions will be
genuinely accurate, and on the unforecastable ones will correctly say so instead of bluffing.

---

## 4. The build plan (no more incremental metric-chasing)

1. **Stage ② — the structural-model compiler.** An LLM call: `question + retrieved context → {mechanism,
   variables[value, est_sd, volatility, timescale], structural_equations, outcome_reader}`. Start with a
   **mechanism library** (bracket / tournament, electorate, committee, single-agent, diffusion, market) the
   LLM *selects and parameterizes*; grow toward the LLM *emitting* equations for novel structures. Pluggable
   backend (Anthropic API in prod, cached in dev) — same pattern as everything else.
2. **Volatility/rate library.** Per-variable per-unit-time volatility from data where we have it (opinion,
   polls, prices, approval), from LLM priors where we don't — so Stage ③ is automatic and calibrated.
3. **Reflexivity & events.** Let forecasts feed back (a predicted blowout depresses turnout) and let the
   retrieved timeline inject dated exogenous shocks over the rollout.
4. **The honest front door.** `simulate(question) → distribution + reducible/irreducible + predictability
   horizon + best action` — one call, the compiler underneath.
5. **Score on a forecastable benchmark.** Validate specifically on questions *inside* the predictability
   horizon (short-horizon, structure-dominated) where the thesis says we should win — not on coin-flips,
   where being right is impossible and being *calibrated* is the only honest target.

The engine, calibration proof, and mechanism examples exist today (EXP-063). The next build is Stage ② —
the compiler — which is where your vision of "map the whole relevant slice of the world and run it forward"
actually gets realized in one call.
