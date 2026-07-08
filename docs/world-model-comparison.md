# What Aaru, Simile, and the generative-agent papers do — and what we learn from them

*A capability-by-capability comparison of our social world model against the two commercial leaders
(**Aaru**, **Simile**) and the two founding research papers (**Generative Agents**, Park et al. 2023;
**Social Simulacra**, Park et al. 2022), read against our own north star: a **general** social world model
a consumer can ask (a) **prediction** questions and (b) **best-action** questions, at both **aggregate/
population scale** and **single-modeled-individual** scale, answered by running the **highest-fidelity
simulation of the relevant slice of the world** and returning the actual best outcome/prediction.*

This document is written to the same standard as the rest of the repo: claims are grounded in the actual
papers/press and in our actual modules and EXP results, and where a competitor's headline is thinner than
it looks, that is said plainly.

---

## 0. The one-paragraph verdict

The four reference systems are, at their core, **one idea executed at four points on a
believability↔calibration spectrum**: an LLM (or LLM-agent society) grounded on real-world data generates
plausible human behavior, and you read an outcome off the crowd of generated agents. **Social Simulacra**
and **Generative Agents** are the honest research end — they explicitly *disclaim prediction* and are
validated on **believability**, not accuracy. **Aaru** and **Simile** are the commercial end — they *claim
prediction* and validate on **a few high-profile correlations** (Aaru: 90% median correlation to one EY
survey; Simile: agents reproduce a person's own survey answers at ~85% of the person's own
test–retest consistency). **None of the four foregrounds the two disciplines our entire repo is built
around: (1) calibrated, no-cheat, time-forward backtesting with a reducible/irreducible split, and (2) a
first-class best-action `do`-layer scored on interventional metrics (policy-regret / CATE-sign).** That gap
is simultaneously our moat and the source of the concrete things we should steal from them. What we should
steal: Simile's **interview-grounded individual agents** and the Generative-Agents **memory-stream +
reflection** substrate (both directly upgrade our *single-individual* regime), Aaru's **"recreate an
expensive survey and beat the stated-intent baseline on realized behavior"** as our killer calibrated demo,
and Social Simulacra's **Multiverse/WhatIf UX** as the front-end for our navigable pivotal-branch object.

---

## 1. What each reference system FULLY does

### 1.1 Generative Agents — Park, O'Brien, Cai, Morris, Liang, Bernstein (UIST 2023; arXiv:2304.03442)

**What it is.** 25 LLM-driven agents in a Sims-like sandbox ("Smallville"). Each agent is an LLM wrapped in
a cognitive architecture that produces **believable individual + emergent social behavior** over ~2 simulated
days.

**The architecture (the part worth stealing):**
- **Memory stream** — a complete, timestamped, natural-language log of everything the agent perceived/did.
- **Retrieval** — to act, the agent pulls memories by a weighted score: **recency** (exponential decay),
  **importance** (an LLM-rated 1–10 "poignancy" of the memory), and **relevance** (embedding cosine
  similarity to the current situation). The three are normalized and summed.
- **Reflection** — periodically (when accumulated importance crosses a threshold) the agent asks itself
  high-level questions about its recent memories, retrieves the relevant ones, and **synthesizes abstract
  insights** ("Klaus is passionate about gentrification research"), which are written back as higher-level
  memories. A reflection *tree*.
- **Planning + reacting** — recursive top-down plans (day → hour → 5–15-min chunks), stored as memory, and
  **re-planned** when new observations warrant.

**What emerged (their headline results):** from a *single* seed ("Isabella wants to throw a Valentine's
party"), with no further scripting: **information diffused** (awareness of Sam's mayoral candidacy grew from
1→8 of 25 agents; awareness of the party 1→12 of 25), **relationships formed** (agent-network density rose
~0.167→0.74 over two days), and agents **coordinated** (5 of the 12 invited actually showed up at the right
time and place). These are genuinely emergent — a mean of independent agents cannot produce them.

**How it was evaluated:** **believability**, via a controlled study — agents were "interviewed" across
self-knowledge, memory, planning, reactions, and reflections, and rated by humans (TrueSkill ranking).
**Ablations** (remove reflection; remove reflection+planning; remove memory) each degraded believability →
every component earns its place. **It was NOT evaluated on predicting any real person or real outcome.**

**Stated limits:** memory-retrieval failures, hallucinated embellishment, instruction-tuned over-politeness,
cost/scale, and — critically — **believable ≠ accurate**; no claim that any agent tracks a real individual.

### 1.2 Social Simulacra — Park, Popowski, Cai, Morris, Liang, Bernstein (UIST 2022; arXiv:2208.04024)

**What it is.** A **design-time** tool ("SimReddit") that populates a *not-yet-launched* online community
from the designer's inputs — **goal, rules, seed personas** — and generates thousands of synthetic users
plus their posts/replies, **including anti-social behavior** (trolls, hustlers, harassment), so the designer
can see failure modes before real users arrive.

**Three features (each a product idea):**
- **Generate** — expand ~10 seed personas → ~1,000 personas, then generate posts and threaded replies via
  GPT-3 prompt chains; rules act as *nudges*, not hard constraints (some agents break them, as in reality).
- **WhatIf** — the designer injects a persona or a moderator action ("what if a troll replies?", "what if a
  mod says X?") and sees how the thread re-develops → **intervention roleplay**.
- **Multiverse** — *re-generate* the same design many times to surface the **breadth of possible outcomes**,
  explicitly refusing a single point prediction.

**How it was evaluated:** **believability** — repopulate 50 real subreddits created *after* GPT-3's cutoff
(leakage-controlled by construction) and ask people to tell the real thread from the generated one.
Participants were near chance (**41% error**, 50% = random). Plus a study of 16 designers who found unforeseen
positive and negative behaviors and iterated their rules.

**Stated purpose and limit (verbatim intent):** *"Social simulacra do not aim to predict what is absolutely
going to happen… perfect resemblance to reality is not the goal."* It is a **prototyping / coverage** tool,
not a forecaster.

### 1.3 Simile — the commercial descendant of both papers (2026)

**Who.** Founded by the actual authors — **Joon Sung Park, Michael Bernstein, Percy Liang**. Raised a
**~$100M Series A (Index Ventures)**. This is Generative Agents → a company.

**What they claim to do (from their own material):**
- **Individual agents as the "quantum unit."** For CVS Health: agents built on **2.9M consented responses
  from 400k+ participants across 200+ behavioral scenarios**, modeled on **real people's interview responses
  and past choices**, used as "safe, accurate stand-ins."
- **A staged capability ladder** — *individuals → journeys over time → interactions → whole markets*:
  static → dynamic → multi-agent. The stated frontier is **multi-agent market simulation** (customers ×
  competitors × partners × policies) with **second-order effects stress-tested before capital is committed.**
- **The research bet (Percy Liang's "age of simulation" post):** high-fidelity models of people +
  environments; **large-scale/multi-scale** simulation (macro + micro; "8 billion people over a year");
  **calibrated probability estimates**; and — notably aligned with us — simulation as a **causal model**
  answering Pearl's ladder: prediction → **intervention (`do`)** → **counterfactual**, with the simulation
  serving as an **interpretable, auditable trace**.
- **Validation shown:** the Generative-Agent-Simulations-of-1,000-People result — agents from **~2-hour
  interviews** reproduce a participant's **GSS answers at ~85% of the participant's *own* two-week
  test–retest rate** (a *relative self-consistency* benchmark). Business claims for CVS are qualitative
  ("faster validation," "sharper NPS drivers," "adherence under real constraints," "competitive
  positioning"), positioned as **pre-screening before costly pilots**.

**The honest read.** Simile has (a) the strongest *individual*-fidelity method in existence
(interview-grounded agents) and (b) the most ambitious *scale/multi-agent* roadmap. Its public validation is
a **self-consistency** number (reproduces *stated survey* answers), not calibrated accuracy on *realized
behavior over time*, and the business wins are framed as insight/pre-screening.

### 1.4 Aaru — via the EY Global Wealth simulation (2025)

**What they claim to do.** "AI simulation" of real-world **populations**: build **synthetic populations** of
data-driven agents from **census + financial + social-media/behavioral data**, give them behavioral
architectures, and have them answer strategic questions at scale ("100,000 personas in hours"). Used by
Interpublic (ad-audience response), Heartland Forward (AI sentiment across 20 states), and EY.

**The flagship validation (EY).** Recreated EY's 2025 Global Wealth survey (3,600 investors, 30+ markets,
normally 6 months) **in one day**: across **53 single-select questions**, **median Spearman correlation
0.90**, RMSE 7.1pp, Euclidean 12.9. The marketing thesis is the **say–do gap**: where the simulation
*diverged* from the survey it was allegedly *closer to real behavior* — inheritance-advisor retention
(survey 82% "will keep advisor" → sim 43% → real-world 20–30%); single-provider preference (survey 69% →
sim 37% → real 33%). Positioning: **behavior, not stated intent**; real-time re-simulation when conditions
change (election, tariff, rate move → 24h re-run); "no PII, unlimited scale, continuous vs point-in-time."

**The honest read.** The EY number is a **single point-in-time rank-correlation on one survey** (plus two
anecdotal say–do divergences retro-validated against industry stats). There is **no calibration curve across
many independent questions, no reducible/irreducible decomposition, no prospective no-cheat backtest, and one
question correlated −37.82%** (reframed as "insight"). It is a compelling *demo of survey-recreation*, not a
demonstrated *calibrated generalizing forecaster*. Aaru also owns the **positioning** we most need to answer:
"simulation beats surveys because it predicts what people *do*."

---

## 2. What OUR system does today (mapped to modules + EXPs, honestly)

Our repo is not an agent-society-and-read-off-the-crowd system. It is a **world-model compiler + calibrated
Monte-Carlo engine + a first-class evaluation harness**, with an agent society as *one mechanism in the
library*, not the whole product.

| Capability | Where | Honest status |
|---|---|---|
| **Question → structural model ("compiler," Stage ②)** | `swm/api/compiler.py`, `selecting_compiler.py`, `world_model.py` | **Built** (EXP-064/065/068). LLM selects a mechanism (bracket / committee / electorate / single_agent / generic_scm) and emits a spec; a validator+repair loop catches malformed specs live (EXP-067/068). Candidate-and-**select** across K structures with agreement reporting. |
| **Calibrated-time Monte-Carlo engine** | `swm/simulation/structural.py` | **Built** (EXP-063). Diffusion SCM, drift·dt + vol·√dt, `variance_decomposition` → **reducible (epistemic) vs irreducible (aleatoric)** split = a *forecastability ceiling*. |
| **Agent society (interaction, emergence)** | `swm/simulation/agent_society.py`, `mean_field.py`, `world/substrate.py` | **Built + honestly scored.** Interaction beats the independent composite **only in specific regimes**: SCOTUS coalition *margin* (EXP-055, MAE 0.168 vs 0.208), contagion/tipping fashion cascades (EXP-072, +42% skill at turning points). On mass GSS opinion, coupling does **NOT** beat the independent mean or persistence (EXP-053/061). The `World`/`Entity`/`Coupling` substrate (EXP-070) is real; whether to wire any two scales is an *empirical* question with a scoreboard. |
| **Grounded population readout (aggregate regime)** | `swm/variables/`, `IndependentPopulationReadout` (ex-`GroundedSimulator`) | **Built + demoted.** Bottom-up compositor: per-person VariableMap → latent-factor + LLM-prior + reliability-weighted readout (EXP-040/048/050), aggregated. Beats the top-down aggregate ~24% on distinctive subgroups. **Explicitly renamed from "simulate" to "readout"** because ∂pᵢ/∂pⱼ=0 (SIMULATION_AUDIT.md). |
| **Individual regime (person as a dynamical system)** | `swm/simulation/individual_agent.py`, `api/individual_simulate.py`, `variables/deep_inference.py` | **Built.** `IndividualAgent` = VariableMap + mutable state (mood/busyness/reciprocity) evolving as contacted. Deep per-person inference from a writing corpus (EXP-069, our scalable analog of Simile's interview) helps monotonically with history depth (~8%). Validated on real CMV persuasion. |
| **Best-action `do`-layer** | `swm/decision/`, `api/action_simulate.py` | **Built (all 7 components).** `argmax_a E[U|do(a)]` = inner Monte-Carlo × outer best-arm racing; typed parameter/structural/**temporal** interventions; mean/quantile/CVaR/constrained risk; confident-winner-or-honest-tie; **sequential policies**. Re-earned on real data: CMV best-message **precision@1 0.739 vs 0.518 = +22pt** (EXP-069). |
| **Navigable outcome object** | `swm/report/navigable.py` | **Built.** Replaces a scalar with distribution + reducible/irreducible + **automatic pivotal-branch discovery** ("37%, and here's the fork"). |
| **No-cheat, time-forward backtest harness + calibration** | `swm/eval/`, `variables/bayes_logistic.py`, `calibrated_weights.py` | **First-class.** As-of leakage guard, SKILL = 1 − loss/loss_baseline vs persistence/momentum/base-rate/market, ECE/CRPS/coverage, PostMortemLog + do-no-harm recalibration, forecastability triage (FORECAST/HEDGE/ABSTAIN). This is the spine none of the four competitors foreground. |
| **Decisive fidelity result** | EXP-073 | On GSS rolling-origin (133 no-cheat forecasts) the calibrated **11-variable forward sim beats persistence +0.107 skill and all baselines** — and **more calibrated variables flipped a loss into a win** (2 vars −0.032 → 11 vars +0.107). The digital-twin bet pays **where the population is modelable and simple baselines are weak.** |
| **Front door (question intake / auto state retrieval)** | ROADMAP stage A/B | **Partial.** Retrieval scaffolding + live_forecast log exist (EXP-058); auto-parse arbitrary NL question → proposition/resolution/horizon and auto-instantiate the population is the remaining keystone. |

**Our honest self-assessment (from SIMULATION_AUDIT.md):** the *shipped default path* was until recently
~85% compositing; the flagship is a mean of independent regressions and has been renamed accordingly. The
genuine-dynamics wins are real but **regime-specific**, and we have *measured* exactly when interaction beats
compositing rather than asserting it always does.

---

## 3. Head-to-head, on the dimensions the product vision demands

| Dimension (from the vision) | Generative Agents | Social Simulacra | Simile | Aaru | **Ours** |
|---|---|---|---|---|---|
| **Aggregate/population prediction** | Emergent, qualitative | Coverage, not prediction | Yes (markets = frontier) | **Yes — the core pitch** | Yes, **calibrated + backtested**; honest about regimes |
| **Single-individual modeling** | Believable persona, not a real person | Personas, not real people | **Strongest (interview-grounded, ~85% self-consistency)** | Population-level, not individual | Built (IndividualAgent + deep inference); **weaker elicitation data than Simile** |
| **Best-ACTION `do`-query** | No | WhatIf (qualitative) | On roadmap (Pearl ladder stated) | Implicit (scenario testing) | **Yes — built, interventionally scored (+22pt, CATE-sign, policy-regret)** |
| **Calibrated uncertainty** | No | Explicitly no ("Multiverse") | Stated goal, not shown | Not shown (1 correlation) | **Yes — ECE/CRPS/coverage, first-class** |
| **Reducible vs irreducible split** | No | No | No | No | **Yes — `variance_decomposition`; a genuine differentiator** |
| **No-cheat, time-forward backtest** | N/A (believability) | Leakage-controlled *believability* | Self-consistency, not forward | **No prospective backtest** | **Yes — the spine of the whole repo** |
| **Mechanism chosen per question** | Fixed (agent society) | Fixed (community sim) | Fixed substrate (agent society) | Fixed (population agents) | **Yes — a compiler with a mechanism library** |
| **Genuine emergence/interaction** | **Yes (their headline)** | Yes (threads) | Yes (multi-agent frontier) | Population interaction claimed | Yes **but only where it beats baselines** (SCOTUS margin, contagion) |
| **Honest "we can't call this"** | N/A | Yes (by design) | No | No | **Yes — forecastability triage / ABSTAIN** |
| **Scale** | 25 agents | ~1,000 personas | **8B ambition, 400k real** | **100k personas/hours** | Modest; engine scales linearly in K, data-bound |
| **Public validation depth** | Deep (believability) | Deep (believability) | One relative benchmark | One correlation | Dozens of no-cheat EXPs incl. honest negatives |

**The single most important row is "calibrated uncertainty / reducible-vs-irreducible / no-cheat backtest."**
On those three, we are alone. Everyone else ships **believable or high-correlation** output; we ship
**calibrated** output that **says when it can't call it.** That is the entire thesis of the audit — and it is
exactly the discipline a *consumer asking "what should I actually do"* needs, because a confident wrong
answer is worse than an honest "this is 55% ± a lot, and here's the pivot."

---

## 4. The one conceptual axis that explains everything: believability ≠ calibration

All four reference systems descend from the observation that **an LLM's training data already contains a huge
variety of human social behavior**, so you can *generate* plausible people and read an outcome off them. That
gives **believability** cheaply. Our repo's founding result (and the published critiques it cites — Bisbee
2024 "means-OK, variance-broken"; Santurkar OpinionQA; Dominguez-Olmedo option-order artifacts) is that
**believable output is systematically mis-calibrated**: LLM personas flatten within-group variance,
caricature subgroups, and are unstable to prompt wording. Aaru's own −37.82% question and Simile's
*self-consistency* (not accuracy) framing are this gap surfacing.

Two of the four (the *papers*) are honest about it and **disclaim prediction**. Two of the four (the
*companies*) **claim prediction** on thin validation. **Our whole architecture is the machine that closes
that gap** — the calibrated weights, the shrinkage that makes "more variables" safe (EXP-041/048/072), the
reducible/irreducible split, the no-cheat harness. This is not a footnote; **it is the product.**

A corollary that directly rebuts a naive reading of our own vision ("simulate all the variables"): **fidelity
is structure + calibrated estimation, not variable count.** We *measured* that adding 12 variables *hurts*
naively (+0.111 log-loss) and only helps once properly shrunk (EXP-072); that bandwagon coupling made GSS
*worse* (−0.15, EXP-061); that the NBA miss was a *wrong-mechanism* bug, not too-few-variables (EXP-063).
None of the four competitors has grappled with this publicly. "Highest-fidelity simulation of all the
variables" is right **only** if "fidelity" means *the right causal structure with humbly-estimated,
uncertainty-carrying variables* — which is precisely what ARCHITECTURE_WORLDMODEL.md Law 1 already encodes.

---

## 5. What we should learn / steal — prioritized, tied to modules and to the two regimes

### Lesson 1 — For the INDIVIDUAL regime: buy fidelity with *elicitation data*, the way Simile does (highest ROI)
Simile's decisive advantage is **not** a cleverer model; it is **~2-hour interviews** (and CVS's 2.9M
consented responses) that give each agent rich first-person grounding, yielding ~85% self-consistency. Our
analog — deep inference over a writing corpus (EXP-069) — helps only ~8% because *a scraped doc is a noisy
trait realization*, exactly as they note their own ceiling is ~85%.
- **Action:** build an **elicitation surface** for the single-person product — a structured onboarding
  interview / questionnaire (or "ingest your own corpus: emails, chats, past decisions") that populates the
  `VariableMap` + `DeepPersonaStore` from *first-person* data, not just behavioral traces. This is the
  data-acquisition flywheel for the individual regime and the highest-leverage fidelity gain available.
- **But hold it to our bar, not theirs:** score it on **realized held-out behavior over time**, not
  self-consistency. That single distinction (accuracy-over-time vs reproduce-your-own-survey) is a
  defensible public claim Simile has not made.

### Lesson 2 — For the INDIVIDUAL regime: adopt the memory-stream + reflection substrate (Generative Agents)
Our `IndividualAgent` has a mutable scalar-ish state; Generative Agents' **memory stream (recency ×
importance × relevance retrieval) + reflection tree** is the SOTA believability substrate for a *single*
modeled person and maps cleanly onto our C.4 "memory" design (episodic log + semantic summaries), which is
currently a stub.
- **Action:** implement `swm/memory/` for real (retrieval-scored episodic stream + periodic reflection that
  writes back synthesized traits into the `VariableMap`), and wire it into `IndividualAgent.response_fn`.
  Validate that reflections *improve calibrated* next-behavior prediction (not just believability) — the
  ablation Generative Agents ran, re-scored on our metric.

### Lesson 3 — For the AGGREGATE regime: Aaru's "recreate an expensive survey, beat the stated-intent baseline" IS our killer calibrated demo
Aaru's EY story is marketing-thin on rigor but **strategically exactly right**: the most legible proof of a
social world model is *reproduce a study that costs \$M and 6 months, in a day, and then show where you're
closer to real behavior than the survey was.* We already have the harness Aaru lacks.
- **Action:** run a **survey-recreation benchmark** with the honest scoreboard we already own — recreate a
  public post-cutoff survey wave (GSS'24 / ANES'24 / a Pew wave), report **calibration + skill vs the
  stated-intent baseline on a realized-behavior holdout**, *with* the reducible/irreducible split and an
  ECE curve across *all* questions (not one correlation). This directly out-rigors Aaru's 0.90 headline and
  is immediately marketable. The say–do gap is a *feature of our calibration*, not a slogan.

### Lesson 4 — For BOTH regimes: Social Simulacra's Multiverse/WhatIf is the front-end for our navigable object
Social Simulacra validated (with users) that people reason better when shown **a breadth of possible
outcomes (Multiverse)** and can **inject interventions (WhatIf)** — not a single number. We *have* the
calibrated engine (`navigable.py` pivotal branches; the `do`-layer) but not their UX.
- **Action:** present the navigable outcome as a **Multiverse with probabilities on the branches** ("in 43%
  of worlds X, here's why; in 57% Y") and expose **WhatIf** as the consumer-facing skin of `action.py`
  typed interventions. Their contribution is the *interaction model for uncertainty*; ours is the *calibrated
  content* to put inside it.

### Lesson 5 — For NOVEL actions with no backtest: adopt Social Simulacra's honesty as an explicit product mode
Predicting response to a *genuinely novel* action/community with no historical analog is our open research
problem #5 and is *exactly* what Social Simulacra is for — and it **refuses to call it prediction.** Our
audit already draws the **Insight vs Prediction** line (E.5, the API `report_type` tag).
- **Action:** make that line a **first-class product mode.** When the compiler + forecastability triage
  determine there's no backtestable analog, return a Social-Simulacra-style **qualitative coverage of
  possible responses tagged `insight`**, never a calibrated number. This is both more honest than
  Aaru/Simile and a genuine capability (coverage) rather than a bluff.

### Lesson 6 — For scale: Simile's multi-scale/multi-agent-market frontier is the right long-horizon bet — pursued our way
Simile's "individuals → journeys → interactions → markets, macro+micro, second-order effects" ladder is the
real generalization path, and it is **our `World`/`Entity`/`Coupling` substrate (EXP-070) verbatim** — we
just insist on a scoreboard for *when* the coupling earns its place (which we've started: SCOTUS null, FOMC
inertia-dominated, contagion wins). 
- **Action:** keep building the substrate **regime-detection-first** — the deliverable is not "8B agents"
  but a **learned map of *when* cross-scale coupling beats separable baselines** (the EXP-070/071/072
  through-line). That map is a research asset none of the competitors is publishing, and it's what makes a
  *general* simulator trustworthy instead of universally-coupled and universally-overconfident.

### Lesson 7 (defensive) — do not chase their headline numbers; own the honest scoreboard
Aaru's 0.90 and Simile's 0.85 are **the wrong target** — one is a single rank-correlation, the other is
self-consistency. If we optimize for "match a survey" we rebuild a compositor (which our audit already warns
recovers a marginal it was fit on). Our differentiated, defensible claims are: **calibrated across many
questions, skill vs persistence at each horizon, interventional lift on real `do(x)` data, and an explicit
"here's what we can't call."** That is the lane both companies leave open (README thesis #1; audit A).

---

## 6. Where we are already ahead (the moat, stated plainly)

1. **The `do`-layer.** A market or an LLM answers *what will happen*; we answer *what should I do and what
   happens if I do it* — `argmax_a E[U|do(a)]`, built and interventionally scored (+22pt on real CMV,
   Upworthy randomized A/B, CATE-sign, policy-regret). Simile *states* the Pearl ladder as an aspiration;
   we have the interventional scoreboard running. **This is the "best action for a desired outcome" half of
   the product vision, and it is our strongest lead.**
2. **Calibration + reducible/irreducible + forecastability triage.** We return *distributions that know their
   own limits* and *abstain* when a question is past its predictability horizon. None of the four does.
3. **No-cheat, time-forward evaluation as the product's spine**, including logged honest negatives — the exact
   discipline the published critiques say the whole category lacks (Larooij & Törnberg 2025: "validation is
   the central challenge").
4. **A compiler, not a single mechanism.** The four competitors force every question through one substrate
   (an agent society / a population). We *select the generative structure per question* — a bracket for a
   championship, an electorate for a referendum, a single-agent dynamical system for a personal reply — which
   is what a *general* world model (aggregate *and* individual, in one front door) actually requires.
5. **We measured the fidelity paradox.** "Model all the variables" is only right as "model the right
   structure with humbly-estimated variables"; we have the experiments (EXP-063/072/041) proving blind
   variable-count *raises* error. That is hard-won knowledge that keeps a high-fidelity simulator from
   becoming a confidently-wrong one.

---

## 7. The synthesis for the product vision

The vision — *a consumer asks a prediction or best-action question; we run the highest-fidelity simulation of
the relevant world and return the actual best outcome* — is, precisely stated, **the world-model compiler +
the calibrated engine + the `do`-layer**, which we largely have. The four reference systems tell us **where
to spend next**, mapped to the two regimes:

- **Individual regime** (single modeled person, best-action for *me*): steal Simile's **elicitation-grounded
  fidelity** (Lesson 1) and Generative-Agents' **memory-stream + reflection** (Lesson 2), held to
  *behavior-over-time* accuracy, wired into the already-built `IndividualAgent` + `do`-layer. This is where
  Simile is ahead and where a focused elicitation surface closes the gap fastest.
- **Aggregate regime** (population-scale prediction): our **survey-recreation-with-honest-calibration**
  benchmark (Lesson 3) is the demo that beats Aaru on its own turf, and our **coupling-regime map** (Lesson
  6) is the trustworthy path to scale that Simile's "8B agents" hand-waves.
- **Both regimes**, front-end: dress the calibrated navigable object in **Multiverse/WhatIf** (Lessons 4–5),
  with an explicit **insight-vs-prediction** mode so we are the one system that neither over-claims (Aaru/
  Simile) nor refuses to predict (the papers) — we predict when we can, calibrated, and say so when we can't.

The competitors have the **believability, the scale ambition, and the go-to-market story.** We have the
**calibration, the interventional `do`-layer, and the honesty.** The learnable moves above import their
believability substrate and their killer demos **into our calibrated frame** — which is the only combination
that makes the general "ask anything, simulate the real world, get the actual best action" vision *true*
rather than merely *believable*.
