# Large-scale backtest verdict (EXP-089/090) — history as the test set

Run over **660 resolved, cutoff-clean** binary forecasting questions from Manifold (post-training-cutoff, so
the model cannot have memorized the outcomes), each with the **crowd probability at a fair as-of lead** as the
baseline. The world-model forecaster: compile the question as-of → simulate → P(YES) (the LLM states variables
+ a mechanism; the *simulation* produces the probability — the decomposition defense against leakage).

## UPDATE — two distinct problems, one was a bug

The first headline ("AUC 0.50, log-loss 1.41") was **two problems stacked**:

1. **A measurement bug (now fixed).** The harness read P(YES) as `p_event = P(readout > 0.5)`, which
   BINARIZES a `calibrated_readout`'s probability (0.7→1.0, 0.5→0.0). This *manufactured* the extremes and
   caused the coin-flip 0.02 (the readout correctly computes 0.5, but "> 0.5" is false at exactly 0.5 → 0 →
   clipped to 0.02). Fixing it (use the readout MEAN) **halved overall log-loss 1.41 → 0.84**, dropped extreme
   predictions **58% → 10%**, and coin-flips now return **0.5**.
2. **The real, remaining problem: zero discrimination.** After the fix, **AUC is still 0.509** — the model
   cannot rank YES above NO better than chance. The raw LLM alone discriminates weakly-but-really (0.559); the
   readout *washes that out* to 0.51. The crowd is 0.789. Calibration can't fix this (it has no signal to
   calibrate), so every ablation config, temperature-scaled, only reaches ≈ base-rate and none beats the crowd.

Root cause of (2): for open-domain questions the LLM **invents** variables and elasticities with **zero
declared uncertainty** (`weight_sd=0, est_sd=0`), so the "simulation" is a static logistic over confabulated
features taken as certain — no latent state, no transitions, no outside-view anchor, no honest ignorance. See
the diagnosis at the bottom.

## The result (pre-fix): the simulation architecture is noise on open-domain questions

| Forecaster | log-loss | AUC (discrimination) | skill vs crowd |
|---|---|---|---|
| **compile→calibrated-readout→simulate** | **1.41** | **0.503 (≈ random)** | **−1.58** |
| direct LLM (leakage meter) | ~0.93 | 0.559 | −0.57 |
| **crowd (baseline)** | **0.55** | **0.789** | — |
| base rate | 0.69 | 0.5 | — |

- **AUC 0.503** — the simulation has essentially zero ability to rank YES above NO. It is statistically
  indistinguishable from random, wrapped in extreme overconfidence (58% of its predictions are >0.9 or <0.1
  vs 14% for the crowd). "Daily Coinflip" → the model says 0.02.
- **The pipeline destroys signal.** The raw LLM alone discriminates weakly but really (0.559); compiling it
  into a calibrated readout and simulating drops that to 0.503. The apparatus is worse than doing nothing.
- **No leakage inflation.** The direct LLM does *not* beat the crowd (−0.57), so on this clean set we are not
  measuring memorization — the honest signal is simply absent.
- **Loses in every category** (election −3.2, culture −5.4, tech −2.5, sports −1.2, …), and even where the
  crowd is unsure (.35–.65) the model adds nothing (−1.07).

## Stage-2 ablation: nothing recovers it (held-out, temperature fit on train)

| config | skill vs crowd (calibrated) | skill vs base |
|---|---|---|
| ensemble model+crowd | −0.31 | +0.06 |
| direct LLM | −0.36 | +0.02 |
| ensemble model+direct | −0.38 | +0.01 |
| readout, top-3 vars | −0.41 | −0.01 |
| grounded readout | −0.43 | −0.03 |
| compiler-chosen mechanism | −0.44 | −0.03 |
| ungrounded readout | −0.44 | −0.03 |

- Every config needed the **minimum** temperature (0.20, the grid floor) — calibration's verdict is "ignore
  the model's confidence entirely, pull to the base rate," i.e. the model knows ~nothing.
- After maximal calibration the best configs only reach **≈ base-rate level**; none beats the crowd.
- **direct LLM > simulation** even after calibration — the simulation is a net negative.
- Grounding the values barely helps (−0.43 vs −0.44 ungrounded), and *fewer* variables is slightly better
  than more — because for open-domain questions the LLM's invented variables/weights carry no real signal.

## Why (this tests the founding thesis directly)

The decompose-into-variables-and-simulate approach only beats direct judgment where the elasticities are
**genuinely data-calibrated** (FOMC +58pt, adoption dynamics — real corpus/vintage weights). For an arbitrary
question we have **no calibrated priors**, so the LLM invents variables and weights, and a logistic over
ungrounded guesses is confidently-random noise. *"Model everything with proper calibration"* is right — and
this proves **proper calibration is the entire game**, which we only possess in a few domains.

## Mandated architectural redirect

1. **Do not simulate open-domain questions.** Route to simulation ONLY where real calibrated priors exist;
   the direct (calibrated) LLM already beats the readout everywhere else.
2. **Default to the crowd/market where one exists** — the honest system defers, and only overrides with a
   simulation that has a *measured* edge on that regime.
3. **The regime router is now empirically mandatory and much narrower than hoped** — it must be trained on
   this loss surface, not asserted.
4. **To ever beat the crowd** the lever is genuine information edge (fresh as-of grounding, real per-domain
   elasticities, ensembling many models), not more simulation machinery. The current apparatus has no edge.

The most valuable thing the project has produced: history-as-backtest caught the whole simulation edifice
being worse than useless on breadth — which no amount of component-level validation ever revealed.

---

# Flywheel turn 2 — the re-architected latent forecaster (EXP-091)

Replaced the discriminative readout with a genuine **latent-state simulation**: base-rate anchor + honest
uncertainty by construction + time-accurate transitions (horizon = real days to resolution; metric questions
diffuse a grounded value with vol·√H, event questions move latent log-odds by decaying driver shocks). The LLM
supplies the outside-view base rate, the state, and honest driver strengths — never the outcome; the
simulation produces P(YES). Re-run on the SAME 660 clean questions:

| | old readout | **latent (re-arch)** | crowd | base |
|---|---|---|---|---|
| log-loss (calibrated) | 0.836 | **0.683** | 0.547 | 0.692 |
| AUC (discrimination) | 0.509 | **0.552** | 0.789 | 0.5 |
| coin-flips | (0.02 bug) | **0.5** (3/4) | 0.5 | — |

- **The confidently-wrong disease is cured:** calibrated log-loss ≤ base rate (0.68), no longer worse than a
  coin toss; coin-flips return 0.5; extreme-prediction rate honest.
- **Discrimination recovered on modelable categories** (real latent state, not confabulation):
  **crypto AUC 0.68, economy 0.60, election 0.60** — vs weak/noise on culture/geopolitics (small n).
- **Still below the crowd (0.79).** The remaining gap is grounding: the backtest HUMBLES metric questions
  (it cannot ground the live current value without leaking), yet crypto still reaches 0.68 AUC — so wiring the
  live grounder into the metric branch (trust=high on a grounded current value) is the lever to actually beat
  the crowd on metric questions in live use.

Verdict: honest calibration + real latent state beats confident confabulation, as predicted. The architecture
is now sound; the frontier is grounding quality + per-domain calibrated elasticities, not more machinery.

---

# Flywheel turn 3 — full experimentation run (EXP-092/093)

Tried multiple first-principles levers on the same 660 questions; kept only what genuinely helped.

**What worked — as-of STATE grounding (kept).** Leakage-free: fetch the price known ON the question's date
(Coinbase historical candles) + the asset's realised volatility, trust=high. Sharpened the modelable slice:
**crypto AUC 0.68 (humbled) → 0.79 (grounded).** The lever is real; the metric branch now simulates the actual
as-of price path.

**What was marginal — ensembling.** `grounded+direct` (latent + direct-LLM) is the best model-only config
(+0.02 AUC overall); mixing the model INTO the crowd is WORSE than the crowd alone → defer where a crowd exists.

**What did NOT work — as-of NEWS (information parity via GDELT).** GDELT covers only ~4% of these niche
prediction-market questions. Public news archives do not contain the specific information the market
aggregates, so news cannot give parity here.

**The verdict on the founding thesis** ("same info + real simulation beats biased human instinct"):
- We still do NOT beat the crowd — even grounded, crypto is 0.79 vs the crowd's **0.90**.
- The reason is precise: we do NOT actually have the same information. For NICHE questions the resolving
  information is specific/insider/community knowledge that is in no public archive; for LIQUID crypto the
  market prices the public state efficiently and reads more than price (flow, sentiment).
- Same-information-superior-computation therefore holds only where information is PUBLIC **and** the market is
  INEFFICIENT — the modelable-public-state regime (macro series, diffusions, under-traded questions), not
  liquid markets aggregating private knowledge.

Net: the architecture is sound and honestly calibrated, and as-of state grounding gives real signal where the
world is public + modelable. Beating a good prediction market is not a machinery problem — it is an
information-access problem, and for most market questions the market's information is not publicly obtainable.

---

# Flywheel turn 4 — inner-crowd ensemble + GDELT social-state (EXP-094)

**Inner-crowd (8 diverse personas, extremized log-odds) on the 660.** Simulate the crowd, not one agent.

| category | single-pass | inner-crowd | real crowd |
|---|---|---|---|
| crypto | 0.678 | **0.817** | 0.903 |
| economy | 0.60 | **0.72** | — |
| election | 0.60 | 0.62 | — |
| tech | 0.52 | **0.45** | — |
| sports | 0.54 | 0.52 | — |
| **overall AUC** | 0.552 | **0.568** | 0.789 |

- **Real win on QUANTITATIVE / modelable domains:** crypto 0.68→0.82 (approaching the crowd's 0.90), economy
  0.60→0.72. The quant + base-rater + domain-expert personas sharpen exactly where the world is modelable.
- **Hurts on SOFT domains** (tech 0.52→0.45, sports slightly) — the bull/bear/contrarian speculation injects
  noise where there is no real structure. So the inner crowd must be applied SELECTIVELY (regime routing).
- **Extremization tuned to 0.8 (<1 = de-sharpen), not >1** — confirming the personas are CORRELATED (same
  LLM), so the classic crowd-extremization that helps independent forecasters does not apply; the gain comes
  from averaging out each persona's framing bias, capped by their shared model.
- Calibration unchanged (log-loss ≈ 0.683); the gain is discrimination, on the right domains.

Net: a genuine lever where the world is quantitative/modelable (crypto now 0.82 vs crowd 0.90 — the closest
we've reached), diluted overall by the soft-domain noise → deploy selectively.

**GDELT bulk social-state index (built, demonstrated).** Corrects the earlier "4% coverage" (a rate-limit
artifact + weakest product). Free bulk daily CAMEO event files → as-of per-country social state (tone,
Goldstein conflict-cooperation, protest/violence/diplomacy rates + trends), leakage-free, no rate limit.
Verified as-of Jan 2025: Ukraine/Russia show ~2x US violence, Russia net-conflictual. The vision-aligned
grounding for the social/geopolitical slice — next: wire it into the forecaster.

---

# Flywheel turn 5 — GDELT wired in, mechanism simulators, and the data thesis (EXP-095)

Three things this turn: (1) wired GDELT social-state grounding INTO the forecaster and measured it; (2) built
the general mechanism-specific simulator framework; (3) answered the strategic questions — inner-crowd
independence and "is the gap just MORE DATA."

## 1. GDELT social grounding — measured, and it is a REAL lever on the social slice (EXP-095)

`GdeltSocialGrounder` detects the question's country (name→FIPS), pulls its as-of GDELT social state
(conflict/violence/protest trajectory, leakage-free), injects a QUANTIFIED state block into the compile prompt,
and adds a GROUNDED escalation driver whose SIGN is the LLM's conflict-polarity read (escalation q → +, ceasefire
q → −) and whose STRENGTH is the measured conflict/violence departure from calm. Run WITHOUT vs WITH grounding
on the 99 clean questions that name a country:

| slice | plain AUC | **grounded AUC** | plain ll | **grounded ll** |
|---|---|---|---|---|
| **CONFLICT/POLITICS** (on-topic, n=15) | 0.696 | **0.768** | 0.648 | **0.629** |
| off-topic (sports/other, n=84) | 0.558 | 0.583 | 0.673 | 0.676 |
| all country-naming (n=99) | 0.563 | **0.596** | 0.670 | 0.670 |

- **The grounding helps exactly where it is on-topic:** on genuine conflict/politics questions, discrimination
  rises **AUC 0.70 → 0.77** and calibrated log-loss improves **0.648 → 0.629**. Measuring the real state of that
  part of the social world makes the forecast of that world better — the founding thesis, confirmed on the most
  vision-aligned slice.
- **Neutral where off-topic** (sports/other: AUC +0.02, log-loss flat-to-slightly-worse) — the LLM's
  `conflict_pushes` mostly no-ops when a country is named incidentally. So GATE it to social/geopolitical
  questions (a regime signal), don't fire it on every country mention.
- Caveat: the conflict subset is small (n=15) and the crowd is perfect there (AUC 1.0), so skill-vs-crowd stays
  negative — the gain is DISCRIMINATION, honest, and directionally clear. The lever is real; scaling the
  measurement needs a bigger conflict corpus.

## 2. Mechanism-specific simulators — specialize on MECHANISMS, not subjects (`swm/api/mechanisms.py`)

The general-world-model way to add per-domain simulators without fragmenting into infinite topics: there are
only ~7 GENERATIVE MECHANISMS by which a binary social outcome is produced. Elections, referendums and
shareholder votes are ONE mechanism (aggregate a share, threshold at a majority); sports, court cases and races
are a CONTEST; a launch, a death and a record are an ARRIVAL. So we built seven tiny parametric Monte-Carlo
simulators on the SHARED honest substrate (base-rate anchor + integrated parameter uncertainty + real horizon),
plus an LLM router that names the mechanism and supplies its grounded params:

| mechanism | covers | grounded by | kernel |
|---|---|---|---|
| aggregation | vote / approval / referendum | polls / current share | threshold on a Normal share (poll error) |
| contest | sports / court / race | ratings / odds | Elo-logistic or grounded win-prob |
| diffusion | price / index / % / count | as-of value + realised vol | GBM (reuses the metric branch) |
| arrival | launch / death / record / first | base rate over a horizon | Poisson survival 1−e^(−λH) |
| whipcount | legislation / treaty / merger | committed votes vs needed | Binomial break of the undecided |
| escalation | war / bank-run / viral / unrest | GDELT social pressure | reinforcing log-odds drift |
| persistence | incumbency / status-quo / on-time | disruption hazard | e^(−hH) status-quo survival |

`mechanism_forecast` compiles → grounds live (as-of price for diffusion, GDELT pressure for escalation) →
routes, and **falls back to the generic latent sim** when the mechanism/params are absent, so it is never worse
than `latent_forecast`, only more specific where it can be. One engine, any binary social question. (23 new
unit tests, all green.)

## 3. The strategic answers

**Inner-crowd independence — can we get independent errors?** Not with the current backend. Independent errors
require different model FAMILIES (distinct pretraining), not personas of one model — which is exactly why the
extremization factor tuned to **0.8 (<1)**: the 8 personas share DeepSeek's weights, so their errors are
correlated and sharpening them amplifies shared bias. Making personas "more extreme" on one model therefore
CANNOT help (and empirically hurts). The real independence path is a multi-MODEL panel (DeepSeek + Qwen + Llama
+ Mixtral + Gemma), and only THEN is extremization >1 valid. We built that path (`model_panel_llms`, model-aware
`ResilientLLM` with DeepSeek-fallback OFF so families stay independent) — but it is currently **billing-blocked**:
the HF router returns `402 Payment Required` for every model family (HF credit exhausted), and DeepSeek-direct
serves only one model. So: possible in architecture, ready in code, gated on HF credit. Until then the
single-model inner-crowd stays a SELECTIVE booster on modelable domains (crypto 0.68→0.82, econ 0.60→0.72).

**Is inner-crowd a drift from the core thesis? — partly, yes.** Inner-crowd is an ensemble-of-JUDGMENTS method
(average many LLM framings); it is orthogonal to "simulate the social world's latent state + transitions." It
earns its place only as the honest fallback where there IS no groundable latent state (soft/unmodelable
questions). The GDELT + mechanism + dataset direction IS the core vision — measure the real state, evolve it
with calibrated transitions, read off the outcome — and it is where the measured edge lives (GDELT +0.07 AUC on
conflict; crypto grounding 0.68→0.79). **Strategic call: invest in the world-model stack, keep inner-crowd as a
gated soft-domain booster.**

**Are there FULL social-world-model datasets? Is GDELT the closest? — GDELT is the closest single feed, but the
real answer is a LAYERED STACK, and yes, the gap is substantially MORE DATA.** No one dataset is a "full social
world model." The social world's grounding decomposes into three layers, and the frontier is grounding on all
three:

- **Event / flow layer** (high-frequency "what is happening now") — **GDELT** (563M events, 1979→, global, the
  broadest + most immediate feed, but noisy machine-coding: poor redundancy/domain accuracy) + **POLECAT** (6.2M
  events 2010→, the ICEWS successor, PLOVER ontology replacing CAMEO, high accuracy + low redundancy) +
  **ACLED** (human-verified violence/protest, georeferenced by day — the gold standard, but narrow). GDELT for
  breadth/immediacy, POLECAT/ACLED for accuracy on the conflict slice.
- **Structural / state layer** (slow-moving latent state of each society — this literally IS the world model's
  latent state) — **V-Dem** (~500 democracy/institution indicators, expert-coded, country-year) + **World Bank
  WDI** (266 economic/social/demographic indicators, 217 countries) + Polity / Fragile-States.
- **Belief / opinion layer** (what people think) — polls/surveys (partly wired) + GDELT tone + social media.

Published country-month stability models that fuse **363 WDI + 129 V-Dem + ACLED** reach ~85% contemporaneous /
~75% next-month accuracy — direct evidence that the layered stack is the calibration substrate. So GDELT is the
closest to a live universal social-event feed (and the right first grounding, now wired + measured), but the
step-function is a STACK: GDELT/POLECAT/ACLED events over a V-Dem/WDI structural state, with mechanism
transitions calibrated on how those actually move. That is the core thesis, fully — "more real-world data to
ground/calibrate on," organized as latent state (V-Dem/WDI) + transitions (calibrated on GDELT/ACLED flow).

Sources: GDELT vs POLECAT comparison (doi.org/10.3390/data11070158), PLOVER/POLECAT (Halterman et al.), V-Dem
(v-dem.net), World Bank WDI, ACLED, and the WDI+V-Dem+ACLED stability-forecasting literature.

---

# Flywheel turn 6 — first external pastcasting benchmark: BTF-3 pilot (EXP-101)

Ran the EXP-095 mechanism forecaster on 50 questions of FutureSearch's public BTF-3 pastcasting
benchmark (all outcomes May–Jul 2026; leakage protocol code-enforced; paired arms: DeepSeek-V4-Flash
[Apr-2026 cutoff ≈ the intended as-of state] and V3-0324 [mid-2024 cutoff, unambiguously clean]).

| | brier | AUC | vs |
|---|---|---|---|
| WMv2 + V4-Flash | 0.263 | 0.497 | FutureSearch SOTA **0.092** / AUC 0.918; const-base 0.192 |
| WMv2 + V3-0324 | 0.337 | 0.508 | same 50 questions |

Both arms at chance discrimination — the EXP-089 result reproduced externally, and simultaneously a
clean No-Evidence contamination probe (leakage inflates; it never nullifies). Damage is concentrated in
8 wrong-side extreme predictions from two families: evidence deficit (Hormuz, BoJ — retrieval, not
reasoning) and confident confabulation in the `aggregation`/`whipcount` kernels, which never inherited
EXP-091's honest-ignorance posture. Model recency (V3→V4, two years of knowledge) bought only 0.073
brier at chance AUC — the vintage is not the constraint; uncertainty discipline + evidence are.
Cutoff forensics: V4 self-reports "May 2025" but is officially Apr 2026 — fine for BTF-3, **disqualifying
for BTF-2 (resolves Oct–Dec 2025)**, which must use old-cutoff models via the new
`swm/api/openrouter_backend.py`. Full report: `experiments/exp101_btf3_pilot.md`.
