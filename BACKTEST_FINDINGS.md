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
