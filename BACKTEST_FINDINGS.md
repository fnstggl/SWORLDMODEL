# Large-scale backtest verdict (EXP-089/090) — history as the test set

Run over **660 resolved, cutoff-clean** binary forecasting questions from Manifold (post-training-cutoff, so
the model cannot have memorized the outcomes), each with the **crowd probability at a fair as-of lead** as the
baseline. The world-model forecaster: compile the question as-of → simulate → P(YES) (the LLM states variables
+ a mechanism; the *simulation* produces the probability — the decomposition defense against leakage).

## The result: the simulation architecture is noise on open-domain questions

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
