# EXP-072 — a real contagion/tipping test: where the coupled dynamic finally beats simple baselines

The prediction from EXP-070/071 was precise: a coupled/shared-world model earns its place only where **(a)
endogenous cross-agent feedback is strong AND (b) simple baselines are weak** — the contagion/tipping
regime. This is the test of that prediction on real data, and it holds.

**Data.** Real SSA-derived baby-name shares — 481 names, 1880–2008. Names are a *pure* social contagion:
they spread by imitation (bandwagon), saturate, and then **crash** (fashion fatigue). Persistence and trend
are structurally wrong at the turns. Forecast each name's share **H=10 years ahead** from its as-of
trajectory only (leakage-free).

**Models.**
- **persistence** — share stays put (the baseline that beat us on FOMC and GSS).
- **trend** — extrapolate the recent slope.
- **contagion** — the **coupled bandwagon+saturation dynamic**: momentum carries the name, but its growth is
  dragged down by its own level (fatigue/over-exposure), so a high-flier decelerates, **peaks, and
  reverses**. Growth depends on prevalence → a genuinely coupled (non-separable) forecast. Two params (rho,
  lambda) fit on TRAIN names, scored on held-out TEST names.

## Results (MAE, percentage-point share)

| slice | n | persistence | trend | **contagion** | winner |
|---|---|---|---|---|---|
| **ALL** | 18,429 | 0.140 | 0.136 | **0.123** | **contagion (+9%)** |
| **TURNING POINTS** (near peak) | 1,083 | 0.264 | 0.570 | **0.152** | **contagion (+42%)** |
| RISING (g > 0.02%/yr) | 1,704 | **0.441** | 0.630 | 0.498 | persistence |
| STABLE | 16,108 | 0.107 | **0.081** | 0.085 | trend |

## What this shows — the coupling finally earns its place

- **At the turning points — the hard cases — the coupled contagion dynamic beats the best simple baseline by
  42%** (MAE 0.152 vs 0.264). This is the first real-data case in the whole project where a coupled,
  non-separable dynamic *substantially and cleanly* beats persistence and trend. It wins precisely because it
  predicts the peak-and-reverse that persistence (says "stays high") and trend (says "keeps rising") both
  get structurally wrong.
- **Overall it wins too (+9%).** Across all 18k forecast points, modeling the cascade beats the simple
  baselines.
- **And it's honest about where it doesn't help.** On STABLE names, trend/persistence are slightly better
  (contagion is a touch too eager to reverse); on RISING-but-not-yet-peaked names, persistence is safest
  (the model sometimes calls the turn early). The coupled dynamic is not a universal improvement — it is the
  right tool *specifically* in the tipping regime, exactly as the EXP-070/071 boundary predicted.

## The through-line, now closed

Across the scored couplings:
- **SCOTUS (EXP-070)** and **FOMC (EXP-071)** — coupling ties/loses, because simple baselines (static
  ideology, policy inertia) are *strong*.
- **Contagion (EXP-072)** — coupling **wins by 42% at the turns**, because simple baselines are *weak* and
  the process is a genuine cascade.

That is the complete, measured answer to *when the shared-world machinery beats separate models*: **in the
contagion/tipping regime.** The discipline of scoring coupled-vs-separate did exactly its job — it found the
regime where the coupling pays, and honestly flagged the ones where it doesn't.

## Tests — `tests/test_contagion.py` (3, all pass)

The contagion dynamic rises-then-reverses (a turning point a linear trend can't make); it beats persistence
at a synthetic turn; a low/flat name stays put. Full suite green.

## Data

`experiments/results/exp072/baby_names.json` — real name-share trajectories (SSA-derived, committed); reruns
offline.
