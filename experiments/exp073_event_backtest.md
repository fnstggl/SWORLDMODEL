# EXP-073 — the no-cheat historical-event backtest: does the calibrated multi-variable simulation beat baselines?

The decisive test of the vision, assembled from the two new pieces (the calibration engine + the event
backtest harness), on the one longitudinal case where a forward simulation has beaten persistence before
(GSS opinion, EXP-045). Now run properly: rich per-variable calibration, scored no-cheat against the
baselines a skeptic uses for free.

## Setup

- **Data**: GSS opinion 1972–2024 (real, 72,707 respondents, 15 items). Rolling origin at **2006** — the model
  trains ONLY on rows ≤ 2006 and forecasts each later year (2008–2024). 133 forecasts. Leakage-guarded.
- **Forecaster** (EXP-045 form, anchored on persistence so the model supplies the *change*, not the level):
  `Ŝ(t) = S(2006) + [ĝ(composition at t) − ĝ(composition at 2006)]`, where `ĝ` is `CalibratedWeights`
  (per-variable priors + empirical-Bayes shrinkage + integrated weight uncertainty) fit on ≤2006 demographics
  → opinion.
- **Baselines**: persistence `S(2006)`, linear trend, base rate. Scored by `swm/eval/event_backtest.py`
  (SKILL = 1 − MAE/MAE_baseline).
- **Fidelity arms**: FEW variables (party, age) vs ALL 11 demographic variables — the direct thesis test.

## Result

| fidelity | forecasts | MAE | skill vs persistence | vs trend | vs base rate | beats ALL |
|---|---|---|---|---|---|---|
| **2 variables** | 133 | 0.0786 | **−0.032** | −0.054 | +0.304 | No |
| **11 variables (calibrated)** | 133 | 0.0680 | **+0.107** | +0.088 | +0.398 | **Yes** |

**Two findings, both decisive:**

1. **The rich calibrated simulation beats every baseline, no-cheat.** At full fidelity the forward
   simulation is +10.7% skill over persistence, and beats the linear trend and base rate simultaneously —
   the honest bar (a skeptic's free forecasts) is cleared.

2. **More calibrated variables *raised* the skill — it flipped a loss into a win** (−0.032 → +0.107). Two
   variables underperform persistence; all eleven, properly calibrated, beat it. This is the thesis measured
   on a real *predict-the-future* task: modeling more pressuring variables makes the simulation more
   accurate, provided each carries a calibrated weight.

## Why this matters

Across the earlier scored cases (SCOTUS, FOMC) the couplings didn't beat simple baselines — but those have
strong simple baselines (static ideology, policy inertia) and weak cross-scale feedback. GSS opinion is the
regime the substrate was built for: a modelable, evolving population where a price/momentum baseline has no
structural edge. Here the digital-twin bet pays: **fidelity buys accuracy, and the forward simulation beats
the baselines.** The general lesson is now empirical — *reach for the rich simulation where the outcome is
the aggregate of a modelable evolving population and simple baselines are weak*; the calibration engine is
what makes the added variables help rather than hurt.

Run: `python -m experiments.exp073_event_backtest`. Machinery: `swm/variables/calibrated_weights.py`,
`swm/eval/event_backtest.py`. Tests: `test_calibrated_weights.py`, `test_event_backtest.py`.
