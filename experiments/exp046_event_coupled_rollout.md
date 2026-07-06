# EXP-046 — Event-coupled population rollout: where period dynamics help (and where they don't)

EXP-045's grounded forward simulation captured the **compositional** component of opinion change (the
population's demographic mix evolves, each cell carries a stance) and beat persistence. It explicitly left
out **period effects** — the within-cohort swings events cause. This couples the two:
`S(t) = S(last) + Δcompositional + Δperiod`, and forecasts the period term from the composition-removed
residual (the aggregate footprint of events), projected forward with damping and optionally distributed
by each cell's responsiveness (the EXP-042 operator).

## Setup (no-cheat, rolling-origin)
GSS, 15 items, 406 forecasts. For each item/test-year, fit only on years `< t`; the period velocity comes
from the past residual series `r(y) = S(y) − ĝ(demographics at y)` only. Damping 0.5, responsiveness by
age group (impressionable-years), estimated globally and pooled across items.

## Result — MAE of the predicted share (lower is better), by horizon

| method | all | 1–3y | **4–7y** |
|---|---|---|---|
| persistence | 0.0288 | 0.0287 | 0.0419 |
| compositional (EXP-045) | **0.0264** | **0.0264** | 0.0288 |
| coupled_uniform (+ period momentum) | 0.0272 | 0.0272 | **0.0251** |
| coupled_gated (+ per-cell responsiveness) | 0.0273 | 0.0273 | 0.0263 |

Change directional accuracy: persistence 0.480, compositional 0.593, coupled_uniform 0.593,
coupled_gated **0.596**.

## The honest findings
1. **Period dynamics help exactly where theory predicts — the long horizon — and hurt where it doesn't.**
   At **4–7 years the coupled model (0.0251) beats compositional-only (0.0288) by 13% and persistence
   (0.0419) by 40%.** Over long spans the secular period drift accumulates into real, forecastable signal.
   But at 1–3 years the period momentum is mostly noise (0.0272 vs 0.0264), and since most forecasts are
   short-horizon, **compositional-only wins overall.** The two dynamics genuinely compose, but their
   balance is horizon-dependent: composition dominates the near term, period drift the far term.
2. **Responsiveness-gating adds nothing over a uniform shock** (0.0273 vs 0.0272). At the aggregate-share
   level, distributing the period shock by per-cell responsiveness — the EXP-042 gating that was
   *mechanistically* real at the individual level — does not improve the population forecast. The
   composition already carries most of the heterogeneity; the extra per-cell modulation of the period
   term is second-order and washes out.
3. **Per the do-no-harm rule, the compositional model (EXP-045) stays the deployed forecaster** — the
   coupling is not a net win. But EXP-046 tells us precisely *when* to switch it on: a horizon-aware
   forecaster should add the period term at long horizons (4y+) and suppress it near-term.

## What it means for the architecture
- The EXP-045 → EXP-046 pair cleanly decomposes opinion change into **composition** (near-term, always
  helpful) and **period drift** (far-term, helpful only there). A production population forecaster should
  weight the period term by horizon — which is the honest, evidence-based coupling rule, not "always add
  events."
- The disappointing gating result sharpens where the individual-level operator (EXP-042) does and doesn't
  transfer to the aggregate: its *mechanism* is real per-person, but aggregate shares are dominated by
  composition, so per-cell event-gating doesn't move the population number.

## Honest limits
- Damping (0.5) and the residual window (8) are fixed at sensible defaults, not tuned leakage-free; a
  horizon-conditional damping schedule (low near-term, higher far-term) is the obvious refinement and
  should widen the long-horizon win.
- The period signal is the *aggregate residual* (events' footprint), not an explicit event calendar;
  wiring real dated events (elections, rulings, shocks) as the period driver is the deeper version.

## Reproduce
`python -m experiments.exp046_event_coupled_rollout` →
`experiments/results/exp046_event_coupled_rollout.json` (committed gzipped GSS cache; ~20 min).
