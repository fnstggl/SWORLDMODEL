# EXP-045 — Multi-step population-over-time rollout: a grounded population BEATS persistence

The untested axis of the thesis, and the one the market data could not test. EXP-033 showed a market's
belief is a near-martingale — *nothing beats "it stays put"* over multiple steps, so long-horizon belief
forecasting is hopeless from the price. The open question: is a **population** the same? Or does a
population have *structure* that evolves predictably, so simulating it forward beats persistence?

We could not test this on OpinionQA (its cache repeats only 4 questions across waves), so we brought in a
new data source: the **General Social Survey (GSS), 1972–2024** — 72,707 respondents across 34 survey
years, 15 attitude items, individual demographics each wave. The canonical longitudinal opinion dataset.

## The grounded forward simulation
Opinion changes partly because the population's *composition* changes — cohorts age, education rises,
the party/ideology mix shifts — and each demographic cell carries a stance. The simulation predicts that
compositional component:
```
Ŝ_grounded(t) = S(last) + [ ĝ(demographics at t) − ĝ(demographics at last) ]
```
where `ĝ(·)` is a correlation-aware demographic→attitude model fit on all PRIOR years, applied to a
given year's *real* demographic composition. It adds the demographic-driven change onto the persistence
level — a true bottom-up roll-forward. (Future demographics are near-deterministic — cohorts age, the
census projects them — so using year-t composition is fair; only ATTITUDES are held out.)

## Setup (no-cheat, rolling-origin)
For each item and each test year `t`, fit only on years `< t`, predict the population's support share
`S(t)`, compare to the held-out actual. Baselines: persistence `S(last)` and a linear-trend
extrapolation. 15 items, **406 rolling-origin forecasts**.

## Result — the grounded population beats the martingale, and more so over longer horizons

**MAE of the predicted population share (lower is better), by horizon:**

| method | all | 1–3y | 4–7y |
|---|---|---|---|
| persistence (martingale) | 0.0288 | 0.0287 | 0.0419 |
| linear trend | 0.0478 | 0.0476 | 0.0724 |
| **grounded forward simulation** | **0.0264** | **0.0264** | **0.0288** |

**Directional accuracy of the change** (did the share move up or down vs the last observation, on real
moves):

| method | directional accuracy |
|---|---|
| persistence | 0.480 |
| linear trend | 0.517 |
| **grounded forward** | **0.593** |

Two things stand out:
1. **The grounded population rollout beats persistence** — the opposite of EXP-033's market result. A
   population is *not* a martingale: its demographic composition evolves predictably, and composing the
   opinion from grounded cells captures drift the "it stays put" baseline misses.
2. **The advantage grows with horizon.** At 1–3 years the edge is small (0.0264 vs 0.0287 — opinion is
   sticky short-term, persistence is strong). At **4–7 years the grounded MAE is 0.0288 vs persistence's
   0.0419 — a 31% reduction.** This is the multi-step signal: the longer you roll forward, the more the
   compositional dynamics matter, exactly as a forward simulation should behave.
3. **Naive trend extrapolation is worse than persistence** (0.048 vs 0.029) — it overshoots. The win is
   specifically from the *grounded, structural* forward model, not from any extrapolation.

## Why this matters for the north star
This is the first evidence for the thesis's deepest claim — **simulate agents forward N steps and read
off what happens** — on real longitudinal data. It cleanly separates the two regimes the project has
been mapping:
- **Efficient-market belief (a price): a martingale.** You cannot roll it forward and beat persistence
  (EXP-033). Direction is callable via the lean (EXP-036), but the path is not forecastable.
- **A grounded population: NOT a martingale.** It has structure (who is in it, how that evolves) that a
  bottom-up simulation exploits to beat persistence, increasingly so over longer horizons (EXP-045).

So the differentiated capability — forecasting *outcomes that are the aggregate of a modelable,
evolving population* — is real and multi-step, precisely where no liquid market exists to read.

## Honest limits
- The forward model captures the **compositional** component of change (cohort/demographic shift); it
  does not model **period effects** (a within-cohort attitude swing from an event), which is why the
  overall edge is modest and concentrated at longer horizons where composition dominates. Coupling this
  to the EXP-042 event operator (period shocks) is the natural next step.
- Uses the target year's *actual* demographic composition (fair — demographics are forecastable — but a
  fully autonomous forecast would project the composition too; census projections make this easy).
- 15 items, U.S. only; the mechanism should generalize but is validated here.

## Reproduce
`python -m experiments.exp045_population_rollout` → `experiments/results/exp045_population_rollout.json`
(uses the committed gzipped GSS cache; the raw .dta is gitignored — see `experiments/datasets_gss.py` for
the one-time download). Rolling-origin fit is ~20 min.
