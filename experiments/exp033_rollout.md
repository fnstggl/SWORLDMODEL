# EXP-033 — Multi-step rollout: does the temporal dynamics hold up over a horizon? (the honest answer)

The one-step operator (EXP-030) wins at t+1 with a known event. The whole "forecast forward / who wins"
ambition rests on rolling that forward over days and weeks. This is the leakage-free test of whether it
holds up — and the answer reshapes the roadmap.

## Setup (no-cheat)
Endogenous per-step dynamics (momentum + mean-reversion, `swm/transition/rollout.py`) fit on TRAIN market
trajectories; then rolled forward on **held-out TEST markets' future arrays** (untouched daily prices
after the target). The event-informed tier uses the EXP-030 LLM impact at step 1 only — because the real
constraint is that **future events are unknown**. 182 Kalshi markets with ≥10 future days.

## Result — persistence (the martingale) beats everything, at every horizon
MAE vs the actual future price, by horizon (days):

| model | h1 | h3 | h5 | h7 | h10 |
|---|---|---|---|---|---|
| **persistence (flat)** | **0.036** | **0.053** | **0.069** | **0.081** | **0.090** |
| momentum extrapolation | 0.038 | 0.064 | 0.090 | 0.113 | 0.128 |
| endogenous (learned) | 0.038 | 0.074 | 0.116 | 0.161 | 0.217 |
| event-informed | 0.039 | 0.074 | 0.116 | 0.161 | 0.217 |

- **Nothing beats persistence at any horizon.** Momentum and learned dynamics are *worse*, and get
  progressively worse with distance (endogenous MAE 0.038 → 0.217 over 10 days).
- **The one-step event edge does not survive rollout** — `event_informed ≈ endogenous` after step 1; the
  edge is local to the step where we actually have the news.
- **Drift:** even persistence's error grows **2.5×** over 10 days (0.036 → 0.090) — the belief genuinely
  moves, driven by events we can't see.
- The Monte-Carlo band is *over*-covered (0.89–0.99 vs target 0.80): it widens too fast — a calibration
  fix, not the headline.

## What this means (the essential finding)
**Prediction-market belief is a near-martingale: today's value is the best forecast of every future
value, and you cannot beat it by extrapolating dynamics — because future moves are caused by future
events you don't know.** Our learned momentum/mean-reversion actively *hurts* (it injects spurious
movement). The world model's forecasting power is **information-bounded, not dynamics-bounded**: the edge
exists exactly and only where we have event information.

## The roadmap consequence
Long-horizon forecasting is therefore **not** a "better dynamics" problem — it is a **future-event
forecasting** problem. To forecast weeks/months out you must model the *distribution of future events*
(and their belief impact), then roll the transition over sampled event paths. This is precisely the
branching-realities problem, and it is the honest bottleneck the general simulator must solve (see
`ROADMAP.md`). Extrapolating the belief curve is a dead end; forecasting the events that move it is the
task.

## Honest limits
- Kalshi, ≤10-day horizons (the data's future arrays cap at ~16 days); months/years are unreachable with
  this benchmark and would need longer trajectories.
- "Beats persistence" is a hard bar precisely because these markets are efficient; on *less* efficient
  belief series (slow-moving social attitudes) endogenous dynamics may carry more — untested here.
- The event-informed tier only had news at step 1; a rollout with sampled *future* events (the roadmap
  build) is the real multi-step test and is not yet done.

## Reproduce
`python -m experiments.exp033_rollout` (SWM-Bench + committed impact signals).
`python -m pytest tests/test_rollout.py`.
