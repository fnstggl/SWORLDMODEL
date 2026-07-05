# EXP-035 — Future-event model + distributional rollout: the honest "simulate forward" capability

EXP-033 proved you cannot beat the martingale on the POINT forecast — the *direction* of future surprises
is unforecastable in an efficient belief series. So this build targets the thing you actually can win and
actually need for decisions: a **calibrated predictive distribution** over the future belief. Scored
properly (CRPS — a proper scoring rule rewarding sharp *and* calibrated forecasts — plus interval
coverage), and iterated to diminishing returns.

## Setup (no-cheat)
`swm/transition/event_model.py`: learns, from TRAIN market trajectories, (a) a near-zero drift and (b) a
**heteroskedastic per-step variance** E[Δ²|state] (variance depends on recent volatility, level, distance
to the 0/1 boundary), then Monte-Carlos the belief forward into a distribution. σ is calibrated on a
train/val split; the known-event direction at step 1 is the EXP-030 LLM impact. Rolled forward on
held-out TEST futures (Kalshi, 10-day horizon).

## Result — a calibrated distribution that beats persistence and a constant band
Mean CRPS (lower is better), and 80%-interval coverage:

| tier | mean CRPS ↓ | CRPS h1 | h5 | h10 | coverage (target 0.80) |
|---|---|---|---|---|---|
| persistence (point) | 0.0673 | 0.036 | 0.069 | 0.090 | — (no distribution) |
| constant-variance band | 0.0595 | 0.030 | 0.062 | 0.077 | ~0.91 (too wide) |
| **event model (heteroskedastic)** | **0.0513** | **0.027** | **0.054** | **0.066** | **0.74–0.82 (calibrated)** |

**The event model beats persistence by ~24% and a constant band by ~14% on CRPS, at every horizon, with
80% intervals that actually cover 80%.** This is the real "simulate forward" deliverable: not a better
point (impossible for a martingale), but a *calibrated distribution* — which is what P(outcome),
intervals, and decisions require.

## What the iteration taught us (v1 → v3, then converged)
1. **v1 (CRPS 0.073, coverage 0.12–0.28 — broken):** fitting *log*-variance on individual squared deltas
   collapsed σ toward zero (Jensen bias — most steps are quiet, so E[log Δ²] ≪ log E[Δ²]).
2. **v2 (CRPS 0.056):** predict **E[Δ²] directly** → correct heteroskedastic σ → beats both baselines.
3. **v3 (CRPS 0.051, coverage ~0.80):** calibrate a global σ-multiplier on a train/val split → sharp *and*
   calibrated. Sweeps then show CRPS is flat around σ×0.5–0.6 → **diminishing returns, converged.**

**The decisive honest findings:**
- **The win is heteroskedastic variance — forecasting *when* volatility hits, not *which way*.** Placing
  uncertainty by state beats a constant band; that is the entire edge.
- **Direction does not help multi-step.** Ablating the LLM impact *improves* CRPS by 0.001 (event-no-
  impact 0.0513 vs 0.0523), and adding endogenous drift monotonically hurts (0.0516 → 0.0574 as drift→1).
  This reconfirms EXP-033: the event's directional effect is local to t+1; over a horizon only its
  *variance contribution* survives. The model is honest about this — it is a calibrated uncertainty
  forecaster, not a direction crystal ball.

## What this means for the general simulator (ROADMAP stage D/E/F)
It reframes the "future-event model." For an **efficient** belief series you cannot and should not
forecast the *direction* of future events — you forecast the **distribution** of their impact (variance +
timing), and the rollout returns calibrated percentages. That is exactly what decisions need: P(belief
crosses a threshold), interval bands that widen correctly, and — with the Monte-Carlo machinery here —
pivotal-branch conditionals. The directional-forecasting fantasy is a dead end where the source is
efficient; the calibrated-distribution build is the achievable, useful thing, and it works.

## Honest limits
- Kalshi, ≤10-day horizon (the data caps futures at ~16 days); the *shape* of the variance term should
  extend to weeks, but longer horizons need longer trajectories to verify.
- The heteroskedastic variance uses trajectory state only; a true event *calendar* (known FOMC/election/
  earnings dates) would place variance even better — the natural next data addition.
- On **inefficient** belief series (slow social attitudes), directional forecasting may add real point-
  accuracy; there the drift head could earn its place. Untested — that is where a general SWM is most
  differentiated from prediction markets.

## Reproduce
`python -m experiments.exp035_event_model` (SWM-Bench + committed impact signals).
`python -m pytest tests/test_event_model.py`.
