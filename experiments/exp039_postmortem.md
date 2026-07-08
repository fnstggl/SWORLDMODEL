# EXP-039 — The live post-mortem loop (Tetlock #8/#10: a leakage-free skill number + perpetual beta)

Two things a deployed forecaster must do that no backtest can:

1. **Produce a contamination-free skill number.** Every dated backtest risks the model recalling the
   outcome (the Halawi trap that caveats EXP-037). A forecast *logged before* its resolution and *scored
   after* has no future to leak. `PostMortemLog` enforces `made_at < resolves_at` and computes skill only
   over such forecasts — leakage-free **by construction**, not by hoping the cutoff holds.
2. **Recalibrate from its own track record (perpetual beta).** As forecasts resolve, fit a calibration
   map on PAST resolved (forecast, outcome) pairs and apply it to FUTURE forecasts — better probabilities
   the longer it runs, past correcting future, no leak.

## Setup (no-cheat)
Each of 91 Kalshi questions is logged with its as-of market belief `target.p` as the system forecast
(the `s_t` our transition operator starts from), `made_at = target time`, `resolves_at = the market's
resolution date` (strictly later), and outcome = the market's near-resolution value (`future[-1] > 0.5`).
Recalibration is fit only on forecasts resolved strictly before the eval window.

## Result

**A. Leakage-free skill (n=91, made_at < resolves_at for every forecast):**

| metric | value |
|---|---|
| Brier | 0.0566 |
| log-loss | 0.2282 |
| ECE | 0.1687 |
| directional accuracy | 0.966 |
| base rate | 0.275 |

This is the first skill number in the project that is contamination-free **structurally** — there is no
future for the model to recall, because the forecast is fixed before the outcome exists. (It scores the
market belief itself here; deployed, you log whatever forecast the engine emits.)

**B. Self-recalibration (do-no-harm perpetual beta):**
- *Real Kalshi track record:* with only ~45 early-resolved forecasts, the do-no-harm guard's held-out
  validation slice is too small to trust, so it **abstains** — recalibrated ECE = raw ECE (0.180),
  **no regression**. Correct behavior: you cannot recalibrate on a track record you don't yet have.
- *Controlled mechanism check* (n=400, a stationary underconfident forecaster hedged toward 0.5 — a
  common real failure): the guard **deploys** and recalibration improves held-out **ECE 0.142 → 0.075**
  and Brier 0.176 → 0.160. The loop recovers calibration from its own record.

## The honest findings
1. **The recalibration guard is the real contribution, not the Platt map.** A naive chronological Platt
   *hurt* on the real data (ECE 0.18 → 0.26): the market belief is already fairly sharp and 45 forecasts
   is far too few — the map overfit a skewed early window and anti-transferred. The fix is a
   **self-validating, do-no-harm** recalibrator: fit on 80% of past-resolved forecasts, validate on the
   held-out 20%, and deploy the map **only** if it beats identity by a meaningful margin *and* the
   validation slice is large enough (≥15) to trust. On thin/already-calibrated history it abstains; on a
   real, persistent miscalibration with enough history it deploys and helps. Perpetual beta that can
   never make a good forecaster worse.
2. **You must earn the right to recalibrate.** The real-data abstention isn't a failure — it's the honest
   Tetlock lesson made mechanical: a forecasting *skill* is trained on an accumulated *scored record*, and
   91 dated market questions (≈45 resolved before the eval window) is below the threshold where
   self-correction is reliable. The mechanism is validated at the scale a deployment reaches.
3. **A log-odds (Platt) map cannot fix a log-odds miscalibration non-trivially** — it absorbs any affine
   distortion of the logit exactly, so an "overconfident (×1.7 log-odds)" stress test is degenerate. The
   meaningful test is a *non-affine* failure (underconfidence shrunk in probability space), which is what
   B uses.

## What it means for the architecture
The loop closes Tetlock's cycle — **measure → learn → recalibrate → measure** — mechanically, and is the
only path in this project to a **structurally** leakage-free skill number (the EXP-037 direction result
carried a recall caveat this design removes). Deployed, the engine logs each forecast, scores it as
questions resolve, and sharpens its probabilities from its own record without ever risking a regression.

## Honest limits
- The real skill number scores the market belief, not yet a from-scratch engine forecast; wiring the
  `QuestionEngine`/`simulate()` output into the log is the deployment step.
- Recalibration is validated at scale on a controlled stationary stream; real track records drift
  (non-stationarity), which the guard handles by abstaining but does not yet *adapt to* (a rolling-window
  or drift-aware recalibrator is the next refinement).

## Reproduce
`python -m experiments.exp039_postmortem` → `experiments/results/exp039_postmortem.json`.
`python -m pytest tests/test_postmortem.py`.
