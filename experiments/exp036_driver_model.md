# EXP-036 — Can we forecast DIRECTION? Yes. (correcting the martingale over-claim)

EXP-033/035 concluded "direction is unforecastable." That was **over-stated** — it was true only for the
setup used (price shape + a *momentum* drift). This experiment finds a real, leakage-safe directional
signal and corrects the record.

## The driver we were missing: the lean
A leakage-safe diagnostic (no LLM, no future data) found: a belief's **lean** — its distance from 0.5 —
predicts the direction of its future move, because **a question resolves toward the side it currently
favors, at roughly its calibration rate.** A belief at 0.8 goes up ~80% of the time. That is a directional
signal knowable *as-of* from the current level — no LLM recall of outcomes (the contamination that
inflated prior LLM-forecasting claims — Halawi et al. were criticized for exactly that).

## Result (Kalshi, no-cheat; directional accuracy on the sign of the 8-day move)
| directional rule | overall accuracy | confident (\|p−0.5\|>0.2) |
|---|---|---|
| coin flip | 0.50 | 0.50 |
| momentum (sign of recent slope) | 0.50 | 0.45 |
| **lean (predict toward the favored side)** | **0.80** | **0.85** |
| learned classifier | 0.53 | 0.58 |

- **The lean predicts direction at 0.80 (test) / 0.61 (train period) — decisively above chance in both
  regimes.** Direction *is* forecastable. Confident beliefs: 0.85.
- **Momentum is useless (0.50/0.45)** — reconfirming EXP-033 that *price* dynamics carry no direction.
- **A learned classifier does not beat the parameter-free lean** (0.53): the lean already *is* the
  calibrated directional signal; adding features fit the weaker train-period relationship and generalized
  worse. (It did find that a resolution cue in the news amplifies the lean — `lean×cue` was its largest
  weight — a real driver interaction, just not enough to beat the raw lean.)

## The honest boundary (this is the key nuance)
This is **directional accuracy** (which way), not a **point-forecast edge** over the martingale (how far).
A calibrated belief already *is* its expected value, so you cannot beat the market's *probability* on
liquid efficient markets — EXP-033/035 stand on that. What EXP-036 adds is that the *direction* implied by
that probability is correct at the calibration rate, which is well above a coin flip and is genuinely
useful.

## What this means for "can we predict direction for any question?" (answering the real question)
Yes — and the mechanism is now clear, with three tiers:
1. **Question WITH a liquid market:** the lean is free (the price), giving 0.6–0.8 directional accuracy
   immediately. You cannot beat the price's probability, but you can read its direction.
2. **Question WITHOUT a market (the general case):** you must *infer the lean* — the belief/probability —
   from the driver variables. This is the VariableMap applied to the question: infer P(outcome) from the
   drivers, and the direction follows at the inferred calibration. This is the front-door build.
3. **To EXCEED market-level direction:** you need information the crowd lacks or prices slowly — richer/
   faster driver variables. This only pays off on **inefficient** questions (slow social attitudes, niche
   topics with no market), which is exactly where a general social world model is differentiated. On
   efficient markets it is provably impossible.

So "model the content of future events" is more inferrable than the martingale framing implied — not
because you predict a coin flip, but because most events resolve toward forecastable underlying drivers,
and the *lean those drivers imply* gives the direction. The residual you truly can't predict is the
surprise component; the better you model the drivers, the smaller it gets.

## Honest limits
- Directional accuracy varies by regime (0.61 train / 0.80 test) — the exact number is unstable; the
  robust claim is "clearly > 0.5," not "0.80."
- It is directional only; no point-edge over efficient markets (by construction).
- The lean must be *known or inferred*. On markets it is given; for arbitrary questions, inferring a
  calibrated lean from drivers is the open build (ROADMAP stage B/D) — and its quality is the whole game.

## Reproduce
`python -m experiments.exp036_driver_model`. `python -m pytest tests/test_direction_model.py`.
