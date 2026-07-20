# EXP-107 — the 25-question BTF-3 stack-up (all fixes, mean-of-3, numeric-actor fidelity)

The honest broad-sample verdict, on identical questions.

## Scores (25 BTF-3 binary questions, base rate 40% YES)

| forecaster | Brier ↓ | acc@0.5 | AUC | mean-p on YES / NO |
|---|---|---|---|---|
| **constant base-rate (0.40)** | **0.240** | — | — | — |
| constant 0.5 | 0.250 | — | — | — |
| **FutureSearch SOTA** (n=22) | **0.176** | 0.77 | **0.79** | — |
| thin mechanism kernel (EXP-101 style) | 0.352 | 0.58 | 0.521 | 0.288 / 0.274 |
| **rich WMv2, all fixes, mean-of-3** | **0.310** | 0.48 | **0.413** | 0.444 / 0.513 |

## The brutal findings (no spin)

1. **Both our approaches LOSE to a constant.** Predicting the base rate (0.40) on every question scores
   Brier 0.240; the rich pipeline scores 0.310 and the thin kernel 0.352. The machinery is net-negative
   versus a single number.
2. **The rich pipeline is ANTI-DISCRIMINATIVE (AUC 0.413 < 0.5).** mean-p on YES outcomes (0.444) is LOWER
   than on NO outcomes (0.513) — it is slightly more confident in the WRONG direction. The thin kernel is
   barely-positive (0.521); the rich machinery actively HURTS discrimination relative to the thin kernel.
3. **SOTA crushes both** (0.176 / AUC 0.79). On "will specific event X happen by deadline" questions — which
   dominate BTF and resolve NO ~60% of the time — SOTA assigns 0.03–0.05; we assign 0.4–0.85. Examples we
   got confidently wrong: Knesset dissolved 0.85 (NO), DOJ lawsuit 0.75 (NO), Google Veo 0.63 (NO),
   Sonko PM 0.59 (NO).
4. **Variance is still large after mean-of-3**: mean per-question spread 0.30 — mean-of-3 is not enough;
   mean-of-8–10 would be needed to stabilise, at 3× the cost.

## Diagnosis: over-prediction of occurrence on a NO-heavy distribution

The single systematic error is over-predicting that specific events happen by their deadline. The grounded
LLM prior I added ELICITS a base rate, but the LLM is optimistic about occurrence; the rich structure then
adds confidence in that wrong direction. The component wins that motivated this wave (BoJ 0.057→0.73 in one
draw, visionOS 0.49→0.56) were real but did NOT generalise — they were the minority of questions where the
LLM's optimism happened to be right (recurrences), and they are swamped by the occurrence questions where it
is wrong.

This reproduces, on a clean 25-question benchmark, the finding the project's own backtests keep reaching
(BACKTEST_FINDINGS, SUPERFORECASTING.md, EXP-089): **the elaborate world-model machinery does not beat
simple base-rate discipline on general forecasting; it is worse.** The lever is NOT more structure. It is
outside-view / status-quo / timeframe-decay discipline — exactly what FutureSearch's simple-prompt
forecaster and the superforecasting literature encode, and what SOTA's 0.03–0.05 on occurrence questions
demonstrates.

## Honest implications

- The fixes this session are real engineering improvements (evidence now flows and is targeted; the prior is
  grounded; empty rollouts are structurally impossible; deprecated paths quarantined; variance is measured
  and reducible). But **on end-to-end BTF accuracy the rich pipeline is worse than a constant and worse than
  the thin kernel** — the machinery's value is auditability / structure / counterfactuals, not calibrated
  point forecasts on general questions.
- The next lever is not another structural fix. It is **base-rate discipline for the occurrence class**:
  "will specific event X happen by deadline D" must default LOW (most don't), overridden only by grounded
  evidence of imminence — the arrival-kernel discipline the thin path has and the rich path dilutes.
- Caveats kept honest: n=25; numeric-actor fidelity (full-LLM actors excluded, but they hurt in EXP-102 and
  are not the fix); mean-of-3 leaves ~0.30 spread; one thin-kernel question errored (24/25). None of these
  caveats rescue the headline — losing to a constant is not a small-sample artifact at this margin.
