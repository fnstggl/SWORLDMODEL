# Superforecasting tenets in the world-model architecture

Tetlock's *Superforecasting* distills what makes forecasters accurate. Most of its principles are
structural choices we can bake into the model rather than hope for. This maps each of the "ten
commandments" to where it lives (or should live) in the architecture — several are already implemented,
a few are the honest next additions.

| # | Tetlock commandment | Where it lives here | Status |
|---|---|---|---|
| 1 | **Triage** — spend effort where it pays (the Goldilocks zone; skip the hopeless and the trivial) | The `simulate()` **abstention + confidence** layer (EXP-024) declines out-of-envelope *inputs*; the learned **`ForecastabilityScorer`** (EXP-038) declines low-signal *questions* — a calibrated "can we call this?" score → FORECAST/HEDGE/ABSTAIN, validated by selective forecasting (Q1 0.85 → Q3 1.00) | ✅ |
| 2 | **Fermi-ize** — decompose a hard question into tractable sub-questions | The **driver decomposition** in `QuestionEngine` (EXP-037); the **VariableMap** decomposes a person into variables | ✅ |
| 3 | **Outside then inside view** — start from the base rate, then adjust for specifics | `QuestionEngine` starts from `base_rate` (reference class) and adjusts via drivers; hierarchical **partial pooling** shrinks entity estimates toward the population prior | ✅ |
| 4 | **Update incrementally** — small Bayesian steps, don't over/under-react | Log-odds accumulation + the **`evidence_shrink`** anti-overreaction term (EXP-037); recency-weighted **EWMA** entity state | ✅ |
| 5 | **Clashing causal forces** — weigh forces pushing both ways (actively open-minded) | Drivers carry **signed** direction; the inference prompt requires **balanced** YES- and NO-pushing drivers | ✅ |
| 6 | **Granular probabilities** — distinguish degrees of doubt, use fine-grained odds | Calibrated continuous probabilities everywhere; **conformal** prediction sets and CRPS-scored distributions (EXP-024/035) | ✅ |
| 7 | **Balance under/overconfidence** — calibration vs resolution | **Calibration grade** (ECE) on every prediction, σ-calibration of the rollout band, conformal coverage (EXP-035) | ✅ |
| 8 | **Post-mortems** — learn from a scored track record | Strict **no-cheat backtests** + logged **honest negatives**, now with a live **`PostMortemLog`** (EXP-039): forecasts logged before resolution + scored after = a **structurally leakage-free** skill number, and a **do-no-harm self-recalibrator** (deploys a Platt map only on a trusted held-out improvement) | ✅ |
| 9 | **Perpetual beta / dragonfly eye** — aggregate many independent views, keep iterating | **Agent swarms** + `forecast_from_views` dragonfly median (EXP-037); iterate-to-diminishing-returns discipline (EXP-035) | ✅ |
| 10 | **Own the cycle** — treat forecasting as a skill to be trained, not a talent | The whole iteration loop: measure → diagnose → improve → re-measure | ✅ |

## The additions now implemented (were the two partial → full gaps)

1. **Triage / forecastability score (commandment 1) — DONE (EXP-038).** EXP-033/036 taught us the ceiling
   is information-bounded: efficient-market points are unbeatable; direction is callable via the lean;
   inefficient/no-market questions are where skill exists. `ForecastabilityScorer` learns
   `P(call is correct | as-of features)` and drives FORECAST/HEDGE/ABSTAIN triage; selective forecasting
   confirms it separates reliable from unreliable questions (Q1 0.85 → Q3 1.00 resolution accuracy). The
   system now *says what it can and can't usefully forecast* and spends effort accordingly.

2. **Live post-mortem loop (commandment 8) — DONE (EXP-039).** `PostMortemLog` logs each forecast, scores
   it as questions resolve, and **self-recalibrates** from its own track record. Two wins: (a) forecasts
   made *before* resolution and scored *after* give a **structurally leakage-free** skill number — the
   clean way past the leakage ceiling that caveats EXP-037; (b) the recalibrator is **do-no-harm** — it
   deploys a Platt map only when a held-out slice of past-resolved forecasts shows a *meaningful* gain,
   abstaining on thin or already-calibrated history, so perpetual beta can never make a good forecaster
   worse. Validated: on a stationary miscalibrated stream it recovers ECE 0.14 → 0.08; on the thin real
   track record it correctly abstains.

## The one place Tetlock and our evidence disagree — and it's instructive
Superforecasters believe structured decomposition beats holistic judgment. EXP-037 found the opposite on
market questions: the LLM's holistic **base rate** carried the accuracy and the explicit **drivers added
nothing** to the number (they added *interpretability*). The reconciliation: the base rate a strong LLM
produces already *is* a decomposition-informed judgment (it has read the reference class and the news),
so the extra explicit decomposition is redundant for accuracy but valuable for **auditability** — which
is what a *decision-support* system needs. We keep the decomposition for the "why," and trust the
holistic base rate for the "what."
