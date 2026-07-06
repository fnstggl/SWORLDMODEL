# Superforecasting tenets in the world-model architecture

Tetlock's *Superforecasting* distills what makes forecasters accurate. Most of its principles are
structural choices we can bake into the model rather than hope for. This maps each of the "ten
commandments" to where it lives (or should live) in the architecture — several are already implemented,
a few are the honest next additions.

| # | Tetlock commandment | Where it lives here | Status |
|---|---|---|---|
| 1 | **Triage** — spend effort where it pays (the Goldilocks zone; skip the hopeless and the trivial) | The `simulate()` **abstention + confidence** layer (EXP-024) declines out-of-envelope queries; a dedicated **forecastability score** (is this in the predictable band?) is the clean addition | ◑ partial |
| 2 | **Fermi-ize** — decompose a hard question into tractable sub-questions | The **driver decomposition** in `QuestionEngine` (EXP-037); the **VariableMap** decomposes a person into variables | ✅ |
| 3 | **Outside then inside view** — start from the base rate, then adjust for specifics | `QuestionEngine` starts from `base_rate` (reference class) and adjusts via drivers; hierarchical **partial pooling** shrinks entity estimates toward the population prior | ✅ |
| 4 | **Update incrementally** — small Bayesian steps, don't over/under-react | Log-odds accumulation + the **`evidence_shrink`** anti-overreaction term (EXP-037); recency-weighted **EWMA** entity state | ✅ |
| 5 | **Clashing causal forces** — weigh forces pushing both ways (actively open-minded) | Drivers carry **signed** direction; the inference prompt requires **balanced** YES- and NO-pushing drivers | ✅ |
| 6 | **Granular probabilities** — distinguish degrees of doubt, use fine-grained odds | Calibrated continuous probabilities everywhere; **conformal** prediction sets and CRPS-scored distributions (EXP-024/035) | ✅ |
| 7 | **Balance under/overconfidence** — calibration vs resolution | **Calibration grade** (ECE) on every prediction, σ-calibration of the rollout band, conformal coverage (EXP-035) | ✅ |
| 8 | **Post-mortems** — learn from a scored track record | Strict **no-cheat backtests** on every experiment, and we log **honest negatives** (EXP-019/029/031/033) — a live self-recalibrating track record is the next step | ◑ partial |
| 9 | **Perpetual beta / dragonfly eye** — aggregate many independent views, keep iterating | **Agent swarms** + `forecast_from_views` dragonfly median (EXP-037); iterate-to-diminishing-returns discipline (EXP-035) | ✅ |
| 10 | **Own the cycle** — treat forecasting as a skill to be trained, not a talent | The whole iteration loop: measure → diagnose → improve → re-measure | ✅ |

## The additions worth making explicit (partial → full)

1. **Triage / forecastability score (commandment 1).** EXP-033/036 taught us the ceiling is
   information-bounded: efficient-market points are unbeatable; direction is callable via the lean;
   inefficient/no-market questions are where skill exists. A `forecastability(question)` score — combining
   the confidence layer, the market-efficiency of the source, and driver coverage — would let the system
   *say what it can and can't usefully forecast*, and spend its effort accordingly. This is the single
   most Tetlock-aligned addition still missing.

2. **Live post-mortem loop (commandment 8).** We backtest, but a deployed engine should log each forecast,
   score it as questions resolve, and **recalibrate** `kappa`/`evidence_shrink`/σ from its own track
   record — perpetual beta made mechanical. (Also the only clean way past the leakage ceiling: forecasts
   made *before* resolution, scored *after*, are contamination-free by construction.)

## The one place Tetlock and our evidence disagree — and it's instructive
Superforecasters believe structured decomposition beats holistic judgment. EXP-037 found the opposite on
market questions: the LLM's holistic **base rate** carried the accuracy and the explicit **drivers added
nothing** to the number (they added *interpretability*). The reconciliation: the base rate a strong LLM
produces already *is* a decomposition-informed judgment (it has read the reference class and the news),
so the extra explicit decomposition is redundant for accuracy but valuable for **auditability** — which
is what a *decision-support* system needs. We keep the decomposition for the "why," and trust the
holistic base rate for the "what."
