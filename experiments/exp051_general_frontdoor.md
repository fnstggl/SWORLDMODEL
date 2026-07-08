# EXP-051 — The general front door: any question flows all the way through

`GeneralSimulator.answer(...)` is the assembled front door — one call that ROUTES a question to the
machinery carrying its signal and FUSES whatever evidence exists (population, news, drivers) in log-odds:
- POPULATION question with a modelled population → `GroundedSimulator` bottom-up aggregation (EXP-050);
- NOVEL question with as-of news → `SemanticStanceJudge` reading for THIS resolution (EXP-047);
- NOVEL question with world knowledge → `QuestionEngine` driver aggregation (EXP-037).

## Result (end-to-end, no-cheat)
**A. Population questions** (routed → bottom-up simulation), scored vs the true held-out GSS share:

| question | predicted | true share |
|---|---|---|
| Should marijuana be legal? | 0.336 | 0.337 |
| Favor the death penalty? | 0.692 | 0.687 |
| Is homosexuality wrong? | 0.572 | 0.576 |
| Abortion legal for any reason? | 0.435 | 0.429 |
| Spend more on the environment? | 0.623 | 0.618 |

**MAE 0.0043** — the population path is well-calibrated end-to-end.

**B. Market questions** (routed → semantic news reading), scored vs the as-of price the judge never saw:
**correlation 0.51** with the price (recovering EXP-047's directional signal through the front door).

## The honest finding
- **The routing works**: one `answer()` sends population questions to bottom-up simulation and market
  questions to grounded news reading, and both produce forecasts end-to-end. That is the "any question
  flows all the way through" assembly.
- **The market path recovers DIRECTION, not LEVEL.** MAE-vs-price is 0.25 because the news stance is
  anchored at base_rate 0.5, while these Kalshi events live near 0.1 — the semantic reading tells you
  *which way*, not the absolute probability, without a reference-class base rate the stance judge does not
  supply. The correlation (0.51) is the honest signal; the level needs the `QuestionEngine` base rate or a
  calibration step wired in. Reported honestly rather than as a flattering MAE.
- **Caveat from the simulation audit**: the population path's excellent MAE is partly marginal recovery
  (`GroundedReadout` shrinks toward the per-question training marginal), so this validates *routing +
  calibration*, not *simulation fidelity* — which needs the interventional / skill-vs-persistence KPIs
  (see `SIMULATION_AUDIT.md`).

## Reproduce
`python -m experiments.exp051_general_frontdoor` → `experiments/results/exp051_general_frontdoor.json`.
`python -m pytest tests/test_general_simulate.py`.
