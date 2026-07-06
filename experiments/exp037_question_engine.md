# EXP-037 — The question→driver→inferred-lean engine (the front door to "any question")

Applies the VariableMap architecture to a QUESTION: infer the DRIVERS acting on its resolution
(base-rate first, Fermi-decomposed, balanced), aggregate them into P(outcome) in log-odds (Tetlock's
incremental Bayesian update), and read off the direction (EXP-036). For a question with **no market**,
this inferred lean IS the forecast — the concrete meaning of "map all the variables acting on a question
and roll forward."

## Setup (no-cheat, leakage-guarded)
An 8-agent swarm inferred each of 160 Kalshi questions' drivers from the **question + as-of news only** —
it was **not** given the market price or the future. `QuestionEngine` aggregates them; `kappa` and the
anti-overreaction `evidence_shrink` are tuned on one half of the questions, evaluated on the other.

## Result (80 held-out questions)
**The front door works as a mechanism:**
- driver-inferred P vs the **market probability the agent never saw**: correlation **0.63**, MAE 0.17;
- inferred lean vs the market's direction: **0.83** directional accuracy;
- inferred lean vs the eventual resolution: 0.81 (leakage-caveated — see below).

So mapping a question to drivers/world-knowledge recovers a **market-consistent probability** and calls
direction well, *without* reading the market. That is the "any question → infer P(outcome)" capability.

## The honest findings
1. **The structured drivers add nothing over the base rate** (Δcorrelation −0.002; the tuner drove
   `evidence_shrink` down to dampen them). The LLM's holistic **reference-class base rate** alone carries
   the accuracy; the Fermi driver decomposition's value here is **interpretability and auditability**
   (you get the breakdown of which drivers and why), not a better number. On these questions, with this
   log-linear pool, explicit drivers are a transparency layer, not an accuracy layer.
2. **Leakage caveat (the Halawi lesson).** The agents' knowledge cutoff covers most of these dated
   questions, so the 0.81 accuracy-vs-eventual-outcome is partly recall, not forecasting. The cleaner
   signal is the **market-consistency** result (0.63 corr / 0.83 direction) — the agent recovered a
   probability for a price it was never shown — but even that could be aided by outcome recall. A
   rigorous forecasting-skill number requires **questions resolving after the model's cutoff** (or live
   tracking), which this benchmark cannot provide.

## What it means for the architecture
- **"Maps zero world variables" is now addressed structurally**: the engine explicitly maps a question's
  driver variables (not just the price's shape). But EXP-037 is honest that, *for accuracy*, the LLM's
  holistic judgment dominates the explicit decomposition — the drivers earn their place on
  **interpretability** (auditable "why"), which is exactly what decisions need, not on raw score.
- **The front door exists**: `QuestionEngine.forecast(question, driver_infer_fn)` turns any natural-
  language question into a calibrated-ish P(outcome) + direction + an auditable driver breakdown —
  usable for the vast space of questions with no liquid market, where there is no price to read.

## Honest limits
- Accuracy validated only as market-consistency (0.63) + leakage-caveated direction (0.81); true
  out-of-sample skill needs post-cutoff questions.
- Drivers don't beat the base rate here; whether richer/retrieved drivers (real fundamentals, not
  LLM-recalled) would add is the open question — and is where inefficient/no-market domains should differ.
- One inference pass per question (dragonfly n_views=1 in this run); multiple independent passes would
  reduce single-view bias.

## Reproduce
`python -m experiments.exp037_question_engine` (committed driver inferences + held-out truth under
`experiments/results/exp037_qe/`). `python -m pytest tests/test_question_engine.py`.
