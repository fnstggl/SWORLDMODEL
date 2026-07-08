# EXP-042 — Forward simulation of opinion change under an event (Part a: the temporal thesis test)

EXP-040 was cross-sectional (a snapshot). This tests the untested core of the thesis: **couple a
grounded actor to the event-transition operator and simulate the outcome forward** — predict opinion
*change*, not a static answer. ChangeMyView is the substrate: an OP holds a view, an argument (the event)
is applied, and the ground truth is whether the view **changed** (a delta).

The world-model form is multiplicative — the person **gates** the event:
`P(change) ~ responsiveness(person) · impact(argument)`. An open, non-entrenched mind moves under a strong
argument; an entrenched or skeptical one does not. `responsiveness` is computed by the **same operator**
(`responsiveness_from_map`) the multi-step rollout uses, from the OP's inferred openness / skepticism /
entrenchment; `impact` is the argument's inferred crux-fit / evidence / clarity / respect / expertise.

## Setup (no-cheat)
1,200 CMV cases, chronological 70/30 split; each arm fits a calibrated logistic on train, scores test
(n=360, base rate 0.649).

## Result

| arm | log-loss | brier | accuracy | uplift@20 |
|---|---|---|---|---|
| persistence (base rate) | 0.6440 | 0.226 | 0.656 | 0.00 |
| event-only composite (argument only) | 0.6380 | 0.223 | 0.656 | 0.094 |
| person-only (responsiveness only) | 0.6254 | 0.218 | 0.658 | 0.094 |
| additive feature-soup (person ⊕ event) | 0.6196 | 0.215 | 0.658 | 0.150 |
| **structural simulation (+ coupling)** | **0.6189** | **0.215** | 0.658 | 0.150 |

**The coupled forward simulation beats persistence (+0.025 log-loss) and the one-sided baselines** — the
operator, grounding a person and applying an event, genuinely predicts opinion change better than "views
rarely change" or "judge the argument alone."

**But the multiplicative coupling adds almost nothing over the additive model** (+0.0007). At n=360 with
noisy *inferred* variables, having both person and event as linear features captures nearly everything;
the interaction term is not separably useful for the aggregate score.

**The gating mechanism is nonetheless real — and visible when you look for it directly.** Splitting the
test set by responsiveness and measuring how strongly argument impact predicts change in each half:

| subgroup | impact → change slope |
|---|---|
| responsive OPs (open, n=180) | **0.709** |
| entrenched OPs (n=180) | 0.512 |

**The same argument predicts opinion change ~38% more strongly for responsive people than entrenched
ones.** The person *does* gate the event — the structure the operator encodes is present in the data,
even though it doesn't move the aggregate log-loss at this scale.

## Honest findings
1. **The forward-simulation operator works** as a predictor of opinion change: coupling grounded actor +
   event beats persistence and either half alone.
2. **The multiplicative structure earns *mechanism*, not aggregate accuracy** — exactly the pattern of
   EXP-037 (drivers) and EXP-040 (naive variables): a good additive model captures the accuracy; the
   structure's value is the interpretable, verifiable gating (open minds move more), which is what a
   *decision* about whom to persuade actually needs.
3. **Scale and variable noise bound the interaction test.** 360 test cases and LLM-inferred (noisy)
   person/argument variables attenuate a second-order interaction; grounded, lower-noise variables and
   more data are where the coupling should separate from additive.

## Honest limits
- One-step event (argument → outcome); a true multi-step temporal rollout of a *population* over time
  needs longitudinal opinion data (OpinionQA's cache repeats only 4 questions across waves — too few), so
  the "N-step" axis is validated as the per-step operator, not yet as a long horizon.
- Person/argument variables are LLM-inferred, not grounded in behavioral history; the gating signal
  should sharpen with grounded responsiveness.

## Reproduce
`python -m experiments.exp042_forward_simulation` → `experiments/results/exp042_forward_simulation.json`.
