# EXP-040 — Does grounded-variable simulation beat the composite? (the north-star test)

This is the experiment that tests the project's actual thesis — not "predict the market price," but
**map each individual's real variables, simulate their decision, aggregate to the outcome, and beat the
crowd composite.** It runs on a real social outcome (how a population answers an opinion question) with
real per-person variables (actual demographics + attitudes), leakage-free.

## The question
For a population answering a question, is it better to
- model the **aggregate** directly — the question's marginal answer rate, "the average person" (the analog
  of reading a market's single number); or to
- **ground and simulate** — map each person's real variables (party, ideology, religion, age, income, …)
  and simulate their individual answer, then aggregate?

And the two sharper sub-questions the north star hinges on:
- **Q2** Does mapping *more* real variables monotonically improve the simulation? ("map ALL the variables")
- **Q3** Does grounded bottom-up composition beat the top-down aggregate where heterogeneity matters
  (distinctive subgroups)?

## Setup (no-cheat)
6,000 respondents split train/test by a stable hash; a shrinkage-naive-Bayes per-person model
`P(answer | real variables)` is fit on **train** respondents only, and **test** respondents' answers are
never seen when predicting. The shrinkage strength is tuned on a **train-internal** hold-out (never
touches test). 353 questions, 6,698 held-out answers.

## Result — the thesis holds, with one crucial caveat

**Q1/Q2 — individual simulation quality by variable richness (tuned shrinkage α=80):**

| variables mapped | log-loss ↓ | accuracy ↑ |
|---|---|---|
| marginal (composite, 0 vars) | 0.6117 | 0.656 |
| + party (1) | 0.5976 | 0.675 |
| + party, ideology, religion (3) | 0.5861 | 0.686 |
| **all 11 real variables** | **0.5854** | **0.690** |

**Grounding each person in their real variables beats the crowd composite** (log-loss 0.612 → 0.585,
accuracy 0.656 → 0.690), and — properly regularized — **more variables help monotonically.** Simulating
the individual from who they actually are is a better model of the outcome than the aggregate number.

**Q3 — aggregate subgroup share, bottom-up grounded vs top-down composite (TV, lower better):**

| | top-down composite | bottom-up grounded |
|---|---|---|
| all subgroups (n=1,184) | 0.1527 | 0.1485 |
| **distinctive subgroups (n=692)** | **0.2253** | **0.1718** |

On subgroups whose true opinion is far from the population average — exactly where *who is in the group*
should matter — composing the outcome from simulated individuals beats the aggregate by **24%**. This
replicates EXP-034's bottom-up win and strengthens it with real per-person variables (not
value-similarity).

## The crucial caveat (and the real lesson for the north star)
**"Map more variables" is only better if you can *estimate* their joint effect.** The *naive* all-11
model (light smoothing) was *worse* than one variable — log-loss 0.92 vs 0.60 — because naive Bayes
**double-counts correlated variables** (party and ideology are collinear: `(democrat, liberal)`,
`(republican, conservative)` dominate) and overfits ~38 train respondents per question. The shrinkage
sweep on train-val makes this explicit:

| shrinkage α | 4 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|
| all-11 log-loss | 0.92 | 0.77 | 0.69 | 0.63 | **0.60** |

Heavier regularization recovers the gain. **The binding constraint is variable *estimation quality*
(handling correlation and limited per-cell data), not variable *count*.** Enumerating more variables is
free; estimating their joint effect without double-counting is the actual engineering frontier. A
simulator that naively piles on variables is *confidently wrong*; one that pools/regularizes turns each
real variable into signal.

## What this means for the project (the honest north-star read)
- **The core thesis is validated on real data:** grounded simulation of individuals from their real
  variables beats the crowd composite, and beats it most where heterogeneity matters. This is the
  differentiated capability — computing an outcome from its constituents in a way a single aggregate
  number cannot — and it is real, not a market-price heuristic.
- **The frontier is estimation, not enumeration.** The path to beating the crowd is *better-estimated*
  variables (real grounded values, partial pooling across questions/people, correlation-aware readouts),
  not longer variable lists. That reframes the roadmap: invest in grounding variables in real data and in
  the estimator, not in adding nominal variables.
- **This is the bottom-up simulate-and-aggregate loop on a real outcome** — the assembled version of the
  machinery (VariableMap → per-person simulation → aggregation) that EXP-034 validated in part.

## Honest limits
- The "real variables" here are 11 survey demographics/attitudes — grounded and real, but a narrow slice
  of a full VariableMap; the win should widen with richer per-person grounding (text/history personas).
- Cross-sectional (one opinion snapshot); coupling the grounded population to the event-transition
  operator (EXP-030/032) for a *forward* simulation of opinion *change* is the next build.
- The outcome is an opinion share, the cleanest real social outcome we can check at scale; validating on
  discrete future events (elections, votes) needs a dataset pairing outcomes with their deciding agents.

## Reproduce
`python -m experiments.exp040_grounded_simulation` → `experiments/results/exp040_grounded_simulation.json`
(uses the committed OpinionQA parsed cache). `python -m pytest tests/test_grounded_simulation.py`.
