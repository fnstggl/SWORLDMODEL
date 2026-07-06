# EXP-053 — Mean-field coupling vs the independent mean (the audit's decisive test)

The simulation audit's #1 mandate: `simulate_population` is `sum(ps)/n` — a mean of independent per-person
readouts (`∂pᵢ/∂pⱼ = 0`). `swm/simulation/mean_field.py` makes aggregation **non-separable** — each agent's
belief evolves toward the population's *current aggregate* (a mean-field coupling term) over steps, so agent
i's outcome depends on agent j through the aggregate. It earns the word "simulate" only if the coupled
roll-forward does something a mean of independent predictions cannot. Tested honestly on two axes.

## A. Controlled cascade — does coupling recover emergence a mean cannot?
A Granovetter threshold cascade (agents adopt when the adoption rate passes their personal threshold) has a
genuine **S-curve trajectory**. A mean of independent predictions is a flat line.

| | trajectory MAE ↓ |
|---|---|
| independent mean (flat) | 0.6013 |
| **coupled mean-field (bandwagon)** | **0.0145** |

```
true:    0.085 → 0.128 → 0.195 → 0.292 → 0.468 → 0.755 → 1.0 → 1.0 …
coupled: 0.085 → 0.128 → 0.191 → 0.287 → 0.430 → 0.645 → 0.968 → 1.0 …
```
**The coupled model recovers the emergent S-curve; the independent mean cannot represent it at all.** This
is genuine simulation of emergence — the output is provably not reproducible by averaging independent
per-agent predictions, because the trajectory's acceleration comes from the feedback (`∂pᵢ/∂pⱼ ≠ 0`).

## B. Real GSS aggregate — does coupling beat the independent mean on the true share?
Holding the per-agent beliefs fixed (population-weighted, so the independent mean is the fair EXP-045-style
baseline), does rolling the coupled dynamics forward improve the predicted share?

| method | MAE ↓ |
|---|---|
| persistence | **0.0280** |
| independent mean (the current flagship) | 0.0634 |
| coupled roll-forward | 0.0681 |

**No.** The coupled aggregation does **not** beat the independent mean, and neither beats persistence at
these horizons. Conformity dynamics slightly *overshoot* the calibrated cross-sectional mean.

## The honest verdict (exactly what the audit set up)
- **Coupling earns "simulate" for DYNAMICS / EMERGENCE, not for the marginal number.** Where the outcome is
  genuinely emergent from interaction (cascades, tipping, S-curves), the coupled model captures a shape a
  composite cannot. Where the outcome is a well-calibrated marginal (a population's opinion share),
  **compositing suffices and coupling adds nothing** — a mean of independent readouts is already near the
  ceiling (EXP-050 MAE 0.0045), and adding interaction only overshoots.
- This is the audit's predicted fork, now *measured*: the honest claim is **"a coupled simulator for
  emergent/interaction-driven outcomes; a calibrated compositor for marginal opinion shares."** The system
  should route to coupling only where interaction genuinely drives the outcome — and we now have the module
  and the test to tell the two regimes apart, rather than overselling one number as "simulation."

## Honest limits
- The cascade is a controlled process (the mechanism validation); real interaction-driven social outcomes
  with ground-truth trajectories are the missing dataset to test coupling *in the wild* (GSS opinion is
  marginal-dominated, so it is the wrong substrate to reward coupling).
- The GSS coupled model used fixed coupling constants; tuning them leakage-free would not change the
  qualitative verdict (conformity is mean-preserving/overshooting on calibrated shares).

## Reproduce
`python -m experiments.exp053_mean_field` → `experiments/results/exp053_mean_field.json`.
`python -m pytest tests/test_mean_field.py`.
