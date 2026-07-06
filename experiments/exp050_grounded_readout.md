# EXP-050 — The unified GroundedReadout + the assembled end-to-end pipeline

The estimation-frontier program had three separate wins — latent factors (EXP-048), LLM world-knowledge
priors (EXP-049), and reliability weighting — each fixing part of the bottleneck. This composes them into
one estimator (`swm/variables/grounded_readout.py`) and wires it into an assembled end-to-end simulator
(`swm/api/grounded_simulate.py`) that finally runs the whole pipeline on a real question.

## The unification (and why the pieces compose cleanly)
- **Structure**: decorrelate the correlated variables into orthogonal latent value factors.
- **World-knowledge prior**: the LLM prior is a coefficient vector in one-hot space; the factors are an
  orthonormal basis V, so the prior projects **exactly** into factor space as `Vᵀ·prior` — the two
  compose without approximation.
- **Reliability weighting**: scale each variable's features by its provenance reliability (data/user=1.0,
  llm=0.55, heuristic=0.3), so a noisy inferred variable's effect is attenuated errors-in-variables style
  — real variables dominate, inferred ones contribute in proportion to trust.
- Plus EXP-041 n-adaptive pooling toward the marginal.

Because the pieces help *conditionally* (factors help low-rank signal; the prior helps thin data), a fixed
recipe can regress — so `fit_auto` **self-configures**, picking the winning combination on a train-internal
hold-out (never worse than its best component, no test leakage).

## Result (GSS, no-cheat)

**A. Compounding — individual prediction at a data-poor N=150 (where estimation quality separates):**

| estimator | log-loss ↓ | accuracy ↑ |
|---|---|---|
| plain (no factors, no prior) | 0.6183 | 0.677 |
| factors only (EXP-048) | 0.6084 | 0.682 |
| prior only (EXP-049) | 0.6134 | 0.682 |
| **GROUNDED (auto-config: factors+prior)** | **0.6070** | **0.683** |

**The unified readout compounds** — it beats *both* single pieces and plain (0.607 vs the best single
0.6084, −1.8% vs plain). `fit_auto` chose `{factors: on, prior: on}` from the train-internal hold-out, so
the win is self-selected, not hand-picked.

**B. Reliability weighting — real + injected noisy-inferred variables:**

| | log-loss ↓ |
|---|---|
| uniform weighting | 0.6111 |
| **reliability weighting** | **0.6078** |

Down-weighting the noisy inferred variables (provenance "llm", 50% corrupted) recovers accuracy that
uniform weighting loses to the noise — the mechanism does what it should: trust grounded variables more.

**C. End-to-end — the assembled pipeline on real questions:** `GroundedSimulator` takes a question + a
held-out population, simulates each person, and aggregates bottom-up to a calibrated support share:

- **MAE vs the true held-out share: 0.0045** over 15 questions.
- Worked example — *"Should marijuana be legal?"* (`grass`): predicted **0.336** vs true **0.337**
  (n≈11,900), with an auditable value-factor breakdown of which axes drove it.

The simulate-the-event pipeline — grounded variables → unified structured/primed estimation → bottom-up
aggregation → calibrated outcome + value decomposition — now runs **end-to-end on a real question**, and
it is well-calibrated.

## The honest findings
1. **Compounding is real but conditional, and the self-configuring readout is what makes it safe.** A
   *fixed* factors+prior recipe regressed at N=800 (the k=3 compression hurt where the signal wasn't
   low-rank); `fit_auto` fixes this by selecting the helpful pieces per dataset, so GROUNDED is never worse
   than its best component and compounds when the pieces genuinely combine. The lesson: there is no free
   lunch across regimes — the right estimator is data-dependent, and the honest fix is to *choose*, not to
   stack blindly.
2. **Reliability weighting is the concrete answer to "the variables aren't real":** it lets grounded and
   inferred variables coexist with the estimate trusting each in proportion to its provenance.
3. **The pipeline is assembled.** For the population/opinion domain, "map grounded variables → simulate →
   aggregate → calibrated outcome" is now one callable, validated end-to-end (MAE 0.0045) — no longer a
   shelf of separate experiments.

## Honest limits
- Validated on GSS demographics; the larger test is the *full* VariableMap (grounded + inferred together),
  where reliability weighting should matter far more than the injected-noise proxy here.
- `simulate_population` needs the question to be one the readout was trained on (for its marginal +
  calibration); a truly novel NL question still routes through the `QuestionEngine`/`semantic_stance`
  front door — wiring those into `GroundedSimulator` is the next integration.
- The compounding margin is modest (−1.8% vs plain); it should widen where variables are more numerous and
  more correlated than 10 demographics.

## Reproduce
`python -m experiments.exp050_grounded_readout` → `experiments/results/exp050_grounded_readout.json`
(deterministic splits via crc32). `python -m pytest tests/test_grounded_readout.py tests/test_grounded_simulate.py`.
