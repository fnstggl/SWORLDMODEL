# EXP-052 — The unified readout on a real full VariableMap: reliability ≠ relevance

EXP-050 validated reliability weighting with *injected* noise (redundant noisy copies of grounded
variables). This tests it on a real mixed-provenance VariableMap — ChangeMyView persuasion, where the map
mixes 8 **LLM-inferred** persuasion variables (openness, crux-fit, evidence, …; provenance "llm") with 5
**grounded** surface variables (message length, links, quotes; provenance "data"/"heuristic").

## Result (CMV, no-cheat temporal split, n=1200)

| estimator | log-loss ↓ | accuracy | uplift@20 |
|---|---|---|---|
| uniform (logistic learns relevance) | 0.6005 | 0.667 | 0.206 |
| reliability-scaled features | 0.6005 | 0.667 | 0.206 |
| reliability soft-prior | 0.6005 | 0.667 | 0.206 |
| decorrelated (latent factors) | 0.6324 | 0.658 | 0.053 |

**Top learned drivers (uniform model) — note the provenance:**

| variable | provenance | weight |
|---|---|---|
| op_openness | **llm** | 0.554 |
| arg_addresses_crux | **llm** | 0.314 |
| op_entrenchment | **llm** | 0.241 |
| arg_expertise | **llm** | −0.214 |
| arg_respectfulness | **llm** | 0.214 |
| op_logwords | data | −0.131 |

## The honest findings — two corrections to EXP-050
1. **Reliability weighting is a no-op under feature standardization.** `LogisticReadout` standardizes
   features (subtract mean, divide by std), so scaling a feature by a constant reliability factor is
   exactly undone — uniform, reliability-scaled, and soft-prior are *identical* (0.6005). Reliability must
   enter the **regularization / prior on the coefficient**, not the feature scale, to have any effect when
   the readout standardizes (in EXP-050 the one-hot features were un-standardized, which is why scaling
   bit there).
2. **Reliability ≠ relevance — and this is the deeper point.** The signal here is carried *entirely by the
   LLM-inferred variables* (op_openness, crux-fit, entrenchment dominate; the grounded surface features
   barely register). Down-weighting inferred variables by provenance — the EXP-050 mechanism — would
   discard the actual signal. The EXP-050 injected-noise proxy was **too kind**: it made inferred variables
   *redundant noise*, the one case where reliability weighting helps; real inferred variables often carry
   the *unique* signal. The correct weighting is **learned relevance** (what the plain logistic already
   does), with reliability as at most a **soft prior the data can override** — never a hard down-weight.
3. **Decorrelation hurts on this map** (0.632 vs 0.600): the persuasion signal is not low-rank (each
   inferred variable carries somewhat independent information), so PCA compression loses it — consistent
   with EXP-050's finding that factors help *conditionally* and the self-configurator would switch them off
   here.

## What this means for the architecture
- **`GroundedReadout`'s reliability weighting should be demoted from a default to a guarded option**: apply
  it only when inferred variables are plausibly redundant with grounded ones, and always let learned
  relevance dominate. The honest general rule is *weight by reliability × learned-relevance*, estimated
  from data, not by provenance alone.
- It also sharpens the simulation audit: even the "grounded variables" story is really *learned-relevance
  regression* — the estimator earns its accuracy by learning which variables matter, grounded or not, not
  by trusting provenance.

## Reproduce
`python -m experiments.exp052_full_variablemap` → `experiments/results/exp052_full_variablemap.json`.
