# Architecture for the peak — what still needs to change

> **STATUS (this session): all six built + validated.** (1) Corpus harvest — **592 elasticities across 15
> outcome-classes from 8 datasets**, committed to `learned_priors.json`, all sign-correct (inflation→hike
> +0.44, unemployment→hike −0.58, ideology=liberal→conservative −0.26, tenure→churn −21) — EXP-076.
> (2) Regime router — routes population/diffusion→rich_sim (0.75–0.83), macro/election/market→baseline
> (≤0.35), learned from the portfolio + a world-knowledge prior — EXP-078. (3) Adaptive fidelity —
> `swm/api/adaptive_fidelity.py` variance-triage (invest calibration only in the high-leverage variables).
> (4) Embedding-keyed registry — `swm/variables/embedding_registry.py` (cross-phrasing transfer, pluggable
> real-embedding backend). (5) Event model — `swm/simulation/event_model.py`, calibrated on FOMC rate jumps:
> **82% interval coverage (nominal 90%) vs persistence's 3%** — EXP-077. (6) Full-covariance weight posterior
> — `bayes_logistic.predict_dist(full_cov=True)`. The sections below are the original design rationale.



*What the general social world model needs to become as accurate as possible — assessed from what the
portfolio backtest (EXP-074) and the calibration work (EXP-072/073) actually measured, not from hope.*

The core loop is now real and, on the right domains, **wins no-cheat**: compile a question → a structural
model whose every variable carries a **calibrated weight-with-uncertainty** → Monte-Carlo integrating both
value and weight uncertainty → a navigable distribution + the best action. On modelable evolving populations
and diffusions it beats the baselines (opinion +0.15, adoption +0.32 skill). Six changes take it to peak.

## 1. Run the calibration HARVEST and persist the registry (the flywheel, turned on)
The registry (`swm/variables/prior_registry.py`) is built and consumed by the compiler
(`calibrated_compiler.apply_registry`), but it starts EMPTY — it is only as good as the elasticities we've
fit into it. **Build a standing job that fits `CalibratedWeights` across every dataset we have (GSS, OQA,
adoption, senate, referenda, FOMC, CMV, …) and commits `learned_priors.json`.** Then every new question's
variables arrive pre-calibrated from real data, and each dataset added tightens the priors for all related
questions. This is the single highest-leverage build: it converts "more data" directly into "more calibrated
default weights everywhere."

## 2. An automatic REGIME ROUTER, learned from the portfolio map
EXP-074 shows fidelity wins in some regimes and not others. The system should **decide per question whether
to run the rich sim, defer to a baseline (persistence/momentum/market), or blend** — a classifier trained on
the portfolio's (features → does-fidelity-beat-baseline) signal. This is the forecastability triage made
data-driven: never lose to a simple baseline by over-simulating, never under-model where fidelity pays.

## 3. Adaptive fidelity IN the loop (variance triage wired to compilation)
The compiler should emit MANY variables, but the runtime should invest precise calibration only where the
outcome is **sensitive** — using the variance decomposition already computed (`variance_contribution`). Wire
it so the compile→run loop auto-selects which variables get data-calibrated weights vs a rough prior, and
prunes near-constant/low-leverage ones. This makes "model everything" tractable at scale: 100 variables in,
compute spent on the ~10 that move the answer.

## 4. Semantic (embedding-keyed) registry, not string keys
`semantic_key` is string-normalized; at scale, elasticities must transfer across *phrasings and domains*
(mood ↔ affect ↔ sentiment; "inflation" ↔ "price growth"). **Key the registry by an embedding of
(variable, outcome-class)** so a weight learned in one domain informs a semantically-near variable in
another. This is what lets calibration compound across the whole corpus rather than per-exact-string.

## 5. A validated EVENT MODEL — the long-horizon / interventional unlock (the real gap)
The World substrate and rollout exist but the future-EVENT model (what pivotal events will fire over the
horizon, and their impact) is the least-validated piece, and it caps long-horizon and many best-action
questions. This is the honest frontier: **a calibrated event-hazard + event→impact model, scored on the
same no-cheat harness.** Until it lands, trust short-horizon / structure-dominated questions and be explicit
that long-horizon is event-limited.

## 6. Full-covariance weight posterior (honest joint uncertainty)
The Laplace posterior is currently diagonal (per-weight SD). For correlated variables (the double-counting
failure mode), a **full-covariance** posterior + joint sampling gives honest joint uncertainty and better
integrated forecasts. A refinement of `bayes_logistic`, not a rewrite.

---

## What is NOT the bottleneck (measured)
- **Variable count** — with calibrated weights + shrinkage, more variables help then plateau (EXP-072); the
  constraint was always weight calibration + uncertainty, not count.
- **The mechanism library** — bracket/committee/electorate/single-agent/generic-scm/calibrated-readout cover
  the scored domains; the compiler picks among them.
- **The action layer** — complete (all 7 components) and validated (+22pt on CMV).

## The one-line peak
A compiler that emits every relevant variable **with a weight calibrated from the accumulated corpus**, a
runtime that **spends fidelity where the outcome is sensitive** and **routes to a baseline where a simple
model already wins**, and a **validated event model** for the horizon — scored continuously on the portfolio
harness so the whole thing keeps getting more accurate as data scales. Items 1–3 are the near-term build;
4–6 are the reach.
