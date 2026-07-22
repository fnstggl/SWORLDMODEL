# WMV2 Phase 7 — Nonlinear Forms: Research & Catalog

The full, machine-readable catalog is `experiments/results/wmv2_phase7_form_registry.json` (31 forms, each
with monotonicity, parameter schema, extrapolation + missing-data behavior, invariants, differentiability,
computational cost, failure conditions, theory, and maturity). This document explains the methodology, the
form↔mechanism compatibility design, and the context/pooling/regime layers. It does **not** restate every
form's JSON.

---

## Research methodology

Candidate forms are drawn from **mechanism theory, primary literature, observed data shape, known invariants,
and prior failure history** — never generated arbitrarily to enlarge a search space (Part 6). A form's
*presence* in the registry means only "a real, evaluable candidate shape with declared invariants." It does
**not** mean validated. Every form carries an honest `maturity`:

| maturity | meaning | example |
|---|---|---|
| `primitive` | standard mathematical shape, evaluable, no scenario claim | `linear`, `u_shaped` |
| `fitted_available` | a fit routine exists and has produced real coefficients | `logistic`, `gam`, `hill`, `logistic_growth` |
| `validated_elsewhere` | earned a held-out win in a specific mechanism | `cloglog_hazard`, `hill` (Higgs) |
| `structural_candidate` | plausible for a mechanism, not yet validated | `inverted_u`, `fatigue`, `hysteresis`, `self_exciting` |

**Promotion requires evidence, not sophistication.** A theoretically elegant form that does not beat the
appropriate simpler baseline on held-out stays `structural_candidate` (Part 27). This is why `inverted_u`
(backfire) and `hysteresis` remain candidates: they are implemented and executable, but not yet supported by a
held-out win in a specific mechanism.

---

## The form catalog (31 forms, by phenomenon)

Each form is **pure-Python evaluable** at runtime (`StructuralForm.eval(params, inputs)`), deterministic,
serializable, and guarded against non-finite output (NaN/inf/overflow → a recorded `FormError`, never a
crash). Grouped by the Part-4 phenomenon they express:

- **Baseline / link:** `linear`, `logistic`, `cloglog_hazard`, `survival_hazard`, `gam`
- **Thresholds & tipping:** `threshold_hard` (Granovetter), `threshold_smooth` (heterogeneous-threshold soft
  step), `change_point`, `hill` (n>1 onset), `hysteresis` (two-threshold bistable band)
- **Saturation & diminishing returns:** `hill`, `michaelis_menten`, `logistic_saturation`, `exp_saturation`,
  `logistic_growth` (Verhulst increment), `finite_population` (SIR-like), `monotonic_spline`
- **Fatigue / habituation / refractory:** `fatigue` (geometric), `habituation` (hyperbolic), `refractory`
  (recovery after rest)
- **Non-monotone:** `inverted_u` (weak-tie / backfire — *candidate*, never assumed), `u_shaped`, `cubic_spline`
- **Reinforcement / self-excitation:** `recurrent_event_hazard`, `self_exciting` (Hawkes — *candidate*,
  quarantined on Higgs), `self_inhibiting`
- **Heterogeneity / regime:** `finite_mixture`, `regime_model`, `hmm_regime`, `mixture_of_experts`,
  `nonlinear_state_space`
- **Interpretable flexible:** `piecewise_linear`, `gam` (spline-basis + link; the additive-nonlinear workhorse)

### Why the GAM is the workhorse for tabular mechanisms
`gam` evaluates `link(Σ linear terms + Σ smooth(x) + Σ interactions)` where each smooth is a fitted
piecewise-linear (hinge) spline basis — a genuine additive-nonlinear model that stays interpretable and
extrapolates linearly at the tails (no black box, no wild spline overshoot). It is the form that won the telco
churn mechanism (below), capturing the declining tenure hazard the additive logistic cannot.

**Interaction handling (correctness note).** Interaction terms carry a *dedicated interaction standardizer*
(`interactions_std[key] = [mu, sd]`) applied identically in the offline fit and the runtime evaluator, so a
raw `tenure×contract` product (magnitude up to ~144) never destabilizes the fit or disagrees with the runtime.
This fixed a fit/runtime scale mismatch discovered during validation.

---

## Mechanism ↔ form compatibility (Part 1 restriction)

Not every form is offered to every mechanism — only causally-meaningful ones. Full map in
`wmv2_phase7_mechanism_form_compat.json`; excerpt:

| Phase-6 family | causally-meaningful forms |
|---|---|
| `attrition_dropout_hazard` | logistic, **gam**, survival_hazard, threshold_smooth, change_point |
| `bass_diffusion` | **logistic_growth**, linear_growth, logistic_saturation, finite_population |
| `complex_contagion_hazard` | hill, cloglog_hazard, exposure_response_hazard, threshold_smooth |
| `content_response_click` | logistic, gam, inverted_u, michaelis_menten |
| `trust_formation` | hysteresis, piecewise_linear, threshold_smooth |
| `hawkes_self_excitation` | self_exciting — **QUARANTINED, do not select** |

---

## Context-conditioning design (Part 2)

`context.ContextVariable` gives each conditioning dimension typed provenance: `source` (observed /
inferred_phase3 / documented_event / derived_from_history / population / network / institution), scale,
temporal validity, missingness policy, transport risk, and two hard leakage flags. `ContextSchema.leakage_audit`
runs **before any value is used** and refuses (a) any variable whose validity window starts after the as-of
cutoff (future context) and (b) any variable flagged `derived_from_outcome`. The LLM may *nominate* candidate
context variables and interactions; it may not decide they are active, estimate an effect, or choose a form —
those are fixed by fitted data and held-out validation.

## Hierarchical partial pooling (Part 3)

`pooling.py` provides empirical-Bayes shrinkage (Gaussian random-effects via a DerSimonian–Laird τ², and a
Beta-Binomial variant for rates). Each group reports its raw estimate, shrunk estimate, shrinkage fraction,
standard error, and an `escaped_prior` flag; a new/out-of-group unit predicts the population mean (full
shrinkage). This is used for the Upworthy per-test CTR baseline (thousands of sparse tests) and is the honest
alternative to fitting an unconstrained model per group. Full hierarchical MCMC is available in the offline
layer where warranted.

## Regime models (Part 10)

`regime_model` (regime index from context/data/Phase-3 — never LLM-invented), `hmm_regime` (marginalizes over
an inferred regime posterior at runtime), and `mixture_of_experts` (a softmax gate over expert forms). Regime
identity must come from data, documented external events, Phase-3 latent inference, or an explicit broad prior;
the forms consume a regime or a regime posterior, they do not fabricate one.

---

## Selected vs rejected forms (evidence-driven)

| Mechanism | selected | rejected / not promoted | why |
|---|---|---|---|
| `attrition_dropout_hazard` (telco) | **gam (+interaction)** | logistic, logistic_interaction | GAM's tenure/charges smooths beat additive logistic on held-out (CI<0) |
| `bass_diffusion` (baby names) | **logistic_growth** | linear_growth | saturation beats non-saturating extrapolation on real trajectories |
| `response_occurrence_hazard` (StackExchange) | **logistic (kept)** | gam, gam_interaction | nonlinear did not beat logistic — Phase-6 null preserved |
| `argument_persuasion_success` (CMV) | **logistic (kept)** | gam, inverted_u (backfire) | nonlinear/interaction null; backfire unsupported |
| `content_response_click` (Upworthy) | **pooled baseline** | gam headline, linear headline | headline effects null; pooled/global CTR dominates |

## Applicability & transport (Parts 14–15)

`applicability.evaluate_applicability` is conservative: missing history/context, insufficient subgroup size,
strong extrapolation, or an unsupported phenomenon all drop the verdict toward `keep_linear` or
`applicable_widened` (with a quantified uncertainty widening and support-grade penalty). `transport_check`
refuses nonlinear transport on regime or outcome-definition mismatch and widens uncertainty otherwise. The
telco cross-contract transfer is the live demonstration: the nonlinear tenure shape did **not** transport
(tenure support barely overlaps across contract types) → the extension is `domain_restricted`, not
production-eligible.

## Unresolved research gaps

- **fatigue / habituation / refractory**: forms implemented and executable, but no committed dataset in this
  run isolates repeated-exposure sensitivity decline cleanly enough to validate them → they remain
  `structural_candidate`. A repeated-contact mobilization or notification dataset would close this.
- **hysteresis**: the two-threshold form is executable but needs a longitudinal trust/adoption series with
  observed activation *and* deactivation to identify the band → `structural_candidate`.
- **self-exciting**: preserved-quarantined on Higgs; a context-specific point-process pack (circadian
  baseline, multi-kernel) could be re-tested *separately* without un-quarantining the original.
- **post-peak decline in diffusion**: `logistic_growth` saturates but cannot fall; a rise-and-fall adoption
  mechanism (e.g. a fashion/obsolescence term) is the next diffusion form.
