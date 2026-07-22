# Lean V2 accuracy run — 5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6-fidelity

**Question:** Will Apple announce visionOS 27 (or a successor major version to visionOS 26) at WWDC 2026?

**Status:** completed | **probability:** 0.0 | **source:** mass_weighted:completed_rollouts+grounded_prior | **grounding:** exploratory | **confidence:** low


## Causal world

- thesis: Apple's reported abandonment of Vision Pro platform and shift to smart glasses, combined with lack of new hardware plans, makes a major visionOS announcement unlikely. However, WWDC tradition and the need to support existing Vision Pro users could still compel a minor update or a nominal major versi
- actors: 3 | action templates: 3

## Shared world conditions

- **apple_os_naming_convention**: Apple transitioned to year-based OS naming starting with WWDC 2025, so the third major release of visionOS was called 'visionOS 26' rather than 'visionOS 3'. — counted rate 0.75 (n=1)
- **apple_vision_pro_sales_decline**: Apple Vision Pro sales have been low and the platform is considered abandoned by Apple internally. — counted rate 0.8333 (n=2)
- **wwdc_annual_os_announcement_pattern**: Apple has historically announced major new versions of its operating systems at WWDC each June. — counted rate 0.8333 (n=2)

## Forecast decomposition

- prior_forecast (grounded prior): 0.8333 (n=2, source=counted_outcome_reference_class)
- simulation_forecast (conditional on resolved mass): 0.0 (resolved mass 1.0)
- simulation probability bounds (residual-widened): [0.0, 0.316] (residual bound 0.316)
- headline_forecast: 0.0 via mass_weighted:completed_rollouts+grounded_prior
- prior/simulation disagreement: 0.8333
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Simulation completion audit

- resolved mass: 1.0 of 1.0 (target ≥0.8 met: True)
- unresolved by cause: {}
- unknown-state terminal mass: 0.0 (must be 0: True)
- missing-mechanism terminal mass: 0.0 (ok: True)
- readiness verdict: ready | round-trip ok: True

## Unresolved mass by cause


## Cost

- calls: 11 | wall: 102.49s | peak nodes: 42
- deliberations: 1 | challenger: False
- limitation: forecast decomposition — grounded prior 0.8333 (counted n=2); simulation-conditional 0.0 (resolved mass 1.00); headline 0.0 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: bounded omitted-state residual 0.316: the interval widens to [0.0, 0.316] — private-state omissions are BOUNDS, never unknown-state worlds
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the mass-based recovery blend (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior); the two inputs are reported separately and never blended by a fixed rule
