# Lean V2 accuracy run — 5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6

**Question:** Will Apple announce visionOS 27 (or a successor major version to visionOS 26) at WWDC 2026?

**Status:** completed | **probability:** 0.875 | **source:** mass_weighted:grounded_reference_prior+grounded_prior | **grounding:** exploratory | **confidence:** low


## Causal world

- thesis: Apple's executive leadership, particularly the Software Engineering VP and the Vision Products Group lead, hold discretion over visionOS release plans. Despite reports of platform abandonment, the institutional inertia of WWDC OS announcements and the need to support existing Vision Pro hardware may
- actors: 2 | action templates: 2

## Shared world conditions

- **apple_os_naming_transition**: Apple transitioned to year-based OS naming starting with WWDC 2025, so the third major visionOS release was called visionOS 26. — counted rate 0.75 (n=1)
- **vision_pro_sales_decline**: Sales of Apple Vision Pro have been low, with the M5 refresh failing to revive interest, and reports indicate Apple has effectively abandoned the platform. — counted rate 0.8333 (n=2)
- **wwdc_annual_os_announcement_pattern**: Apple has historically announced major new versions of its operating systems at WWDC each June. — counted rate 0.875 (n=3)

## Forecast decomposition

- grounded prior: 0.875 (n=3, source=counted_outcome_reference_class)
- simulation-conditional: 0.0 (resolved mass 1.0)
- combined: None via combiner_unavailable_range_only
- prior/simulation disagreement: 0.875
  - no leakage-audited reliability combiner is fitted — prior and simulation are reported separately with the feasible combined range; no fixed blend is applied and Lean V2 must not become default on this basis

## Unresolved mass by cause


## Cost

- calls: 9 | wall: 84.5s | peak nodes: 24
- deliberations: 1 | challenger: False
- limitation: forecast decomposition — grounded prior 0.875 (counted n=3); simulation-conditional 0.0 (resolved mass 1.00); headline 0.875 via combiner_unavailable_range_only. Prior and simulation are reported separately and never blended by a fixed rule.
- limitation: no leakage-audited prior↔simulation reliability combiner is fitted — the headline is the simulation-conditional forecast; the grounded prior is reported separately; this is a stated reason Lean V2 must not become the default yet
