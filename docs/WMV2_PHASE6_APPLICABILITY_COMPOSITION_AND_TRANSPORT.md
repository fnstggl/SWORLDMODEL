# WMV2 Phase 6 — Applicability, Composition & Transport

The compiler must select a mechanism because it **answers the required causal process** in a **compatible
scenario**, never because a name sounds relevant. Three engines enforce this.

## 1. Applicability engine (`registry/applicability.py`)

`score_applicability(record, scenario)` returns per-axis subscores in [0,1] (domain_match, population_match,
time_scale_match, variable_compatibility, data_availability, institutional_compatibility, **evidence_quality**)
plus hard-exclusion reasons and a transport widening. Evidence quality is earned by the validation *history*,
not the citation list: citation-only floor 0.15; a verified `published_estimate` 0.40; passed
`posterior_predictive` 0.45; passed `held_out` 0.70; passed `transfer` 0.90; a family whose only record is a
failure floors at 0.05.

**Per-process selection** (`select_for_process(store, process, scenario)`) is the Phase-6 fix for the
Phase-1 flaw:

1. `_process_match(family, process)` — 1.0 if the family DECLARES `answers_processes ∋ process`, 0.6 on a
   content-token overlap (stopwords like "after" filtered so `X_after_Y` ≠ `Z_after_W`), 0.3 on ontology
   overlap, else **0.0** (not a candidate).
2. Applicability refines: `combined = process_match · (0.4 + 0.6 · applicability)`.
3. Returns winner + `competing[]` (runners-up, kept as hypotheses) + `rejected[]` (with reasons).

Selectable statuses: `locally_validated, transfer_validated, production_eligible, domain_restricted,
research_encoded`. Domain-restricted/research-encoded enter at Tier 4 (published mechanism, widened).

### Adversarial selection tests (Part 6)

`tests/test_wmv2_phase6.py::test_adversarial_incompatible_family_not_selected` pins that diffusion families
score `_process_match == 0` for `offer_response` and never appear in the selected/competing set; the
before/after eval shows selection **validity 22.7% → 100%**.

## 2. Transport-risk engine (`registry/transport.py`)

`assess_transport(pack_context, scenario)` scores 12 weighted axes (population, domain, platform, geography,
period, **outcome_definition**, action_space, exposure_process, network_structure, time_scale, measurement,
**intervention_type**) and returns a decision + a widening factor:

- axis risk: exact match 0.0; substring 0.15; token overlap 0.3; weak overlap 0.6; no overlap 0.9; one side
  unspecified 0.5 (pessimistic); neither side specifies it 0.15 (not a known mismatch).
- `transport_direct` (risk ≤ 0.12, widen ×1) → `transport_widened` (widen 1 + 2·risk) → `experimental` →
  **`reject`** when a DECISIVE axis (`outcome_definition`, `intervention_type`) mismatches, or a required
  variable is missing.

Widening inflates **parameter/log-odds uncertainty**, never the point estimate — a transported pack is *less
certain*, not more extreme. Example (from the forensic traces): GGL turnout → an election scenario yields
`transport_widened` (×1.61); a lab-economic-game → political-donation with an outcome-definition flip yields
`experimental`/`reject`.

## 3. Composition engine (`registry/composition.py`)

`compose(selections)` takes the per-process selections and produces a plan that:

- **Detects double-counting:** every process maps to an EFFECT CHANNEL (adoption, participation, belief,
  relationship, decision, attention). If >1 distinct family writes one channel via different processes, the
  highest-precedence writer is the single-writer and the others become competing hypotheses — preventing
  double-counted exposure/persuasion/social-influence.
- **Preserves competing mechanisms:** when several families answer a process and evidence does not
  distinguish them, they are kept as competing hypotheses to be branched across (NOT averaged), so mechanism
  disagreement propagates into terminal dispersion.
- **Assigns precedence:** production > transfer > local > domain_restricted > research_encoded > generic.
- **Flags cross-timescale conflicts:** same channel written by mechanisms at incompatible declared time
  scales.

Pinned by `test_composition_double_counting_and_competing` (two diffusion families on the `adoption` channel
→ single-writer kept + double-count flagged; the simple-contagion runner-up preserved as competing).

## 4. What is recorded in the plan

`WorldExecutionPlan.provenance` now carries `per_process_selection` ({process → selected family, status,
competing, n_candidates}) and `mechanism_composition` (ordered writers, competing, double_counting,
conflicts). This makes selection auditable and is the basis for the forensic traces.
