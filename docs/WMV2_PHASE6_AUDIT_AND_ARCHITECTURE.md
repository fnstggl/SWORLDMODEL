# WMV2 Phase 6 — Audit & Architecture

## 1. Ruthless current-registry audit (Part 0)

At Phase-6 start the registry (`swm/world_model_v2/registry/`, built by `build_registry.py`) held **47
families / 6 packs**. The audit's verdict, per plane (regenerated facts in
`experiments/results/phase6_research/_audit_notes.md`):

- **Code plane: strong.** All 47 families resolve to a callable transition with a `test_ref` → all reach
  `implemented`. This was never the gap.
- **Evidence / posterior plane: thin.** Only 6 packs existed, and 3 of those stored `"value": "see
  artifact"` pointers rather than the real fitted numbers.
- **Validation plane: very thin.** Only 3 families had any passed validation; 41 had none.
- **Production plane: 2 families** (`engagement_momentum_persistence`, `social_preference_population`); 1
  `locally_validated` (`exposure_response_hazard`); 1 `quarantined` (`hawkes_self_excitation`, preserved
  held-out failure).
- **Compiler flaw (the load-bearing finding):** `compiler.py` called `rank_mechanisms(scenario)` once and
  reused `selected[0]` — a single scenario winner — for **every** required causal process, then stamped
  `has_domain_pack=True` on all of them. High tiers, but the family usually did not answer the process
  (measured: only **22.7%** valid selections).

Disposition summary (full per-family audit rows in the priority matrix, §3): retain all executable families;
**repair** the "see artifact" packs (embed real coefficients); **quarantine** stays for Hawkes; **rewrite**
the compiler selection path (per-process); **do not** treat dataset availability as a mechanism map.

No prior artifact, benchmark, null, or quarantine was deleted or overwritten — Phase-6 files are additive.

## 2. Exact compiler → registry → runtime call path (after Phase 6)

```
natural-language question
 → compile_world()                                   swm/world_model_v2/compiler.py
   → _scenario_descriptor(raw)                        {domain, population_kind, time_scale, available_state, …}
   → for each required_causal_process:
       registry.select_for_process(store, process, scenario)      registry/applicability.py
         → _process_match(family, process)             ANSWERS-this-process gate (not name similarity)
         → score_applicability(family, scenario)       per-axis subscores + transport widening
         → winner + competing[] + rejected[]
       fallback.select_tier_for_process(process, winner, competing)   status → Tier 1–7 + support grade
   → registry.composition.compose(selections)          double-counting, competing, precedence, conflicts
   → WorldExecutionPlan(mechanism_choices, fallbacks_used, provenance{per_process_selection, mechanism_composition})
 → build_world → WorldState                            materialize typed state
   → event fires → operator.run(world, event, rng):    BehavioralMechanismOperator / FeatureHazardOperator
       applicable → propose (read typed state, bind pack values, apply transport widening on log-odds)
       → validate → apply → StateDelta (typed change + provenance + uncertainty)
   → temporal rollout → terminal-state projection → SimulationResult
```

The generic tier-6/7 `generic_outcome_prior` remains the terminal safety net: it writes the readout ONLY if
a domain mechanism left it unset, so a validated mechanism always takes precedence.

## 3. Three-layer data model (Part 1)

1. **Universal family** — `MechanismRecord` (`registry/record.py`): structural causal form, formal
   description, typed inputs/outputs, required state, parameter schema, applicability + `answers_processes`,
   citations (with transport limits), lifecycle status, code/test refs, known failure modes.
2. **Domain parameter pack** — `ParameterPack`: a family bound to a real empirical context (domain,
   population, dataset, period) with parameter **values carrying a source label + uncertainty** and a
   validation history. `enforce_uncertainty()` bans a bare point for `assumed/unsupported/experimental`
   sources.
3. **Scenario instance** — built by the compiler at runtime: the pack's values bound to concrete state paths
   as a `hazard_spec` (behavioral) or feature-hazard spec, carrying the transport widening and provenance.
   Executed by an operator → `StateDelta`.

Lifecycle (enforced by `promotion_blockers`, `store.set_status`): `proposed → research_encoded → implemented
→ locally_validated → transfer_validated → production_eligible`, with `domain_restricted` (valid only in
declared contexts) and `quarantined`/`rejected` demotions. Gates are strict: `locally_validated` needs a
**passed** held-out/PPC/transfer (a recorded-but-failed check does not count); `production_eligible` needs a
passed held-out/transfer **and** is blocked by any on-record **failed** transfer.

## 4. Ingestion pipeline & estimation core

`registry/ingestion.py` provides the shared pure-Python estimators (`fit_logistic`, `fit_bernoulli_hazard`
cloglog survival GLM, `profile_frailty`, `compare_forms`, `paired_bootstrap_delta`) and the pipeline verbs
(`register_published_mechanism`, `fit_pack_from_data`, `record_failure`, `promote`). `registry/families/
hazard.py` (`fit_feature_hazard`, `FeatureHazard`) and `families/behavioral.py` add the Phase-6 transitions.
`experiments/wmv2_phase6_fits.py` is the resumable real-data fitting + held-out harness; the registry reads
its committed coefficients so **no number is hand-typed**.

## 5. Storage, versioning, hashes, migrations

- Committed machine-readable state: `registry/data/registry.json` + `packs.json`, each wrapped with a
  SHA-256 integrity block; loading verifies the hash and refuses a corrupted registry.
- Phase-6 artifacts: `registry/data/priority_matrix.json`, `studies.json`, `coefficients.json` (same hashed
  envelope).
- Migration note: two new statuses (`research_encoded`, `domain_restricted`) and one new
  `ApplicabilityRule.answers_processes` field are backward-compatible (defaults keep old records valid); the
  store's lean-mirror maps the new statuses to calibration levels.

## 6. Downstream dependency boundaries (honest)

Phase 6 consumes the evidence/posterior interfaces available through Phase 1–2. Where a fuller Phase-2/3
capability is not yet available, the typed input contract is defined and the dependency recorded rather than
faked:

- **Posterior service:** packs carry point + broad-prior uncertainty; a full posterior-sample service (SBC,
  hierarchical partial pooling at runtime) is a Phase-2/3 dependency. Contract: pack `values[param] =
  {value, sd|lo/hi, source, method, dataset}`.
- **Evidence-to-parameter estimator:** the compiler binds packs to instances from the scenario descriptor;
  an automatic evidence→pack estimator is out of scope. Contract: `select_for_process` consumes the Phase-1
  `WorldExecutionPlan` scenario draft.

No fake evidence or posterior system was introduced.

## 7. Artifact index

| Artifact | What |
|---|---|
| `swm/world_model_v2/registry/data/registry.json` / `packs.json` | committed family + pack registry (hashed) |
| `.../data/priority_matrix.json` | 49-family research-first priority matrix |
| `.../data/studies.json` | 93/97 verified primary sources |
| `.../data/coefficients.json` | 91 verified reported coefficients |
| `experiments/results/wmv2_phase6_fits.json` | real-data fits + held-out (telco/SE/CMV/Upworthy) |
| `experiments/results/wmv2_phase6_selection_eval.json` | before/after fallback-tier + selection validity + ablations |
| `experiments/results/wmv2_phase6_forensic_traces.json` | 10 traces (5 executed end-to-end) |
| `experiments/results/wmv2_phase6_validation_summary.json` | four family + six pack counts, per-family planes |
| `experiments/results/wmv2_phase6_failures.json` | 7 preserved negative results |
| `experiments/results/phase6_research/cluster_*.json` | 8 primary-literature research clusters |
| `tests/test_wmv2_phase6.py` | 15 tests (families, execution, selection, transport, composition, gates) |
