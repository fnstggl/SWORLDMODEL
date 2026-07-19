# Core-architecture acceptance report (§42) — machine-generated

Generated 2026-07-19T20:53:51Z by `experiments/core_arch_acceptance.py`. **27/27 gates green.**

- ✅ `boundary_contract_suite` (suite): 18 passed in 0.10s
- ✅ `outside_world_suite` (suite): 19 passed in 0.12s
- ✅ `bounded_cognition_suite` (suite): 35 passed in 0.11s
- ✅ `truncation_suite` (suite): 21 passed in 0.07s
- ✅ `mechanism_spec_suite` (suite): 19 passed in 0.19s
- ✅ `invariant_enforcement_suite` (suite): 34 passed in 1.28s
- ✅ `cross_domain_fixture_suite` (suite): 10 passed, 1 skipped in 0.14s
- ✅ `phase13_integration_suite` (suite): 15 passed in 0.25s
- ✅ `qualitative_actor_suite` (suite): 28 passed in 0.38s
- ✅ `generated_world_suite` (suite): 17 passed in 1.19s
- ✅ `structural_ensemble_suite` (suite): 35 passed in 1.85s
- ✅ `combined_runtime_suite` (suite): 9 passed in 0.44s
- ✅ `forensic_traces_exist` (trace): 6 live case artifact(s)
- ✅ `actual_llm_calls_used` (trace): calls per case: [('case1_founder_platform.json', 126), ('case2_personal_message.json', 82), ('case3_institutional_promotion.json', 268), ('case4_capacity_bottleneck.json', 102), ('case5_population_ado
- ✅ `explicit_boundaries_default` (trace): world_boundaries section present on ensemble-route live results
- ✅ `residual_outside_world_default` (trace): outside_world section present on live results
- ✅ `boundary_criticism_ran` (trace): critic stages present in live boundary generation traces
- ✅ `no_numeric_actor_fallback_live` (trace): verification.numeric_actor_fallbacks == 0 on every live case
- ✅ `no_generic_prior_write_live` (trace): no broad-prior terminal writes on live cases (suppressions are legal and surface as under_modeled)
- ✅ `truncated_weight_visible` (trace): ≥1 live case carries visible truncated weight with halted branches (case6 is designed to)
- ✅ `monoculture_surfaced` (trace): model_family_monoculture reported on every live case
- ✅ `cognition_stage_traces_live` (trace): bounded-cognition records present on live cases
- ✅ `under_modeled_surfacing_works` (trace): under_modeled subtypes appear when gaps exist (or no gaps existed)
- ✅ `production_never_sets_baseline_markers` (code): no production assignment of the §19/§28 markers
- ✅ `result_contract_statuses` (code): ('completed', 'completed_with_degradation', 'clarification_required', 'execution_failed', 'temporally_truncated', 'under_modeled', 'truncated')
- ✅ `branch_status_vocabulary` (code): ('active', 'completed', 'absorbed', 'quiescent', 'truncated_actor_budget', 'truncated_event_budget', 'truncated_context_budget', 'truncated_boundary_budget', 'truncated_missing_mechanism', 'truncated_
- ✅ `mechanisms_declare_read_write_sets` (code): 38 specs; 0 without any declared I/O
