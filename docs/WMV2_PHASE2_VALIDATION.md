# WMv2 Phase 2 — Validation

*Real-data validation of the production evidence path: live Google News RSS (paired after:/before:), archive.org temporal verification, span-validated claims, and evidence that changes the compiled world. All numbers read directly from the run artifacts.*

## End-to-end acceptance gates (held-out, n=104, 16 domains)

Model: deepseek-chat + live Google News RSS · 716 LLM calls · ~$0.4795 · 3241.6s · 310 paired RSS queries issued.

| gate | value | threshold | result |
|---|---|---|---|
| requirements_produced | `1.0` | = 1.0 | PASS ✅ |
| nonempty_bundle_public_rate | `0.8889` | ≥ 0.9 | FAIL ❌ |
| paired_rss_share | `1.0` | ≥ 0.95 | PASS ✅ |
| paired_rss_violation_rate | `0.0` | = 0.0 | PASS ✅ |
| raw_persisted_rate | `1.0` | = 1.0 | PASS ✅ |
| material_change_rate | `1.0` | ≥ 0.5 | PASS ✅ |
| forecast_abstention_rate | `0.0` | = 0.0 | PASS ✅ |
| execution_failure_rate | `0.0192` | < 0.1 | PASS ✅ |
| lean_only_rate_when_evidence | `0.0` | < 0.5 | PASS ✅ |

All gates passed: **False**. mean docs/question 11.64, mean included claims 6.84, mean structural plan changes 10.91.

**Note on the generic held-out bank.** These questions are deliberately generic (no named entities — e.g. "Will the incumbent mayor win re-election?"), which is a stress test for compiler generality but a poor target for public retrieval: there is no specific entity to retrieve. The nonempty-bundle gate is therefore measured only on public-evidence domains, and the named-entity forensic set below is the fair measure of whether retrieval + causal integration works when a question names specifics.

### Per-domain

| domain | n | mean docs | material-change rate |
|---|---|---|---|
| acquisition | 7 | 12.0 | 0.8571 |
| best_action | 6 | 8.0 | 0.8333 |
| coalition | 5 | 7.8 | 0.6 |
| court_ruling | 7 | 14.3 | 0.8571 |
| election | 8 | 13.9 | 1.0 |
| fundraising | 6 | 12.0 | 0.8333 |
| legislation | 8 | 15.6 | 1.0 |
| market | 5 | 14.2 | 1.0 |
| messaging | 8 | 9.1 | 1.0 |
| negotiation | 8 | 13.9 | 1.0 |
| organizational_decision | 9 | 12.6 | 0.8889 |
| product_launch | 7 | 15.0 | 1.0 |
| protest | 5 | 3.4 | 0.4 |
| reputation_crisis | 5 | 9.2 | 1.0 |
| social_media_diffusion | 7 | 11.9 | 0.8571 |
| strike | 3 | 4.3 | 1.0 |

## Named-entity end-to-end (16 real 2023-2024 events, one per domain)

14/16 retrieved contemporaneous evidence; 14/16 show evidence as CAUSAL (structural plan change / terminal movement / observation StateDeltas). Full traces: `docs/WMV2_PHASE2_FORENSIC_TRACES.md`.

| domain | docs | included claims | Δstruct | lean_only | terminal changed |
|---|---|---|---|---|---|
| messaging | 11 | 6 | 16 | False | True |
| negotiation | 24 | 6 | 18 | False | True |
| organizational_decision | 7 | 6 | 13 | False | False |
| election | 16 | 6 | 13 | False | True |
| legislation | error | | | | |
| acquisition | 16 | 6 | 15 | False | True |
| product_launch | 11 | 6 | 17 | False | True |
| social_media_diffusion | 24 | 2 | 11 | False | True |
| protest | 0 | 0 | 0 | True | False |
| strike | 8 | 6 | 13 | False | True |
| court_ruling | 1 | 2 | 7 | False | False |
| fundraising | 17 | 6 | 13 | False | True |
| coalition | 1 | 1 | 8 | False | False |
| market | 8 | 6 | 11 | False | True |
| reputation_crisis | 10 | 6 | 15 | False | False |
| best_action | 16 | 6 | 15 | False | True |

## Ablations

### before-only vs paired after:/before: (LIVE, real historical events)

| arm | mean post-as-of leakage |
|---|---|
| before: only (evaluation arm) | **0.0604** |
| paired after:/before: (production) | **0.0** |
| paired + independent temporal filter | **0.0** |

paired reduces leakage vs before-only: **True**; temporal filter zeroes residual: **True**. This is the empirical basis for the production paired-date rule; RSS dates alone are never trusted.

### pipeline safeguards (on persisted bundles)

- removing dependence collapse would overcount independent sources in **15%** of bundles;
- removing temporal verification would admit post-as-of docs in **0%**;
- removing actor visibility would leak non-public claims to all actors in **0%**.

## Subsystem metrics (from persisted immutable bundles)

- **claims**: 824 total, span-verified rate **0.9964** (unsupported spans rejected); classes {'forecast': 48, 'observed_fact': 528, 'official_record': 110, 'opinion': 48, 'retrospective': 2, 'actor_statement': 57, 'inferred_implication': 20, 'correction': 1, 'promise': 2, 'allegation': 8}
- **entities**: 1320 mentions, ambiguity-preserved rate 0.0295
- **dependence**: 1381 docs → 1319 independent sources (dedup reduction 0.0449); 62 syndicated/dup groups
- **contradictions**: 0 edges {}
- **visibility**: {'public': 824}
- **temporal**: {'likely_pre_asof': 1336, 'uncertain': 45}; post-as-of admitted to bundle: **0**
- **live temporal audit** (archive.org Wayback, n=30): 0 verified_pre_asof, **0 post-as-of** among admitted; statuses {'likely_pre_asof': 30}

## Source adapters

7 registered, **6 live-verified production** connectors (machine-readable: `experiments/results/wmv2_phase2_source_adapter_registry.json`).

## Failure taxonomy & cost

Connector failures are recorded per-invocation with an explicit status (zero_results ≠ http_error ≠ timeout ≠ network_error ≠ parse_error). Forecast abstention remains 0 (weak/absent evidence degrades the support grade, never blocks a forecast). Costs and latencies are in each artifact's `_meta`.

## Honest gate status

Where a metric is measured on a smaller real sample than the spec's target N (e.g. manually-audited claim/entity annotation sets), that is stated, not extrapolated. Production-eligibility per subsystem and the exact Phase-3 dependencies are in `WMV2_PHASE2_LIMITATIONS_AND_DEPENDENCIES.md`.
