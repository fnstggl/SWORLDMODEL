# WMv2 Phase 1 — Validation (no-abstention, B12/B13/B15)

*Real-LLM validation of the production Phase-1 path: every coherent question SIMULATES; epistemic weakness lowers the support grade, never refuses. All numbers are read directly from the run artifacts (JSON) — nothing hand-entered. Companion machine-readable summary: `experiments/results/wmv2_phase1_validation_summary.json`.*

**Protocol.** 104 held-out natural-language questions across 16 domains, **no scripted plans** — the compiler builds its own plan for each and the shipped `pipeline.simulate()` produces the `SimulationResult`. Model: deepseek-chat · 104 calls · ~$0.2411 · 1358.0s. Resumable, deterministic given the cache.

## B13 acceptance gates — ALL PASSED ✅

| gate | value | threshold | result |
|---|---|---|---|
| valid plan (compiled) | `1.0` | ≥ 0.95 | ✅ |
| materialize (world built) | `1.0` | ≥ 0.9 | ✅ |
| complete rollout + bound readout | `0.9231` | ≥ 0.85 | ✅ |
| forecast abstention (coherent Q, no forecast) | `0.0` | = 0.0 | ✅ |
| clarification (incoherent only) | `0.0` | < 0.05 | ✅ |
| execution failure (engineering) | `0.0288` | < 0.1 | ✅ |
| provenance status present | `1.0` | = 1.0 | ✅ |
| fallback names its tier | `1.0` | = 1.0 | ✅ |
| unsupported precision (field stamped observed) | `0.0` | < 0.02 | ✅ |
| LLM-minted terminal probability | `0.0` | = 0.0 | ✅ |
| no scenario keyword router (static) | `True` | = True | ✅ |

**Forecasts produced: 101/104 (97%).** Simulation-status histogram: `{'completed_with_degradation': 101, 'execution_failed': 3}`. Support-grade histogram: `{'highly_speculative': 100, 'exploratory': 4}`.
 Failure taxonomy (execution_failed only): `{'terminal_readout_unbindable': 3}`.

### Reading the grades
On the **general path** the highest defensible mechanism tier is 6 (generic structural, `exploratory`) or 7 (competing structural hypotheses, `highly_speculative`), because no held-out-validated *domain* parameter pack (tiers 1–4) applies to these arbitrary questions. The support grade honestly reports that: a from-scratch general social simulation without a validated domain mechanism is exploratory/speculative, and the forecast is a correspondingly broad prior with wide dispersion and explicit limitations — not a confident number. Sharpening requires the domain packs and evidence assimilation (Phase 3), out of Phase-1 scope. Phase 1's claim is generality + no-abstention + honesty, which the gates above establish.

## Per-domain coverage

| domain | n | forecast rate | complete rate | grades |
|---|---|---|---|---|
| acquisition | 7 | 100% | 100% | exploratory, highly_speculative |
| best_action | 6 | 67% | 50% | highly_speculative |
| coalition | 5 | 100% | 100% | exploratory, highly_speculative |
| court_ruling | 7 | 100% | 100% | highly_speculative |
| election | 8 | 100% | 100% | highly_speculative |
| fundraising | 6 | 100% | 100% | exploratory, highly_speculative |
| legislation | 8 | 100% | 100% | highly_speculative |
| market | 5 | 100% | 100% | highly_speculative |
| messaging | 8 | 100% | 100% | highly_speculative |
| negotiation | 8 | 100% | 100% | highly_speculative |
| organizational_decision | 9 | 100% | 100% | highly_speculative |
| product_launch | 7 | 100% | 100% | highly_speculative |
| protest | 5 | 100% | 80% | exploratory, highly_speculative |
| reputation_crisis | 5 | 100% | 60% | highly_speculative |
| social_media_diffusion | 7 | 86% | 71% | highly_speculative |
| strike | 3 | 100% | 100% | highly_speculative |

## B12 ablations — component contributions

Each question compiled ONCE; the compiled plan is transformed per ablation (no extra LLM calls). k=24 questions.

| ablation | forecast | complete | exec-fail | dispersion | struct-H |
|---|---|---|---|---|---|
| full_compiler | 100% | 100% | 0% | 0.996 | 1.337 |
| no_fallback_hierarchy | 0% | 0% | 4% | 0.0 | 0.0 |
| no_readout_repair | 4% | 4% | 96% | 0.041 | 0.057 |
| no_structural_hyps | 100% | 100% | 0% | 0.999 | 0.0 |
| no_sensitivity_margin | 100% | 100% | 0% | 0.996 | 1.337 |

**Contribution vs. full compiler** (what breaks when a component is removed):

- **no_fallback_hierarchy**: forecast −100%, complete −100%, exec-fail +4%, Δdispersion -0.996, Δstruct-H -1.337. expected: forecast/complete rate COLLAPSES — the fallback resolver is what guarantees every coherent question forecasts (core no-abstention lever).
- **no_readout_repair**: forecast −96%, complete −96%, exec-fail +96%, Δdispersion -0.955, Δstruct-H -1.28. expected: execution_failure INCREASES on questions whose LLM readout was unbindable — repair is what guarantees the terminal reads out.
- **no_structural_hyps**: forecast −0%, complete −0%, exec-fail +0%, Δdispersion 0.003, Δstruct-H -1.337. expected: structural entropy → 0 and dispersion narrows — the component that represents structural uncertainty as competing particles.
- **no_sensitivity_margin**: forecast −0%, complete −0%, exec-fail +0%, Δdispersion 0.0, Δstruct-H 0.0. expected: negligible forecast change — fidelity planning affects compute allocation, not whether a forecast is produced.

## B15 forensic traces

16/16 domain traces produced a forecast; full per-domain traces (every intermediate structure) in `docs/WMV2_PHASE1_FORENSIC_TRACES.md` + `experiments/results/wmv2_phase1_forensic_traces.json`.

## Verdict

All B13 acceptance gates pass: the production compiler produces an honest, executable, terminal-state forecast for 97% of 104 held-out questions across 16 domains, with **zero forecast abstentions**, through one generic path with no domain hard-coding and no LLM-minted probabilities. Weakness is carried by the support grade, not by refusal. Historical Session-1 results (`WMV2_COMPILER_VALIDATION.md`) are preserved unedited; see `WMV2_NO_ABSTENTION_MIGRATION.md`.
