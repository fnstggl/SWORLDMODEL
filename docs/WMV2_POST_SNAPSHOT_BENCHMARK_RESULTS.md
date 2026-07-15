# World Model V2 Post-Snapshot Benchmark Results

## Verdict

The benchmark was executed and scored, but the strict completion gate failed. It produced exactly 100 independent representative worlds, 400 primary V2 forecasts, 400 fair baseline rows, 4,400 phase records, 400 cutoff-safe evidence capsules, a 160-row locked score opened once, and a separate 60-world/120-row causal diagnostic.

The system is not production eligible. The decisive reasons are:

- Phase 2 degraded with `evidence_error: AttributeError` on all 400 scored rows despite being formally recorded active;
- one required Phase 4 controlled row was blocked;
- Phase 8 and Phase 11 had zero meaningful controlled ablation effects;
- the market-informed V2 comparison was not run;
- the model arm is mutable Tier C, not immutable Tier A or provider-attested Tier B;
- the full repository suite has 12 failures;
- locked performance is weak and sensitive to model-memory strata.

## Execution counts

| Item | Result |
|---|---:|
| Source events examined | 3,000 |
| Binary-eligible contracts | 1,430 |
| Independent eligible worlds | 1,174 |
| Selected representative worlds | 100 |
| Calibration forecasts | 160 |
| Validation forecasts | 80 |
| Locked forecasts and scores | 160 |
| Total primary V2 forecasts | 400 |
| Baseline rows | 400 |
| PhaseExecutionRecords | 4,400 |
| Evidence capsules | 400 |
| Causal worlds / cutoff rows | 60 / 120 |
| Locked outcome-store reads | 1 |

All 400 primary rows used Tier C, the same scored runtime fingerprint, 11 phase supervisors, terminal `WorldState` readout, and an OS-isolated model endpoint. Formally, all relevant records were active, all irrelevant records were explicit no-ops, and no representative record was blocked. The Phase 2 internal error invalidates the stronger full-runtime claim.

## Temporal safety and leakage

The selected arm is `causally_blinded_historical`. The exact hosted DeepSeek version was not immutable, no provider knowledge cutoff was available, and the selected historical questions are therefore not claimed to postdate a verified model snapshot or cutoff.

All evidence bytes were available by their forecast cutoff. The forecaster could not access outcomes, mappings, unblinded vaults, or the open internet. Before outcomes opened, 344/400 rows were clean-blinded and 56/400 were contamination-susceptible. Locked final strata were 92 clean, 28 susceptible, and 40 known-contaminated rows.

## Phase 12

Calibration candidates were fit only on 140 pre-open-clean calibration rows from 35 worlds. Selection used only 76 pre-open-clean validation rows from 19 worlds. `task_family_conditioned` beat identity on validation:

| Method | Validation Brier | Validation log loss |
|---|---:|---:|
| Identity | 0.245672 | 0.684470 |
| Selected task-family conditioning | 0.233571 | 0.661645 |

The validation gain did not transfer. On the locked set, calibration worsened Brier from 0.246966 to 0.256230 and log loss from 0.687077 to 0.708343.

## Locked predictive metrics

| Arm | Rows | Worlds | Brier | Log loss | AUROC | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 raw | 160 | 40 | 0.246966 | 0.687077 | 0.492647 | 0.581250 | 0.0562 |
| V2 calibrated | 160 | 40 | 0.256230 | 0.708343 | 0.525575 | 0.575000 | 0.0179 |
| Constant 0.50 | 160 | 40 | 0.250000 | 0.693147 | 0.500000 | 0.425000 | 0.0750 |
| Domain base rate | 160 | 40 | 0.259772 | 0.716214 | 0.466752 | 0.450000 | 0.1450 |
| Direct DeepSeek | 160 | 40 | 0.243866 | 0.678828 | 0.581602 | 0.506250 | 0.1159 |
| Call-matched ensemble | 160 | 40 | 0.255148 | 0.713071 | 0.550671 | 0.556250 | 0.1012 |
| Observer panel | 160 | 40 | 0.253332 | 0.701579 | 0.565537 | 0.550000 | 0.0981 |
| Analogical retrieval | 160 | 40 | 0.233121 | 0.645512 | 0.592631 | 0.537500 | 0.1064 |
| Market midpoint | 71 | 24 | 0.150621 | 0.483433 | 0.822464 | 0.774648 | 0.0872 |

V2 raw was extremely unsharp: mean absolute distance from 0.50 was 0.025078. Its calibration slope was 1.011452, but AUROC was below 0.50.

## Paired world-clustered comparisons

Differences are V2 minus baseline; negative favors V2. Every displayed 95% interval crosses zero.

| Comparison | Brier difference [95% CI] | Log-loss difference [95% CI] | Point result |
|---|---:|---:|---|
| V2 raw vs 0.50 | -0.003034 [-0.010534, 0.004557] | -0.006071 [-0.021267, 0.009310] | Slight V2 advantage; inconclusive |
| V2 calibrated vs 0.50 | 0.006230 [-0.035596, 0.047589] | 0.015195 [-0.077268, 0.113508] | Calibrated V2 worse; inconclusive |
| V2 raw vs domain base rate | -0.012806 [-0.051973, 0.026843] | -0.029137 [-0.122823, 0.058204] | V2 advantage; inconclusive |
| V2 raw vs direct DeepSeek | 0.003100 [-0.047121, 0.047269] | 0.008248 [-0.115532, 0.120587] | V2 worse; inconclusive |
| V2 raw vs call-matched ensemble | -0.008182 [-0.065479, 0.044355] | -0.025995 [-0.179972, 0.104149] | V2 advantage; inconclusive |
| V2 raw vs observer panel | -0.006367 [-0.060383, 0.043124] | -0.014502 [-0.149782, 0.104720] | V2 advantage; inconclusive |
| V2 raw vs analogical retrieval | 0.013845 [-0.025492, 0.053174] | 0.041565 [-0.049167, 0.130628] | V2 worse; inconclusive |
| V2 raw vs market midpoint | 0.091767 [-0.007633, 0.172985] | 0.194482 [-0.136137, 0.439723] | Market much better at point estimate; inconclusive interval |

The market comparison uses only the 71 rows with a valid contemporaneous snapshot. The required market-informed V2 comparison is `not_run`; the design forbade a post-hoc blend.

## Model-memory sensitivity

| Locked stratum | Rows | Worlds | Raw Brier | Raw log loss | Raw AUROC | Raw accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Clean-blinded | 92 | 25 | 0.252413 | 0.697982 | 0.489796 | 0.478261 |
| Contamination-susceptible | 28 | 10 | 0.252411 | 0.697973 | 0.500000 | 0.464286 |
| Known-contaminated | 40 | 10 | 0.230625 | 0.654365 | 0.500000 | 0.900000 |

The all-row headline is not robust to model-memory risk. The clean-only result is worse than the 0.50 Brier benchmark, while known-contaminated rows look much easier.

## Causal-coverage results

This is a controlled behavior diagnostic, not real-world causal accuracy. It has 0 real-world trajectory-labeled worlds because the source corpus lacks independently archived intermediate trajectory labels.

| Phase | Independent worlds | Applicable rows | Full active rate | Any meaningful effect | Terminal effect | Delta-count effect | Delta-sequence effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| Actor policy | 43 | 86 | 0.988372 | 0.988372 | 0.534884 | 0.988372 | 0.988372 |
| Mechanism registry | 35 | 70 | 1.000000 | 1.000000 | 0.585714 | 1.000000 | 1.000000 |
| Nonlinear dynamics | 10 | 20 | 1.000000 | 1.000000 | 0.750000 | 1.000000 | 1.000000 |
| Persistence | 60 | 120 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Populations | 25 | 50 | 1.000000 | 1.000000 | 0.360000 | 1.000000 | 1.000000 |
| Networks | 12 | 24 | 1.000000 | 1.000000 | 0.541667 | 1.000000 | 1.000000 |
| Institutions | 52 | 104 | 1.000000 | 1.000000 | 0.740385 | 1.000000 | 1.000000 |
| Recompilation | 15 | 30 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Phase 4's one blocked row, plus the complete absence of Phase 8 and Phase 11 effects, means the diagnostic does not show every phase working.

## Cost, latency, and tests

The 160 locked V2 forecasts recorded 641 model calls and 2,938.591 seconds of summed row latency. Auditable USD cost is unavailable because the mutable API response did not expose billed usage.

The full suite result was 1,067 passed, 12 failed, 2 skipped, and 11 warnings in 172.95 seconds. The stacked PR adds or changes zero test files, as requested. The strict test gate fails; the exact nodes and classifications are in `full_test_suite_report.json`.

## Explicit answers

1. Temporal tier: Tier C, `causally_blinded_historical`.
2. Immutable exact model version: no.
3. Every question after a documented model snapshot/cutoff: no such snapshot/cutoff claim is available under Tier C.
4. Exactly 100 representative worlds: yes.
5. Exactly 400 primary V2 forecasts: yes.
6. Every row invoked all supervisors: yes, 11 records per row.
7. Every relevant phase executed: formally yes, substantively no because Phase 2 degraded internally.
8. Every irrelevant phase explicitly no-op: yes in the representative records.
9. Relevant phases blocked: zero in representative records; one Phase 4 row in causal coverage.
10. Terminal probability from terminal `WorldState`: yes for all 400 representative rows.
11. Live internet unavailable: yes; only the allowlisted model endpoint was reachable.
12. Outcomes inaccessible until governed scoring: yes.
13. Baseline evidence parity: yes on 400/400 rows.
14. Phase 12 fit only on calibration: yes.
15. Phase 12 selected only on validation: yes.
16. Locked test opened once: yes.
17. V2 versus direct DeepSeek: V2 raw was slightly worse; interval inconclusive.
18. V2 versus call-matched ensemble: V2 raw was slightly better; interval inconclusive.
19. V2 versus markets: V2 raw was much worse at the point estimate on 71 rows; interval crossed zero. Market-informed V2 was not run.
20. Every causal phase working: no; Phase 4 had one blocked row and Phases 8/11 had zero effects.
21. Production eligible: no.
22. Remaining work: prospectively rerun all forecasts under the repaired fingerprint, repair Phase 4/8/11 causal behavior, add the market-informed arm before outcomes, reconcile tests, obtain a stronger model temporal tier, and validate on real-world trajectory labels plus additional providers.

## Requirement matrix

| Requirement | Required | Achieved | Evidence | Pass/Fail |
|---|---|---|---|---|
| Independent representative worlds | 100 | 100 | `frozen_selection_manifest.json` | Pass |
| Cutoffs per world | 4 | 4 | three canonical forecast files | Pass |
| Primary V2 forecasts | 400 | 400 | three canonical forecast files | Pass |
| Temporal tier | Causal blinding | Causal blinding | temporal audit + probes | Pass |
| Phase records | 11 on every row | 4,400 total | canonical forecasts | Pass |
| Relevant phases active without internal error | All | Phase 2 error on 400 rows | canonical forecasts | Fail |
| Irrelevant phases explicit no-op | All | All representative records | canonical forecasts | Pass |
| Blocked representative phases | 0 | 0 | canonical forecasts | Pass |
| Terminal source | Terminal WorldStates | All 400 | canonical forecasts | Pass |
| Evidence cutoff safety | All 400 | All 400 | evidence capsule manifest | Pass |
| Outcome/network isolation | Required | Proven | isolation manifests | Pass |
| Fair required baselines | 400 | 400 | canonical baselines | Pass |
| Phase 12 governance | Calibration then validation | Correct | fit + selection artifacts | Pass |
| Locked open count | 1 | 1 | exclusive ledger | Pass |
| Market-informed V2 comparison | Required | Not run | locked score | Fail |
| Causal protocol | 60 worlds, 120 rows | Complete | causal artifact | Pass |
| Every causal phase active | Required | One Phase 4 block | causal artifact | Fail |
| Meaningful effect for every causal phase | Required | Phase 8/11 zero | causal artifact | Fail |
| Full repository suite | 0 failures | 12 failures | suite report | Fail |
| No test files in stacked diff | 0 | 0 | suite report + git diff | Pass |

| Phase | Invoked on every row | Relevant recall | False causal activation | Blocked rate | Meaningful ablation effect | Status |
|---|---|---:|---:|---:|---:|---|
| Phase 1 compiler | Yes | N/A | N/A | 0.000000 | N/A | Formal pass |
| Phase 2 evidence | Yes | N/A | N/A | 0.000000 | N/A | Fail: internal error on 400 rows |
| Phase 3 posterior | Yes | N/A | N/A | 0.000000 | N/A | Formal pass |
| Phase 4 actor policy | Yes | 0.981982 | 0.044944 | 0.000000 representative; 1/86 causal | 0.988372 | Fail: one causal block |
| Phase 6 registry | Yes | 1.000000 | 0.064516 | 0.000000 | 1.000000 | Pass |
| Phase 7 nonlinear | Yes | 0.976190 | 0.012658 | 0.000000 | 1.000000 | Pass |
| Phase 8 persistence | Yes | N/A | N/A | 0.000000 | 0.000000 | Fail: no effect |
| Phase 9 populations | Yes | 1.000000 | 0.000000 | 0.000000 | 1.000000 | Pass |
| Phase 9 networks | Yes | 1.000000 | 0.019737 | 0.000000 | 1.000000 | Pass |
| Phase 10 institutions | Yes | 1.000000 | 0.010309 | 0.000000 | 1.000000 | Pass |
| Phase 11 recompilation | Yes | 0.978261 | 0.000000 | 0.000000 | 0.000000 | Fail: no effect |
