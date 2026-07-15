# World Model V2 Post-Snapshot Benchmark Architecture

## Scope and status

This benchmark is a stacked, separate PR built from `claude/world-model-v2-full-activation-replay-v2`. It does not modify PR #100, does not start Phase 13, and does not merge either PR. The representative run is bound to runtime fingerprint `79537cdec279fd8f`. The intermediate forward repair at `b1da180` had fingerprint `66c735b4201edc17`; after integrating the latest PR #100 head, the final repaired runtime has fingerprint `1a77f7a553aba15d`. Neither repaired fingerprint has a product-performance claim.

The execution produced all required forecasts and a single-open locked score, but it fails the strict completion standard. Phase 2's capsule adapter degraded on every primary row, Phase 4 had one blocked controlled-ablation row, Phases 8 and 11 had no meaningful controlled ablation effect, the market-informed comparison was not run, and the final merged-tree repository suite has 3 failures. The authoritative status is `experiments/results/post_snapshot_benchmark/exact_completion_gate_report.json`.

## Temporal-safety decision

The model arm is DeepSeek V4 Flash through the hosted `deepseek-v4-flash` alias.

- Tier A was rejected because the hosted model had no immutable serving revision, documented knowledge cutoff, or auditable server-side weight identity. A downloadable checkpoint exists, but the required local inference infrastructure and an official cutoff were unavailable.
- Tier B was rejected because the provider did not attest a stable hosted version plus knowledge cutoff.
- Tier C, `causally_blinded_historical`, was selected. It uses stable pseudonyms, transformed dates, stripped identifying text, blinded evidence, and six probes per row.

Tier C reduces and measures model-memory risk; it does not prove the model lacked outcome knowledge. The run must not be described as an immutable-snapshot benchmark.

## Representative benchmark

The pool builder examined 3,000 source events, retained 1,430 binary-eligible contracts, clustered them into 1,174 independent worlds, and selected exactly 100 without outcome access. The selected domain counts are 30 sports, 25 other, 20 crypto, 15 geopolitics, 5 politics, 3 culture, 1 technology, and 1 weather/science.

Worlds were ordered chronologically by question-open time and frozen into:

- 40 calibration worlds × 4 cutoffs = 160 rows;
- 20 validation worlds × 4 cutoffs = 80 rows;
- 40 locked worlds × 4 cutoffs = 160 rows.

The 100-world selection, clustering map, split map, and 400 forecast cutoffs were immutable before model execution. A failed selected row could be repaired and retried but never replaced.

## Evidence and isolation

Each of the 400 cutoffs has one immutable, blinded capsule. The manifest records each source byte hash and its first-proven availability timestamp; all 400 pass the cutoff rule. Original records and pseudonym mappings remain outside the forecast mount.

Forecast and baseline workers ran under macOS Sandbox. The child could read code, its blinded input shard, and its capsule. It could not read any resolution store, pseudonym map, unblinded vault, or canonical source archive. Network access was denied except through a loopback CONNECT relay restricted to `api.deepseek.com:443`. The API credential moved from parent memory through an inherited anonymous pipe and was not placed in argv, environment variables, logs, or artifacts.

## Full-system row contract

Every primary row contains exactly 11 `PhaseExecutionRecord` objects for:

1. compiler;
2. evidence;
3. posterior;
4. actor policy;
5. mechanism registry;
6. nonlinear mechanisms;
7. persistence;
8. populations;
9. multilayer networks;
10. institutions;
11. dynamic recompilation.

The only ordinary no-op is `no_op_causally_irrelevant`. A relevant phase must be `causally_active`; `blocked_*` and `execution_failed` fail qualification. Terminal probabilities are empirical readouts from terminal `WorldState` distributions, and direct terminal rate modulation is prohibited.

The frozen qualification implementation had a defect: its capsule adapter lacked fields required after evidence recompilation, and the core supervisor trusted an earlier `executed=True` marker after an `AttributeError`. Consequently, all 400 Phase 2 records were formally `causally_active` while preserving `evidence_error: AttributeError` internally. The final audit treats all 400 as degraded and fails the strict full-runtime gate. A forward-only code repair completes the adapter contract and maps core exceptions to `execution_failed`; it does not change the scored corpus.

## Baseline parity

Every representative row has these baselines:

- constant 0.50;
- domain base rate fit at the independent-world level;
- direct single-call DeepSeek;
- direct ensemble with exactly the same call budget as V2;
- observer panel;
- analogical retrieval;
- contemporaneous market midpoint where available.

All model baselines receive the identical blinded question and capsule hash. Required model arms complete on 400/400 rows, and the ensemble is exactly call-matched on 400/400. The representative V2 arm was prospectively market-blind; a market-informed V2 arm was not generated, and no post-outcome blend was permitted.

## Calibration and locked scoring

Phase 12 candidates were fit only on the pre-open-clean subset of calibration rows: 140 rows from 35 worlds. Selection used only the pre-open-clean validation subset: 76 rows from 19 worlds. `task_family_conditioned` strictly beat identity on validation Brier and log loss and was frozen before locked execution.

The final scorer was frozen after a pre-open static repair added the missing constant-0.50 arm. It verified all forecast, baseline, calibrator, support, market, and leakage-probe hashes before creating an exclusive ledger. The locked resolution store was read once. The ledger records one read, its resolution-store hash, and the frozen score hashes.

## Separate causal-coverage benchmark

The causal diagnostic is not representative accuracy. It uses 60 controlled simulated worlds, two cutoffs per world, and matched common-randomness ablations. Every ablation measures three targets: terminal-distribution total variation, StateDelta count, and StateDelta sequence hash.

It covers at least 10 independent worlds for actor policy, mechanism registry, nonlinear dynamics, persistence, heterogeneous populations, multilayer networks, institutions, and natural structural change. The source corpus lacks independently archived real-world intermediate trajectory labels, so the diagnostic has 0 real-world worlds and must not be used as a causal-accuracy claim.

## Resumability and immutability

Forecasts and baselines use keyed immutable attempt files plus atomically rebuilt canonical JSONL views. Successful attempts are never overwritten; retries remain available for forensics. Calibration, validation, locked forecasts, locked baselines, scorer inputs, scoring code, and the single-open ledger were committed at separate checkpoints.
