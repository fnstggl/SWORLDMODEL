# World Model V2 Post-Snapshot Benchmark Forensics

## Frozen identities

| Item | Identity |
|---|---|
| Scored runtime fingerprint | `79537cdec279fd8f` |
| Forward repaired runtime fingerprint | `66c735b4201edc17` |
| Representative selection | `4d4acab326144a794f7e531e0232bf87ede662aaa40f5644cf3d0886d4dea822` |
| Locked forecast file | `5005c8fe098f59feb157dabd34db37f3716c43d6b0cb26764735c12f9e0e1e6c` |
| Locked baseline file | `8acf4924d07db7b1f133060aaf0c823904dae970f1416f0a9d2e05e4169eacf8` |
| Final scorer source | `31b8596cee962dd2dd526a99203606871791ec19896fab68f3c67d7a75746ce8` |
| Locked resolution store, learned at the one open | `d0566aa66bbaa0d682d8f8871f68e2bbfa820c20b6a48c326ddcbc96fbce74c7` |
| Locked aggregate score | `5a2edd4f721e4b77985939be3a924f25d03978fe0f0f01f92fa20385686ea239` |
| Locked scored rows | `b07ec369d295ae504234bd4cc85632e2b18fb22e728a760bf3957d7b4d4c62ee` |
| Causal diagnostic | `0a77d86ee0d5d441bf1e43fa08e69a27d936a171d6c14aaf12ed662ddcdc1e14` |
| Exact completion audit | `ceabf2c54145ccea868ab27c57919badf16725853a0877f0c02384f97ca5322c` |

## Scorer access proof

The final scorer freeze states that the ledger and prior score artifact were absent and that the outcome-store hash was deliberately unknown. At `2026-07-15T17:43:23.772046Z`, the scorer created the exclusive ledger only after verifying 160 locked forecast hashes, 160 locked baseline hashes, the Phase 12 artifacts, market snapshots, and all leakage probes. It completed at `2026-07-15T17:43:24.674757Z` with `read_count: 1`.

Calibration outcomes were opened for candidate fitting, validation outcomes for method selection, and locked outcomes once for final scoring. Locked outcomes were never used for code changes, calibration, model selection, retries, or row replacement.

## Representative row trace

Locked row `pm_510803` at cutoff `2026-05-30T14:52:07.669085Z` provides a compact trace:

- forecast SHA-256: `22493839470ddccb04146e78ddb74ad727fabc322f63e0746860a6a80de5a46c`;
- evidence byte SHA-256: `6a61ef8b8897608c3af90c7166e6b81ac37369cb9cb130ad6c76510d0976a762`;
- terminal particles: 80;
- StateDeltas: 800;
- terminal readout variable: `outcome`;
- raw P(YES): 0.475;
- active records: compiler, evidence, posterior, actor policy, mechanism registry, persistence, populations, and institutions;
- explicit causal no-ops: nonlinear mechanisms, networks, and recompilation.

This row, like all 400 primary rows, carries 11 records, zero formally blocked relevant phases, and a terminal-world-state readout. It also preserves the Phase 2 `evidence_error: AttributeError` degradation described below.

## Phase 2 qualification defect

All 400 primary rows have:

- formal Phase 2 status: `causally_active`;
- internal Phase 2 validation status: `evidence_error: AttributeError`;
- empty Phase 2 error list because the runtime caught the exception;
- formal `full_system_qualified: true`.

The failure occurred after the capsule adapter had supplied claims but before evidence-conditioned plan replacement completed. The adapter omitted `requirement_coverage` and `actor_visibility`, while the supervisor retained an earlier executed flag. The scored results remain immutable and fail the strict integration gate. Commit `b1da180` adds the missing adapter fields and makes any core `_error:` state an `execution_failed` record for future runs. That creates a new runtime fingerprint and invalidates reuse of the scored Phase 12 artifacts.

## Phase 11 controlled example

Controlled world `court_6`, cutoff 0, row SHA-256 `c70118133372f58f1a7e364f8a5ca42892e693365f7a8c767b4758ea5b6ed84b`, labeled Phase 11 relevant and recorded it `causally_active`. Its matched Phase 11 removal produced:

- identical affirmative probability: 0.4848 versus 0.4848;
- identical StateDelta count: 528 versus 528;
- identical StateDelta sequence SHA-256;
- total variation: 0.0.

The same zero-effect pattern holds for all 30 Phase 11 cutoff rows across 15 independent controlled worlds. The representative corpus assessed Phase 11 as irrelevant on all 400 rows, so its only substantive test here is the separate controlled diagnostic—and it failed.

## Phase 4 blocked example

Controlled world `legvote_2`, cutoff 1, row SHA-256 `59c8ba368712c966ee7b99670189581e00494eb8f62bac3321f18f143875a4b8`, returned `blocked_no_mechanism` for Phase 4. The matched removal produced no change in terminal distribution, StateDelta count, or sequence. The other 85 applicable Phase 4 rows showed a meaningful effect, for a rate of 0.988372.

## Phase 8 ablation result

Phase 8 was labeled relevant and `causally_active` on all 120 controlled rows. Yet removing it changed none of the three targets on any row. This is preserved as a scientific failure, not interpreted as evidence that default-on persistence works.

## Model-memory forensics

Before outcomes opened, the six-probe audit classified 344/400 rows `clean_blinded` and 56/400 `contamination_susceptible`. After the one locked open allowed outcome-dependent validation of the name-only probe, the locked rows split into:

- 92 clean rows across 25 worlds;
- 28 contamination-susceptible rows across 10 worlds;
- 40 known-contaminated rows across 10 worlds.

Raw V2 Brier was 0.252413 on clean rows versus 0.246966 on all rows. Known-contaminated rows had 0.90 directional accuracy, demonstrating that the all-row result is sensitive to model memory. All frozen worlds remain visible; none was silently replaced.

## Preserved failures and retries

Earlier activation and preflight failures remain in `failure_taxonomy.json`, including terminal-readout failures, missing required activations, and a compiler empty-decomposition failure. The causal run also preserved a transient `legvote_5` compiler failure before a successful retry. Immutable row shards retain every attempt while canonical views retain one successful row per key.

The final merged-tree full suite completed with 1,077 passed, 3 failed, 2 skipped, and 11 warnings. The remaining failures are a missing dataset registry, a pre-existing backtest-toggle mismatch, and missing `fastapi`. The earlier pre-merge child run—1,067 passed and 12 failed—is preserved in the same machine report; PR #100's newer runtime resolved the stale phase-contract failures. Per the requested stacked-PR constraint, the diff contains zero test files.
