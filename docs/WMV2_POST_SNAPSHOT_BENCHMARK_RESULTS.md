# WMV2 Post-Snapshot Benchmark — results (locked test opened ONCE)

**Machine-readable source of truth:** `experiments/results/replay_v3/scores_v3.json`,
`forecasts_{calibration,validation,locked_test}.jsonl`, `locked_access_log.json`,
`runtime_freeze_manifest.json`. This doc summarizes; the artifacts govern.

## Exact counts

- Tier: **B `provider_attested_post_cutoff` + causal blinding + six probes** (deepseek-v4-flash, released
  2026-04-24, open-weights reference hf@60d8d707; thinking disabled, temp 0.2).
- Vault: **exactly 100 worlds** (frozen chronological selection from a 148-world eligible pool; window
  2026-04-25→2026-07-12; correlated contracts clustered at the archive's event level; 100/100
  timestamp-matched market snapshots). Splits chronological: 40 cal / 20 val / 40 locked.
- Forecasts: **400 attempted; 388 fully qualified** (all 11 phase records, zero blocked relevant phases,
  terminal from world states). **12 preserved failures** (9 phase-4 integration blocks under compile
  variance, 3 compiler/serving crashes) — the "all 400 complete" gate is **FAILED honestly**; failed rows
  were retried 2-3× without outcome access and remain in the audit table.
- Runtime frozen (fingerprint `9ba1511f5a47a619`) after a 12/12-clean preflight; one mid-wave engineering
  repair (relation-name normalization) was made **without accessing any outcome**, the runtime refrozen,
  and every earlier forecast invalidated + regenerated.

## Leakage census (per-row probes; clean = clean_blinded only enters the headline)

cal+val: 140 clean_blinded / 82 contamination_susceptible / 13 known_contaminated.
locked: **91 clean rows (34 worlds)** / 61 contamination_susceptible / 1 known_contaminated / 0 tampered.
The recognition probe fires often (markets reference real named entities) — susceptible rows are excluded
and censused, not hidden.

## Phase 12

Fit on calibration worlds only (identity/Platt/isotonic); selected on validation only: **identity** (no
method beat identity — recorded as the honest outcome; V2 raw = V2 calibrated on the locked test).

## Locked-test comparisons (91 clean rows, 34 worlds; world-clustered bootstrap CIs)

| Arm | Brier ↓ | 95% CI | log loss | AUROC | dir. acc |
|---|---|---|---|---|---|
| **Full V2 (raw = calibrated)** | **0.2465** | [0.224, 0.270] | — | 0.555 | 0.582 |
| Market midpoint @cutoff | 0.2476 | [0.198, 0.295] | 0.686 | 0.581 | 0.505 |
| Constant 0.5 | 0.2500 | — | 0.693 | 0.500 | 0.418 |
| Calibration base rate | 0.2616 | [0.242, 0.280] | 0.716 | 0.500 | 0.418 |
| Observer panel (same evidence/model) | 0.2806 | [0.192, 0.380] | — | 0.572 | 0.571 |
| Analogical retrieval | 0.2946 | [0.237, 0.350] | 0.789 | 0.534 | 0.451 |
| Call-matched ensemble (3× direct) | 0.3470 | [0.252, 0.454] | — | 0.440 | 0.571 |
| Direct single-call DeepSeek | 0.3644 | [0.243, 0.502] | — | 0.423 | 0.582 |

**Honest verdict.** The full unified V2 **beats every same-evidence LLM baseline** (direct, ensemble,
panel, analogical) and the base-rate baseline on Brier, and is nominally ahead of the timestamp-matched
market midpoint (0.2465 vs 0.2476) — but its CI **includes the 0.50-constant baseline (0.25)**, its AUROC
(0.555) shows weak discrimination, and the margin over the market is not significant. **The system is NOT
production eligible** under the pre-registered gates (must beat the 0.50 baseline decisively). The result
is preserved as-is; this locked benchmark version is consumed and will not be tuned against.

## Incomplete components (reported, not hidden)

- **Causal-coverage benchmark**: vault found only 5/60 eligible coverage worlds in the post-snapshot
  window (pool exhausted; Kalshi/Metaculus proxy-blocked) — NOT executed; matched phase ablations under
  the redesigned (post-rate-modulation-removal) runtime remain to be re-measured there.
- **Tier A immutable-checkpoint arm**: blocked (no GPU inference); unblock path in the model audit.
- **Secondary robustness arm**: not run (single primary arm only).
- OS-level forecaster network isolation: PARTIAL (process-level; recorded per row).

## Reproduction

```
PYTHONPATH=. python experiments/replay_v3/build_vault3.py
PYTHONPATH=. python experiments/replay_v3/run_benchmark.py --capsules
PYTHONPATH=. python experiments/replay_v3/run_benchmark.py --split calibration|validation|locked_test
REPLAY_SCORER=1 PYTHONPATH=. python experiments/replay_v3/score3.py [--open-locked]  # locked opens ONCE
```
