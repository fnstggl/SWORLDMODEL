# WMV2 Post-Snapshot Benchmark — architecture

## Model-snapshot logic (temporal-safety tiers)

Verified live: the DeepSeek API serves `deepseek-v4-flash`; open weights exist at
`hf:deepseek-ai/DeepSeek-V4-Flash@60d8d707…` (created 2026-04-22; official release 2026-04-24). Full audit:
`experiments/replay_v3/model_temporal_safety_audit.json`.

- **Tier A (immutable checkpoint)**: blocked — 284B params, no GPU inference here; the hosted alias is not
  proven byte-identical to the checkpoint. Recorded with the exact unblock path; did NOT stop the benchmark.
- **Tier B (provider-attested post-cutoff) — SELECTED**: exact live alias + officially documented release
  date, and **every benchmark question opened after 2026-04-24** (archive server timestamps), so the
  ordering `S < Q ≤ T < R` holds per row. Even silently-updated serving weights cannot contain outcomes of
  events that had not happened at release unless retrained on them post-release — which the probes measure.
- **Belt-and-braces**: the primary arm is ADDITIONALLY causally blinded and runs all six leakage probes per
  row (`provider_attested_post_cutoff_blinded`). The HF repo shows post-release activity
  (lastModified 2026-06-22), so serving mutation is treated as a live, measured risk — never assumed away.

## Full-system row contract

Every V2 row: frozen capsule → blinding → six probes → `simulate_world(blinded_q, prebuilt_bundle=…)` with
`deepseek-v4-flash` (thinking disabled, recorded) → 11 PhaseExecutionRecords (derived from branch logs) →
qualification: record coverage = 11, **zero blocked relevant phases**, `terminal_source =
terminal_world_states`, p_yes from the terminal distribution (no LLM writes it) → fair baselines on the
byte-identical capsule text + blinded question with the same model (direct single call, call-matched
ensemble of 3, observer panel of 3, analogical retrieval) → frozen audit row (hash stamped). A row failing
qualification carries `failure_reason` and never silently disappears.

## Representative vs causal-coverage separation

- **Representative vault** (`experiments/replay_vault_v3/events.json`): frozen server-side eligibility
  (opened ≥ 2026-04-25, resolved ≤ 2026-07-12, ≥12d lifetime, ≥$1k volume, definitive binary resolution),
  chronological fill with a domain cap — **no phase quotas, no hand-selection**; correlated contracts
  grouped into one world at the archive's own event level. Chronological splits: earliest 40 calibration /
  next 20 validation / latest 40 locked. 4 cutoffs per world at 15/40/65/88% of archived lifetime with
  exact at-or-before market ticks.
- **Causal-coverage vault** (`coverage_events.json`): frozen category rules over the REMAINING pool;
  proves phase behavior (2+ cutoffs, trajectory targets, matched ablations); never merged into the
  accuracy headline.
- The eligible pool in the 2.5-month post-snapshot window is recorded honestly in
  `candidate_pool.json`; Kalshi/Metaculus are proxy-blocked (see `blocker_decision_records.json`). If the
  pool cannot fill 100 worlds, the exact shortfall is reported — no ineligible event is substituted.

## Evidence & isolation

Capsules (Wikipedia revision-pinned + Wayback bytes; sha256; `first_proven_available_at ≤ cutoff` enforced
at access) are frozen to disk BEFORE forecasting; the replay path consumes `prebuilt_bundle` only (no
retrieval call sites). Resolutions + trajectory targets live in the sealed store (`REPLAY_SCORER=1`,
single-open locked log, freeze-hash verification). The LLM API is the forecaster's only network
dependency; OS-level egress lock is unavailable here — recorded per row, isolation reported PARTIAL.
Design review: `benchmark_scientific_design_review.json`; adversarial review:
`benchmark_red_team_report.json`; blocker records: `blocker_decision_records.json`.
