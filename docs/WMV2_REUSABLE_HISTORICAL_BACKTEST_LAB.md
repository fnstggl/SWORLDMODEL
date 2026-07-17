# WMv2 Reusable Historical Backtest Laboratory

Permanent home: `historical_backtests/` (see its README for commands). This document records the
design contract so future WMv2 versions rerun the same benchmark without being able to touch the
answers.

## What it is

A walk-forward, outcome-isolated harness that executes the **complete production World Model V2
simulation** — the canonical facade `swm.world_model_v2.unified_runtime.simulate_world`, the same
call graph a live user question takes — for every `(question, forecast_cutoff)` row of a sealed
historical benchmark, using a **period-bounded open-weight model** through a strictly pinned
OpenRouter endpoint, then scores the sealed predictions against what actually happened.

## The canonical production call graph (documented at build time, enforced per row)

`simulate_world` at the frozen commit runs: Phase-1 universal compiler → Phase-2 as-of evidence
(injected frozen ReplayBundle; live retrieval never occurs in benchmark rows) →
evidence-conditioned recompile → Phase-3 posterior over hidden state → Phase-9/10
population/network/institution instantiation + rule normalization → fidelity layer (resolution
criterion, actor decomposition, scheduled facts, grounded mode-scoped stances with graded
control) → activation synthesis + trajectory depth → canonical mode graph + pathway processes +
decision structures → event-time conversion (hazard rounds, absorption monitor, persistence,
stance reviews, capacity attrition, contested channels) → Phase-11 recompilation → the one
terminal funnel (Phase-8 persistence rollout firing Phase-4 actor policies with actor-local
views, feasibility, typed actions; Phase-6/7 registry + nonlinear operators; Phase-10
institutional decisions) → ≥200-particle rollout → first-passage readout (binary deadline
questions are `P(yes)=F(deadline)` with polarity) → mandatory phase supervision
(PhaseExecutionRecords for the full current phase contract, 11 phases at freeze).

Per-row **full-run proof** (stored in every ledger row): entrypoint, runtime fingerprint, commit,
all PhaseExecutionRecords with statuses and StateDelta counts, operator delta census, actor-action
delta counts, particle count, terminal source, integration failures, fallbacks. The
**qualification gate** (`framework/qualify.py`) fails any row missing a phase record, with a
blocked causally-relevant phase, without the first-passage readout, below the 200-particle floor,
or with expected-but-absent actor decisions. Sentinel tests prove (a) the legacy simplified
entrypoint (`pipeline.simulate`) is never touched, and (b) removing one required phase fails the
row rather than passing silently.

## Temporal integrity

1. **Model boundary**: `model_release < question_open <= cutoff < resolution`, release-based
   (`effective_temporal_boundary` = checkpoint public release), fail-closed
   (`models/registry.assert_temporal_ordering`).
2. **Evidence**: deterministic queries (proper nouns + criterion nouns + date window; no frontier
   model performs ANY case-dependent operation); GDELT/RSS are URL discovery only; every news item
   must resolve to Wayback **archived bytes** captured ≤ cutoff (server-verified timestamps,
   hashed) or is rejected; Wikipedia = exact revision ≤ cutoff (revid proof); contamination scrub
   (post-cutoff dates, resolution language) on every item; capsules frozen + sealed before
   simulation; `first_proven_available_at <= cutoff` re-enforced at access.
3. **Fitted parameters**: `framework/packs.py` — monthly survival-pack snapshots fit only on
   markets resolved before the snapshot boundary; `load_pack(as_of)` selects the latest snapshot
   before the cutoff and fails closed if any training row resolved at/after it; intention-HR and
   coupling effects run on documented priors with
   `fallback_reason: insufficient_pre_cutoff_fit_data` recorded (no labeled pre-cutoff corpus
   exists yet). Regression test: a future-resolved case cannot affect an earlier forecast.
4. **Runtime freeze**: `framework/freeze.py` writes commit, phase contract, registry hash, vault
   sha, capsule shas, pack shas, seed, particle floor before locked forecasting; scorer verifies.

## Outcome isolation

`resolution_store.py` raises at import without `REPLAY_SCORER=1`; the runner asserts it is not in
a scorer process and holds no reference to the store (sentinel test); question vault contains no
outcomes (asserted at load); scoring is a separate process; every outcome access appends to
`outcome_access_ledger.jsonl`; rotating-locked outcomes open exactly once per runtime
(`locked_access_log.json`), then the split is CONSUMED.

**Limitation stated plainly**: isolation binds the code and the model, not the developers' memory
of public history. Reusable-regression results are engineering evidence
(`REUSABLE_DEVELOPMENT_BACKTEST`); scientific claims require a fresh rotating holdout
(`ROTATING_SEALED_HOLDOUT`) or the live forward vault.

## Reuse across WMv2 versions

Results are append-only under `results/<benchmark>/runtime_<commit>/`. Rerun with
`tools/run_benchmark.py`; compare with `tools/compare_runs.py` (aggregate, by scale/domain,
qualification and failure rates, per-case deltas, cost). A CI-suitable subset run is
`--split reusable_regression --limit 10 --no-baselines`; full OpenRouter runs are manual by
design (cost is never grounds for reducing the pipeline — too-expensive models are
diagnostic-only or not executed).
