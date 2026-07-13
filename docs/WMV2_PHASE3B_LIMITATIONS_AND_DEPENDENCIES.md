# WMv2 Phase 3B — Limitations and Dependencies

*What this repair run does NOT establish, where it is under-powered, and what later phases must not assume.
Written to be read alongside `WMV2_PHASE3B_FAILURE_ANALYSIS.md`, `WMV2_PHASE3B_ARCHITECTURE_AND_REPAIR.md`,
and `WMV2_PHASE3B_REAL_VALIDATION.md`. Null and negative findings are preserved.*

## Statistical power (the central limitation)

The mandate's target was ≥150 resolved questions total and ≥75 in the locked test. This run did not reach
that scale: the diagnostic/development set is **23** questions and the locked final test is **~34** questions
(one production-path run per arm is ~60–90 s of live retrieval + LLM, so a 75-question multi-arm locked test
was not achievable within this run's compute/latency budget). Consequences:

- A paired-bootstrap 95% CI on ~34 questions is **wide**. A null result on the locked test is therefore
  **"not demonstrated," not "disproven"** — the run can fail to detect a real small improvement.
- Per-domain breakdowns have 1–5 questions per domain and are **directional only**; no domain-level acceptance
  claim is made.
- The verdict is reported with this power caveat attached. If the locked test does not clear the
  pre-registered gates, the honest conclusion is **Phase 2 remains the production default** and the repair is
  **not** production-eligible — not that Phase 3 is fixed.

## What the repair is, and is NOT

- The repair is a **calibration + learned Phase-2/Phase-3 stack + evidence-quality gate + real reference-class
  priors**. It is fit on the DEV split and frozen before the locked test.
- It is **not** a from-scratch scenario-specific causal-latent re-architecture. Part C of the mandate asks for
  typed latent states (actor intent, institutional authority, procedural feasibility, …) with their own
  observation models and mechanism consumers. Building and *fitting* those to real data is a multi-corpus
  effort beyond this run; the generic outcome-rate posterior is retained as the Phase-3 signal but is now
  **calibrated, gated, and subordinate to Phase-2** rather than overriding it. Typed causal latents are a
  documented dependency for a future run, not a claim made here.

## Observation models (Part E) — partial

- The directional observation models are **calibrated by temperature/shrinkage** (a global `gamma`, a
  no-information mixture, an optional posterior temperature) fit on the DEV split. This is a real calibration,
  but it is **not** a hierarchically pooled, per-claim-class / per-source / per-horizon likelihood fit on a
  large labeled corpus. That larger fit needs a labeled historical claim→outcome corpus that this run does not
  build. The single global calibration is the honest, data-supported subset.

## Reference priors (Part D) — curated, not a live connector

- `phase3b_reference_priors.py` is an **auditable curated table** of class-level base rates with provenance
  (period, eligibility, sample size, transport risk), using only class-level pre-as-of frequencies — no
  outcome-specific hindsight. It is **not** a live base-rate connector, and its coverage is limited to the
  evaluation domains. Rounded, conservative counts; a reviewer can check each entry. Domains without a matching
  class keep the generic prior (honestly labeled).

## Reproducibility and drift

- The numeric posterior pipeline is deterministic given frozen (plan, bundle, tags, seed); the offline grid
  posterior reproduces the production particle posterior to <0.01 mean absolute error.
- **Live retrieval drifts**: re-running the production path on the diagnostic questions produces different
  evidence than the committed backtest (news changes over wall-clock). The committed negative backtest is
  **frozen and preserved**; the fresh capture is used only as the DEV substrate. The locked test records its
  own retrieval date and is scored on its own frozen forecasts.

## Leakage

- The locked test uses **event-family- and temporally-disjoint** questions (different institutions, entities,
  contests, countries, leagues) so no diagnostic family recurs. The as-of strict-retrieval discipline
  (paired `after:`/`before:`, per-document temporal verification, claim-level leakage audit) is unchanged from
  Phase 2, and a stratified manual audit is re-run. Residual risk: RSS publication dates can be imprecise, and
  a widely-reported outcome can be alluded to in pre-as-of coverage; the manual audit checks a sample but not
  every document.

## What later phases MUST NOT assume

- Do **not** assume Phase 3's posterior should override Phase 2. The diagnosed failure mode is exactly that
  override. The consuming interface is the **selected/blended** forecast, with a safe Phase-2 fallback.
- Do **not** treat the generic outcome-rate posterior as a calibrated real-world probability. It is a
  degraded fallback signal, calibrated and gated.
- Do **not** cite DEV-set improvement as validation — it is fit there. Only the locked-test number counts, and
  only with its power caveat.

## Open dependencies (for a future, larger run)

1. A labeled historical claim→outcome corpus to fit hierarchical, per-claim-class observation models (Part E).
2. Typed scenario-specific causal latents with operational definitions, priors, observation models, and
   mechanism consumers (Part C).
3. A live reference-class base-rate connector with versioned data sources (Part D).
4. ≥75-question locked test (and ≥150 total) for adequately powered per-domain acceptance (Parts I/K).
5. Genuine as-of crowd/prediction-market baselines where licensing permits (Part J).
