# WMV2 Post-Snapshot Benchmark — forensics

## Representative complete row (locked wave)

`pm_510376 @ 2026-06-05` (frozen audit row in `forecasts_locked_test.jsonl`): capsule with archived
revision-pinned bytes (sha256 + `first_proven_available_at` < cutoff, capsule access rule enforced) →
blinded question + evidence → six probes recorded (prompts + outputs) → full supervised runtime → 7-8
causally-active phases, remainder explicit no-ops, **zero blocked**, `terminal_source=terminal_world_states`,
p_yes=0.3699 → baselines on the byte-identical evidence → freeze hash. Scorer verified every hash before
outcome access (0 tampered across 400 rows).

## Engineering repairs (all without outcome access; each → refreeze + regeneration)

1. Phantom blocked phases: the supervisor re-derived relevance AFTER synthesis mutated plan text (an
   institutional "threshold rule" phrase lexically fired the nonlinear detector). Fix: the single
   pre-synthesis relevance verdict is reused by the supervisor.
2. `rate_modulation` (Part 4): removed from the terminal resolver entirely; consumer state is consumed by
   MECHANISMS (institutional member propensity; the aggregate-outcome realization operator) inside the
   event loop; prohibition regression test added. Prior ablation passes that relied on the resolver channel
   are treated as invalidated.
3. Organizations as strategic actors: P4 blocked when only org-type entities were declared; the actor
   substrate now includes institutions/companies (the action ontology always had those families).
4. Relation-name normalization: compiler rel names outside the registry were dropped at materialization,
   leaving the diffusion mechanism an empty graph (`blocked_no_mechanism`); now normalized onto registered
   layers, never dropped.

## Preserved failures (12/400; taxonomy in the audit table)

- 9 × `blocked_relevant_phases:[phase4_actor_policy]` — phase-integration failure under compile variance
  (decision event synthesized but not consumed in specific compiled shapes); retried 2-3×; preserved.
- 3 × `phase_record_coverage_0_of_11` — compiler/serving exceptions (execution_failed); retried; preserved.
- No failure was repaired or replaced after outcome access; no world was substituted for performance.

## Locked access proof

`experiments/results/replay_v3/locked_access_log.json` — exactly one open (timestamp + calibrator
`identity` + row count recorded); the scorer refuses a second open. The fit/select scoring pass that
preceded locked generation ran WITHOUT `--open-locked` and read only calibration/validation outcomes.

## Leakage probe examples

- `known_contaminated` (1 locked row): the name-only probe stated the actual resolution with confidence —
  excluded from the clean census, preserved in the audit.
- `contamination_susceptible` (61 locked rows): the recognition probe identified the real event behind the
  blinded packet (market questions name public entities); excluded from the headline.
- The no-evidence probe on blinded twins concentrates near 0.5 — blinding removes memorized signal.

## What was NOT run (see results doc)

Causal-coverage benchmark (eligible pool too small in-window), Tier-A immutable arm, secondary robustness
arm. Prior smoke/pilot artifacts (18-world v1, replay_v2 infra rows) were never mixed into this benchmark.
