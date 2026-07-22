# WMV2 Temporal Replay v2 — forensics

Representative complete traces; all numbers in machine-readable artifacts
(`experiments/results/activation200/activation200.json`, `experiments/results/replay_v2/audit_rows.jsonl`,
`experiments/results/integration/phase11_shock_validation.json`).

## Trace 1 — a supervised replay row, end to end (smoke; non-benchmark)

World `pm_10262` (an archived, resolved market question), cutoff 2024-09-04, arm `blinded_current_llm`:
1. Frozen capsule: 3 archived items (Wikipedia revision-pinned; e.g. `oldid`-addressed bytes,
   sha256-hashed, `first_proven_available_at` = revision timestamp < cutoff; the capsule raises on any
   post-cutoff item — tested).
2. Blinding: question + every claim text pseudonymized through the world's stable mapping; years stripped;
   mapping stored forecaster-side (no outcomes), sealed for the scorer.
3. Six probes recorded (name-only, recognition, no-evidence, identity-permutation,
   counterfactual-evidence, temporal-fact) with full prompts and outputs.
4. `simulate_world(blinded_q, prebuilt_bundle=capsule)`: compiler → evidence-conditioned recompile →
   posterior over the archived claims → relevance gate (structured `causal_dependencies` signals) →
   synthesis → supervised rollout. PhaseExecutionRecords: **10 phases causally_active, 1 explicit no-op
   (`phase9_networks`: no transmission dependency), 0 blocked**; StateDeltas per phase recorded; terminal
   from terminal world states (`p_yes=0.6351`); freeze hash stamped.
5. The diagnostic unblinded arm on the same capsule: 11 active, `p_yes=0.475` — the arms genuinely differ.

## Trace 2 — a blocked phase failing loudly (the supervision contract)

During infrastructure validation the posterior phase raised (`ReplayBundle` lacked the tagger's
claim-triple contract). The row did NOT silently degrade: `phase3_posterior → blocked_invalid_contract`,
`fully_integrated=false`, the row carried `failure_reason=blocked_relevant_phases:[phase3_posterior]`, and
the defect was fixed rather than papered over. This is the designed behavior: a relevant phase cannot
disappear.

## Trace 3 — activation-gate ablations (200-question corpus)

Every executed phase on every relevant question ran a matched ablation (same compiled plan deep-copied,
same seed, common randomness; only the target phase's requirement forced off). Effects were measured on
terminal shift OR StateDelta-trajectory change; per-row deltas are in `activation200.json`. Ablation
effect rates were 1.0 for all measured phases — removal visibly changes execution.

## Trace 4 — Phase 11 (retained shock corpus)

8 injected structural shocks (new actor/institution, dated+sourced rule change, authority/coalition
change, exogenous shock, outcome-space change, persistent network restructuring) through the real
controller: recall 1.0; 10 controls including 6 adversarial near-misses (actor alias, future-dated rule,
transient outage, irrelevant actor, unsourced rule, known institution): false 0.0; migration gates 32/32.
**Natural-trigger validation inside full replay remains OPEN** (benchmark blocked) — injected shocks are
not counted as the only long-term Phase-11 proof; the replay forecaster records Phase-11 records per row
for when the benchmark runs.

## Failed / excluded rows census (this run)

- `pre_cutoff_checkpoint` rows: ALL emitted as `arm_a_blocked_external` (the honest reason the 800-count
  is unreachable; see model_registry.json).
- Early smoke rows with `blocked_relevant_phases` (posterior contract defects): preserved in git history;
  final smoke rows show 0 blocked.
- Activation-200: 1/200 compile-time error (recorded verbatim in the artifact).
- Vault: `network_diffusion` causal-category quota 0/10, `structural_change` 3/10 — source-archive gap,
  recorded in events.json header.
