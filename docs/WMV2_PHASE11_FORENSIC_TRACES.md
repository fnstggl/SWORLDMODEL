# WMV2 Phase 11 — Forensic Traces

Full machine-readable traces: `experiments/results/phase11/forensic_traces.jsonl` (complete
`RecompilationTrace` records) + `forensic_index.json` (index). Regenerate with
`PYTHONPATH=. python -m experiments.wmv2_phase11_traces`. Each trace carries: observations → diagnostics →
trigger posterior → scope + alternatives → all candidates + component scores + rejections → decision →
migration mapping → event-queue diff → emitted records → plan mixture → lineage → terminal effect → checksums.

## Coverage (spec §29)

| Requirement | Delivered |
|---|---|
| Trigger families | **10** (rule_change, new_actor, authority_change, coalition_change, network_restructuring, outcome_space_change, exogenous_shock, evidence_contradiction, mechanism_regime_change, impossible_event) |
| No-recompile-correct cases | **10** (controls + safety cases) |
| Injected migration-failure → rollback | **2** (source preserved) |
| Real-grounded traces | 2 (nuclear option + stable control) |
| Adversarial / semi-synthetic traces | 21 |
| Parameter-only / local-structural / full-plan scopes | present (institution_ruleset, actor, relationship, full_plan) |
| Multi-hypothesis retention | present (plan mixture size ≥ 2) |

*Honest gap:* real-side traces = 2 (tied to the corpus real-episode shortfall in `WMV2_PHASE11_VALIDATION.md`);
the ≥12-real-trace target is unmet.

## Trace 1 — REAL: US Senate "nuclear option" (2013), a genuine dated+sourced rule change

- **Episode** `realNO2013` (real-grounded): as-of 2013-01-03; the model streams routine cloture observations,
  then on the first reveal ≥ 2013-11-21 sees an **external** observation declaring the rule change
  (`effective_date` = 2013-11-21, source "S.Res. 2013 nuclear option"). Post-change regime: nomination-cloture
  success jumps from ~0.55 to ~0.90.
- **Diagnostic → trigger:** eligible (external evidence); `d_rule_change` fires (dated + sourced, not future);
  trigger posterior `{rule_change: 0.94}`. Alternatives enumerated (future-dated-not-in-force rejected because
  effective < now).
- **Scope:** `institution_ruleset` (smallest sufficient); global escalation not taken (local structural).
- **Candidates + scoring:** `cand::current` (retained, weight 0.09), `cand::minimal` (add the evidenced rule,
  weight 0.46), `cand::alt_branch` (competing hypothesis, weight 0.46). The LLM did not choose — component
  scores did (minimal: residual-reduction 0.66, evidence-fit 1.0, continuity 0.85; current: residual-reduction
  0, evidence-fit 0.06). `recompile_warranted = True`.
- **Migration:** additive; object-retention **1.0**, time-reversal **0**, duplicate-event-rate **0**,
  invariants_ok **True**; 24 orphans recorded (the rule targets an institution absent in the bare world → the
  rule is retained at plan level, recorded, not dropped).
- **Events emitted:** `recompile_triggered → recompile_candidate_generated → recompile_decision →
  plan_migrated → recompile_completed`.
- **Continued execution + terminal:** the adopted structure governs the substrate (broad prior over the new
  regime); the run recovers to terminal **0.897** vs the realized post-change outcome **0.90**. A no-recompile
  run stays near the stale 0.55 regime. Simulation time stayed monotonic; the recompile records did not move
  the terminal — continued execution did.

## Trace 2 — REAL negative control: stable Senate period (2003–04)

- **Episode** `realStable2003` (real-grounded, unchanged): routine observations only, no rule change.
- **Result:** **0 recompiles** (no eligible external structural evidence); terminal 0.611 ≈ true 0.60. This is
  the "usually zero recompiles" property on a real stable period — the system does not manufacture a
  structural change from ordinary variation.

## Trace 3 — Adversarial: new actor vs. alias

- `new_actor` changed episode → `d_new_actor` fires (causally relevant, not a known alias) → scope `actor` →
  minimal candidate adds the entity with a **broad** latent prior (no access to others' private history) →
  migration retention 1.0 → recover.
- `advF_alias` safety case: an observation declaring actor `S_jr` which resolves (via `aliases`) to known
  `subject` → **no trigger, 0 recompiles**. Renaming is not a new actor.

## Trace 4 — Adversarial: impossible event → global escalation

- `impossible_event` episode: an **out-of-support** observation (`representable=False`) → `d_impossible_event`
  (severity 0.95) → fusion classifies **global_structural** → scope escalates to `outcome_contract`/`full_plan`
  (a local update cannot repair invalid global structure) → full-recompile candidate scored + migrated with
  the source-only structure explicitly orphaned (recorded), low continuity by design.

## Trace 5 — Adversarial safety: unsourced + future-dated rule reports

- `advF_false_rule` (unsourced) and `advF_future_rule` (`effective_date` in the future): both **rejected** by
  the detector / static validation → **0 recompiles**. The system does not accept an ungrounded or not-yet-in-
  force rule.

## Trace 6 — Adversarial: single noisy in-support surprise

- `advF_noise`: one `simulation_internal`, representable observation with a large residual → **not eligible** →
  **0 recompiles**. Ordinary low-probability sampling is Phase-3 posterior updating, not model failure.

## Trace 7 & 8 — Injected migration failure → atomic rollback

- Two `injected_migration_failure` cases: the candidate build raises mid-activation. The
  `RecompileTransaction` **rolls back** to the source checkpoint — `activated=False`, `rolled_back=True`,
  `source_preserved=True`. The only valid world is never partially mutated; the run continues the current plan
  with a recorded degraded status.

## Counterfactual baseline (per §29)

For every changed trace, the counterfactual **B0 (no recompilation)** on the identical stream is scored in
`eval.json`: mean changed-Brier 0.0286 vs B5 0.0058 (improvement 0.0228, 95% CI [0.0156, 0.0301]). The
**B2 full-reset** counterfactual recompiles but discards continuity (scope 0.29) and is beaten by B5's minimal
migration (0.0058 < 0.0085). These contrasts are what make each recompilation's value legible.
