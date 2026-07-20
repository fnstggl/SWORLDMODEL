# World Model V2 — the ONE canonical path (and what is quarantined)

Established after the EXP-102 forensic audit, in which a benchmark run silently used a legacy inner
entry and lost the calendar layer, evidence conditioning, institution normalization and activation
synthesis — producing broad-prior forecasts from fully compiled worlds.

## Canonical (use these — nothing else)

| Layer | Entry | What it guarantees |
|---|---|---|
| Public facade | `swm.facade` | routes every simulation request to `simulate_world` |
| The one V2 entry | `swm.world_model_v2.unified_runtime.simulate_world` | compiler → evidence bundle (or caller-supplied `evidence=` text) → Phase-3 posterior → institution normalization → scheduled-reality/recurring-events calendar → fidelity + actor intentions → activation synthesis (structure→readout binding) → event-time conversion → Phase-11 recompilation → one rollout funnel → phase supervision |

Since the validation-gate removal, `simulate_world` also guarantees:

* **Run-everything**: experimental mechanisms execute labeled (widened uncertainty, exploratory
  support grade) — never rejected into a broad-prior fallback.
* **Structure decides the outcome**: `institutional_decision` / `population_aggregation` /
  `actor_action_aggregation` events are synthesized from declared components; `generic_outcome_prior`
  is a no-op safety net that writes only if nothing else resolved the readout.
* **Truncation robustness**: the decompose prompt emits execution-critical keys first; a bounded
  continuation call recovers keys lost to a token cap; recovery is recorded in provenance
  (`truncated_reply` / `truncation_recovered_keys`).
* **Calendar reality**: dated public facts AND recurring institutional calendars (annual conferences,
  meeting schedules, release cadences — past instances cited strictly before `as_of`) execute
  deterministically and feed the outcome mechanism.
* **Actor knowledge scoping**: actors know what their real counterpart would publicly know as of the
  date (public history, own org's routines/calendar, domain expertise) — with a hard time boundary at
  `as_of` and no access to other minds. Scheduled facts are injected into actor prompts.
* **No hidden budget cliff**: `SWM_ACTOR_MAX_CALLS` lifts the actor-cognition call budget per run.

## Quarantined (legacy — do not build on these)

| Module | Status | Why |
|---|---|---|
| `swm.world_model_v2.pipeline.simulate` | DeprecationWarning on call | the bare inner funnel; skips evidence, posterior, criterion parsing, fidelity, mode graphs, event-time. Hardened (`harden_general_path`) so even legacy callers cannot reach broad-prior-only degradation, but NOT the production entry. |
| `swm.world_model_v2.phase3_pipeline` | banner | orphan Phase-3 entry, superseded by `simulate_world` |
| `swm.world_model_v2.phase9_pipeline` | banner | orphan Phase-9 entry, superseded by `simulate_world` |
| `experiments/wmv2_forecastbench_run.py` "V2 arm" | banner | a crowd-anchored rescaling, not the simulator; must never be reported as simulator performance |

## Rules going forward

1. Every benchmark, backtest, or product claim about "the simulator" runs `simulate_world` — nothing else.
2. A new capability layer is DONE only when it is wired into `simulate_world`'s funnel and visible in the
   run's `active_component_manifest` / `operator_delta_census`; a layer that exists but isn't on the
   canonical path is treated as not existing (that's exactly how the visionOS calendar miss happened).
3. Unvalidated mechanisms run labeled; blocking is reserved for mechanisms that are impossible,
   contradictory, consume post-`as_of` information, or write the answer directly.
