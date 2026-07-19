# Event-driven temporal architecture (World Model V2)

**World Model V2 now advances through real causal events and elapsed-time processes. Actors do
not reconsider merely because a generic review interval passed.**

This document describes the production temporal path after the event-driven replacement. It
does not claim temporal accuracy or calibration: the validation harness exists
(`benchmarks/temporal/`), the resolved-case corpus is empty, and calibration is **not**
claimed anywhere (see *Known limitations*).

## Real timestamps vs realistic temporal causality

The pre-existing system already used real unix timestamps and an event queue — but much of the
timing *inside* the queue was synthetic: 2–14 evenly spaced "periodic strategic review" events
for the six highest-sensitivity actors (cadence `horizon/6`), fixed 60 s direct / 1 h public
delivery, a fixed 30-minute reconsideration timer, five invocations per actor, cascade depth 8,
frontier cap 8, daily background ticks, hazard rounds on an even grid, and same-timestamp
events resolved by queue insertion order. Those were scheduler conveniences. The replacement
makes timing itself *causal*:

    real initial time and real calendar
    → real scheduled facts, commitments, deadlines, and process stages
    → scenario-generated temporal model             (temporal_compiler → temporal_model)
    → events occur when their real causal triggers occur
    → information travels through actual channels    (ChannelTemporalModel stages)
    → actors notice information according to their situation (profiles, attention)
    → actors reconsider only when something creates a reason  (DecisionTrigger)
    → decisions and actions take situation-specific time
    → institutions advance through scenario-specific stages
    → continuous processes evolve over the exact elapsed interval (advance_interval)
    → simultaneous events interact without insertion-order artifacts (microstep batches)
    → the world advances event by event until the real horizon

The old periodic scheduler survives ONLY as a token-gated ablation
(`legacy_ablations.legacy_periodic_review_ablation`) for old-vs-new benchmarks; enforcement
tests (`tests/test_temporal_enforcement.py`) fail the build if production can reach it.

## Scenario temporal models (`temporal_model.py`, `temporal_compiler.py`)

`compile_temporal_model` is a default-on LLM stage in `unified_runtime.simulate_world`
(content-addressed cache; one compilation per scenario). It receives the generated causal
world, actors/institutions/relations, scheduled facts, evidence, as-of/horizon, user context
and intervention, and generates a `ScenarioTemporalModel`: IANA timezones, civil calendars,
per-channel stage timings, per-actor temporal profiles with latent availability hypotheses,
institutional stage machines, continuous processes, deadlines, dependencies, **sourced**
recurring obligations (recurrence without a source is refused), decision-trigger sources,
simultaneity rules, correlated temporal latents, uncertainties and honest unknowns. Two
independent critics check the twelve §4 failure classes (missing processes, unrealistic
speed/slowness, missing sleep/stages/implementation lag, wrong simultaneity/ordering, invented
precision, missing deadlines/calendar effects, synthetic recurrence); repairs apply as typed
patches. Every call is traced (stage, prompt hash, response, repairs, accepted/rejected).

Timing values are `TimingSpec`s — exact timestamp, bounded range, qualitative regime,
calendar expression, event dependency, or **unresolved** (never silently coerced to a number).
The documented regime bands (`TIMING_REGIMES`) are the sampling semantics of the qualitative
vocabulary, deliberately wide, sampled per particle from particle-rooted streams.

## Decision triggers

Every actor decision event carries a `DecisionTrigger` (type, causal parents, what was
observed, why decision-relevant, why now, provenance). Trigger types are an open vocabulary
(`TRIGGER_TYPE_EXAMPLES` documents examples: newly noticed information, direct request,
deadline approaching, institutional stage reached, promised follow-up, action completed or
failed, threshold crossed, condition became true, sourced recurring responsibility,
self-scheduled revisit). No trigger → no decision event → no actor call: a one-year quiet
simulation makes zero actor calls; a one-hour crisis may make many
(`tests/test_temporal_invariants.py::test_inv13_14…`, `…inv15…`).

## Delivery ≠ attention

Information travel is staged: transmitted → delivered/**available** → (moderation, exposure
for broadcast) → **noticed** → read → decision-relevant. Delivery publishes to the information
plane and buffers *availability*; nothing enters an actor's information set until an attention
event collects it (invariants 17/18). Attention comes from the actor's own generated profile:
channel-checking cycles (per-particle anchored phase — several items arriving before one check
coalesce into ONE ordered bundle and ONE actor invocation, §20), tz-aware sleep/active
windows, urgency interrupts, relationship priority, and the particle's sampled latent state
(traveling/crisis workload stretch the cycle; the latent persists for the whole particle and
is shared across matched counterfactual arms).

## Actor availability and chosen timing

Actors choose *when* to act: immediate, at a real time, after an event, before a deadline, or
deferred to a condition. A deferral with a calendar expression compiles through the actor's
own `CivilCalendar` ("tomorrow_morning" is their timezone's morning); a condition deferral
registers a watcher that fires when the awaited event occurs; unresolvable intent stays
recorded as an unresolved timing mechanism — never auto-scheduled 30 minutes later
(`generated_world.compile_actor_deferral`).

## Institutional time

Institutions carry generated stage machines (`InstitutionalProcessModel`): entry conditions,
responsible holders, per-stage durations (TimingSpecs), working calendars, deadlines, outputs,
next stages. Stages schedule **only when entered** (invariant 22); a submission resolves the
generated stage chain to place the decision process, and authority holders receive their
decision opportunity when the matter reaches them in the queue (per-particle queue position),
with an `institutional_stage_reached` trigger. Known statutory timing should come from
evidence; unknown timing stays a range.

## Continuous-time evolution (no daily ticks)

`temporal_runtime.advance_interval(world, start_ts, end_ts, …)` advances every declared
continuous process over the EXACT interval between events: analytic forms
(exponential decay/approach, linear drift) exactly, the logistic form by adaptive internal
integration whose grid is invisible (no events, no actor decisions, error bound recorded).
Contested attrition drains capacity per elapsed day (`attrition_rate_per_day`), replacing
`attrition_per_review` (old fitted packs convert at the legacy ~21-day cadence). One 10-day
gap ≡ ten 1-day gaps, and produces one interval advance, not ten synthetic events.

## Hazard scheduling (continuous-time first passage)

Evenly spaced `hazard_round` grids are replaced by `temporal_hazards.CumulativeHazardState`:
per (particle, process) one persistent Exp(1) threshold (particle-rooted → shared across
matched arms), piecewise cumulative intensity from the fitted family curve (calibrated chains
use Λ_total = −ln(1−target), exactly mass-conserving), live modulation by the sampled stance
hazard ratio × consumed causal state. When written state touches a process's declared read
set, accumulated hazard and threshold are PRESERVED and only the remaining time re-projects
(stale crossings are generation-invalidated, never double-fired). Persistence-window criteria
still write provisional absorptions; a collapse resumes the process with a fresh threshold
segment above the accumulated exposure. Exact scheduled facts remain exact events.

## Simultaneous events (microstep batches)

The queue pops the FULL batch at the earliest timestamp. Explicit causal parents/dependencies
layer the batch into microsteps; independents evaluate in canonical content order (insertion
order carries no semantics — permuting the schedule yields identical terminals, invariant 32);
an event whose declared reads intersect same-microstep writes defers one microstep; same-time
causal descendants (follow-ups at the same timestamp) run in the next microstep (invariant
33); two events writing one path in one microstep is an EXPLICIT simultaneity conflict —
resolved by a scenario simultaneity rule when one names the mechanism, else recorded loudly as
unmodeled (invariant 34). The heap's `seq` survives only as heap stability inside the pop.

## Phase 13 and personal-reaction timing

Phase 13's matched engine (`MatchedRolloutEngine`) runs the same temporal loop with
stream-partitioned CRN randomness; temporal latents, sampled schedules, calibrated targets and
first-passage thresholds seed from the PARTICLE ROOT, so matched arms share one temporal
reality except where the action causally changes it. Action follow-ups travel their channel
(no universal +1 s); plan steps are change-triggered (state watches) with dependent steps
firing the instant parents complete; timing variants anchor to real scenario times (deadlines,
scheduled facts, decision points) instead of a synthetic mid-horizon point. The
personal-reaction route models send → channel delivery → the person's real attention →
decision, with distinguishable outcomes: responded / read_but_deferred / unread_by_horizon
("no response yet" is never conflated with "ignored"), real history timestamps when supplied,
and user-supplied schedule context strongly informing the profile.

## Cost behavior

Cost follows causal activity, not horizon × cadence: the cost benchmark
(`experiments/temporal_cost_benchmark.py`, artifact
`artifacts/temporal/cost_benchmark.json`) measures a 9-month quiet scenario dropping from 31
actor invocations (30 periodic) to 1, while a 2-day crisis keeps all 18 genuinely-triggered
invocations. Temporal compilation adds ~4 LLM calls per scenario (2 stages + 2 critics),
cached content-addressed. Attention bundling reduces per-item invocations. Cost control never
truncates reality: no actor caps, no cascade-depth models, no numerical fallback.

## Truncation behavior

Safety budgets exist only to protect the service (`DEFAULT_BUDGETS`, `max_events`), sit far
above natural quiescence, and REACHING one marks the branch
`simulation_status="temporally_truncated"` with the pending events, unprocessed actors, why
they matter, and support-grade downgrade recorded (`TemporalRunStats.truncation`; §12).
Remaining actors are never converted to numerical policies and the cascade is never marked
naturally complete.

## Result contract

Every result carries `provenance.temporal_runtime` (§27): temporal model id/hash, timezone
and calendar assumptions, known scheduled facts, generated processes, event counts by type,
actor invocations by actor and trigger, delivery→attention / attention→decision delay
quantiles, same-time batches and microsteps, simultaneity conflicts, cancellations and
re-projections, pending-at-horizon, truncations, unresolved timing mechanisms, timing support
classification, and temporal-compilation LLM calls.

## Known limitations

- **Not temporally calibrated.** The backtest harness and metrics exist
  (`benchmarks/temporal/harness.py`), but no resolved-case corpus has been scored; regime
  bands and priors are documented, unfitted.
- The temporal compiler can fail or return partial structure on a live backend; the run
  degrades LOUDLY (`degraded: temporal_compilation_failed` / labeled broad bands) and
  delivery/attention fall back to wide per-particle regime bands, never to fixed constants.
- Read/write-set conflict detection uses declared paths plus post-hoc delta writes; an
  operator that reads state it does not declare can evade same-microstep deferral (the
  write-write conflict record still fires).
- Guard-condition state watches are coarse (objects/resources/information writes), so a
  watched step may re-check on unrelated changes (bounded by `max_condition_checks`).
- Correlated latent effects on attention use documented multipliers
  (`LATENT_STATE_CHECK_FACTOR`) — regime semantics, not fitted values.
- Exposure processes for very large populations still route through per-actor delivery; a
  population-level aggregation mechanism exists (phase 9) but is not attention-integrated.
