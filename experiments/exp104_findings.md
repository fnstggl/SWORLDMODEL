# EXP-104 — the 5 BTF-3 questions through the CORRECT WMv2 path (`unified_runtime.simulate_world`)

The honest baseline. EXP-102 used the deprecated `pipeline.simulate` (now quarantined); this uses the real
top entry that threads evidence, Phase-3 posterior, Phase-10 institutions, scheduled-reality and the actor
rollout through one plan. ~2.4 h, ~265 LLM calls/question.

## Scores (n=5 — anatomy, NOT a benchmark claim)

| forecaster | Brier | acc@0.5 |
|---|---|---|
| thin mechanism kernel (EXP-101) | 0.3455 | 2/5 |
| **deprecated WMv2 path (EXP-102)** | **0.2605** | 3/5 |
| **correct WMv2 path (EXP-104)** | **0.3925** | 1/5 |
| FutureSearch SOTA (same 5) | 0.1645 | 4/5 |

| qid | question | p | actual | brier | census (real structure ran) |
|---|---|---|---|---|---|
| 7279494c | BoJ June hike | 0.057 | YES | 0.89 | institutional_decision, actor_action_aggregation |
| 5c0765ed | visionOS 27 @ WWDC | 0.493 | YES | 0.257 | + population_aggregation (mis-framed as timing) |
| 741b4bed | Wale PM | 0.613 | YES ✓ | 0.150 | institutional_decision, population_aggregation |
| 017e64ef | Hormuz ≥50 transits | 0.514 | NO | 0.264 | hazard_round, nonlinear_state_step |
| cfb43147 | Banxico unanimous | 0.366 | YES | 0.402 | institutional_decision |

## What is FIXED (architecture) vs what is BROKEN (accuracy)

**FIXED — the fail-closed collapse is gone.** `generic_outcome_prior` as terminal resolver appears NOWHERE.
Every question runs real structure: boards vote (`institutional_decision`), populations aggregate,
actors act and bind (`actor_action_aggregation`), `fully_integrated: true`, `phase_integration_failures:
[]`. The deprecated-path collapse to a ~0.5 broad prior does not happen on the correct path.

**BROKEN — the rich structure ran BLIND, so it is LESS accurate than the crude paths on this sample.**
Root cause, from provenance: `phase2_evidence: executed=true, n_events: 0`. The evidence layer ran and
observed **zero** events → Phase-3 posterior had nothing to reweight (`structural_posterior: null`) →
the outcome-determining processes fell to `structural_process_prior` (tier-6 broad LLM priors) → forecasts
hug ~0.5, systematically under-confident on YES (4/5 of these resolved YES; only Wale crossed 0.5).

**The evidence layer is not network-blocked — it is dropping what it retrieves.** The Google News
connector, called directly as-of the visionOS question date, returns 10 real articles including
*"visionOS 27 Is Looking Like a Maintenance Release."* (which confirms the YES). Yet the full run observed
0 events. The retrieved items come back with unparsed `published_at`, so the temporal-admissibility filter
almost certainly excludes all of them. **Decisive evidence is retrieved and then silently discarded.**

Secondary bug: visionOS binary ("will Apple announce visionOS 27 at WWDC") was compiled with
`readout_repaired: true`, `readout_var: absorbed_at` — mis-framed as a first-passage *timing* question, so
p = CDF-at-horizon = 0.493 (a coin), not P(announced | WWDC happens ≈ 1).

## Reprioritized fixes (the baseline reorders them)

1. **Evidence pipeline: stop dropping retrieved news** (NEW #1 lever). Fix date parsing / temporal
   admissibility so retrieved items actually become observed claims and reach Phase-3. This is a real
   system bug, not sandbox, not n=5 — and it is the single biggest accuracy lever here.
2. **Calendar/scheduled-reality from MODEL KNOWLEDGE** (Task #4). WWDC-ships-visionOS-every-June is
   model-knowledge (no network needed); it must enter as a high-weight scheduled fact + entailment.
3. **Fix the binary-vs-timing misframe** (compiler outcome contract) so recurring-announcement binaries
   are not read as first-passage timing.
4. **Actor info-set: legitimate public as-of facts** (Task #5).
5. Fail-open-with-uncertainty gate (Task #3) — lower priority: the structure already runs; the problem is
   what feeds it.

Structure running was necessary but not sufficient. The lever is now clearly EVIDENCE + PRIORS feeding the
structure, exactly the direction the whole project's backtests keep pointing at.

## EXP-105/106 addendum — paired re-run after the first fix wave (evidence gate, recurrence calendar,
## grounded prior, rollout retry)

| q | EXP-104 | EXP-106 diag | EXP-105 paired | actual | SOTA |
|---|---|---|---|---|---|
| visionOS 27 @ WWDC | 0.493 | — | **0.563** (brier 0.191 < 0.257) | YES | 0.91 |
| BoJ June hike | 0.057 | **0.727** | **0.085** (brier 0.838) | YES | 0.74 |

Two honest findings:
1. **visionOS crossed to the correct side** (0.56, was 0.49) — the recurrence-aware calendar + grounded
   prior moved it the right way, but far less than the component-level signals implied (prior ~0.9,
   fact_entailment 0.98). The first-passage machinery + structural hypotheses still dilute the calendar.
2. **Run-to-run variance is now the dominant error mode.** BoJ, same code, same seed, two runs: 0.727
   (correct) vs 0.085 (badly wrong). The temp-0.2 compile draws different worlds per run and the forecast
   swings by ±0.6. This is the BTF-1 finding (single forecasts unreliable; mean-of-5 cuts SD ~2/3) and
   FutureSearch's #1 disclosed lever (mean of multiple runs) reproduced in our own stack. A single-run
   number from this pipeline is noise-dominated and must not be interpreted alone.

Implications, in order: (a) mean-of-K (K≥3) aggregation belongs in the measurement harness NOW and in the
production path as an option; (b) BoJ's persistent low side is the evidence-targeting gap (the hawkish
May-2026 signals are never retrieved — queries target actors, not the decisive fact), the sanctioned next
fix; (c) component-level wins do not compose linearly through the funnel — every future fix gets measured
end-to-end, mean-of-K, before any claim.
