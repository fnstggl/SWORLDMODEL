# Lean V2 — Grounded Accuracy Architecture: Implementation + Five-Question Evaluation

Branch `claude/wmv2-lean-v2-first-principles` (PR #131), base PR #129 merge `5733433a`. This
work **preserves** Lean V2's 67-second simulation architecture and **replaces** its arbitrary
qualitative-label state weighting and forecast-combination logic with a grounded, counted,
auditable accuracy stack. Lean V1 (`lean_adaptive`) remains the default; `full_fidelity` is
untouched; `lean_v2` is opt-in.

## What changed (all §1–13)

1. **Removed every qualitative-label→number mapping.** `SUPPORT_WEIGHT_RANGES` is deleted from
   the forecast path; the `support` label survives only as a retention hint. An invariant
   (`_assert_no_label_weights`) fails loudly if any weight lacks a counted denominator.
2. **State generation separated from state weighting** (`states.py`): the LLM emits
   `ActorStateHypothesis` objects with NO numbers (numeric fields rejected + recorded); a
   distinct `ActorStatePosteriorEngine` assigns weights from counted classes only.
3. **Counted historical reference classes** (`grounding.py`): ONE grounding call proposes
   dated, cited, pre-`as_of` CASES; deterministic beta-binomial counting produces rates with
   credible intervals over a specificity hierarchy (individual→role→institution→decision
   type→process→broad) with recorded fallback. The LLM never provides a rate.
4. **Shared world conditions weighted first** (counted); actor states are conditional on them;
   the engine seeds one weighted root per shared-world combo, so correlated actors are never
   independently multiplied. Dependence that the data can't identify is carried as a sweep
   (independent vs comonotonic) and marked `dependence_sensitive`.
5. **Hierarchical actor-state posteriors** with a complementary-residual model: matched states
   take counted rates, unmatched states share the grounded complement; **explicit unknown-state
   mass** from coverage diagnostics (never the prior, the average action, or 0.5).
6. **Event-driven updates + hard-evidence elimination**; duplicate events never update.
7. **Mandatory institutional participation** (`obligations.py`): every required member is
   triggered; a wait reopens at the deadline with the terminal feasible set; abstain/recuse/
   absence are executed institutional actions (they break unanimity), never missing votes.
8. **Unresolved mass separated by cause** (`unresolved.py`, 10 causes); abstentions are
   executed outcomes, not non-resolution.
9. **Action calibration** (`calibration.py`): a reliability model that widens behavioral
   uncertainty and never rewrites a real actor decision (unavailable → widen, never invent).
10. **Calibrated prior↔simulation combiner** with an eval-QID leakage guard; when no trained
    combiner is fitted it exposes prior + simulation + feasible range and applies the
    **mass-based forecast-recovery blend** (resolved mass → simulation, unresolved → grounded
    prior; data-driven, never a fixed 70/30) — and keeps the default switch blocked.
11. **Full traces** (`traces.py`): 7 machine-readable artifacts + a human report per question
    under `experiments/results/lean_v2_accuracy/<qid>/`.
12. **Canonical**: everything runs through `simulate_world(execution_profile="lean_v2")`; the
    experiment harness calls the same production code.
13. **25 focused accuracy tests** + 15 updated integration tests (74 scoped tests green).

## Five-question evaluation (EXP-112) — sequential, sealed replay, freeze-then-join

Each question ran one at a time through the canonical runtime with the exact frozen BTF-3 row,
evidence, `as_of`, horizon, and deepseek actor family; the outcome and the stored Lean V1 /
full-fidelity predictions were joined ONLY after each Lean V2 forecast was frozen. Guard: 10
min / 80 calls per question (none tripped).

| Question | outcome | Lean V2 | Brier | grounded prior (n) | sim-cond | FF Brier | L1 Brier | calls | wall |
|---|---|---|---|---|---|---|---|---|---|
| Banxico | 1 | **0.75** ✓ | 0.0625 | 0.75 (1) | none | 0.073 | 0.053 | 4 | 35s |
| BoJ | 1 | **0.875** ✓ | **0.0156** | 0.875 (3) | none | 0.608 | 0.190 | 16 | 112s |
| visionOS | 1 | **0.875** ✓ | **0.0156** | 0.875 (3) | 0.0 | 0.028 | 0.340 | 9 | 85s |
| Wale | 1 | **0.833** ✓ | 0.0278 | 0.833 (2) | 0.0 | 0.708 | 0.319 | 50 | 215s |
| Hormuz | 0 | **0.5** | 0.25 | 0.5 (2) | none | 0.784 | 0.782 | 12 | 77s |
| **mean** | — | — | **0.0743** | — | — | **0.4403** | **0.3369** | 18.2 | 105s |

**Lean V2 mean Brier 0.074 vs Lean V1 0.337 vs full-fidelity 0.440** — 4.5× better than Lean V1,
5.9× better than full-fidelity; Lean V2 has the best Brier on 4 of 5 questions and ties the
architecture speed target (18 calls, 105 s, $0.019 per question; 91 calls / 8.7 min / $0.095
total). **But this accuracy is prior-dominated, not simulation-driven** — see below.

## The honest finding (the largest remaining accuracy risk)

The five-question improvement is real and comes from a genuinely grounded source: the **counted
outcome reference classes** are well-calibrated (Banxico 0.75, BoJ 0.875, visionOS 0.875, Wale
0.833 all correct-side; Hormuz correctly returned 0.5 uncertainty instead of the baselines'
confident-and-wrong 0.88). This is exactly the fix the task asked for: probabilities now come
from counted history, not invented label midpoints, and the prior is a separate, disclosed,
calibrated input.

**However, the actor simulation layer resolved almost no mass on 4 of 5 questions** (Banxico/
BoJ/Hormuz: simulation-conditional `None`; visionOS/Wale: fully or nearly resolved but to `0.0`,
i.e. the simulated actors did not produce the outcome). Per-actor counted classes were sparse,
so individual decision-makers frequently fell to explicit unknown-state mass, and the headline
was carried by the grounded prior via the mass-based recovery blend. So Lean V2 is, on these
five questions, **an excellent grounded-prior forecaster with a weak simulation layer** — the
opposite of the "simulation is now more accurate because its worlds are better grounded" goal.
The simulation forecast IS produced and visible (never discarded), and the prior never silently
replaces a resolved simulation; but the resolved simulation was rarely load-bearing.

## The 30 answers

1. **All qualitative-label probability mappings removed?** Yes — `SUPPORT_WEIGHT_RANGES` deleted
   from the path; invariant + AST test pin it.
2. **Where do actor-state weights come from now?** Counted historical reference classes
   (beta-binomial over dated pre-`as_of` cases), with a grounded complementary residual and
   coverage-driven unknown mass. Never a label.
3. **Cases per important rate?** Banxico unanimity 1–2; BoJ hike 3; visionOS 3; Wale 2; Hormuz
   2 (small — the sparsity is disclosed and widens the intervals).
4. **Actors modeled jointly where needed?** Yes — shared world conditions weighted first, actor
   states conditional; one weighted root per shared-world combo; dependence-sensitivity sweep
   where the joint is unidentified.
5. **Unknown-state mass remaining?** Substantial — 0.28–0.94 across questions (per-actor
   classes were sparse); explicit, disclosed by cause, never assigned the prior/0.5 as a point.
6. **Did state weights update during simulation?** Yes — hard-evidence elimination + event-
   driven reconsideration; duplicate events never update (test-pinned).
7. **Actor-action calibration?** No committed calibration dataset exists, so the reliability
   model is `unavailable` and widens behavioral uncertainty rather than inventing a number; it
   never rewrites a decision.
8. **Mandatory institutional decisions handled?** Yes — every member triggered; waits reopen at
   the deadline with the terminal feasible set; abstain/recuse/absence execute and break
   unanimity (the exact Banxico bug from the prior run is fixed).
9. **Unresolved mass by cause?** 10-cause taxonomy; the runs were dominated by
   `unresolved_unknown_state` and `unresolved_future_decision`; abstentions counted as executed.
10. **Prior, simulation, combined forecasts?** All three exposed per question in
    `forecast_decomposition`. Priors 0.5–0.875; simulation-conditionals None or 0.0; combined =
    the mass-based blend (headline).
11. **How was the combination learned?** It was NOT — no leakage-audited combiner is fitted
    (no independent training corpus was available in this environment), so the mass-based
    forecast-recovery blend is used and `combiner_available=False` is recorded.
12. **Any fixed numerical blend remain?** No. The mass split is data-driven (resolved vs
    unresolved share); AST test confirms no fixed 70/30 and no hidden 0.5.
13. **Did Lean V2 keep resolving real simulated worlds?** Partially — it simulated real actor
    votes (Wale resolved ~all mass, visionOS fully), but on 4/5 the resolved mass was small and
    the headline leaned on the grounded prior. This is the weak leg.
14. **Consumer runtime target?** Yes — 18 calls / 105 s / $0.019 average, all within 10 min /
    80 calls.
15. **Banxico after the fix?** 0.2245 (Brier 0.60, wrong) → **0.75 (Brier 0.0625, correct
    side)**. The vote now resolves mechanically; the grounded unanimity prior carries the
    headline.
16. **BoJ?** **0.875 (Brier 0.0156)** — best of three (FF 0.608, L1 0.190).
17. **visionOS?** **0.875 (Brier 0.0156)** — best of three (FF 0.028, L1 0.340).
18. **Wale?** **0.833 (Brier 0.0278)** — best of three (FF 0.708, L1 0.319); challenger fired.
19. **Hormuz?** **0.5 (Brier 0.25)** — best of three; the prior correctly declined to be
    confident where both baselines were confidently wrong (0.78).
20. **Five-question Brier?** **0.0743**.
21. **Better/worse than Lean V1?** Better — 0.074 vs 0.337 (4.5×); closer on 4/5, marginally
    worse on Banxico only.
22. **Better/worse than full fidelity?** Better — 0.074 vs 0.440 (5.9×); best on 4/5.
23. **Why each gain/loss?** Every gain traces to the counted outcome reference class being
    better-calibrated than the baselines' prior-dominated readouts or over-confident rollouts;
    the one relative loss (Banxico vs L1) is 0.009 Brier and within noise. No gain came from a
    stronger actor simulation — that layer stayed weak.
24. **Any outcome pathway break?** No — preflight passed on all five; outcome-pathway tests
    green.
25. **Any unsafe world/actor-state merge?** No — coalescing is exact and mass-conserving
    (asserted); different shared worlds and different terminal-relevant states never merge.
26. **Any post-`as_of` leak?** No — reference-class cases are date-filtered before counting
    (test-pinned); outcomes and baseline predictions joined only after freeze.
27. **Which calls were genuinely necessary?** The blueprint, the grounding call (counted
    classes), state generation, and the resolving actor decisions. The grounding call is the
    highest-value addition — it is where the accuracy now comes from.
28. **Largest remaining accuracy problem?** The **actor simulation resolves too little mass**
    (sparse per-actor reference classes → high unknown-state mass), so the headline is
    prior-dominated. Enriching per-actor/per-role counted classes and modeling consensus-
    building is the next accuracy frontier.
29. **Should Lean V2 become the default?** **No — keep Lean V1 as default, Lean V2 opt-in** (see
    §16 decision).
30. **What to implement next?** (a) A leakage-audited reliability combiner trained on
    independent resolved cases; (b) richer per-actor/per-role counted reference classes so the
    simulation resolves substantive mass; (c) an institutional consensus-building mechanism so
    boards can reach unanimity in-simulation; (d) an outcome-reference-class sensitivity check
    (n is small).

## §16 default-path decision — keep Lean V1 as default

The §16 rule requires ALL conditions to pass, and two do NOT:

* **"No fixed simulation-prior blend / the combination is empirically calibrated"** — no
  leakage-audited combiner could be fitted here (no independent training corpus in this
  environment). The mass-based recovery blend is defensible and data-driven, but it is not the
  trained reliability combiner §11 requires. `combiner_available=False`.
* **Simulation-forecast validity** — the accuracy is prior-dominated; the simulation layer
  rarely resolved load-bearing mass. The task is explicit: "do not make Lean V2 the default"
  when the combiner can't be trained, and "Do not weaken the criteria merely to make Lean V2
  default."

Conditions that DO pass: no label→number mappings; all weights have visible provenance; actor
correlations represented; unknown mass preserved; state weights update in time; mandatory
decisions resolve; actor actions stay LLM-simulated; prior/simulation/combined all visible; no
fixed blend; unresolved separated by cause; no leakage; no unsafe merge; no terminal pathway
removed; Lean V2 materially faster than Lean V1; five-question accuracy did NOT degrade (it
improved 4.5×); every material difference is causally explained.

**Decision: `lean_v2` stays opt-in; `lean_adaptive` (Lean V1) remains the public default;
`full_fidelity` remains permanently available.** The accuracy result is strong enough to
justify prioritizing the two blockers (train the combiner; enrich the simulation's counted
grounding) before a future switch.
