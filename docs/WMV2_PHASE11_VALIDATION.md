# WMV2 Phase 11 — Validation

All numbers regenerate from `experiments/wmv2_phase11_eval.py` on the frozen corpus. Gates were written to
`experiments/results/phase11/gates.json` **before** evaluation and are not weakened here.

## 1. Corpus & episode construction (spec §21–§23)

`experiments/wmv2_phase11_corpus.py` → `corpus.jsonl` (+ `corpus_manifest.json`). Each episode = a question +
initial plan + initial posterior + an ordered observation stream (with reveal times) + a change/no-change label
+ affected-scope label + a downstream outcome to score. Observations replay in chronological order; the system
never sees the change label or future outcome.

| Property | Value | Target (§21) | Met |
|---|---|---|---|
| Total episodes | **135** | 120 | ✅ |
| Changed / unchanged-control | 81 / 54 | meaningful both | ✅ |
| Real-grounded episodes | **2** | 60 | ❌ (honest gap) |
| Adversarial / semi-synthetic | 133 | 60 | ✅ |
| Trigger families | 10 | 8 | ✅ |
| Domains | 9 | 6 | ✅ |

- **Adversarial / semi-synthetic** (real `WorldState` particles, controlled generating process,
  `wmv2_phase11_substrate.py`): changed episodes across 10 families × 8 domains + 54 unchanged negative
  controls + the §23 safety cases (unsourced rule report, future-dated rule, alias-not-new-actor, transient
  outage, single-noisy-surprise) that the safe system must *not* recompile on.
- **Real-grounded** (documented real change + sources): the **2013/2017 US Senate "nuclear option"** —
  a real, dated, sourced rule change making nomination-cloture a simple majority, with the real post-change
  regime shift — plus a real stable-period unchanged control (2003–04, no cloture change). Encoded with
  sources in the `grounding` field.
- **Splits** (`splits.json`, frozen by id hash, no leakage): train 72 / calibration 18 / validation 15 /
  test 30. Trigger thresholds are fit on **calibration** only (64 unchanged-control residuals, 90th-percentile
  false-alarm-controlled) — never on test.

**Honesty (spec §21):** the corpus does not reach 60 real episodes; the manifest reports `real=False`, the
real-record *replay* arm is named as remaining expansion, and this is **not** relabelled as final real
validation. The build is resumable.

## 2. Baseline arms (spec §24)

All arms are the *same* `RecompilationController` with one component ablated (identical episode streams):
B0 no-recompile · B1 parameter-only · B2 full-reset · B3 LLM-only (no evidence scoring gate) · B4 oracle
trigger/scope (from labels) · **B5 full Phase 11** · B6 current+branch. Plus ablations: no-fusion,
no-scope-selection (always full recompile).

## 3. Results on the frozen test split (30 episodes: 21 changed, 9 controls)

| Arm | Trigger recall | Precision | FPR (controls) | Scope exact/equiv | Changed Brier | Control Brier |
|---|---:|---:|---:|---:|---:|---:|
| **B5 full Phase 11** | 0.762 | **1.00** | **0.00** | 0.688 | **0.0058** | 0.0006 |
| B0 no-recompile | 0.00 | – | 0.00 | – | 0.0286 | 0.0006 |
| B2 full-reset | 0.667 | 1.00 | 0.00 | 0.286 | 0.0085 | 0.0006 |
| B4 oracle | 0.762 | 1.00 | 0.00 | **1.00** | 0.0058 | 0.0006 |

- **Predictive recovery:** B5 − B0 mean changed-Brier improvement **0.0228, 95% CI [0.0156, 0.0301]** (paired
  bootstrap) — the CI strictly favours Phase 11. B5 also beats **full-reset** (0.0058 < 0.0085) while being far
  **less destructive** (scope 0.688 vs 0.286; full reset discards continuity).
- **No regression on unchanged controls:** control Brier is identical across arms (0.0006); B5 does **0**
  recompiles on the 9 controls (FPR 0.00).
- **Credible family wins:** **7** families where B5 beats B0 on ≥ half the changed episodes (≥ 3 required):
  rule_change, new_actor, authority_change, coalition_change, network_restructuring, outcome_space_change,
  exogenous_shock.
- **Migration invariants (aggregate over 126 migrations):** time reversals **0**, duplicate events **0**, lost
  valid events **0**, min object-retention **1.0**, 144 orphans explicitly recorded (rule ops orphan to
  plan-level when the test world lacks the institution object — recorded, never silently dropped).
- **Determinism:** replay parity **True** (same inputs/seed → identical triggers + terminal).

## 4. Ablations (spec §26)

Run and preserved in `eval.json` (`ablation_no_fusion`, `ablation_no_scope_selection`) alongside B0–B6.
No-scope-selection (always full recompile) reproduces the full-reset destructiveness (scope ≈ full_plan) and
underperforms B5's minimal-scope migration on continuity; no-fusion removes the dependence-collapse + false-
alarm control that keeps FPR at 0. B0/B1/B2/B3/B6 are the removed-trigger / parameter-only / full-reset /
LLM-selects / branch-only ablations. Each removed component is causally present in the evaluated episodes, so
these are empirical, not ornamental.

## 5. Acceptance gates (frozen; scored honestly)

| Gate | Result | Basis |
|---|:--:|---|
| Safety (determinism parity, atomic rollback, no leakage) | **PASS** | replay parity True; injected-failure rollback restores source; future-dated rules rejected |
| Migration (0 time-reversal / duplicate / lost; retention ≥ 0.999; lineage integrity) | **PASS** | aggregate invariants all clean |
| Predictive (beats no-recompile after change, CI favours P11; ≥3 family wins; no control regression) | **PASS** | CI [0.0156, 0.0301]; 7 wins; beats full-reset with less loss |
| Trigger (recall ≥ 0.85, precision ≥ 0.80, FPR ≤ 0.10) | **FAIL** | recall **0.76** < 0.85 (precision 1.0 ✓, FPR 0.0 ✓) |
| Scope (exact/equivalent ≥ 0.75) | **FAIL** | **0.69** < 0.75 |
| Real held-out validation (≥ 60 real episodes) | **FAIL** | 2 real-grounded |

The trigger/scope shortfalls are concentrated in the pure-diagnostic families (impossible / mechanism-regime /
evidence-contradiction) on the numeric substrate; they are reported, not tuned away (tuning on test is
forbidden and thresholds come from the calibration split).

## 6. Four-status verdict

| Status | Result | Evidence |
|---|:--:|---|
| Software implemented | **Yes** | 10 production modules (`swm/world_model_v2/phase11/*`); 16 detectors; typed contracts; migration; lineage; controller |
| Executes end to end | **Yes** | controller runs the full pipeline on real `WorldState` particles: external structural evidence → detect → fuse → scope → candidates → validate → atomic migrate → score → mixture → continue → terminal; recompile events emitted; 27 tests pass |
| Empirically validated | **Partial** | predictive + migration + safety gates pass on the constructed corpus (B5−B0 CI favours P11; 0 invariant violations; determinism); trigger-recall + scope gates fall short; real-episode target unmet |
| Production eligible | **No** | a required gate failing ⇒ not eligible: real held-out validation unmet, trigger/scope short, real-V2 execution adapter + persistent-store resume not wired |

A large implementation with partial empirical results is labelled exactly that: **software implemented +
executes end to end + empirically validated only on the constructed corpus + not production eligible.**

## 7. Direct answers (spec §34)

1. Detects genuine inadequacy automatically? **Yes** — from external/out-of-support/verified-structure
   observations + validated diagnostics, not labels.
2. Avoids recompiling for ordinary noise? **Yes** — a single in-support surprise never triggers; controls FPR
   0.00; a no-external-evidence run does 0 recompiles.
3. Distinguishes parameter drift from structural change? **Yes** — drift (level move, residuals stable) →
   `parameter_only`; structural failure → mechanism/hypothesis/contract scope.
4. Chooses a causally sufficient scope? **Partially** — exact/equiv 0.69 on test (oracle 1.0), below the 0.75
   gate; global invalidations correctly escalate.
5. Generates multiple defensible revised plans? **Yes** — current + minimal + alternative + full, retained as a
   mixture.
6. Plan scoring beats LLM-only selection? **Yes** — B3 (LLM-only, no scoring gate) over-recompiles; B5's
   evidence scoring retains the current plan when unwarranted (FPR 0).
7. Preserves valid actors/histories/evidence/resources? **Yes** — additive migration retention 1.0; split
   partitions, merge sums-once.
8. Migrates networks and institutions correctly? **Yes** for the additive cases (edges/rules added, existing
   preserved); institution objects absent in a bare world are orphaned to plan level (recorded).
9. Migrates posterior uncertainty, not just means? **Yes** — particles reweighted + renormalized, ESS reported,
   broad priors for new variables.
10. Prevents duplicate/lost events? **Yes** — 0 duplicates, 0 lost (dedup by signature; classification).
11. Prevents time reversal? **Yes** — 0 (events behind sim-time dropped; clock monotonic).
12. Rolls back a failed migration? **Yes** — atomic transaction restores the source snapshot; tested.
13. Retains structural uncertainty when candidates tie? **Yes** — normalized mixture, not top-1.
14. Revised structure affects later execution? **Yes** — `post_migration` makes the adopted structure govern
    the substrate; B5 recovers where B0 cannot.
15. Improves real held-out predictive recovery? **On the constructed corpus, yes** (CI favours P11); a genuine
    real-record held-out set is not yet built.
16. Outperforms no recompilation? **Yes** — 0.0228 Brier improvement, CI [0.0156, 0.0301].
17. Outperforms/matches full reset while preserving continuity? **Yes** — 0.0058 < 0.0085 and far less
    destructive.
18. Avoids degrading unchanged episodes? **Yes** — identical control Brier; 0 false recompiles.
19. Deterministic and reproducible? **Yes** — replay parity True; seed-driven; no wall-clock/RNG identity.
20. Ready to merge? **Yes** as an honestly-scoped Phase 11 (code + execution + partial validation); **do not
    merge** until forensic review (per the run instruction).
21. Production eligible? **No** — see the verdict.

## 8. Anti-scaffolding answers (spec §33, representative)

Traced concretely in `WMV2_PHASE11_FORENSIC_TRACES.md`. Summary: the triggering observation is the specific
external evidence item (e.g. the dated+sourced rule publication); the trigger probability is the reproducible
`severity·persistence·independence` squash (rule_change ≈ 0.94); alternatives (noise / drift / not-yet-in-force)
are enumerated per detector; scope sufficiency + rejected smaller/larger scopes are recorded in each trace;
candidates include the current plan and the LLM's grounded proposals bounded by the static battery; component
scores (residual reduction, evidence fit, continuity, complexity …) select the winner, not the LLM; migrated
vs transformed vs orphaned state is enumerated in the `MigrationPlan`; particle reweighting + ESS are recorded;
pending events carry disposition + reason codes; simulation time is monotonic; the injected-failure traces show
rollback; replay reproduces the result. What failed: trigger recall + scope on pure-diagnostic families, and
the real-episode target — both preserved above. Why more than a callback/restart: it detects inadequacy from
diagnostics + external evidence, selects a *minimal* scope, scores *competing* plans (LLM cannot choose),
*migrates* the running posterior + events + history atomically with rollback, retains a plan *mixture*, and
continues the *same* timeline — none of which a callback or a full restart does.

## 9. Regression

Inherited base (`9f20591`) baseline: **977 passed, 2 failed** (missing `data/dataset_registry.json`; missing
`fastapi`) — both pre-existing + environmental. Phase 11 adds 27 passing tests; the full-suite result +
introduced-vs-inherited classification is in the PR description.
