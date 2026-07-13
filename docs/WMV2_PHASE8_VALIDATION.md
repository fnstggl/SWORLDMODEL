# WMv2 Phase 8 — Validation

*Datasets · splits · power analysis · baselines · metrics · ablations · real results · failures ·
four-status phase grading.*

Companion to `WMV2_PHASE8_PERSISTENCE.md` (architecture) and `WMV2_PHASE8_MIGRATION.md` (schema/contract).
Every result below is produced by pure compute (no LLM) over real, cached data, and is reproducible from the
commands in the migration doc. All prior positive, null, and negative results are preserved — nothing was
overwritten.

The grading rule this doc obeys: **do not claim lift because the output changed.** Lift is claimed only when
a *paired bootstrap CI excludes zero in the favorable direction*, and it is graded against the strongest
baseline, not the weakest. Nulls are preserved.

---

## 1 — Datasets

| Track | Dataset | Real? | Provenance | Longitudinal structure |
|---|---|---|---|---|
| A | OmniBehavior (Kuaishou user traces) | ✓ | HF `jiawei-ucas/OmniBehavior`, CC-BY-NC-SA 4.0; cache `data/omnibehavior/` (92 users) | repeated individual behavior (per-user event sequence) |
| B | Enron email | ✓ | CMU tarball → `enron_comm_edges.json` (70 nodes, 592 dyads) | dyadic/relationship interaction over time |
| C | US Senate S117 roll-calls | ✓ | Voteview → `congress_S117_bills.json` (882 scored) | institutional process / prior decisions |

Datasets are not committed (repo policy); acquisition commands + hashes are in the migration doc.

---

## 2 — Splits & leakage control

* **Track A — time-forward + person-disjoint + sequence-disjoint.** Per-user chronological 70/30 (train
  prefix / test suffix). 20 % of users held **entirely** out of training (person-disjoint transfer). Each
  target's prior history uses only `evs[:idx]` events **strictly before** it (sequence-disjoint by
  construction); the target's own outcome is never a model input. Filter hyperparameters (`decay`,
  `prior_strength`) fit on **train only** via a grid that *includes* `decay=1` (no momentum) — the test set
  is never tuned.
* **Track B — median time-split, dyad-disjoint.** First-half interaction history → predict a second-half
  interaction; each dyad scored on its own future.
* **Track C — time-forward.** Roll-calls processed in sequence; each vote predicted from prior decisions only.

The Phase-8 event log enforces the gate physically: `events_as_of(t, mode="filter")` returns
`observed_time ≤ t`; `mode="smooth"` (retrospective) is a separate call and is never used for an as-of
forecast. Missing `as_of` is refused.

---

## 3 — Power analysis (before grading)

Paired design → the precision is set by the **empirical sd of the per-observation Brier *difference***, not
the marginal Brier sd (the prior round's documented mistake). Computed *before* interpreting each ablation.

| Comparison | n | se(paired Δ) | power @ observed effect | MDE₈₀ | adequately powered? |
|---|---|---|---|---|---|
| A: persist − memoryless (main) | 7074 | 0.00079 | **1.00** | 0.0022 | **yes** |
| A: persist − memoryless (person-disjoint) | 216 | 0.00167 | **1.00** | 0.0047 | **yes** |
| C: persist − no-history | 882 | — | **0.43** | — | **no (null is uninformative alone)** |

---

## 4 — Baselines (strong, appropriately separated)

All arms run through the **same** shared-world path (history → filter → materialize → readout), so the
comparison isolates the mechanism, not the plumbing:

* **B0 no-history** — global base rate (no user level).
* **B_userrate (memoryless)** — hierarchical-shrunk per-user rate, momentum off (`decay=1`, zero events) —
  the *persistent user level* without recent-history momentum. **This is the strong baseline the win is
  graded against.**
* **B_persist (full)** — the same anchor + the actor's real prior history through the decayed Beta-Bernoulli
  filter. Differs from B_userrate *only* in momentum → the paired Δ isolates the momentum contribution, and
  removing history collapses B_persist onto B_userrate (the causal ablation).
* Track B adds a **frequency** baseline (fraction of prior windows active).

---

## 5 — Metrics & real results

### Track A — OmniBehavior engagement (the headline: the win, through the shared world)

Cohort: **71 users, 7074 test events**, real action rate 0.206. Train burstiness `momentum_lift = 6.777`
(p_hot 0.477 vs p_cold 0.070 — persistence structurally present). Train-fit selected `decay=0.85` (< 1 →
momentum ON) over the no-momentum `decay=1` in the grid.

| arm | Brier | log-loss | AUROC | ECE |
|---|---|---|---|---|
| B0 no-history | 0.16370 | 0.509 | 0.500 | 0.044 |
| B_userrate memoryless | 0.10295 | 0.339 | 0.857 | 0.034 |
| **B_persist shared-world** | **0.09109** | **0.298** | **0.895** | **0.025** |

**Paired ablations (Brier Δ, negative = persistence better):**
* persist − memoryless: **−0.01186, CI95 [−0.01341, −0.01026]** — excludes 0, favorable, **power 1.0**.
* persist − no-history: −0.07261, CI95 [−0.0769, −0.0680].
* **Person-disjoint transfer** (persist − no-history on 216 held-out-person events): **−0.01889,
  CI95 [−0.02181, −0.01561], power 1.0** — transfers to new people.

This is stronger than the prior standalone predictor (−0.0065): the decayed Beta-Bernoulli filter is a better
momentum model than the inline formula, and it runs **through** WorldState materialization rather than
bypassing it. B0 = 0.16370 matches the prior harness's `A_nohist` = 0.1637 exactly — a faithful port.

### Track B — Enron dyadic link persistence (honest weak)

n = 251 first-half-active dyads; second-half activity rate 0.55.

| arm | Brier | AUROC |
|---|---|---|
| no-history | **0.24952** | 0.500 |
| frequency | 0.37572 | 0.500 |
| persist shared-world | 0.26376 | 0.504 |

persist − frequency: −0.1120 [−0.1417, −0.0836] (beats frequency). persist − no-history: **does NOT beat the
base-rate baseline** (0.264 > 0.250; AUROC ≈ 0.50). **Verdict: honest weak** — Enron's late-2001 collapse is
a regime change that washes out dyadic momentum. Preserved, not overclaimed.

### Track C — Senate roll-call pass persistence (honest null)

n = 882. Base pass rate 0.71. persist − no-history: **−0.00399, CI95 [−0.00845, +0.00015], power 0.43**.
AUROC 0.61 (some discrimination) but the Brier improvement is **not significant and underpowered**.
**Verdict: null** — pass/fail is bill-driven, not chamber momentum.

---

## 6 — Ablations (Part 17, through the shared world)

Engagement task, held-out Brier Δ vs the full system (`experiments/results/phase8/ablations.json`):

| ablation | Brier | Δ vs full | changes execution? |
|---|---|---|---|
| A21 full persistence | 0.08966 | — | (ref) |
| A5 no persistent latent (base rate) | 0.16092 | +0.07126 | ✓ (largest — the user level) |
| A9 no person-specific adaptation | 0.10677 | +0.01711 | ✓ |
| A1 no history | 0.10125 | +0.01159 | ✓ |
| A4 perfect memory, no decay (momentum off) | 0.09879 | +0.00913 | ✓ (momentum beats perfect recall) |
| A2 last event only | 0.09654 | +0.00688 | ✓ |
| A18 truncated window k=3 | 0.09245 | +0.00278 | ✓ |
| A8 no hierarchical shrinkage | 0.08967 | +0.00001 | **near-ornamental (honest)** |

No arm is fully ornamental *except* hierarchical shrinkage on this cohort, which we flag honestly. The
`A4` result is the direct mechanism test: **the forgetting/momentum filter beats perfect memory** (`decay=1`).

**Cross-family causal checks** (families the engagement task can't exercise — does removing that family's
history change the materialized state?): `trust` ✓ (violation history moves the posterior), `institutional_stage`
✓ (appeal path ≠ direct path), `resource_level` ✓ (spend history moves the level). All change execution — none
ornamental. These are causal-change checks on the real filters, **not** faked as engagement arms.

---

## 7 — Failures & preserved nulls

* **Prior `n=48` OmniBehavior null preserved** (`wmv2_omnibehavior_v2.json`): Brier ~0.128, all ablation CIs
  span 0 — persistence added nothing at that sample size. The `n=7074` result *reverses* it at power; both
  are kept.
* **Prior at-chance pilot preserved** (`omnibehavior_eval.json`, n=72, raw-LLM Brier 0.4213).
* **Track B weak / Track C null** preserved above.
* **Hierarchical-shrinkage near-ornamental** on this cohort — preserved as an ablation finding.
* **Enron regime change** documented as the cause of Track B weakness, not hidden.

---

## 8 — Four separate phase statuses (Part 21)

Never one "complete" label. Graded per state family.

### 8.1 SOFTWARE IMPLEMENTED — **yes (all 9 families)**
Typed spec registry, immutable event log, sequential filters, checkpoint/restore/migration/lineage,
materialization, in-world transitions, universal entry, 28 tests. All present and passing.

### 8.2 EXECUTES END-TO-END — **yes (all 9 families)**
Each family filters → materializes into a real WorldState field → is consumed by the ActorView/policy/
feasibility path → emits `StateDelta` + `PersistentStateDelta`. The universal
`simulate_with_persistence` compiles a real question, ingests history, materializes, and rolls out
(`universal_path_trace.json`). Cross-family causal checks confirm consumption.

### 8.3 EMPIRICALLY VALIDATED — **one family (engagement_propensity)**
`engagement_propensity` (behavioral/momentum persistence): adequately-powered real held-out win through the
shared world, time-forward **and** person-disjoint (both CIs exclude 0, power 1.0). Every other family is
**executes-end-to-end but empirically unvalidated** here — no defensible labeled longitudinal dataset was run
for trust / commitment / institutional-stage / resource / reputation / risk / relationship / habit as
*outcomes*. Track B (dyadic) is a weak result; Track C (institutional) is a null.

### 8.4 PRODUCTION ELIGIBLE — **one family, with caveats (engagement_propensity)**
Meets code + execution + validation + provenance + uncertainty + reliability gates: fitted transition params
(train-only), uncertainty propagates, actor visibility enforced, checkpoint/replay deterministic, real
held-out validation, transport limits recorded (OmniBehavior domain; Kuaishou platform), ablation establishes
causal relevance. **All other families are experimental / quarantined** — implemented and executable but not
production-eligible. Phase 8 is **not** promoted as a whole; grading is per-family.

---

## 9 — Anti-scaffolding answers (Part 22, abbreviated; full detail above)

1–6. Real scenario-specific persistent objects (engagement/trust/stage/resource posteriors) were constructed
from real histories (OmniBehavior/Enron/congress/Phase-2 evidence), carrying posterior mean+sd+lineage;
materialized into `latent_state`/`network.edge.trust`/`resources`; consumed by the reinforcement/habit/
reciprocity/feasibility mechanisms; each mutation emitted a `PersistentStateDelta`.
7–9. History changed actor views (recall + policy_state), action distributions (Trace A/B), and terminal
readouts (Brier deltas).
10–14. Validated on OmniBehavior (Track A, powered) with person-disjoint transfer; the earlier `n=7074` win
was **re-derived through the shared world** and beaten; the `n=48` null is preserved.
15–18. Track B weak, Track C null; 8 families remain executes-but-unvalidated; hierarchical shrinkage is
near-ornamental on this cohort; non-engagement transition params are labeled reference/broad priors, not
claimed supported.
19. Checkpoints replay deterministically (parity ✓, integrity ✓).
20. Phase 8 is production-eligible for **one** family only; the rest are quarantined.
21. Remaining work: labeled longitudinal datasets for trust/commitment/institutional outcomes; fitted (not
reference) transition packs; larger action-distribution effect via a fitted persistence policy pack; lineage
trimming for checkpoint size.
