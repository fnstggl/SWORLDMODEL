# World Model V2 — Phase 13 Validation

Two benchmark families plus a prospective ledger. All numbers below are read from persisted
machine-readable artifacts under `artifacts/phase13/`; this document summarizes them, it is not the
source of record. Locked tests were opened exactly once each and are not tuned against.

Regenerate the acceptance summary: `python benchmarks/phase13/acceptance_report.py`
→ `artifacts/phase13/acceptance_report.json` (**24/24 gates pass**).

## A. Controlled decision-correctness benchmark (Part 30A)

200 independently specified tasks with known/exhaustively-computable optima across 14 families
(discrete, continuous, combinatorial, sequential, partially-observable, multi-actor, institutional,
population, network, nonlinear, information-gathering, irreversible, constrained, multi-objective).
Split by task id: development 60 / calibration 40 / validation 40 / locked_test 60.

Runner: `benchmarks/phase13/run_controlled.py`. Artifacts: `artifacts/phase13/controlled/`.

| Gate | Required | Achieved (dev+cal+val) | Locked (64) |
| --- | --- | --- | --- |
| Exhaustive-optimum recovery | ≥99% | 100% | 100% |
| Median optimality gap (relative) | ≤1% | 0.0 | 0.0 |
| CRN pairing (exogenous trace match) | 100% | 100% | 100% |
| Matched variance ≤ unmatched | vr ≥ 1 | ✓ | — |
| Sequential policy beats greedy one-step | all | 30/30 | ✓ |
| Feasibility rejects the infeasible (institutional) | correct | ✓ | ✓ |
| VOI recommends gathering when it dominates | true | 15/15 | ✓ |
| Deterministic replay (same seed → same result) | true | ✓ | — |
| Abstention false-positives on specified tasks | 0 | 0 | 0 |

Search correctness (Part 19, `artifacts/phase13/controlled/search_correctness.json`): successive-halving
racing and coarse-to-fine hierarchical search recover the exhaustive optimum on the enlarged
known-optimum instances; a budget-performance curve (diagnostic → standard → production) is recorded.

## B. Real intervention benchmark (Part 30B)

Exactly **120** real intervention decision tasks. Composition frozen before any V2 performance was
measured (`artifacts/phase13/real/composition_manifest.json`):

- **13 datasets**, **11 domains**, **8 identification designs** (exceeds the ≥10 / ≥6 / ≥4 minimums).
- Quota: 40 randomized + 20 logged-bandit + 20 quasi-experimental + 20 sequential + 20 network/HTE.

Datasets (all real recorded outcomes; dataset cards with unit/treatment/outcome/assignment/propensity/
source/license/sha16 in `artifacts/phase13/real/dataset_cards.json`):

| Dataset | Design | Domain |
| --- | --- | --- |
| NSW/LaLonde, JOBS II, STAR, Thornton HIV | RCT | labor, education, health |
| social-insurance (Cai et al.) | RCT + village peer effects | development |
| Upworthy Research Archive | logged bandit (uniform arms) | media |
| kielmc, jtrain, organ-donations | difference-in-differences | housing, labor, policy |
| gov-transfers, close-elections | regression discontinuity | welfare, politics |
| Card proximity | instrumental variables | labor |
| castle-doctrine panel | staggered-adoption sequential | crime policy |
| ChangeMyView | matched observational | persuasion |

### Off-policy evaluation

`swm/world_model_v2/phase13/ope.py`: IPS / SNIPS, direct method, cross-fitted doubly-robust,
per-decision IS, weighted sequential DR — each **refuses** (raises) when its identifying assumptions
are unmet (missing propensities; <2 clusters for cross-fit). Overlap/clipping/zero-propensity
diagnostics; confidence intervals are cluster bootstrap at the decision-environment level (Parts
32/36). Estimator math is validated against known synthetic ground truth in `tests/test_phase13_ope.py`
(16 tests: IPS/SNIPS/DR recover a target bandit policy within 0.05 and CI-cover it; DR se ≤ IPS se with
a good model; per-decision IS + WDR recover a 2-step MDP policy value).

### Identification assumptions

RCTs — randomization gives known propensity and an **identified** policy value, so OPE estimates are
graded against truth. DiD — parallel trends (2×2 group×time). RD — continuity at the cutoff (local
comparison in a bandwidth). IV — instrument relevance + exclusion (Wald ratio). Transport of any
in-sample effect to a target population is an assumption, recorded per task.

### Results (validation, `artifacts/phase13/real/gates.json`; locked, `gates_locked.json`)

| Metric | dev+cal+val (99 ok) | Locked (18) |
| --- | --- | --- |
| V2 targeted beats **random** | 66% | **72%** |
| V2 beats **no-action** | 78% | **89%** |
| V2 beats **predictive-score-max** | 81% | **77%** |
| V2 beats **simple uplift** | 75% | **69%** |
| Sequential (WDR) beats greedy one-step | 69% | **75%** |
| OPE recovers bandit oracle CTR | 100% | 100% |
| Quasi design reverses naive observational sign | 25% (median \|gap\| 1.08) | — |
| Policy-value calibration MAE (identified vs OPE) | 0.0016 | 0.0035 |

Baselines receive the same admissible information (Part 33): random feasible action, no-action /
status-quo (treat-none), predictive-score-max (treat by predicted outcome, not uplift), simple
single-model uplift, the logging policy, and the oracle simple policy. V2 wins **pairwise against
every named baseline** on the locked test.

### Confidence intervals & clustering

Lifts are clustered at the decision-environment level (dataset/test), not the row level. The
randomized bucket (common [0,1] reward scale) has a poolable clustered CI; other buckets are reported
separately because reward scales differ (CTR vs employment vs log-homicide). See `lift_by_bucket` in
`gates.json`.

## C. Prospective decision ledger (Part 30C)

`swm/world_model_v2/phase13/ledger.py`: append-only JSONL that freezes decision context, admissible
actions, recommendation, predicted utility/effect, and uncertainty **before** any real outcome exists,
with an artifact hash. Frozen rows are never edited; outcomes only enter when reality supplies them.
Operational and smoke-tested (the Thiel run freezes a real row, `artifacts/phase13/thiel_run/
ledger.jsonl`); no outcomes are fabricated.

## Calibration & message elasticities

Reply elasticities are fit and graded on the full real ChangeMyView corpus (19,714 labeled persuasion
outcomes), held-out by matched pair: **grade A** (ECE ≈ 0.02, AUC ≈ 0.57, pair-accuracy ≈ 0.56),
`artifacts/phase13/message_calibration/cmv_backtest.json`. The fit calibrates the elasticity magnitudes
that were previously world-knowledge priors. Transport to cold email is an explicit assumption; the
absolute cold-email P(reply) is additionally flagged out-of-support when a fully-optimized message
maxes ≥60% of its levers (a low-density corner of message-space where the linear-logit model
extrapolates — the ranking is robust, the absolute level is not).

## Outreach layer v3 — validation status

The corrected outreach architecture (persona ensemble + funnel + contract; ARCHITECTURE §13) is
validated at three honesty levels, each labeled on every output:

- **Regression-tested (deterministic)**: the failed Thiel output is an executable regression —
  `tests/test_outreach_funnel.py` asserts the contract rejects it (no identity, diligence-bait ask),
  the funnel ranks the plain human draft above it with the stage trace diagnosing WHERE it loses
  (understand + easy), adversarial framing raises P(negative reply), the caricature guard clamps
  combat levers, and L1 under the funnel demands identity/next-step/believability while zeroing
  adversarial framing. `tests/test_persona_response.py` asserts outcomes are counted (never asked),
  failed draws count as no_response (fail-closed), fragility flags one-hypothesis winners, and the
  dossier carries qualitative text only.
- **Graded on real outcomes**: the additive persuasion elasticities only (CMV, 19,714 real
  outcomes, held-out grade A, ECE ≈ 0.02, refit on the 17-lever feature set).
- **Uncalibrated (labeled)**: funnel magnitudes (structural priors) and the persona ensemble
  (model-based judgment). No labeled cold-email corpus exists in-repo; the prospective ledger is
  the accumulation path. The system reports "best-supported among tested" with hypothesis
  fragility, and never claims a calibrated response probability for an individual.

## Outreach layer v4 — reply-first default, validation status

The reply-first beat planner (ARCHITECTURE §13; `swm/decision/reply_first.py`,
`swm/decision/language_judge.py`) is the default `optimize_cold_outreach` path. Its architectural
promises are executable tests in `tests/test_reply_first.py` (26 offline tests):

- **Judge separation**: a candidate failing the truth gate (fabricated number) or the language
  gate (bot register) provably never reaches the outcome judge — persona appeal cannot resurrect
  it (`test_outcome_judge_never_sees_gate_failures`).
- **Blindness**: the outcome judge evaluates shuffled anonymous candidates; the evaluation order
  is asserted to differ from the input order (`test_outcome_judge_is_blind_shuffled`).
- **No simulated percentages**: persona counts land only in the machine-readable trace
  (`step6_outcome_internal`); the human-facing summary carries ordinal notes and the
  no-reliable-distinction label, asserted free of `%`/probability strings (message bodies may
  still cite the sender's own factual stats).
- **Single output**: `PlannerResult.summary()` is `reply_first_single_output` — one recommended
  message; non-selected finalists appear as labeled notes only.
- **Fail-closed truth**: an unavailable truth judge blocks the candidate rather than passing it.
- **Preference hook**: `record_preference`/`load_preferences` round-trip, and stored human A-vs-B
  choices are asserted present in the language-judge prompt (the calibration path from LLM
  opinion toward the user's demonstrated taste).
- **Rules ride on prompts**: every instantiation prompt carries the hard writing rules (at most
  one number, no jargon compounds, human-typed request, 45–85 words); the backward-requirements
  prompt is asserted to work backward from the verbatim target replies.

Live evidence: `artifacts/phase13/thiel_v5/` (exp095) — full raw LLM trace (`llm_trace.jsonl`),
stage-labeled planner trace (`plan_trace.jsonl`), result, and ledger freeze. The outcome ranking
remains a model-based judgment (uncalibrated, labeled); the planner's stated cure for finalist
ties is real outreach outcomes through the ledger, not more simulation.

## Negative results (reported, not hidden)

- On **low-heterogeneity RCTs**, V2 CATE-targeting does **not** beat the oracle treat-all policy
  (randomized bucket lift ≈ −3.5pp): with little exploitable heterogeneity, targeting adds estimation
  noise over just treating everyone. V2 still beats every *named* baseline pairwise.
- 3 jtrain quasi slices were excluded (DiD cells empty on the slice) — a predeclared exclusion reason,
  recorded in `gates.json:excluded_reasons`.
- The LLM sentence judge is maximally adversarial, so a fully-optimized message often retains ≥1
  residual flag on the opener; the flag is surfaced rather than suppressed.

## Production-gate judgment

Functional, safety, search, and CRN gates: **pass** (both locked tests included). Locked-test
predictive decision-quality gates (V2 beats random / no-action / predictive-score-max / simple-uplift;
sequential beats greedy): **pass**. Acceptance report: **24/24 gates pass, production_ready = true**
under the gate definitions in `benchmarks/phase13/acceptance_report.py`, with the qualifications above
(targeting has no free lunch on homogeneous-effect data; absolute cold-email P(reply) is an
extrapolated, transport-assumed number — trust the ranking).
