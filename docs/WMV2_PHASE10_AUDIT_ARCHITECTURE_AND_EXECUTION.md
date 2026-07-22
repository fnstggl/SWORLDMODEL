# WMV2 Phase 10 — Audit, Architecture & Execution

## 1. Ruthless current-institution audit (Part 0)

Before Phase 10 the institutional path was `swm/world_model_v2/institutions.py`:

- **Executable (good):** `RuleSystem.validate_action` runs BEFORE any transition, so an LLM-proposed action
  that violates a rule is refused; `run_vote` executes thresholds; 7 executable rule kinds (decision_right,
  deadline, budget, eligibility, procedure, capacity, quorum), failing closed on unknown kinds.
- **The core gap:** rules are **LLM-authored** — `materialize.py` reads `plan.institutions =
  [{id, rules:[{kind, params}]}]` straight from the compiler's LLM output. No evidence, no source
  provenance, no as-of versioning, no family/template/instance layering, no authority graph, no stage graph,
  no agenda/matter model, no information boundaries, no queues/appeals/escalation, no Phase-3 institutional
  uncertainty, no competing rule models.

Disposition: **retain** the executable `RuleSystem` (the constraint substrate); **build on top** an
evidence-backed FAMILY/TEMPLATE/INSTANCE layer that reconstructs real institutions and executes through the
same WorldState/StateDelta path. Nothing was overwritten; `institutions.py` is unchanged.

Phase 1 (compiler/WorldState/StateDelta), Phase 3 (`phase3_posterior/pipeline/priors/...`), and Phase 6
(mechanism registry) are used directly, not duplicated.

## 2. Before / after architecture

```
BEFORE:  compiler(LLM) → plan.institutions[{rules}] → materialize → RuleSystem → validate_action / run_vote
AFTER:   compiler → _select_institutions(process, scenario, as_of, jurisdiction)   [by causal need]
           → institutions_v2 store: FAMILY (structural) → TEMPLATE (real, evidence-backed, as-of versioned)
           → InstitutionInstance (scenario-bound) + InstitutionRuntime (authority + stage + decision engines)
         WorldState event → InstitutionOperator:
           authorize (block if unauthorized) → check stage permits → run decision engine (real votes)
           → advance stage → StateDelta → schedule next event → write terminal quantity
```

## 3. Exact compiler → institution → runtime path

```
compile_world(question, as_of, ...)                          swm/world_model_v2/compiler.py
 → required_causal_processes (incl. institutional ones)
 → _select_institutions(raw, processes, as_of)                → institutions_v2.compile.select_institution
     → families that ANSWER the process (answers_processes)   (no keyword routing)
     → templates valid AS-OF, jurisdiction match, verified evidence → tier 1–4
 → provenance["institution_selection"] = {process: {family, template, tier, ...}}
 → build_world → WorldState
 → Event(etype="institutional_action", payload={institution: InstitutionRuntime, action, decision})
 → InstitutionOperator.run: propose(authorize) → apply(decide + advance stage + StateDelta + terminal)
 → rollout → terminal projection → SimulationResult
```

## 4. Family / template / instance model (Part 1)

- **`InstitutionFamily`** (`record.py`) — reusable structural pattern: roles, `AuthorityEdge` graph, permitted
  actions, information rights, `Stage` graph, threshold/resource/deadline/enforcement/appeal semantics,
  `answers_processes`, `code_ref` (executable procedure in `families.py`).
- **`InstitutionTemplate`** — one real institution for a period: `valid_from/valid_to`, `RuleRecord`s each
  linked to an `EvidenceRecord`, thresholds/quorums, informal-practice layer, `rules_as_of(as_of)`.
- **`InstitutionInstance`** — scenario-bound: actor/role/matter bindings, current stage, resources, queue
  state, competing models, posterior weights, support grade, fallback tier.

## 5. Engines

- **Authority** (`authority.py`) — `AuthorityGraph.authorize` blocks an action unless the actor's role holds
  the required authority for the matter/subject/stage; **advisory ≠ decision authority**. `InformationBoundary`
  filters observations so an actor cannot condition on sealed/privileged info (Part 6).
- **Decisions** (`decisions.py`) — quorum (majority = floor(n/2)+1), simple/absolute/super-majority, plurality,
  unanimity, weighted, recusal, abstention-vs-absence, tie-break, veto + override (Art I §7).
- **Procedure** (`procedure.py`) — `Matter` + agenda ops, general `StageEngine` (acyclic check), capacity-
  constrained `ResourceQueue` (real timing effect).
- **Evidence** (`evidence.py`) — as-of `active_rules`, deterministic `validate_rule` (Part 4), `leakage_audit`,
  `amendment_chain`.

## 6. Phase 3 & Phase 6 integration (LIVE — continuation)

- **Phase 6 (live):** `experiments/wmv2_phase10_predict.py` implements the genuine end-to-end path — as-of
  party composition → a Phase-6 partisan actor policy → the institution's matter-aware threshold engine →
  **StateDelta via the real `InstitutionOperator`** → terminal outcome probability. It is OUT-OF-SAMPLE
  (train Congress 117 → test 118) and leakage-safe (the target vote's own counts are never inputs), and is
  reported SEPARATELY from procedural reconstruction (forward prediction is honestly modest: acc 0.83, Brier
  0.132 vs base-rate 0.144; procedural reconstruction on real votes is 96.3% and validates rule EXECUTION,
  not forecasting).
- **Phase 3 (live):** `institutions_v2/particles.py` draws REAL posterior weights over competing institutional
  interpretations from the merged Phase-3 engine (`infer_compositional_posterior` — a Dirichlet over a
  hypothesis simplex) and `execute_competing_models` runs each hypothesis separately, aggregating a weighted
  terminal distribution (incompatible rules never averaged). **No Phase-10-local pseudo-posterior was
  created** — the weights come from the Phase-3 engine. Information boundaries still enforce that a Phase-3
  actor view cannot see unavailable info; templates still record where posteriors are needed.
- **Automatic rule extraction (live):** `institutions_v2/extract.py` — source text → LLM candidate WITH a
  verbatim source span → source-span grounding → deterministic `validate_rule` → typed rule. The LLM proposes;
  it cannot establish an unsupported rule (macro precision 1.0 / recall 0.83 on two real documents).

## 7. Storage, versioning, hashes, migrations

Committed machine-readable registry: `institutions_v2/data/{families,templates}.json`, each wrapped in a
SHA-256 integrity envelope. Templates carry `content_hash` + `version`. New package is additive
(backward-compatible with `institutions.py`); no migration of existing worlds required.

## 8. Artifacts index

| Artifact | What |
|---|---|
| `swm/world_model_v2/institutions_v2/data/families.json` / `templates.json` | family + template registry (hashed; gitignored — regenerated from `build.py`) |
| `experiments/results/phase10/wmv2_phase10_replay.json` | real Senate historical replay + ablations + leakage |
| `experiments/results/phase10/wmv2_phase10_court_replay.json` | real SCOTUS/SCDB replay — decision 99.4% + non-voting term-deadline timing 99.6% |
| `experiments/results/phase10/wmv2_phase10_referendum_replay.json` | real Swiss referendum replay — form regularity + out-of-sample forecast + non-voting cadence |
| `experiments/results/phase10/wmv2_phase10_predict.json` | out-of-sample predictive path vs procedural reconstruction (metric separation) |
| `experiments/results/phase10/wmv2_phase10_extraction.json` | automatic rule extraction vs verified ground truth (P 1.0 / R 0.83) |
| `experiments/results/phase10/wmv2_phase10_forensic_traces.json` | WorldState execution + counterfactuals |
| `experiments/results/phase10/wmv2_phase10_summary.json` | counts + per-template planes |
| `experiments/results/phase10/wmv2_phase10_failures.json` | preserved negatives |
| `experiments/results/exp074/referenda.json` | committed real Swiss federal referendum source (704 votes) |
| `experiments/results/phase10/voteview/` · `.../scdb/` | cached VoteView CSVs / SCDB zip (real data, gitignored, re-downloaded on demand) |
| `swm/world_model_v2/institutions_v2/particles.py` · `extract.py` | competing-model execution (live Phase-3) · automatic rule extraction |
| `tests/test_wmv2_phase10.py` | 28 acceptance tests (18 base + 10 continuation) |
