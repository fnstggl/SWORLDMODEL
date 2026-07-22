# WMv2 Production Architecture

*The one universal social-world-model execution path, after the production round. Every V2 forecast flows
through it; there is no top-level hybrid bypass. Statistical models, fitted functions and solvers appear
ONLY as internal implementations of typed mechanisms inside the shared world.*

## The path

```
natural-language question + as-of + optional intervention + evidence + compute budget
  │
  ▼  swm/world_model_v2/compiler.py : compile_world           [LLM PROPOSES, validator TYPES]
  │    · LLM proposes decomposition (entities, populations, institutions, relations, quantities,
  │      information, latents, events, hazards, mechanisms BY REGISTRY ID, readout, scenario tags)
  │    · registry-vetted + EXECUTABLE-operator-gated mechanism acceptance (unexecutable → rejected loudly)
  │    · production-registry applicability scoring recorded as selection provenance
  │    · outcome contract with readout REQUIRED; CompileAbstention on untypeable slices
  │    → WorldExecutionPlan (versioned, prompt-hashed, evidence-bundle-hashed)
  │
  ▼  swm/world_model_v2/evidence.py + leakage_audit.py         [EVIDENCE PLANE]
  │    · typed EvidenceBundle: per-item URL, dual timestamps, content+bundle hash, credibility,
  │      actor visibility; as-of gate (ZERO slack default) quarantines post-as-of/undated items
  │    · leakage auditor: resolution terms, future dates, retrospective language, duplicate collapse,
  │      timestamp-basis grade → per-question leakage report; hard leaks recommend exclusion
  │
  ▼  swm/world_model_v2/materialize.py : build_world           [WORLD-STATE PLANE]
  │    · typed entities/populations/network/institutions/quantities/information
  │    · PROVENANCE HONESTY: LLM proposals enter `inferred`, never `observed`
  │    · LOUD FAILURE: unknown fields→typed latent_state (kept+recorded); unknown relations/rule-kinds
  │      dropped+recorded; MaterializeAbstention on dangling readout / no executable mechanism
  │    · closed executable rule-kind registry (unknown kinds fail closed)
  │
  ▼  swm/world_model_v2/init_state.py + inference_layer.py     [POSTERIOR PLANE]
  │    · evidence→posterior: hierarchical shrinkage (person←segment←population), evidence-conditioned
  │      latents with provenance, correlated latents (range-correct), structural hypotheses as
  │      per-particle mechanism/parameter assignments
  │    · coherent joint particle sampling; each particle a causally-distinct world
  │
  ▼  swm/world_model_v2/rollout.py : RolloutEngine             [EXECUTION PLANE]
  │    · event-driven REAL calendar time (scheduled + hazards + background-over-elapsed)
  │    · per operator: applicable → propose → institution-validate → apply → StateDelta
  │    · endogenous action→event chains (operators emit validated follow-up events)
  │    · policy contract: LLM may NOT mint behavioral probabilities on the production path;
  │      FittedDecisionOperator (utility+QRE / anchored-logistic / hierarchical rates) is the default;
  │      uniform fallback is loudly flagged; LLM-minting survives only behind an experimental opt-in
  │    · optional filtered rollout: assimilate timed observations → reweight branches → resample
  │
  ▼  swm/world_model_v2/contracts.py : OutcomeContract.project [TERMINAL READOUT]
  │    · the number is READ from terminal states (weighted frequencies / quantiles), never an LLM
  │    · option-space coverage: terminal values outside the declared space are `unresolved_share`,
  │      not answer mass (a silent no-op world cannot read as a confident answer)
  │
  ▼  swm/world_model_v2/calibration.py                         [PRODUCT OUTPUT CONTRACT]
       · conditioned calibration (train/val-fit-only, versioned, partial-pooling)
       · signal-driven abstention (5 grades: supported..unresolvable)
       · uncertainty decomposition (structural vs state/parameter vs evidence)
       · direct-forecast critic (flags disagreement, NEVER overwrites the simulation)
       → raw p · calibrated p · confidence grade · abstention · uncertainty decomposition ·
         sensitivity · omitted high-impact variables · structural disagreement · calibration provenance
```

## The five planes, enforced

| Plane | Where | Enforcement |
|---|---|---|
| CODE | `swm/world_model_v2/*` | typed APIs, versioned, dependency-free core; 720+ tests |
| EVIDENCE | `evidence.py`, `leakage_audit.py` | as-of gate, content/bundle hashes, leakage report; string evidence flagged `unaudited` |
| POSTERIOR | `init_state.py`, `inference_layer.py`, `posterior.py` | distributions with provenance; no unsupported point precision; range-correct sampling |
| WORLD-STATE | `materialize.py`, `state.py` | scenario-specific typed objects at runtime; provenance honesty; loud omissions |
| EXECUTION | `rollout.py`, `transitions.py` | StateDelta per transition; endogenous events; institution validation before apply |

## Anti-bypass invariants (tested)

- No scenario-level branches in `swm/world_model_v2/` (AST-pinned test).
- The compiler is the ONLY constructor of production plans; benchmark adapters load input / define the
  externally-given resolution / format output / score — they may NOT construct the world or select
  mechanisms.
- The general path was exercised with a **real LLM** on ≥100 held-out NL questions (16 domains) — see
  `docs/WMV2_COMPILER_VALIDATION.md`. Prior to this round it had *never* run against a real LLM.
- No mechanism is fitted outside the world and returned as the final label; fitted estimators are internal
  transition implementations (Higgs hazard, choice QRE, persistence momentum) whose predictions are read
  from terminal survival/choice state.

## Four-status honesty

Every component reports FOUR statuses, never one "complete":
**software-implemented** / **executes-end-to-end** / **empirically-validated** / **production-eligible**.
The acceptance-gate table in `docs/WMV2_HISTORICAL_BENCHMARK_RESULTS.md` (final report) records all four
per capability. Most components are software-implemented + execute-e2e; empirical validation is concentrated
where real held-out data exists (diffusion, choice, persistence, inference-recovery, calibration, evidence);
production-eligibility is earned only by passed held-out/transfer records (2 mechanism families to date).

## Versioning & reproducibility

- `WorldExecutionPlan` carries version, parent_version, prompt_hash, evidence_bundle_hash, scenario,
  registry selection provenance.
- Machine-readable registry (`registry/data/{registry,packs}.json`) is integrity-hashed (sha256) with
  atomic writes and a corruption check on load.
- All benchmarks are deterministic under fixed seeds; long runs (Higgs cohort, compiler generality) are
  resumable via committed caches.
- Cost + latency metered per run; every result file records llm_calls / est_cost_usd / runtime_s.
