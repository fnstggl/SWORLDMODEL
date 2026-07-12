# WMv2 Production Gap Audit — Phase 0 (complete)

*Verified against branch `claude/world-model-v2-production-64qwa6` @ 853935c. Method: direct code
inspection by the primary agent, integrating five completed subagent audit reports (Appendix A lists
which findings were accepted directly, independently verified, modified, or rejected). Empirical claims
cite committed artifacts. This document governs the dependency order for the rest of the program.*

---

## 0. The one-sentence verdict

The V2 runtime (typed state, events, deltas, terminal readout) is real and honest, but **nothing that
produced a benchmark number has ever flowed through the general path** — the compiler has never been run
against a real LLM, evidence never enters as typed data, hidden state is never inferred from evidence,
populations/observation/assimilation layers are inert or orphaned, the domain-general decision operator
lets the LLM mint probabilities, and one benchmark's "V2" arm bypasses the world entirely.

## 1. Capability classification (all 33)

Statuses: **PE** production-executable · **EU** executable-unvalidated · **ET** executable-toy ·
**PI** partially-implemented · **IO** interface-only · **HC** hardcoded · **BS** benchmark-specific ·
**AB** absent.

| # | Capability | Status | Key evidence (file:line) |
|---|---|---|---|
| 1 | Question parsing | **EU** | `compiler.py:103-142`; only callers are tests with scripted `llm=lambda` (`test_world_model_v2.py:354,374`) — zero real-LLM executions repo-wide |
| 2 | Causal-world construction | **PI** | scenario-agnostic core pinned by `test_no_scenario_branches_in_v2_source`; but `materialize.py:37` stamps every LLM-proposed field `status="observed"`; `:39-40,:53-54,:66-67` silently drop non-schema fields/relations/quantities (comment at :54 claims "loudly" — false) |
| 3 | Actor discovery | **PI** | one LLM call over `evidence[:4000]` (`compiler.py:113`), proposals pass through unvalidated (`:196`); no dedup/alias/citation check; fabrications inherit "observed" |
| 4 | Population construction | **ET** | `population.py:58-104` real allocation math; **inert**: no rollout path reads `world.populations`; heterogeneity = hardcoded `N(0.5,0.2)` per dimension (`materialize.py:47`) |
| 5 | Institutional discovery | **PI** | enforcement core real & wired (`institutions.py:64-71` ← `transitions.py:89-97,193-199`); but only 5 rule kinds implemented; **any unknown kind silently returns (True,"")** (`institutions.py:24-55` falls through); `deadline` KeyErrors/TypeErrors on missing/string `by_ts`; no discovery-from-evidence |
| 6 | Graph construction | **ET** | `network.py:61-87` typed edges but O(E) scans, no referential integrity; unregistered relations silently dropped (`materialize.py:50-54`); no adapters from real relational data; visibility/latency fields unread |
| 7 | Evidence retrieval | **PI** | real as-of retrieval exists in V1 (`engine/retrieval.py`: dated Google News RSS, Wikipedia revision-as-of; probed live by subagent); **four disjoint stacks; none wired into V2** — the compiler receives a caller-pasted string |
| 8 | As-of verification | **PI** | hard gates exist on toy stores (`retrieval/asof_store.py:50-82`, `retrieval/corpus.py:45-86` raise LeakageError); live paths rely on per-call `search_fn` convention; ~1-day slack windows; no snapshots/content-hashes on any forecast path |
| 9 | Hidden-state inference | **PI** (inference itself **AB**) | `init_state.py:106-139` is prior *sampling* (records/correlations/coherence real); no code maps evidence→posterior; `inference/filter.py` is an 11-line stub `IMPLEMENTED=False`; `CorrelationRule.adjust` defaults lo/hi to [0,1] → biased shifts for non-unit-range latents (e.g. responsiveness [0.5,1.8]) |
| 10 | Structural uncertainty | **IO** | `uncertainty.py` pipeline orphaned (test-only callers); no competing world hypotheses at runtime; all particles share one mechanism set; `posterior.py` docstring claims otherwise |
| 11 | Mechanism selection | **PI** | selection = `if mid in registry` (`compiler.py:147`); 3 of 9 lean entries have `operator=""` (`mechanisms.py:72-80`) and are **silently skipped** at `materialize.py:91-93` → plans "accepting" them execute nothing; new `registry/` (this round) has scoring but is **not yet wired into compile_world** |
| 12 | Mechanism applicability | **IO** (live path) | `required_state`/`domains`/`invariants` declared but never read (grep-verified); a plan accepting `institutional_vote` with zero institutions compiles and runs to `{'None': 1.0}`; new `registry/applicability.py` implements real scoring, unwired |
| 13 | Parameter estimation | **BS** | every reference world hand-writes its own fitter (`enron.py:123`, `behavior_games.py:92-158`, `higgs.py:113`); all point estimates; the only parameter uncertainty anywhere is a hardcoded lognormal sd 0.4 (`higgs.py:161`); new `ingestion.py` estimation core exists, adoption pending |
| 14 | Actor cognition | **EU** | full pipeline ran on Enron with real LLM (660 metered calls); **validation null**: no structured level beats fitted metadata (C6 vs E1 Δ+0.00627 CI[−0.006,+0.019], n=120/60 badly underpowered); action-split constants hardcoded (`actor_cognition.py:195-216`); `p_engage` silently truncates on feature-length mismatch (`:154`) |
| 15 | Typed action policies | **PI** | fitted anchored-logistic path real in reference worlds; **the domain-general `AgentDecisionOperator._policy` (`transitions.py:167-188`) has the LLM directly return the action distribution** — probability minting, contradicting the module contract; `llm=None` → uniform coin-flip |
| 16 | Interaction | **PI** | runtime supports shared multi-actor state (tests pass); benchmarked "interaction" is belief simulation *inside one actor's* closed-form policy (`behavior_games.py:193,205`); Enron worlds have ONE entity; operators cannot emit follow-up events → no action→exposure→decision chains |
| 17 | Belief updating | **ET** | `BeliefUpdateOperator` real, bounded; constants hardcoded priors (0.9 step, ad-hoc compat term, `transitions.py:254-256`); exercised only by tests; never validated against belief data |
| 18 | Trust/relationship dynamics | **ET→PI** | `relationship_strength` genuinely inferred from interaction history (`actor_cognition.py:270-275`), bounded transitions persist and modulate within Enron rollouts (`enron.py:528-534`, verified); no trust violation/repair; transition constants unvalidated priors |
| 19 | Persistent state | **PI** | within-rollout persistence real (attention via `last_observation_ts`, relationship shifts, deferred replies — `enron.py:370-600`); cross-episode persistence absent; OmniBehavior "persistence" = a momentum *feature* in a logistic (`omnibehavior.py:100-114` — no world runtime at all); n=48 underpowered (Δ−0.00003 ns) |
| 20 | Network diffusion | **PI** | mechanisms real and (this round) nonlinear with fitted hazards executing event-driven survival (`registry/families/diffusion.py`; result: gap to fitted logistic closed, Δ−0.000192 CI[−0.000525,+0.000145]; nonlinearity beats linear Δ−0.00253 sig); runs on a fast path outside WorldState objects (parity-tested); **no in-runtime network exposure operator**; graph layer itself is toy (cap 6) |
| 21 | Institutional execution | **PI** | `validate_action` + `run_vote` + `InstitutionalVoteOperator` real and wired; 5 kinds only; silent-pass on unknown kinds (verified); no stages/quorums/vetoes/appeals/capacity; no evidence pipelines for real institutions |
| 22 | Exogenous events | **PI** | scheduled+hazard queue real (heap, re-arming, `rate_fn`, `events.py:87-126`); `endogenous` scheduling registered but **unsupported** — `RolloutEngine` gives operators no way to queue events; no shock/regime-change mechanism beyond registered names |
| 23 | Observation models | **ET** | two synthetic families with invented constants (`observation.py:56-116`); process-global registry; never attached by compiler; unused by any forecast path |
| 24 | Posterior assimilation | **ET** | `ParticlePosterior` machinery real (ESS, systematic resampling, ancestry, PPC) but **disconnected**: `WorldBranch.weight` never mutated anywhere; rejuvenation assumes [0,1] ranges (`posterior.py:117-131` corrupts non-unit latents); missing-latent likelihood 0.5 is a category error |
| 25 | Dynamic recompilation | **IO** | `recompile()` (`compiler.py:210-219`) = re-run compiler + version chain; **no triggers** (no regime/contradiction/ESS-collapse detection), no state migration, no competing revised plans |
| 26 | Counterfactual execution | **EU** | matched-seed cloned-world interventions with per-particle P(best)/regret real (`rollout.py:93-126`); exercised by unit tests only; no benchmark with real (quasi-)randomized interventions has run through it |
| 27 | Best-action search | **PI** | V1 `decision/best_action.py` racing (successive elimination) real but legacy; V2 `evaluate_interventions` enumerates caller-given arms — no search, no sequential policies, no robustness-across-hypotheses reporting |
| 28 | Calibration | **PI** | V1 grade-or-abstain registry + fitted logit shrink real (`engine/calibrate.py`); **V2 results never pass through any calibration**; no domain/horizon/mechanism-status calibration for V2 |
| 29 | Abstention | **PI** | `CompileAbstention` real at compile time; V1 dossier abstain real; no sensitivity-driven abstention (high-sensitivity `assumed` fields never trigger it despite `Provenance` docstring), no runtime uncertainty-based abstention in V2 |
| 30 | Direct-model criticism | **PI** | baselines exist as named facade architectures (`facade.py:96-125`, call-spy pinned); no integrated per-forecast critic comparing V2 vs direct/ensemble/panel with disagreement diagnosis |
| 31 | Historical forecasting | **PI/BS** | corpus real (Manifold+Polymarket resolved binaries, crowd reconstructed at as-of, `eval/forecasting_corpus.py`); **no evidence snapshots** (question-text only); the ForecastBench "V2" arm is `1-(1-platt(crowd))^(g^(1-rem))` (`wmv2_forecastbench_run.py:123-135`) — **a closed-form crowd rescale that never touches the world: a top-level bypass**; runner not resumable |
| 32 | Forward ledger | **PI** (PE for V1 needs) | append-only versioned locks real (`engine/forward_ledger.py`, invariants verified in header+`data/forecast_log.jsonl`); V2 lock fields absent (plan hash, mechanism versions, particle count, evidence bundle hash, state posterior) |
| 33 | Production observability | **PI** | per-run cost/latency meters in runners; `resilient_llm` retries; **no trace IDs, no structured logs from the runtime, no concurrency safety, no resumability** (except the new Higgs cohort cache), no corruption checks (except the new registry store) |

## 2. End-to-end chain trace (where it breaks)

The only general path is `facade.forecast(architecture="world_model_v2")` → `compile_world` →
`run_from_plan` → `{build_world, queue_builder_from_plan, operators_from_plan}` → `WorldModelV2Run.run` →
`OutcomeContract.project`. Link-by-link:

| Link | Real implementation? | Break |
|---|---|---|
| question → parse | one LLM JSON call, validated shallowly | never run with a real LLM; no readout-binding, horizon>as_of, or option-space cross-checks |
| retrieval → evidence | **not wired** | evidence is a caller-pasted string truncated `[:4000]`; V1 retrieval stacks unused by V2 |
| compile → plan | typed validation real (mechanisms/contract/latents/events) | entities/populations/relations pass through unvalidated; latent confidence hardcoded 0.3; sensitivity is advisory LLM output |
| plan → world | typed materialization | provenance fabrication ("observed"), silent drops, hardcoded heterogeneity, populations built then never read |
| world → posterior(t0) | prior sampling with correlations/coherence | **no inference from evidence anywhere**; correlation adjust biased off [0,1] ranges |
| mechanisms → operators | registry lookup | 3/9 entries execute nothing, silently; applicability metadata dead |
| actor view → decision | `observable_view` boundary real | policy = uniform or LLM-minted probabilities; fitted policies exist only in hand-built worlds |
| action → world | StateDelta machinery real, institution-validated | operators cannot emit events → no interaction chains |
| time | real event-driven calendar time | — (solid) |
| observations → assimilation | machinery exists | never called from any forecast path; weight stays 1.0 |
| terminal → readout | projection real, weighted | readout can dangle (None becomes an outcome); options unvalidated vs readout codomain |
| readout → calibration/abstention | — | V2 outputs are raw; no calibration, no post-compile abstention |

## 3. Benchmark call-path table (what each benchmark actually exercised)

| Benchmark | Through compile→materialize? | Through WorldModelV2Run? | What was really tested | Bypassed |
|---|---|---|---|---|
| Enron (I/C ladders) | NO (hand-built) | YES (`v2_predict`, `v2_predict_actor`) | fitted hazard integration, within-rollout persistence, typed action split, interpretation channel | compiler, evidence layer, inference, interaction (1 entity), assimilation, calibration |
| BehaviorBench | NO | YES (trace path `v2_game_world`) + fast path | fitted latent populations, within-policy partner simulation, trembling fits | compiler, evidence, inference, assimilation; partner response not causally load-bearing |
| OmniBehavior | NO | **NO** (pure function `v2_engagement_predict`) | fitted per-user/type rates + momentum feature + latent jitter | the entire V2 runtime |
| Higgs (incl. new nonlinear) | NO | NO (fast path, parity-tested vs closed form) | exposure/contagion hazard forms, rollout, latent q | runtime objects, graph layer, compiler, inference |
| Upworthy | NO | partially (population world, per prior round) | surface-feature population response, fitted weights | compiler, evidence, semantic channel (harmful), inference |
| ForecastBench subset | NO | **NO** | Platt + deadline exponent **on the crowd price** | **everything — labeled V2, is a crowd rescale (Rule-2 violation to fix)** |

## 4. Root-cause analysis (ranked)

The portfolio pattern to explain: `raw LLM < structured simulation ≤ specialized fitted model`, with V2
winning only in transfer.

1. **RC1 — The general path is unexercised.** (architectural centrality: max; evidence: grep/call-path,
   §3.) Every number comes from hand-built worlds or bypasses, so the compiler/inference/evidence links
   have never faced empirical pressure, and "generality" is untested marketing. *Symptoms*: caps 1–3, 31.
2. **RC2 — No evidence→posterior inference.** Hidden state = broad priors + values mislabeled
   "observed". A simulation whose initial state carries no more information than a prompt cannot beat a
   fitted model that consumed the training distribution. *Symptoms*: caps 9, 10, 23, 24; "world-state
   inference remains weak."
3. **RC3 — Mechanism forms too rigid, parameters benchmark-welded.** Linear hazards lost to a logistic
   because the *form* was wrong (now demonstrated fixable: nonlinear hazard closed the gap); no shared
   estimation layer, no packs, no uncertainty on parameters. *Symptoms*: caps 11–13, 20; Higgs/Upworthy.
4. **RC4 — Policy-layer contradiction.** The general path mints LLM probabilities (documented
   miscalibration: ECE 0.16–0.33 on Enron); the calibrated fitted-policy machinery lives only inside
   reference worlds; no hierarchical pooling/person shrinkage/utility+QRE in the general runtime.
   *Symptoms*: caps 14–15; "fitted task models still win."
5. **RC5 — Semantic features enter unvalidated.** No per-feature acceptance gate exists, so 18 dims ride
   on plausibility; Upworthy showed they can significantly hurt. *Symptom*: cap 14; "semantic
   interpretation often adds no value or causes harm."
6. **RC6 — Evidence/as-of is not a typed, audited layer.** String-pasting + convention-based as-of +
   slack windows + no snapshots/hashes = both leak risk and information poverty. *Symptoms*: caps 7–8, 31.
7. **RC7 — No integrated calibration/abstention/critic for V2 outputs.** *Symptoms*: caps 28–30.
8. **RC8 — Structural under-testing of interaction/persistence.** One-entity worlds, no operator-emitted
   events, n=48 persistence cohort. *Symptoms*: caps 16–19, 22.
9. **RC9 — Inert declared state (populations, observation models) and vacuous institutional guarantees**
   (silent-pass rule kinds). *Symptoms*: caps 4–5, 21, 23.
10. **RC10 — Observability/resumability gaps** make long benchmarks fragile and results hard to audit.
   *Symptom*: cap 33.

Ruled *out* as root causes: benchmark mismatch (the portfolio was well-matched per mechanism);
simulation-resolution limits (event-driven time is solid); LLM capability per se (structure beat raw LLM
everywhere).

## 5. Dependency-ordered execution plan

Foundational fixes come first because everything downstream inherits them. Each item lists
(depends → unblocks · deliverable · acceptance test).

**Tier A — foundational architecture (now)**
- **A1 Provenance & loud-failure fixes**: compiler-proposed values enter as `inferred/assumed` with
  prompt-hash method; silent drops become recorded omissions or abstentions; readout-binding check;
  closed rule-kind registry (unknown kind → abstain/experimental, never silent pass); registration-time
  operator resolution (no empty-operator "accepted" mechanisms); CorrelationRule range fix.
  (∅ → everything · code+tests · negative-compile suite passes; fabricated-provenance test fails before/passes after)
- **A2 Registry wiring**: `compile_world` selects via `registry/applicability.py` scoring over the
  production store; packs bind at compile; statuses gate execution. (A1 → C,D,E · code+tests · scoring
  visible in plan provenance; quarantined family never selected)
- **A3 Policy-path repair**: `AgentDecisionOperator` LLM-minting demoted to experimental; default policy
  = fitted/utility+QRE layer (policy.py) behind the same typed-action interface. (A1 → E,F · code+tests ·
  general path produces calibrated distributions on a held-out slice)
- **A4 Endogenous events**: operators may propose follow-up events (validated, queued) → real
  action→exposure→decision chains. (A1 → interaction tests · code+tests · two-actor chain test where B's
  decision depends on A's realized action)

**Tier B — evidence & grounding (Phase 2)** — typed `EvidenceBundle` (per-item URL, timestamps, content
hash, credibility, visibility), as-of gateway with zero-slack default, leakage auditor (resolution terms,
future dates, retrospective language, duplicates), wired into `facade`/compiler. (A1 → D,G,H · code+tests+
auditor reports · auditor catches planted leaks; V2 compile consumes a bundle not a string)

**Tier C — posterior inference (Phase 3)** — evidence-conditioned latent estimators with hierarchical
shrinkage; assimilation wired into rollout (weights actually update; rejuvenation/likelihood fixes);
structural hypotheses as per-particle mechanism/parameter assignment with BMA readout. (A1,A2 → D,F ·
code+tests · hidden-state recovery: posterior beats prior on synthetic truth; model-recovery concentrates
weight on the generating mechanism)

**Tier D — production compiler (Phase 1)** — two-stage parse/decompose with cross-checks; actor/population
grounding statuses; omission log + abstention risks; **first real-LLM generality run: ≥100 held-out cases
across 16 scenario classes** with automated validity metrics + rubric review. (A1,A2,B → F,G · code+cases+
`docs/WMV2_COMPILER_VALIDATION.md` · report actor/institution recall, mechanism precision/recall, invalid-action rate, abstention correctness)

**Tier E — mechanism library (Phase 6/7)** — finish family implementations with real fits on in-repo data
(choice families validated on BehaviorBench; diffusion done; trust/reciprocity from trust games;
DeGroot/bounded-confidence/latent-expressed honestly `implemented`+synthetic-PPC only where no local data
exists); packs with provenance; committed machine-readable registry. (A2 → F · registry JSON + tests ·
promotion gates enforced; no empty production entries)

**Tier F — dependency-ordered benchmarks** — (i) BehaviorBench policy run (validates A3/E; $0);
(ii) OmniBehavior scaled cohort with power analysis first (persistence at power, Phase 8);
(iii) interaction chain test via A4; (iv) compiler generality (D); (v) **historical forecasting (Phase 15)
through the real compile→materialize→rollout path** on the largest defensibly-as-of subset (wiki-revision-
grounded), with product-realistic baselines B0–B7 and no benchmark-specific predictors; resumable.
(C,D,E → G,H)

**Tier G — calibration/abstention/critic (Phase 12) + forward ledger v2 (Phase 16)** — domain/horizon
calibration fitted on train/val; sensitivity-driven abstention; per-forecast critic; v2 lock fields.
(F(i,iv) → H)

**Tier H — acceptance gates + docs (Phases 17-18)** — the 13 documents, honest final table, 17 questions.

**Deferred (explicitly incomplete unless reached)**: institutional evidence pipelines (Phase 10 beyond
rule-kind closure), network co-evolution, platform ranking families with local data, best-action
real-intervention benchmarks (Upworthy A/B arms can serve as quasi-experimental validation), dynamic
recompilation triggers (Phase 11), full 1,000-question corpus.

## 6. Premature-work classification (intervention §4)

Work created before this audit completed (all committed at 853935c, none deleted):

| Artifact | Classification |
|---|---|
| `swm/world_model_v2/registry/{record,store,applicability,ingestion}.py` | **likely reusable** — is the Tier-A2/E fix path; needs wiring into compiler |
| `registry/families/diffusion.py` + `reference/higgs.py` additions | **likely reusable** — mechanism family + fits; parity-tested |
| `experiments/wmv2_higgs_nonlinear_run.py` + `experiments/results/wmv2_higgs_nonlinear.json` | **benchmark artifact (valid)** — protocol identical to prior run (same cohorts/seeds); result stands: nonlinearity closes the logistic gap (Δ−0.000192 ns vs H1; −0.00253 sig vs linear); Hawkes held-out FAILURE preserved; *scope caveat*: fast-path execution outside WorldState objects, like the prior round |
| `swm/world_model_v2/policy.py`, `registry/families/choice.py` | **likely reusable** — Tier-A3/E core; validation run pending (Tier F-i) |
| `experiments/wmv2_behaviorbench_policy_run.py` | **retain for later review** — benchmark harness, not yet run |
| `experiments/results/higgs_nl_cache/` | **temporary artifact** (gitignored) |

## Appendix A — subagent findings integration

Five subagent reports were harvested before the workflow was stopped (raw JSON preserved at the session
scratchpad `audit_agent_results.json`). Coverage: capabilities 1–16 + terminal readout + call-path notes.
Capabilities 17–33 were audited directly by the primary agent (no subagent coverage).

**Accepted directly (precise file:line evidence, low controversy):** population inertness + hardcoded
heterogeneity (cap 4); network toy status + silent relation drops (cap 6); retrieval stack inventory and
live as-of probes (caps 7–8); orphaned status of `uncertainty.py`/`observation.py`/`posterior.py`/
`inference/filter.py` (caps 10, 23, 24); `fit_action_policy`/interpretation pipeline description and
Enron-ladder null numbers (cap 14, cross-checked against `wmv2_enron_actor_ladder.json` metadata quoted in
the finding); parameter-estimation inventory (cap 13); `hidden_state_latents` correlation-range bug (cap 9;
formula verified by reading `CorrelationRule.adjust` defaults).

**Independently verified by the primary agent (high-impact):** provenance fabrication + all silent drops
(`materialize.py:37,39-40,53-54,66-67`); empty-operator mechanisms skipped (`materialize.py:91-93`,
`mechanisms.py:72-80`); institutions silent-pass fall-through (`institutions.py:24-55`); LLM probability
minting in `AgentDecisionOperator` (`transitions.py:140-188`); ForecastBench V2-arm bypass
(`wmv2_forecastbench_run.py:123-141`); compiler never real-LLM-exercised (callers grep); Enron
single-entity worlds + within-rollout persistence (`reference/enron.py` full read); OmniBehavior runtime
bypass (`reference/omnibehavior.py:100-114`); event-queue realness + missing endogenous emission
(`events.py`); facade contract (`facade.py`).

**Modified:** cap 9 reclassified — the subagent's "partially-implemented" holds for the *prior-sampling
machinery*, but inference proper is **absent**; this audit records both. Cap 18 upgraded from toy toward
PI: relationship strength is genuinely data-inferred and its transitions are persisted and read back in
Enron level≥5 (verified), though the transition constants remain unvalidated priors.

**Rejected:** none material. (One subagent note read the BehaviorBench partner simulation as "defensible
as beliefs"; this audit keeps the harder framing: partner response exists in the trace path but is not
causally load-bearing, so shared-world interaction remains unproven.)

**Remaining uncertain (flagged, not asserted):** the exact severity of the 1-day as-of slack on live
Google News retrieval (needs the slack-exploit test in Tier B); whether `run_vote`'s threshold semantics
match real institutional supermajority rules across edge cases (Tier E test).
