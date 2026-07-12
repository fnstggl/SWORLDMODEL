# WMv2 ruthless foundation audit — what is REAL vs interface vs toy

*Line-by-line self-audit of `swm/world_model_v2/` at the foundation commit, BEFORE further implementation.
Classification scale: fully executable · executable-but-toy · interface-only · hardcoded · deterministic
placeholder · unparameterized · unvalidated · absent.*

## Capability classification

| Capability | Status | Honest detail |
|---|---|---|
| Universal state-value envelope (`StateField`+`Provenance`) | **fully executable** | value-or-dist, sample(), full envelope fields. BUT status semantics are only partially enforced in execution (observed fields aren't sampled because only latents enter `InitialStateModel` — there is no *general* guard preventing a latent record pointing at an observed field). |
| Provenance semantics affect execution | **partially executable** | Sampling stamps `sampled`; deltas stamp `derived`. Assimilation-side semantics (observed never perturbed, inferred vs assumed treated differently) were **absent** → now implemented in `posterior.py`. |
| Entity schema + extension registry | fully executable | arbitrary keys rejected; extensions typed. |
| Population allocation & particles | executable, **unparameterized** | machinery real; every weight/heterogeneity in tests is hand-set. No dataset-derived population yet. |
| Relation graph + registry | fully executable | but nothing FITS edge strengths from data; all priors. |
| Institutional rules | fully executable (5 rule kinds) | deterministic vote execution real; legal/agenda/capacity kinds beyond the 5 are **absent**. |
| Quantities | fully executable | registry + clamps; no observation models attached (see below). |
| Information ledger / actor-specific views | fully executable | `visible_to` is the only policy input; decay idempotent per interval. |
| Outcome contracts + terminal projection | fully executable | refuses to run without a readout. |
| Event queue, real time, hazards | fully executable | exponential hazards only; **elapsed-time scaling of transition magnitudes is toy** (background attention drift is a hardcoded mean-reversion prior; no per-mechanism temporal integration tests like "hazard over 30d = integrated probability"). |
| Transition operator contract + registry | fully executable | propose→validate→apply→delta enforced by `run()`. |
| Agent decision policy | executable, **behaviorally unvalidated** | typed-action boundary real; `llm=None` → uniform policy. NOTHING shows the LLM (or any policy) matches real human action distributions. |
| Belief update | executable-but-toy | bounded rule with invented constants (0.9, 0.5 factors) — labeled priors, never fitted. |
| Relationship update | executable-but-toy | bounded ±0.25 shifts; constants unfitted. |
| Resource update | fully executable | conservation real. |
| Background dynamics | **deterministic placeholder** | mean-reversion to 0.7 attention; memory half-life 10d — invented priors, flagged as such. |
| Mechanism discovery/estimation/validation (2B) | fully executable | typed proposals, 8-level hierarchy, rejection real, broadening real. BUT the registry holds **9 entries, ~6 executable** — an "empty library system" as charged. No fitted mechanism exists yet. |
| World compiler | executable, generality **unproven** | one scripted held-out scenario passed. Live-LLM decomposition on 5 held-out classes: **absent**. |
| Fidelity planner | executable-but-toy | deterministic thresholds (0.35 sensitivity cut) — arbitrary, unvalidated. |
| Dynamic recompilation | **interface-only** | `recompile()` re-runs the compiler with lineage; no trigger machinery (regime detection, new-actor detection) exists. |
| Rollout engine | fully executable | shared state, deltas, background interleave. |
| Matched counterfactuals | fully executable | cloned worlds + matched seeds + P(best)/regret. |
| **Observation model** | **ABSENT** → built now (`observation.py`) | there was NO latent→measurement→noisy-observation→evidence chain. |
| **Particle posterior** (weights/likelihood/ESS/resampling/ancestry) | **ABSENT** → built now (`posterior.py`) | particles were equal-weight samples counted at the end; no assimilation. |
| Direct-forecast critic | **absent in V2** | baselines exist on the base branch; disagreement analysis unwired. |
| V2 calibration / grade / abstention wiring | **absent** | v1 registry exists; V2 outputs are raw frequencies. |
| Logging / forward-ledger wiring for V2 | **absent** | ledger exists (v1); V2 runs don't lock into it yet. |

## What each acceptance test actually proves (and does not)

- Tests 1–5, 7, 11–13: **software invariants** — typing, deltas, persistence, info asymmetry, rule rejection,
  clock monotonicity, quarantine, naming. They prove plumbing. They prove nothing about realism.
- Test 6/8/9 (`_mini_run`): proves hidden state propagates to outcome distributions and readout comes from
  terminal states. The 50/50 attention prior and 95/5 policy are **invented toy numbers**; the policy is a
  hand-coded subclass, not an LLM; the scenario is handcrafted. **Plumbing, not realism.**
- Test 10: matched counterfactuals beat baseline — real property, toy world.
- Test 15 (novel scenario): the decomposition was **scripted JSON**, not a live LLM; it proves the compiler
  validates and executes a plan it did not author. Generality with live decomposition remains unproven.
- Phase 2B tests: real properties (rejection, broadening, conflicts) on synthetic proposals.

**Net: the foundation is a domain-general executable ontology + runtime. It is NOT a populated, learned,
calibrated world model. No V2 prediction has ever been compared against a real outcome.** The decision
threshold stands: typed latent state + event-driven evolution must beat same-model+same-evidence on a real
held-out dataset (Enron reply world first) before any migration.
