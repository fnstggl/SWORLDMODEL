# WMv2 Mechanism Registry

*Machine-readable: `swm/world_model_v2/registry/data/registry.json` + `packs.json` (integrity-hashed).
Rebuild: `PYTHONPATH=. python -m swm.world_model_v2.registry.build_registry`. This document is the
human-readable index and the honest status ledger.*

## Three-layer structure

1. **Universal family** — structural form + causal semantics + executable transition (`code_ref`).
2. **Domain parameter pack** — parameters estimated for a defined population/context/period, WITH
   posterior uncertainty and provenance; transport risk widens uncertainty when moved.
3. **Scenario instance** — the compiler binds a pack to concrete world paths at runtime.

## Lifecycle (enforced by promotion gates, not labels)

`proposed → implemented → locally_validated → transfer_validated → production_eligible`, with
`quarantined` / `rejected` demotions. Gates (`MechanismRecord.promotion_blockers`):
- **implemented**: `code_ref` resolves to a callable + `test_ref` present + formal description.
- **locally_validated**: + ≥1 parameter pack + a held-out/posterior-predictive validation record.
- **transfer_validated**: + a transfer validation record.
- **production_eligible**: + a supporting citation AND a **PASSED** held-out/transfer record. *A paper
  citation or a synthetic test alone can never make a mechanism production-eligible.*

## Status ledger (47 families, all with executable transitions — zero empty names)

| status | n | meaning |
|---|---|---|
| **production_eligible** | 2 | passed held-out + transfer on real data |
| **locally_validated** | 1 | held-out recorded (result may be null/negative) |
| **quarantined** | 1 | held-out FAILED; preserved, excluded from production |
| **implemented** | 43 | executable transition + tests; empirical validation pending |

### production_eligible (the earned bar)

| family | domain | validation |
|---|---|---|
| `exposure_response_hazard` | social-media diffusion | Higgs held-out: closes gap to fitted logistic (Δ−0.000192 [−0.000525,+0.000145]); nonlinearity beats linear Δ−0.00253 [−0.00341,−0.00169] (both `wmv2_higgs_nonlinear.json`) |
| `engagement_momentum_persistence` | platform engagement | OmniBehavior held-out n=7074: Δ−0.0065 [−0.0092,−0.0035] vs memoryless user-rate (power 0.993); person-disjoint transfer Δ−0.027 [−0.032,−0.021] (`wmv2_persistence_power.json`) |

### quarantined (failure preserved)

| family | why |
|---|---|
| `hawkes_self_excitation` | held-out count forecast FAILED to beat Poisson on the Higgs stream (MAE 1098.9 vs 973.0); constant background + single exponential kernel underfit the burst. Preserved, not deleted. |

### locally_validated

| family | note |
|---|---|
| `social_preference_population` | BehaviorBench held-out recorded: beats raw LLM/elicitation but LOSES to per-game histogram in-distribution (0.099 vs 0.038); LOGO transfer mixed (wins 3 games, fails public_goods). Honest — not promoted. |

## Ontology coverage (47 families / ~40 target)

`diffusion` 7 · `relationship` 6 · `influence` 4 · `learning` 4 · `participation` 3 ·
`attention`/`bargaining`/`decision`/`institutional`/`measurement`/`memory`/`observation`/`platform` 2 each ·
`belief`/`coalition`/`exogenous`/`interpretation`/`network`/`norm`/`resource` 1 each.

Maps to the 40 required core families: observation/exposure, attention allocation, memory/decay,
message interpretation, utility choice, quantal response, reinforcement, belief learning, EWA, habit,
trust formation/violation/repair, reciprocity, relationship strengthening/decay, obligation/norm,
bargaining, negotiation concession, coalition, voting/turnout, mobilization, donation, simple contagion,
complex contagion, social reinforcement (via complex contagion), threshold/tipping, DeGroot, bounded
confidence, latent-vs-expressed, Hawkes, information aging, finite-population saturation, network rewiring,
platform examination/position bias, platform ranking, resource depletion, institutional approval/veto,
agenda/stage control, measurement/reporting, exogenous shocks. **All present with executable transitions.**

## Honest four-status for the registry as a whole

- **software-implemented**: YES — 47 families, all with executable transitions, tests, applicability
  rules, uncertainty representation.
- **executes-end-to-end**: YES — families are selected by the compiler (applicability scoring) and execute
  inside WorldState (diffusion/choice/persistence proven; the rest execute in unit tests and the
  compiler-generality run).
- **empirically-validated**: PARTIAL — 2 families passed held-out+transfer, 1 has recorded held-out
  (negative), 1 quarantined (failed); the other 43 have executable transitions + published forms but no
  held-out validation yet. This is stated per-family, not hidden.
- **production-eligible**: 2 of 47. The gate is real; most families are honestly "implemented," not
  "production."

## Parameter packs (6, distinct empirical contexts — not duplicated config)

`higgs_2012_rumor` (Twitter Higgs), `higgs_2012_stream` (Hawkes, failed), Hill/linear Higgs packs,
`behaviorbench_moblab` (MobLab games), `omnibehavior_kuaishou` (Kuaishou traces). Each carries fit method,
dataset, transport note, and validation history. **Failed packs are preserved, never overwritten.**

## Ingestion pipeline

`registry/ingestion.py` implements: register published mechanism (requires citation with transport
limits) → encode transition → attach study/dataset → fit pack (shared estimation core: logistic,
Bernoulli-hazard GLM, frailty profile likelihood, functional-form comparison) → held-out validation →
record failure → promote/quarantine. `compare_forms` reports ALL candidates (no silent winner-only).
