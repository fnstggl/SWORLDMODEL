# WorldModelV2 — Phase 0 ground-truth audit

*Verified from live runtime code (imports, constructors, branches) on `claude/world-model-v2` @ the fork point
of `claude/agent-engine-on-main` (7ba9ee6). Not from documentation. Full call-graph trace in
`docs/AUDIT_PART_A_WIRING.md`; this table reclassifies every component in the V2 taxonomy.*

## Classification table

| Component | Entry | V2 classification | Notes |
|---|---|---|---|
| **Observer panel** (binary deliberation default) | `front_door._panel` → `observer_panel.py` | **independent forecast ensemble** | 10 role-prompted calls, same evidence, no shared state, no interaction, single shot. **PRESERVED FINDING: the prior binary "FULL" arm was this ensemble — it was NEVER a society simulation and must never be called one.** |
| **Society rollout** | `front_door._society` → `society.py` | **prompt-based role-play with a shared public string** | Real cast + dated rounds + sampled public signal, but: state = a prose `public` string; persona "memory" (B7) = prompt-appended text; decisions are re-prompts, not typed actions; no machine-readable deltas; no institutional rules; no resources. Pooled n=87: adds nothing on deliberation. |
| **Individual mode** | `front_door._individual` → `individual.py` | **prompt-based role-play (sampled latent states)** | K latent states × reps of one person; states are prose; no relationship/commitment state; no follow-up timing; no persistence. |
| **Artifact optimizer** | `front_door._artifacts` → `actions.py` | **prompt-based role-play + paired ranking + iterative search** | Real candidate texts, fixed persona panel, engage/ignore decisions at temperature. No matched-world counterfactuals in the V2 sense (no cloned latent worlds, no shared seeds). |
| **Diffusion simulator** | `front_door._diffusion` → `diffusion.py` | **partial executable transition mechanism (unvalidated)** | Monte-Carlo cascade on a synthetic heavy-tailed graph IS executable state outside the LLM; but archetype propensities are LLM-sampled without calibration, the graph is synthetic, and it runs as an isolated answer engine (no shared world). Never graded on real cascades. |
| **Parametric kernels** (7 mechanisms) | `swm/api/mechanisms.py` via `_parametric_binary` | **executable transition mechanisms (structurally sound, parameter-starved)** | Honest Monte-Carlo forms (Poisson hazard, whipcount binomial, poll-error aggregation). Accuracy unmeasured; leak-free path compiles ground=False so contests degrade to p=0.50 (13/46 in EXP-099). |
| **`calibrated_readout` logistic** | `swm/api/compiler.py:262` | **LEGACY — the banned pattern** | Logistic over LLM-invented variables+elasticities. Unreachable from human-deliberation paths (test-pinned); still reachable via announcements → parametric. Must stay quarantined; V2 must not inherit it for human behavior. |
| **Legacy tree** (`swm/transition/*`, `swm/variables/*`, `swm/simulation/*`, `swm/worlds/*`) | not imported by engine | **legacy, unwired** | Dead from the production path; do not resurrect. |
| **Grounding** (`grounding.py` + `retrieval.py`) | `SceneGrounder` | **true structured pipeline (kept)** | As-of news + as-of Wikipedia revisions, leak-guards, directional standing, 5-axis grounding score. This is real infrastructure V2 builds on. |
| **Contract layer** (`engine/contract.py`) | validate/check | **true typed validation (kept, extended in V2)** | 6 outcome families, pre/post validation. |
| **Calibration + flywheel + forward ledger** | `calibrate.py`, `flywheel.py`, `forward_ledger.py` | **true structured infrastructure (kept)** | OOS temperatures, horizon buckets, temporal holdout, append-only versioned locks. |
| **Router** | `router.py` | **question-level classifier — the architecture V2 REJECTS as core** | people/parametric + binary_kind; keeps working as baseline/fallback; must not be V2's ontology. |
| **Ablation arms** (B0–B10, EXP-097–101) | `swm/eval/*` | **true measurement harness (kept as Phase 9 baselines)** | Pooled n=87: whole-stack vs one grounded call Δ=−0.0009 (p=0.98); B7/B8 worst arms. This is the bar V2 must beat. |

## The core deficiency V2 exists to fix

Nothing in the current system maintains **a world that exists as executable state outside the LLM context**:
no typed beliefs/resources/relationships/commitments, no machine-readable state deltas, no per-actor
information sets, no executable institutional rules, no event queue on real calendar time, no terminal-state
readout (every probability is ultimately an LLM output, pooled). The measured consequence: the "simulation"
layers add exactly zero over one grounded call, because they are the same LLM reading the same prose harder.

## What V2 inherits unchanged (baselines & infrastructure)

Grounding stack, contract layer, calibration/flywheel/ledger, the parametric kernels (as *validated numerical
mechanisms* behind the V2 mechanism registry), the panel/society/individual/diffusion paths (as **Phase 9
baselines**, correctly labeled), and every ablation result — including the negative ones.
