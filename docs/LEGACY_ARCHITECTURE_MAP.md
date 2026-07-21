# Legacy architecture map + freeze policy

*Verified from imports/call graphs (see `docs/AUDIT_PART_A_WIRING.md` for the full v1 trace and
`tests/test_world_model_v2.py` for the AST-enforced boundaries).*

## The boundary (enforced NOW)

- **One public door:** `swm/facade.py` — every run names its architecture explicitly (`world_model_v2` or
  `baseline:*`). No ambiguous default. Product-eligibility contract recorded on every run; a V2 run that
  executed legacy code RAISES.
- **AST-enforced imports** (`test_import_boundary_ast_v2_never_imports_legacy`,
  `test_facade_is_the_only_legacy_door_for_new_code`): `swm/world_model_v2/` can never import legacy engines/
  compilers/routers/logistic/ODE modules; new top-level code reaches baselines only through the facade.

## Module classification

| Module | Class | Current use | V2 replacement | Allowed future use | Removal prereq |
|---|---|---|---|---|---|
| `swm/engine/front_door.py` + panel/society/individual/actions/diffusion | **evaluation baseline** | `baseline:observer_panel_v1` / `baseline:society_v1` via facade; the Phase-9 ablation arms | compiler+rollout (once validated) | named baselines only | V2 beats them held-out per class |
| `swm/engine/grounding.py`, `retrieval.py`, `contract.py`, `calibrate.py`, `flywheel.py`, `forward_ledger.py` | **current production dependency (shared infrastructure)** | evidence, leak-guards, calibration, ledger — V2 consumes these too (they are not "engines") | n/a — kept | unrestricted | never (shared) |
| `swm/api/compiler.py` (`calibrated_readout` logistic) | **unsafe legacy** | reachable only via `baseline:parametric_v1` / v1 announcements | mechanisms registry + validated kernels | explicit baseline adapter ONLY | V2 numerical mechanisms graded |
| `swm/api/world_model.py` + mechanisms kernels | evaluation baseline / mechanism donors | `baseline:parametric_v1`; the 7 kernels are being re-registered as V2 mechanisms | `world_model_v2/mechanisms.py` | baseline + kernel donor | kernels ported + graded |
| `swm/engine/router.py` | **superseded by V2** (compiler) | v1 baseline routing only | `compiler.py` | baseline only | V2 compiler validated live |
| `swm/transition/*`, `swm/variables/*`, `swm/simulation/*`, `swm/worlds/*` | **dead code** | not imported by any live path | n/a | none | Stage B deletion after one release cycle |
| `swm/experimental/*` (TRIBE, behavior models) | quarantined research | never imported by engines (test-pinned) | n/a | research only | keep quarantined |

## Freeze policy

1. **No new features in V1.** Baseline-preservation and critical bug fixes only.
2. All new state, mechanisms, transitions, compilers, simulations → `swm/world_model_v2/`.
3. An unsupported V2 capability **abstains or marks itself experimental** — it never silently falls back to
   legacy machinery (facade raises on contamination).
4. Baseline runs are `product_eligible=false`, always.

## Staged cleanup plan

**Stage A (done in this commit):** logical quarantine · AST import enforcement · explicit named baseline
adapters in the facade · deprecation notes here · eligibility metadata on every run.

**Stage B (only after a real reference-world benchmark):** move retained baselines → `swm/baselines/`;
delete the dead `transition/variables/simulation/worlds` trees (prereq: one release cycle with the AST guard
proving nothing imports them); replace old public entry points with compatibility wrappers pointing at the
facade; update docs. Each deletion lists its prerequisite test in the PR that performs it.
