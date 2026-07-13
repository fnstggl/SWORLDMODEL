# WMv2 Unified Runtime — Forensic Traces
*Full-chain traces from the ONE canonical `simulate_world` path. Each shows the phases that executed on the shared plan/world, the plan lineage, the posterior, dynamic-recompilation activity, and the terminal — so a reviewer can confirm the output is the integrated simulation, not a wrapper around separate predictors. Machine-readable: `experiments/results/unified/traces.json`.*

## econ — Will the US Federal Reserve cut interest rates at its September 2024 meeting?
- as_of **2024-09-10**, horizon **2024-09-19**; status **completed_with_degradation**, support **exploratory**
- plan lineage: **2** plan version(s), **0** recompilation trace(s); posterior consumed **True**; evidence bundle `3e80810fea2a`
- **active-component manifest**:
    - `phase1_compiler`: EXECUTED — always required
    - `phase2_evidence`: EXECUTED — 11 as-of claims
    - `phase3_posterior`: EXECUTED — 8 eff obs; prior 0.410→post 0.373
    - `phase4_actor_policy`: EXECUTED — operator in plan
    - `phase6_registry`: omitted — no operator of this phase selected by the compiler
    - `phase7_nonlinear`: omitted — no operator of this phase selected by the compiler
    - `phase8_persistence`: EXECUTED — no prior history — broad-prior rollout
    - `phase9_populations`: omitted — compiler declared populations but not as PopulationSpec — not yet inst
    - `phase9_networks`: omitted — no multilayer network declared by the compiler
    - `phase10_institutions`: omitted — no operator of this phase selected by the compiler
    - `phase11_recompilation`: EXECUTED — 0 recompile(s) over 11 obs; eligible=11
- **terminal raw P(yes) = 0.3788**
- old Phase-12 calibrator: **INCOMPATIBLE** (unified runtime changed the distribution)
- limitations: degraded: support grade exploratory; 1 fallback mechanism(s) used
- latency 48.48s

## geopolitics — Will Bashar al-Assad's government fall in Syria in 2024?
- as_of **2024-11-25**, horizon **2024-12-31**; status **completed_with_degradation**, support **exploratory**
- plan lineage: **2** plan version(s), **0** recompilation trace(s); posterior consumed **True**; evidence bundle `0da387227ccb`
- **active-component manifest**:
    - `phase1_compiler`: EXECUTED — always required
    - `phase2_evidence`: EXECUTED — 10 as-of claims
    - `phase3_posterior`: EXECUTED — 8 eff obs; prior 0.302→post 0.244
    - `phase4_actor_policy`: omitted — no operator of this phase selected by the compiler
    - `phase6_registry`: omitted — no operator of this phase selected by the compiler
    - `phase7_nonlinear`: omitted — no operator of this phase selected by the compiler
    - `phase8_persistence`: EXECUTED — no prior history — broad-prior rollout
    - `phase9_populations`: omitted — compiler declared populations but not as PopulationSpec — not yet inst
    - `phase9_networks`: omitted — no multilayer network declared by the compiler
    - `phase10_institutions`: omitted — no operator of this phase selected by the compiler
    - `phase11_recompilation`: EXECUTED — 0 recompile(s) over 10 obs; eligible=10
- **terminal raw P(yes) = 0.3**
- old Phase-12 calibrator: **INCOMPATIBLE** (unified runtime changed the distribution)
- limitations: degraded: support grade exploratory; 1 fallback mechanism(s) used
- latency 55.459s

## finance — Will Nvidia announce a stock split in 2024?
- as_of **2024-05-01**, horizon **2024-06-30**; status **completed_with_degradation**, support **exploratory**
- plan lineage: **2** plan version(s), **0** recompilation trace(s); posterior consumed **True**; evidence bundle `7afbb13e9f5f`
- **active-component manifest**:
    - `phase1_compiler`: EXECUTED — always required
    - `phase2_evidence`: EXECUTED — 8 as-of claims
    - `phase3_posterior`: EXECUTED — 5 eff obs; prior 0.590→post 0.629
    - `phase4_actor_policy`: EXECUTED — operator in plan
    - `phase6_registry`: omitted — no operator of this phase selected by the compiler
    - `phase7_nonlinear`: omitted — no operator of this phase selected by the compiler
    - `phase8_persistence`: EXECUTED — no prior history — broad-prior rollout
    - `phase9_populations`: omitted — compiler declared populations but not as PopulationSpec — not yet inst
    - `phase9_networks`: omitted — no multilayer network declared by the compiler
    - `phase10_institutions`: omitted — no operator of this phase selected by the compiler
    - `phase11_recompilation`: EXECUTED — 0 recompile(s) over 8 obs; eligible=8
- **terminal raw P(yes) = 0.5493**
- old Phase-12 calibrator: **INCOMPATIBLE** (unified runtime changed the distribution)
- limitations: omitted (negligible-sensitivity): employee_stock_plan_details; degraded: support grade exploratory; 1 fallback mechanism(s) used
- latency 43.885s

## sports — Will India win the 2024 ICC Men's T20 Cricket World Cup?
- as_of **2024-06-20**, horizon **2024-06-29**; status **completed_with_degradation**, support **exploratory**
- plan lineage: **2** plan version(s), **0** recompilation trace(s); posterior consumed **True**; evidence bundle `591e5f4632fc`
- **active-component manifest**:
    - `phase1_compiler`: EXECUTED — always required
    - `phase2_evidence`: EXECUTED — 16 as-of claims
    - `phase3_posterior`: EXECUTED — 8 eff obs; prior 0.500→post 0.472
    - `phase4_actor_policy`: omitted — no operator of this phase selected by the compiler
    - `phase6_registry`: omitted — no operator of this phase selected by the compiler
    - `phase7_nonlinear`: omitted — no operator of this phase selected by the compiler
    - `phase8_persistence`: EXECUTED — no prior history — broad-prior rollout
    - `phase9_populations`: omitted — compiler declared populations but not as PopulationSpec — not yet inst
    - `phase9_networks`: omitted — no multilayer network declared by the compiler
    - `phase10_institutions`: omitted — no operator of this phase selected by the compiler
    - `phase11_recompilation`: EXECUTED — 0 recompile(s) over 12 obs; eligible=12
- **terminal raw P(yes) = 0.475**
- old Phase-12 calibrator: **INCOMPATIBLE** (unified runtime changed the distribution)
- limitations: degraded: support grade exploratory; 1 fallback mechanism(s) used
- latency 54.856s
