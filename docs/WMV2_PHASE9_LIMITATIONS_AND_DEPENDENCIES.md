# WMv2 Phase 9 — Limitations & Dependencies

*What Phase 9 does NOT yet do, what it depends on, and the exact continuation manifest. Nothing here is buried:
these are load-bearing caveats for anyone deciding whether to rely on posterior population/network inference.*

---

## 1. Four status axes (never collapsed to "complete")

| capability | implemented | executes e2e | empirically validated | production-eligible |
|---|---|---|---|---|
| Compositional segment-weight posterior | ✅ | ✅ | ✅ **real GSS** (poststrat 0.018→0.002) + synthetic recovery | ⚠️ conditional |
| Segment-conditional (correlated) traits | ✅ | ✅ | ✅ real GSS (beats independent) | ⚠️ conditional |
| Per-edge existence posterior (log-odds) | ✅ | ✅ | ✅ **real congress** link-pred 0.999 + synthetic (ECE 0.02) | ⚠️ present-only approximation |
| SBM community posterior | ✅ | ✅ | ✅ real congress party recovery 0.98 + planted recovery | ✅ for the recovery it does |
| Graph structural posterior | ✅ | ✅ | ✅ real (4-faction) + synthetic (0.998) | ✅ |
| Missing-edge posterior | ✅ | ✅ | tested (unobserved pairs stay uncertain) | ✅ |
| Multilayer execution (StateDeltas, gating, visibility) | ✅ | ✅ | ✅ real-graph ablations (0.474 vs 0.017) | ✅ for the substrate |
| LLM-driven graph discovery wired to the compiler | ❌ | — | — | ❌ **documented dependency** |
| Temporal edge evolution (≥5 transitions) | ❌ | — | — | ❌ **follow-up** |
| 2nd real graph dataset (Enron) | ❌ (loader exists) | — | — | ❌ **resumable** |
| Correlated multivariate traits via copula/latent-class | partial (segment-conditional) | ✅ | ⚠️ | ⚠️ |
| Learned latent graph representation | ❌ (declared kind) | — | — | ❌ needs trained encoder |

**Bottom line on production eligibility:** the population + network posterior subsystems are
software-complete, execute end-to-end on the universal `simulate_populations_networks` path, and are validated
on **real data (1 population dataset, 1 graph dataset) + synthetic recovery**. They are **NOT production
eligible**: LLM-driven discovery is not wired to the compiler, only one real graph dataset was used, and
temporal edge evolution is absent. Ship as **exploratory/transfer-grade** (the support-grade axis enforces
this), never as a calibrated production population/graph model.

---

## 2. Specific limitations

1. **Compiler/LLM discovery boundary (gate 5 NOT met).** The subsystems take TYPED evidence (survey counts,
   `EdgeObservation`s) as input. The LLM-driven front-end that proposes nodes/edges/relation-layers/hypotheses
   from a natural-language question + the Phase-2 bundle and emits those typed observations is NOT wired into
   `compiler.py`/`phase3_pipeline.py` this run. Consequence: the ≥100-question ×12-domain discovery evaluation
   (gate 5) was not run. The inference/execution engine it would feed is complete and tested.
2. **Present-only edge observation approximation.** `EdgeObservation` models are presence-only: seeing a record
   updates the existence posterior; the ABSENCE of an expected record is only partially modeled (the
   `absence_of_expected_interaction` / `edge_expiration` classes exist but a general missing-data/exposure
   mechanism is not wired). Consequence: under VARIABLE observation exposure the posterior is overconfident in
   the mid-range (measured ECE 0.21 in a mis-specified generator; the fully-specified both-readings engine is
   calibrated at ECE 0.02). A missing-data model (each potential observation is a Bernoulli whose non-firing is
   informative) is the fix.
3. **One real graph dataset.** Congress co-voting validated community recovery + link prediction. The Enron
   email loader (`experiments/datasets_enron.py`, streams the CMU tarball) is present but not run this session
   (heavy fetch). A 2nd real graph (gate 6) is resumable via that loader.
4. **Structural posterior is scale-dependent.** On the full 100-node congress graph it prefers 4-faction; on a
   40-node dense subgraph it prefers 1-bloc. Both are correct BIC-driven results, but the takeaway is that the
   structural posterior reflects the subgraph it is given — the discovery/pruning stage (which fixes the
   subgraph) materially affects it.
5. **Temporal edge evolution absent (gate 9).** Edges are static within a run; the ≥5 typed edge transitions
   (trust gain/loss, alliance formation/defection, edge expiration, rewiring) are not implemented.
   `EdgeObservation` carries valid-time fields but no transition mechanism consumes them yet.
6. **Correlated traits are segment-conditional, not full-covariance.** Segment↔trait correlation is captured
   (validated on GSS); within-segment trait-trait correlation via a copula / latent-class model is declared but
   not wired on the general path.
7. **Population-scale hybrid graphs.** The SBM/block model handles mesoscale structure; the explicit-critical-
   node + aggregate-population HYBRID for very large graphs is a documented representation, not built.
8. **Diffusion substrate is one concrete mechanism.** The multilayer execution demonstrates causal consumption
   via information/behavior diffusion + authority/communication gating. Other layer mechanisms (resource
   transfer, jurisdiction enforcement, reporting escalation as multi-hop routing) have typed hooks but the
   diffusion substrate is the fully-exercised one.

---

## 3. Dependencies
- **Phase 3** — `infer_compositional_posterior` / `infer_edge_posterior` (added to `phase3_posterior.py`),
  typed observation models + dependence collapse (`phase3_observation.py`), representation kinds
  (`phase3_representation.py`). Phase 9 is ONE posterior engine with Phase 3; the validated Phase-3 outcome-rate
  path is untouched and still passes.
- **Phase 2** — the evidence bundle would supply the typed observations in the wired-discovery version
  (dependence groups, source reliability, timestamps).
- **Phase 1 compiler** — would request populations/networks + return the plan diff (Part T; the request/return
  surface is the wiring gap in limitation #1).
- **Real datasets** — GSS cache (`experiments/results/exp045_gss/gss_parsed.json.gz`, committed) and the
  congress co-voting graph (`experiments/results/phase9/congress_covote_S117.json`, committed, fetched from
  voteview). Python 3.11, dependency-free `swm` core (SBM EM, Dirichlet posterior, log-odds engine hand-rolled;
  no numpy/scipy).

---

## 4. Continuation manifest (exact next steps)
1. **Wire LLM discovery → compiler** (gate 5): extend `compiler.py` to request populations/networks; a
   discovery agent proposes nodes/candidate edges/relation-layers/hypotheses from the question + Phase-2 bundle,
   emitting `EdgeObservation`s + survey observations; run the ≥100-question ×12-domain evaluation.
2. **Enron real graph** (gate 6): run `experiments/datasets_enron.py` to build the email communication graph;
   validate link prediction + community recovery; add as the 2nd real graph.
3. **Missing-data edge model** (limitation #2): make each candidate observation a Bernoulli whose non-firing
   updates the posterior; re-measure ECE on the typed edge models (target ≤ 0.1 under variable exposure).
4. **Temporal edge evolution** (gate 9): implement ≥5 typed edge transitions consuming valid-time + posterior
   parameters; show future graph changes future behavior; deterministic replay.
5. **Copula / latent-class within-segment correlation** (limitation #6) and **hybrid population-scale graph**
   (limitation #7).

## 5. Reproducibility commands
```
python -m pytest tests/test_wmv2_phase9.py -q
PYTHONPATH=. python experiments/wmv2_phase9_population_validation.py    # real GSS
PYTHONPATH=. python experiments/wmv2_phase9_network_validation.py       # real congress (cached graph)
PYTHONPATH=. python experiments/wmv2_phase9_ablations.py                # ablations + forensic trace
```
Same seed + committed caches reproduce the posterior + terminal hashes. No PR was merged.
