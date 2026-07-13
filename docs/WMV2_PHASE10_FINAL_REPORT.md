# WMV2 Phase 10 — Final Report (brutally honest)

Phase 10 builds a **production institutional world-modeling system**: real institutions reconstructed from
verified evidence, temporally versioned, executed through the Phase-1 WorldState / StateDelta path, blocking
invalid actions and materially changing terminal outcomes. All numbers regenerate from committed artifacts
(`experiments/wmv2_phase10_report.py`); the registry is `swm/world_model_v2/institutions_v2/`.

## 1. Before / after

| Quantity | Before | After |
|---|---|---|
| Institutional representation | flat LLM-authored rule list (`institutions.py` `RuleSystem`) | **FAMILY → TEMPLATE → INSTANCE**, evidence-backed, as-of versioned |
| Institutional families | 0 (7 executable rule *kinds*) | **8** executable structural families |
| Real evidence-backed templates | 0 | **3** (US Congress Art I, Delaware board DGCL §141, SCOTUS Rule of Four) |
| Temporally versioned | no | **3/3** |
| Execute through WorldState → StateDelta | rule-validation + votes only | **3/3** (authorize → decide → StateDelta → terminal) |
| Historically replayed on real data | no | **1** (US Senate, 1626 real roll-calls, 96.3%) |
| Production-eligible | — | **1** (legislative) |
| LLM authors rules? | **yes** (the core gap) | **no** — rules come from verified evidence; the LLM cannot establish a rule |

## 2. The Part-31 answers (honest)

1. **Genuine institutional families?** 8, all executable (legislative, hierarchical approval, collective vote,
   court+appeal, agency, queue service, corporate board, moderation+appeal).
2. **Real templates reconstructed?** 3, each core-verified against the primary source this run.
3. **Temporally versioned?** 3/3 (valid_from dates; as-of filtering + amendment chain in `evidence.py`).
4. **Execute through WorldState/StateDelta?** 3/3 — `InstitutionOperator` authorizes → runs the decision
   engine → emits StateDelta → schedules future events → writes the terminal quantity.
5. **Block invalid actions?** Yes — an unauthorized actor's action is blocked and mutates nothing
   (`test_invalid_action_blocked_mutates_nothing`); advisory ≠ decision authority.
6. **Validated on real historical processes?** 1 — the US Senate legislative institution reconstructs
   **96.3%** of 1626 real VoteView roll-call outcomes by executing the correct evidence-backed thresholds.
7. **Production eligible?** 1 (legislative): verified Art I evidence + temporal versioning + executable +
   historical replay + leakage audit + compiler integration + forensic trace.
8. **Strongest?** `us_congress_legislative` — the only one with a real historical replay + ablation showing
   the institutional rules are load-bearing (+19.25%).
9. **Structural only?** The agency / queue / election / moderation FAMILIES are executable but have **no
   verified template** this run (evidence gap, preserved as a failure) → they select at Tier 3.
10. **Domain restricted?** Delaware board (DGCL §141 default; bylaws may override) and SCOTUS cert (informal
    custom) are `executable`, evidence-backed but not historically replayed.
11. **Quarantined / 12. Failed?** 3 preserved negatives (`wmv2_phase10_failures.json`): the naive-cloture
    negative, the agency/queue/election/moderation evidence gap, the SCOTUS count-not-fraction limitation.
13. **Compiler identifies institutions by causal need?** Yes — `_select_institutions` in the compiler calls
    `select_institution(process, scenario, as_of, jurisdiction)`; no `if domain == legislature`.
14. **Correct as-of rule version?** Yes — `active_rules(template, as_of)` filters; the replay's matter-type
    versioning (the nuclear option) is the load-bearing example.
15. **Phase-3 uncertainty enters institutional execution?** **Partially** — information boundaries filter what
    an actor may observe (so a Phase-3 actor view cannot condition on sealed info), and templates record the
    posterior-needed points (secret cert votes, informal agenda control). A **live Phase-3-posterior-into-
    institution run is defined by contract but not wired end-to-end this run** (documented remaining work).
16. **Phase-6 mechanisms operate inside institutional constraints?** **By contract** — the institution
    supplies the valid action set + authority + information + stage + timing; the decision engine consumes
    votes that would come from Phase-6 actor policy. In the replay, votes are the REAL recorded votes; a live
    Phase-6-actor-vote-into-institution run is remaining work.
17. **Deadlines/thresholds/resources/queues materially affect outcomes?** Thresholds: yes (counterfactuals +
    replay). Queues: yes (`queue_capacity_service` delays completion). Deadlines/resources: engines exist and
    are tested, but are not yet exercised in a full historical replay.
18. **Formal vs informal separated?** Yes — SCOTUS Rule of Four is stored as an **informal custom**, distinct
    from formal law; legislative leadership agenda control is recorded as informal, not a formal passage rule.
19. **Competing rule models preserved?** The schema (`InstitutionInstance.competing_models`, rule
    `alternatives`, `procedural_uncertainty`) exists and is populated (e.g. the cert count ambiguity, the
    2/3-of-quorum override note); multi-particle institutional branching is contract-defined, not yet run.
20. **Information boundaries enforced?** Yes (`InformationBoundary.filter_observations`; tested).
21. **Invalid actions blocked?** Yes (tested).
22. **StateDelta emitted?** Yes (tested).
23. **Later procedural events generated?** Yes — the operator schedules the next stage's event.
24. **Materially affect terminal outcomes?** Yes — same votes, different rule → different terminal
    (counterfactuals) and 96.3% real-outcome reconstruction.
25. **Temporal leakage prevented?** Yes — `leakage_audit` proves the active reconstruction excludes
    post-as-of rules and later outcomes (tested; clean on the replay).
26. **Software implemented?** Yes. **27. Executes end to end?** Yes (legislative). **28. Evidence backed?**
    Yes (3 core-verified templates). **29. Historically validated?** Yes for 1 (legislative, 96.3%).
    **30. Production ready?** 1 template is production-eligible; the SYSTEM is a real first tranche, **not** a
    complete cross-domain institutional platform.
31. **What remains before Phase 11?** (a) evidence-backed templates for the other 5 categories (agency,
    queue, court docket, election, moderation) with real dockets/processing-time data; (b) live
    Phase-3-posterior and Phase-6-actor-policy wiring into an institutional run (contracts defined); (c)
    multi-particle competing-rule-model execution; (d) deadline/resource historical replays; (e) LLM rule
    extraction from source text with the deterministic validator in the loop (the validator exists; the
    extraction front-end is not wired).

## 3. Honest institutional table

| Template | Evidence backed | Temporally versioned | Software impl | Executes | Blocks invalid | Real reconstruction | Historical replay | Production eligible | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| us_congress_legislative | ✓ (Art I §5/§7) | ✓ | ✓ | ✓ | ✓ | ✓ (96.3%) | ✓ (1626 votes) | **✓** | passage/threshold focus; committee gatekeeping is informal/posterior |
| delaware_board_default | ✓ (DGCL §141) | ✓ | ✓ | ✓ | ✓ | schema only | — | — (executable) | statutory default; real bylaws override |
| scotus_certiorari | ✓ (Rule of Four, informal) | ✓ | ✓ | ✓ | ✓ | schema only | — | — (executable) | count-of-4 not a fraction; votes secret |
| agency / queue / election / moderation (families) | — | — | ✓ (family) | ✓ (family) | ✓ | — | — | structural | **no verified template** (evidence gap, preserved) |

## 4. Verdict

- **Software implemented:** YES.
- **Executes end to end:** YES for the legislative institution (authorize → decide → StateDelta → terminal).
- **Evidence backed:** YES — 3 templates core-verified against primary sources (US Constitution, Delaware
  statute, FJC/SCOTUSblog); no fabricated legal text.
- **Historically validated:** YES for 1 (Senate replay 96.3% on 1626 real votes, leakage-clean, ablation-
  confirmed rules are load-bearing).
- **Production ready:** 1 template is genuinely production-eligible; the system is an **honest first tranche**
  of production institutional modeling — the universal architecture + engines + evidence pipeline + one
  deeply-validated institution — **not** the full multi-domain platform. Graded: **architecture + execution +
  one real historically-validated institution complete; cross-domain evidence coverage and live Phase-3/6
  institutional wiring remain.**

## 5. Reproducibility
```
PYTHONPATH=. python -m swm.world_model_v2.institutions_v2.build      # committed families.json/templates.json
PYTHONPATH=. python -m experiments.wmv2_phase10_replay              # real Senate historical replay + ablations
PYTHONPATH=. python -m experiments.wmv2_phase10_execute             # WorldState execution + counterfactuals + traces
PYTHONPATH=. python -m experiments.wmv2_phase10_report              # counts + preserved failures
PYTHONPATH=. python -m pytest tests/test_wmv2_phase10.py            # 18 acceptance tests
```
Artifact index in `WMV2_PHASE10_AUDIT_ARCHITECTURE_AND_EXECUTION.md` §Artifacts.
