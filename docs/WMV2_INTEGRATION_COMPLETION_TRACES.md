# World Model V2 — Integration Completion: Traces & Direct Answers

Concrete execution traces for the one verified execution-level fix, plus direct answers to the
mandate's per-phase questions. Machine-readable backing:
`experiments/results/integration/activation.json` (per-question emission + rule-executability),
`experiments/results/integration/activation_chain_audit.json` (per-phase chain break),
`experiments/results/integration/runtime_fingerprint.json`, and
`experiments/results/integration/phase12_compatibility.json`.

## Trace 1 — Phase 10 rule executability (the fix that landed)

Question `sen_confirm` ("Will the Senate confirm the nominee before the recess?"):

1. `compile_world` emits `institutions = [senate, judiciary_committee]`, each with rules of kinds
   `voting_rule` / `confirmation_process` / `committee_vote`.
2. **Before the fix:** none of those kinds are in
   `EXECUTABLE_RULE_KINDS = {budget, capacity, deadline, decision_right, eligibility, procedure, quorum}`,
   so `materialize` drops every rule → `executable_rule_count(plan) == 0` → the RuleSystem is
   empty (ornamental). The institution is declared but cannot execute.
3. `normalize_institution_rules(plan)` rewrites `voting_rule→quorum`,
   `confirmation_process→procedure`, `committee_vote→procedure`, preserving `params` and stamping
   `_original_kind`.
4. **After the fix:** `executable_rule_count(plan) == 4`; `completeness_diagnostics(plan) == []`
   (the `institution_declared_but_no_executable_rule` flag clears).

This is wired **default-on** in the unified runtime, so any institution-bearing question now
carries executable institution rules into execution. Corpus-wide before/after counts are in
`activation.json → aggregate.phase10_executability_fix`.

## Trace 2 — completeness validator on an ornamental institution

For a plan with an institution whose only rule is `{"kind": "voting_rule"}` and no
`institution_action` operator, `completeness_diagnostics` returns **two** high-severity issues:
`institution_declared_but_no_executable_rule` and `institution_declared_but_no_operator`. This is
the deterministic Part-J check that would have caught the Phase‑10 break automatically.

## Trace 3 — runtime fingerprint / corpus demotion

`runtime_fingerprint()` → `fingerprint_hash = 8cf389ba0ec96da8` (deterministic across calls).
`corpus_status("8cf389ba0ec96da8") == "product_eligible"`;
`corpus_status(<any old hash>) == "diagnostic_only"`. The old Phase‑12 corpus/calibrator are
therefore **INCOMPATIBLE** and may not be used as product-performance evidence.

## Direct answers to the per-phase questions

**Phase 4 (actor policy).** Does an actor operator fire when a question needs strategic-actor
reasoning? Conditionally. The compiler names `production_actor_policy` for clearly-strategic
questions; the gap is questions where it never identifies a strategic actor. Emission recall vs.
the independent actor-required labels is in `activation.json`. Full execute+ablation activation to
the ≥95% gate is **not** claimed — reported as a failed gate with the continuation path in the
validation doc.

**Phase 6 (mechanism registry).** Does a registry family operator fire for mechanism-required
questions? Rarely — the compiler falls back to `generic_outcome_prior`. Failed gate; continuation:
map causal process → applicable registry family + operator.

**Phase 7 (nonlinear).** Do the nonlinear operators fire? The operators are registered but the
compiler emits neither the operator name nor a matching scheduled event, so they do not fire.
Failed gate; continuation: detect nonlinear structure → add operator + matching event.

**Phase 9 populations.** Are populations emitted and consumed? Emission is question-dependent and
there is **no causal consumer** aggregating population state into the terminal. Failed gate;
continuation: emit an aggregation operator that moves the terminal.

**Phase 9 networks.** Are relations modeled as semantic layers with consumers? Relations enter
`world.network` but distinct layers and layer-specific consumers do not yet exist. Failed gate;
continuation: typed multilayer layers + consumers.

**Phase 10 institutions.** Do declared institutions execute? **Yes, now** — rule normalization
makes their rules executable (verified 0→4 above); this is the one execution-level fix that
landed. The remaining gate (institution_action operator + institutional_action event carrying an
InstitutionRuntime, and ≥90% institution StateDelta) is reported honestly as not-yet-met.

**Phase 11 (recompilation).** Does it fire on genuine surprise? The controller runs on every
execution and fires only on real surprise, but it is **not** yet validated on an adversarial
shock/migration corpus. Reported as a failed gate; continuation: injected-shock corpus verifying
trigger → revision → migration → continuation with migration integrity ≥98%.
