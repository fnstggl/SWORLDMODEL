"""The Lean V2 orchestrator — reached ONLY through
`unified_runtime.simulate_world(..., execution_profile="lean_v2")`.

Stage order (each checkpointed; a failed idempotent stage retries once; nothing relaunches):

    evidence_preparation → blueprint (ONE call, cross-run cached) → blueprint_validation
    (deterministic) → targeted_repair (≤1 call, only on failures) → actor_authority_slicing
    → consequence_template_construction (deterministic) → answerability_preflight
    (three-valued; unanswerable STOPS before any actor call, with the best defensible
    labeled forecast and the exact blocking gap) → weighted causal waves (the engine)
    → conditional challenger (deterministic triggers; localized fork shares the decision
    cache so unchanged history costs zero calls) → terminal projection → forecast recovery
    (availability ≠ grounding; unresolved mass disclosed, never renormalized away).

The ConsumerComputeBudget gates every external call at the single gateway; exhaustion
finalizes with completed state + labels + skipped-work disclosure — never 0.5, never a
relaunch."""
from __future__ import annotations

import time as _time

from swm.world_model_v2.forecast_recovery import attach_recovery, recover_forecast
from swm.world_model_v2.lean_decision_cache import DecisionEquivalenceCache
from swm.world_model_v2.lean_v2 import LEAN_V2_VERSION
from swm.world_model_v2.lean_v2.blueprint import (compile_blueprint, norm, repair_blueprint,
                                                  validate_blueprint)
from swm.world_model_v2.lean_v2.budget import BudgetLedger, ConsumerComputeBudget
from swm.world_model_v2.lean_v2.challenger import (DISAGREEMENT_SPREAD,
                                                   build_challenger_blueprint,
                                                   decide_challenger, should_replicate)
from swm.world_model_v2.lean_v2.checkpoints import CheckpointStore
from swm.world_model_v2.lean_v2.compile_cache import CompilationCache
from swm.world_model_v2.lean_v2.consequences import (TemplateExecutor, manifest as
                                                     consequences_manifest,
                                                     precompile_templates)
from swm.world_model_v2.lean_v2.calibration import (ForecastReliabilityCombiner,
                                                    ForecastReliabilityFeatures,
                                                    GroundedPriorForecast,
                                                    SimulationConditionalForecast,
                                                    load_action_reliability)
from swm.world_model_v2.lean_v2.engine import EngineConfig, WaveEngine
from swm.world_model_v2.lean_v2.gateway import BudgetExhausted, LLMGateway
from swm.world_model_v2.lean_v2.grounding import gather_grounding
from swm.world_model_v2.lean_v2.mechanisms import recover_mechanism
from swm.world_model_v2.lean_v2.obligations import build_obligations
from swm.world_model_v2.lean_v2.preflight import run_preflight
from swm.world_model_v2.lean_v2.readiness import (CANONICAL_TERMINAL_KEY,
                                                  canonicalize_terminal_writers,
                                                  simulation_readiness)
from swm.world_model_v2.lean_v2.slice import backward_slice
from swm.world_model_v2.lean_v2.state_completeness import (ensure_actor_state_completeness,
                                                           reversal_focused_search)
from swm.world_model_v2.lean_v2.states import (ActorStatePosteriorEngine,
                                               generate_actor_states, validate_hypothesis_set)
from swm.world_model_v2.lean_v2.traces import write_traces
from swm.world_model_v2.lean_v2.unresolved import (UnresolvedLedger,
                                                   classify_unresolved_reason)
from swm.world_model_v2.lean_v2.worlds import WeightedBranchCoalescer
from swm.world_model_v2.result import SimulationResult


def simulate_world_lean_v2(question: str, *, as_of: str, horizon: str = "",
                           intervention: str = "", evidence: str = "", user_context=None,
                           prior_checkpoint=None, compute_budget=None, seed: int = 0,
                           llm=None, execution_policy: dict = None,
                           trace_level: str = "standard", config=None,
                           prebuilt_bundle=None) -> SimulationResult:
    t0 = _time.time()
    policy = dict(execution_policy or {})
    v2 = dict(policy.get("lean_v2") or {})
    budget = ConsumerComputeBudget(**dict(v2.get("budget") or {}))
    ledger = BudgetLedger(budget)
    gateway = LLMGateway(strong_llm=llm, light_llm=v2.get("light_llm"), ledger=ledger,
                         backend_fingerprint=str(v2.get("backend_fingerprint") or ""))
    cache = CompilationCache(persist=bool(v2.get("persistent_cache", True)),
                             persistent_dir=v2.get("persistent_cache_dir"))
    ckpt = CheckpointStore()
    prov: dict = {"runtime": LEAN_V2_VERSION, "structural_mode": "ensemble"}
    lean_v2_prov: dict = {}

    def _finish(res: SimulationResult) -> SimulationResult:
        lean_v2_prov.update({
            "budget": ledger.manifest(), "gateway": gateway.manifest(),
            "compile_cache": cache.manifest(), "checkpoints": ckpt.manifest()})
        res.provenance = {**prov, **(res.provenance or {}), "lean_v2": lean_v2_prov}
        res.latency_s = round(_time.time() - t0, 3)
        return res

    def _failed(taxonomy: str, msg: str) -> SimulationResult:
        return _finish(SimulationResult(
            question=question, simulation_status="execution_failed",
            failure_taxonomy=taxonomy, limitations=[msg[:240]]))

    if llm is None:
        return _failed("unavailable_service", "lean_v2 requires an LLM backend: real actor "
                                              "decisions are never replaced with numerics")

    # ---------------- 1. evidence preparation (sealed replay honored) -------------------
    def _prep_evidence() -> str:
        if prebuilt_bundle is not None:
            rows = []
            claims = getattr(prebuilt_bundle, "claims", None) or []
            for c in claims:
                if isinstance(c, dict):
                    rows.append(f"- {str(c.get('supporting_span') or c.get('text', ''))[:260]}")
            if rows:
                return "\n".join(rows)[:2600]
        return str(evidence or "")[:2600]
    evidence_text = ckpt.run_stage("evidence_preparation", _prep_evidence)

    # ---------------- 2-4. blueprint → validators → ≤1 targeted repair ------------------
    try:
        bp, bp_cached = ckpt.run_stage("blueprint_creation", lambda: compile_blueprint(
            question=question, as_of=as_of, horizon=horizon, evidence_text=evidence_text,
            user_context=str(user_context or ""), intervention=intervention,
            gateway=gateway, cache=cache))
        ledger.record_structural_model()
    except BudgetExhausted as e:
        return _failed("timeout", f"budget refused the blueprint call ({e.dimension}) — "
                                  f"nothing defensible exists yet")
    except Exception as e:  # noqa: BLE001
        return _failed("invalid_execution_plan", f"blueprint compile failed: {e}")

    fails = ckpt.run_stage("blueprint_validation", lambda: validate_blueprint(
        bp, as_of=as_of, horizon=horizon, evidence_text=evidence_text))
    repair_record = {"attempted": False}
    if fails:
        try:
            bp2, fails2, repair_record = ckpt.run_stage(
                "targeted_repair", lambda: repair_blueprint(
                    bp, fails, as_of=as_of, horizon=horizon,
                    evidence_text=evidence_text, gateway=gateway, cache=cache))
            bp, fails = bp2, fails2
        except BudgetExhausted:
            pass
    lean_v2_prov["blueprint"] = {"hash": bp.raw_response_hash, "from_cache": bp_cached,
                                 "causal_thesis": bp.causal_thesis[:300],
                                 "n_actors": len(bp.actors),
                                 "n_action_templates": len(bp.action_templates),
                                 "validation_failures_final": fails[:10],
                                 "repair": repair_record,
                                 "dropped_grounded_rates":
                                     bp.validation.get("dropped_grounded_rates", [])}

    # ---------------- 4b. terminal writer canonicalization (the mapping-bug fix at the
    # source): every predicate-terminal writer SETS the one canonical key the terminal
    # reads, BEFORE templates compile — a completed simulation can never again be discarded
    # because the writer and the evaluator disagreed on a key or label spelling
    lean_v2_prov["terminal_canonicalization"] = ckpt.run_stage(
        "terminal_canonicalization", lambda: canonicalize_terminal_writers(bp))

    # ---------------- 5. slice + 6. consequence templates -------------------------------
    sl = ckpt.run_stage("actor_authority_slicing", lambda: backward_slice(bp))
    executor = TemplateExecutor(ckpt.run_stage(
        "consequence_template_construction", lambda: precompile_templates(bp, cache)), bp)
    lean_v2_prov["slice"] = sl.manifest()

    # ---------------- 6b. GROUNDING: counted reference classes + shared conditions -------
    kept_actor_dicts = [bp.actor_by_id(aid) for aid in sl.kept_actors
                        if bp.actor_by_id(aid)]
    grounding = ckpt.run_stage("grounding", lambda: gather_grounding(
        question=question, as_of=as_of, evidence_text=evidence_text,
        actor_ids=sl.kept_actors, gateway=gateway, cache=cache))
    lean_v2_prov["grounding"] = grounding
    lean_v2_prov["weight_invariant"] = _assert_no_label_weights(grounding)

    # ---------------- 6c. STATE GENERATION (no numbers) + counted posteriors ------------
    shared_cids = list((grounding.get("shared_world_conditions") or {}).keys())
    states_by_actor, state_rejections, state_meta = ckpt.run_stage(
        "state_generation", lambda: generate_actor_states(
            question=question, as_of=as_of, evidence_text=evidence_text,
            actors=kept_actor_dicts, shared_condition_ids=shared_cids,
            gateway=gateway, cache=cache))
    hard_evidence_ids = _hard_evidence_ids(prebuilt_bundle)

    # ---------------- 6d. ACTOR-STATE COMPLETENESS (the hard invariant) -----------------
    # "the actor's true private state is unknown" is the REASON to simulate multiple
    # weighted worlds, never a reason to stop one: every consequential actor must exit this
    # stage with a valid, non-empty, weighted state set (4-attempt recovery ladder), plus
    # ONE batched probe for omitted reversal-capable states. An empty set can NEVER reach
    # rollout; an inadequate CACHED artifact is invalidated so no future run replays it.
    consequential = _consequential_actors(bp, sl.kept_actors)
    completed, completeness = ckpt.run_stage(
        "actor_state_completeness", lambda: ensure_actor_state_completeness(
            bp=bp, consequential_actors=consequential, states_by_actor=states_by_actor,
            grounding=grounding, evidence_text=evidence_text,
            hard_evidence_ids=hard_evidence_ids, gateway=gateway, budget_ledger=ledger))
    completeness.reversal_search = reversal_focused_search(
        bp=bp, completed=completed, evidence_text=evidence_text, gateway=gateway,
        budget_ledger=ledger)
    if (state_meta or {}).get("from_cache") and completeness.empty_sets_detected:
        if cache.invalidate("actor_state_generation", state_meta["deps"]):
            completeness.cache_invalidations += 1
    states_by_actor = {**states_by_actor, **completed}
    actor_residuals = {aid: rec.residual_r for aid, rec in completeness.actors.items()}
    lean_v2_prov["state_recovery"] = completeness.manifest()

    posterior_engine = ActorStatePosteriorEngine(grounding)
    validated, grounded_weights, gw_by_combo, shared_combos, state_prov = \
        _build_grounded_weights(bp, states_by_actor, posterior_engine, grounding,
                                hard_evidence_ids)
    lean_v2_prov["actor_states"] = {aid: [h.as_dict() for h in hs]
                                    for aid, hs in validated.items()}
    lean_v2_prov["state_posteriors"] = state_prov
    lean_v2_prov["state_generation_numeric_rejections"] = state_rejections
    lean_v2_prov["shared_condition_worlds"] = [
        {"combo": c, "weight": w} for c, w in shared_combos]
    lean_v2_prov["actor_residual_bounds"] = {
        aid: (gw.get("unknown") if gw else None)
        for aid, gw in grounded_weights.items()}
    obligations = ckpt.run_stage("mandatory_obligations",
                                 lambda: build_obligations(bp, grounding))
    lean_v2_prov["obligations"] = {k: o.as_dict() for k, o in obligations.items()}
    map_combo = shared_combos[0][0] if shared_combos else {}

    # ---------------- 6e-bis. DELIBERATIVE INSTITUTION-VOTE RESOLUTION (D7 + D8 + D14) ------
    # An institution vote is resolved by a deliberative sub-simulation over the FAITHFUL roster
    # (D7) with grounded initial positions (D8) and grounded convergence (D14), tallied
    # seat-weighted against the REAL threshold — never by counting independent per-member votes
    # against a rescaled threshold. This becomes the simulation-conditional forecast for
    # institution_vote terminals; the wave engine still runs for dynamics and audit.
    inst_terminal = None
    if str(bp.terminal.get("kind") or "") == "institution_vote":
        inst_terminal = ckpt.run_stage(
            "deliberative_institution_vote",
            lambda: _resolve_institution_vote_terminal(
                bp, question, evidence_text, grounding, validated, gw_by_combo, shared_combos))
        lean_v2_prov["institution_terminal"] = inst_terminal

    # ---------------- 6e. MISSING-MECHANISM RECOVERY (predicate terminals) --------------
    # a numeric-threshold terminal with no mechanical writer gets the 5-attempt bridge
    # ladder BEFORE rollout — `missing_mechanism = 1.0` is never an accepted end state
    # while a grounded bounded mechanism can be built (and when none can, the failure
    # carries its proof). An event_occurs terminal that ALREADY has a canonical-key writer
    # needs no bridge (occurrence via the writer; non-occurrence resolves NO by horizon) —
    # a recovered mechanism must never displace that complete pathway.
    tk = str(bp.terminal.get("kind") or "")
    has_canonical_writer = any(
        e.get("kind") == "set_state"
        and (e.get("params") or {}).get("key") == CANONICAL_TERMINAL_KEY
        for t in bp.action_templates for e in (t.get("effects") or []))
    mechanism, mech_manifest = None, {"attempted": False,
                                      "why": "terminal pathway already complete"}
    if tk != "institution_vote" and (tk == "state_predicate" or not has_canonical_writer):
        mechanism, mech_manifest = ckpt.run_stage(
            "mechanism_recovery", lambda: recover_mechanism(
                bp, cause="terminal_pathway_construction", evidence_text=evidence_text,
                as_of=as_of, gateway=gateway, cache=cache, budget_ledger=ledger,
                world_condition_keys=shared_cids))
    lean_v2_prov["mechanism_recovery"] = mech_manifest

    # ---------------- 7. three-valued answerability preflight ---------------------------
    pre = ckpt.run_stage("answerability_preflight", lambda: run_preflight(
        bp, as_of=as_of, horizon=horizon, consequence_templates=executor.templates))
    lean_v2_prov["preflight"] = pre.as_dict()

    # ---------------- 7b. SIMULATION READINESS GATE (proof before rollout) --------------
    # rollout may not begin until the compiled world is PROVEN able to reach its measured
    # outcome: weighted states for every consequential actor, triggers + feasible actions,
    # institutional deadlines, a terminal writer in the question's units, and the synthetic
    # YES/NO round-trip through the SAME evaluator + recovery path the live run uses.
    # `repairable` triggers targeted repair and ONE re-check; `not_ready` stops LOUDLY with
    # the exact structural reason — never a silent prior-only forecast.
    def _readiness():
        return simulation_readiness(
            bp=bp, consequential_actors=consequential, completed_states=completed,
            grounded_weights=grounded_weights, obligations=obligations,
            executor=executor, mechanism=mechanism, shared_combos=shared_combos)
    ready = ckpt.run_stage("simulation_readiness", _readiness)
    if ready.verdict == "repairable":
        applied = _apply_readiness_repairs(bp, ready, obligations=obligations,
                                           consequential=consequential)
        re_ready = _readiness()
        re_ready.repairs_applied = applied
        ready = re_ready
    # a terminal round-trip that STAYS broken after repair is a HARD stop: a simulation whose
    # completed distribution cannot map to the measured answer must never silently proceed
    # and fall back to the prior (the BoJ/visionOS class). The exact failing checks are
    # reported; rollout is refused.
    if not (ready.round_trip or {}).get("ok"):
        ready.verdict = "not_ready"
    lean_v2_prov["readiness"] = ready.as_dict()
    if ready.verdict == "not_ready":
        return _finish(_stopped_not_ready(question, bp, ready, pre,
                                          structural_fails=fails, grounding=grounding))
    if pre.verdict == "unanswerable" and ready.verdict != "ready":
        return _finish(_stopped_before_rollout(question, bp, pre, evidence_text,
                                               structural_fails=fails, grounding=grounding))

    # ---------------- 8. weighted causal waves (the primary run) ------------------------
    engine_cfg = EngineConfig(
        max_workers=int(v2.get("max_workers") or 6),
        behavioral_replicate_index=0)
    decision_cache = DecisionEquivalenceCache()
    coalescer = WeightedBranchCoalescer(max_nodes=budget.max_weighted_nodes)
    primary = WaveEngine(bp=bp, kept_actors=sl.kept_actors, promotable=sl.promotable,
                         executor=executor, gateway=gateway, budget_ledger=ledger,
                         compile_cache=cache, config=engine_cfg, coalescer=coalescer,
                         decision_cache=decision_cache, structural_model="primary",
                         grounded_weights=grounded_weights, obligations=obligations,
                         shared_world=map_combo, shared_world_combos=shared_combos,
                         grounded_weights_by_combo=gw_by_combo,
                         mechanism=mechanism, actor_residuals=actor_residuals,
                         consequential_actors=consequential)
    try:
        eng = ckpt.run_stage("causal_waves_primary",
                             lambda: primary.run(as_of=as_of, horizon=horizon),
                             idempotent=False)
    except BudgetExhausted as e:
        eng = primary.result
        # account the CURRENT node population honestly (nothing invented, nothing dropped)
        primary._finalize(primary._final_nodes or [])
        lean_v2_prov["budget_stop"] = {"during": "primary_waves", "dimension": e.dimension}
    lean_v2_prov["engine_primary"] = primary.manifest()

    # ---------------- 9. genuinely conditional challenger --------------------------------
    total = eng.yes_mass + eng.no_mass + eng.unresolved_mass + eng.truncated_mass
    unresolved_share = (eng.unresolved_mass + eng.truncated_mass) / total if total else 1.0
    ch_decision = decide_challenger(bp, p_mid=eng.p_mid,
                                    weight_sensitive=eng.weight_sensitive,
                                    unresolved_share=unresolved_share,
                                    evidence_text=evidence_text)
    challenger_engine = None
    if ch_decision.triggered:
        est = max(4, len((eng.decisions_manifest or {}).get("templates", [])) or 8)
        ok, why = ledger.can_afford(what="structural_challenger", est_calls=est + 1,
                                    structural_model=True)
        if not ok:
            ch_decision.skipped_reasons.append(f"budget: {why}")
        else:
            try:
                ch_bp = ckpt.run_stage("challenger_blueprint",
                                       lambda: build_challenger_blueprint(
                                           bp, ch_decision, gateway=gateway, cache=cache))
            except BudgetExhausted:
                ch_bp = None
            if ch_bp is not None:
                ledger.record_structural_model()
                # the challenger's changed variants need grounded weights too; recompute for
                # the challenger blueprint's variant set (counted classes, never labels)
                ch_states = {aid: [_variant_to_hypothesis(aid, v)
                                   for v in (ch_bp.actor_by_id(aid) or {})
                                   .get("private_state_variants", [])]
                             for aid in sl.kept_actors if ch_bp.actor_by_id(aid)}
                _v2, ch_gw, ch_gw_combo, _sc, _sp = _build_grounded_weights(
                    ch_bp, ch_states, ActorStatePosteriorEngine(grounding), grounding,
                    hard_evidence_ids, install_variants=False)
                ch_engine_obj = WaveEngine(
                    bp=ch_bp, kept_actors=sl.kept_actors, promotable=set(sl.promotable),
                    executor=TemplateExecutor(dict(executor.templates), ch_bp),
                    gateway=gateway, budget_ledger=ledger, compile_cache=cache,
                    config=engine_cfg,
                    coalescer=WeightedBranchCoalescer(max_nodes=budget.max_weighted_nodes),
                    # LOCALIZED FORK: the shared decision cache serves every decision context
                    # unchanged by the challenger's delta — zero calls until divergence
                    decision_cache=decision_cache if ch_decision.mode == "localized_fork"
                    else DecisionEquivalenceCache(),
                    structural_model="challenger",
                    grounded_weights=ch_gw, obligations=obligations,
                    shared_world=map_combo, shared_world_combos=shared_combos,
                    grounded_weights_by_combo=ch_gw_combo,
                    mechanism=mechanism, actor_residuals=actor_residuals,
                    consequential_actors=consequential)
                try:
                    challenger_engine = ckpt.run_stage(
                        "causal_waves_challenger",
                        lambda: ch_engine_obj.run(as_of=as_of, horizon=horizon),
                        idempotent=False)
                    lean_v2_prov["engine_challenger"] = ch_engine_obj.manifest()
                except BudgetExhausted as e:
                    lean_v2_prov["budget_stop"] = {"during": "challenger_waves",
                                                   "dimension": e.dimension}
    lean_v2_prov["challenger"] = ch_decision.as_dict()

    # ---------------- 10. replicate policy (recorded; no automatic reruns) ---------------
    rep_allowed, rep_reason = should_replicate(
        status="completed" if unresolved_share < 0.001 else
        ("partially_resolved" if unresolved_share < 0.999 else "unresolved"),
        p_mid=eng.p_mid, unresolved_share=unresolved_share,
        requested_behavioral_replicates=int(v2.get("behavioral_replicates") or 1),
        terminal_mechanism_failed=bool(eng.unresolved_reasons.get(
            "state_predicate_not_mechanically_bound")))
    lean_v2_prov["replicate_policy"] = {"allowed": rep_allowed, "reason": rep_reason,
                                        "ran": False}

    # ---------------- 10b. SIMULATION COMPLETION AUDIT (explicit optimization target) ----
    # after the run: how much world mass reached the exact terminal outcome, and WHY any
    # rest did not. Repairable causes get one bounded post-run repair (a mechanism recovered
    # against the now-known blocking cause re-evaluates ONLY the unresolved worlds); the
    # acceptance targets are recorded pass/fail with reasons — never silently waved through.
    audit = _completion_audit(eng, mechanism=mechanism, mech_manifest=mech_manifest)
    if audit["repair_recommended"] == "mechanism_recovery" and mechanism is None:
        mech2, mm2 = recover_mechanism(
            bp, cause=audit["dominant_unresolved_cause"], evidence_text=evidence_text,
            as_of=as_of, gateway=gateway, cache=cache, budget_ledger=ledger,
            world_condition_keys=shared_cids)
        lean_v2_prov["mechanism_recovery_post_run"] = mm2
        if mech2 is not None:
            mechanism = mech2
            eng = primary.resume_with_mechanism(mech2)
            lean_v2_prov["engine_primary"] = primary.manifest()
            total = eng.yes_mass + eng.no_mass + eng.unresolved_mass + eng.truncated_mass
            unresolved_share = (eng.unresolved_mass + eng.truncated_mass) / total \
                if total else 1.0
            audit = _completion_audit(eng, mechanism=mechanism, mech_manifest=mm2)
    lean_v2_prov["completion_audit"] = audit

    # ---------------- 11. terminal projection + calibrated combination -------------------
    res = ckpt.run_stage("terminal_projection", lambda: _assemble_result(
        question, bp, eng, challenger_engine, unresolved_share, evidence_text,
        grounding=grounding, obligations=obligations, lean_v2_prov=lean_v2_prov,
        inst_terminal=inst_terminal))
    lean_v2_prov["consequences"] = consequences_manifest(executor)
    finished = _finish(res)
    # ---------------- 12. full traces + human report ------------------------------------
    try:
        loc = write_traces(v2.get("qid") or "adhoc", gateway_rows=gateway.rows,
                           lean_v2_prov=lean_v2_prov, result_dict=finished.__dict__)
        lean_v2_prov["trace_dir"] = loc
    except Exception as e:  # noqa: BLE001 — a trace-writing failure never costs the forecast
        lean_v2_prov["trace_error"] = f"{type(e).__name__}: {e}"[:160]
    return finished


# ---------------------------------------------------------------------- grounding helpers
def _assert_no_label_weights(grounding: dict) -> dict:
    """The §1 invariant: fail loudly if any world/state weight was derived solely from a
    qualitative label. Grounding weights come only from counted reference classes; this scans
    the grounding output for the forbidden numeric-from-label pattern."""
    violations = list(grounding.get("numeric_rejections") or [])
    # every shared/actor rate must trace to a counted denominator, not a label
    for cid, sc in (grounding.get("shared_world_conditions") or {}).items():
        prov = ((sc.get("table") or {}).get("provenance") or {})
        if prov.get("rate_mean") is not None and (prov.get("denominator") or 0) == 0:
            violations.append({"where": f"shared:{cid}",
                               "why": "rate present with zero counted cases"})
    ok = not violations
    assert ok, f"label-derived weight invariant violated: {violations[:3]}"
    return {"ok": True, "label_derived_weights_found": 0,
            "rule": "every weight traces to a counted reference class or is explicit unknown "
                    "mass; no qualitative label is mapped to a number"}


def _hard_evidence_ids(bundle) -> set:
    """Claim ids that constitute HARD evidence (can eliminate a state). In the sealed-replay
    bundle every benchmark-background claim is span-verified public fact → eligible as hard."""
    ids = set()
    for c in (getattr(bundle, "claims", None) or []):
        if isinstance(c, dict) and c.get("span_verified"):
            ids.add(str(c.get("claim_id")))
    return ids


def _consequential_actors(bp, kept: list) -> list:
    """The actors whose private state and decisions the terminal outcome actually runs
    through: terminal-institution members, triggered deciders, terminal writers. These are
    the actors the completeness invariant is HARD for."""
    term = bp.terminal
    inst_members = set()
    if term.get("kind") == "institution_vote":
        inst = bp.institution_by_id(term.get("institution_id")) or {}
        inst_members = set(inst.get("members") or [])
    triggered = {d.get("actor_id") for d in bp.decision_triggers}
    writers = {aid for t in bp.action_templates
               if t.get("writes_terminal") or any(e.get("kind") in ("record_vote",
                                                                    "set_state")
                                                  for e in (t.get("effects") or []))
               for aid in (t.get("actor_ids") or [])}
    out = [aid for aid in kept if aid in inst_members or aid in triggered
           or aid in writers]
    return out or list(kept)


def _apply_readiness_repairs(bp, ready, *, obligations: dict, consequential: list) -> list:
    """Targeted, deterministic repairs for a `repairable` readiness verdict — writer-key
    canonicalization, deadlines derived from the terminal, missing triggers scheduled.
    Anything deeper stays `not_ready` and stops loudly."""
    applied = []
    needed = {r.get("repair") for r in ready.repairs_needed}
    if {"canonicalize_writers_or_mechanism_recovery", "terminal_mapping_repair"} & needed:
        rec = canonicalize_terminal_writers(bp)
        applied.append({"repair": "canonicalize_terminal_writers", "record": rec})
    if "derive_deadline_from_terminal" in needed:
        ev = str(bp.terminal.get("evaluation_day") or "")[:10]
        for ob in obligations.values():
            if not ob.deadline_day and ev:
                ob.deadline_day = ev
                applied.append({"repair": "derive_deadline_from_terminal",
                                "institution": ob.institution_id, "deadline": ev})
    if "schedule_mandatory_trigger" in needed:
        ev = str(bp.terminal.get("evaluation_day") or "")[:10]
        triggered = {d.get("actor_id") for d in bp.decision_triggers}
        obligated = {m for ob in obligations.values()
                     for m in (ob.required_participants or [])}
        for aid in consequential:
            if aid not in triggered and aid not in obligated and ev:
                bp.decision_triggers.append(
                    {"actor_id": aid, "when_day": ev, "etype": "deadline",
                     "situation": "the decision deadline has arrived — act now"})
                applied.append({"repair": "schedule_mandatory_trigger", "actor": aid,
                                "day": ev})
    return applied


def _completion_audit(eng, *, mechanism, mech_manifest) -> dict:
    """§completion: the run-level accounting of HOW MUCH world mass reached the exact
    terminal outcome and why the rest did not — with the acceptance targets evaluated
    pass/fail. This is the simulation's explicit optimization target, not a side effect."""
    led = _build_unresolved_ledger(eng)
    total = eng.yes_mass + eng.no_mass + eng.unresolved_mass + eng.truncated_mass
    resolved = eng.yes_mass + eng.no_mass
    resolved_share = resolved / total if total else 0.0
    by_cause = dict(led.by_cause)
    unknown_state_mass = by_cause.get("unresolved_unknown_state", 0.0)
    missing_mech_mass = by_cause.get("unresolved_missing_mechanism", 0.0)
    provider_mass = by_cause.get("unresolved_provider_failure", 0.0)
    dominant = max(by_cause.items(), key=lambda kv: kv[1])[0] if by_cause else ""
    proven_unavoidable = bool((mech_manifest or {}).get("failure_proof")) \
        and missing_mech_mass > 0
    repair = ""
    if missing_mech_mass > 0.02 and mechanism is None and not proven_unavoidable:
        repair = "mechanism_recovery"
    acceptance = {
        "terminal_unknown_state_mass": round(unknown_state_mass, 6),
        "terminal_unknown_state_ok": unknown_state_mass <= 1e-9,
        "terminal_missing_mechanism_mass": round(missing_mech_mass, 6),
        "terminal_missing_mechanism_ok": missing_mech_mass <= 1e-9 or proven_unavoidable,
        "provider_failure_mass": round(provider_mass, 6),
        "provider_failure_ok": provider_mass <= 1e-9,
        "resolved_share": round(resolved_share, 4),
        "resolved_target_met": resolved_share >= 0.8,
        "resolved_hard_floor_met": resolved_share >= 0.6 or proven_unavoidable,
    }
    acceptance["all_ok"] = all(acceptance[k] for k in
                               ("terminal_unknown_state_ok",
                                "terminal_missing_mechanism_ok", "provider_failure_ok",
                                "resolved_target_met"))
    return {"resolved_mass": round(resolved, 6), "total_mass": round(total, 6),
            "unresolved_mass_by_cause": by_cause,
            "dominant_unresolved_cause": dominant,
            "proven_unavoidable": proven_unavoidable,
            "repair_recommended": repair,
            "engine_completion_rounds": eng.completion_audit,
            "acceptance": acceptance}


def _stopped_not_ready(question, bp, ready, pre, *, structural_fails, grounding) -> \
        SimulationResult:
    """The readiness gate refused rollout: a LOUD structural stop carrying the exact failing
    checks — never a silent prior-only forecast, never an invented probability. The counted
    grounded prior (when one exists) is reported honestly labeled."""
    gp = _outcome_prior(grounding or {})
    rec = recover_forecast(distribution={}, options=bp.resolution.get("options") or None,
                           unresolved_mass=1.0, prior_mean=gp.p,
                           prior_source_class="reference_class" if gp.p is not None else "")
    failing = [c for c in ready.checks if not c.get("ok")]
    blocking = "; ".join(f"{c['check']}: {c['note']}" for c in failing[:5]) \
        or "readiness verdict not_ready"
    res = SimulationResult(
        question=question, simulation_status="under_modeled",
        support_grade="exploratory",
        under_modeled_subtypes=["under_modeled_nonhuman_mechanism"],
        under_modeled_components=[{"component": c["check"], "kind": "readiness",
                                   "why": c["note"], "sensitivity": "decisive"}
                                  for c in failing[:6]],
        limitations=[
            f"SIMULATION READINESS GATE STOP (not_ready): {blocking}",
            "rollout was refused BEFORE any actor call — the world cannot yet be proven to "
            "reach its measured outcome; the failing checks above are the exact repairs "
            "needed (this is a loud structural failure, never a silent prior-only forecast)"])
    if structural_fails:
        res.limitations.append("unrepaired validator failures: "
                               + "; ".join(f["what"][:80] for f in structural_fails[:4]))
    if rec is not None:
        attach_recovery(res, rec, override_probability=True)
    return res


def _variant_to_hypothesis(actor_id: str, v: dict):
    from swm.world_model_v2.lean_v2.states import ActorStateHypothesis
    st = v.get("state") or {}
    return ActorStateHypothesis(
        actor_id=actor_id, state_id=str(v.get("variant_id") or ""),
        beliefs=list(st.get("beliefs") or []), goals=list(st.get("goals") or []),
        stances=list(st.get("stances") or []), pressures=str(st.get("pressures") or ""),
        relationships=dict(st.get("relationships") or {}),
        action_if_state=str(v.get("action_if_state") or ""),
        reversal_capable=bool(v.get("reversal_capable")),
        aligned_condition=dict(v.get("aligned_condition") or {}))


def _shared_combos(posterior_engine, cap: int = 6) -> list:
    """Cartesian product of the shared-condition states, weighted by their counted rates,
    capped to the highest-weight combos (mass renormalized + disclosed)."""
    import itertools
    axes = posterior_engine.shared_condition_worlds()   # [(cid, {state: w}, prov, affects)]
    if not axes:
        return [({}, 1.0)]
    combos = []
    for point in itertools.product(*[[(cid, s, w) for s, w in weights.items()]
                                     for cid, weights, _p, _a in axes]):
        combo = {cid: s for cid, s, _w in point}
        weight = 1.0
        for _cid, _s, w in point:
            weight *= w
        combos.append((combo, weight))
    combos.sort(key=lambda cw: -cw[1])
    combos = combos[:cap]
    z = sum(w for _c, w in combos) or 1.0
    return [(c, round(w / z, 6)) for c, w in combos]


def _build_grounded_weights(bp, states_by_actor, posterior_engine, grounding,
                            hard_evidence_ids, *, install_variants: bool = True):
    """Validate hypotheses, install them as blueprint variants, and compute COUNTED per-actor
    weights for the MAP combo and every shared-world combo. Returns
    (validated, grounded_weights, gw_by_combo, shared_combos, provenance)."""
    validated: dict = {}
    for aid, hyps in states_by_actor.items():
        inst_rules = [i for i in bp.institutions if aid in (i.get("members") or [])]
        v = validate_hypothesis_set(aid, hyps, institution_rules=inst_rules,
                                    hard_evidence_ids=hard_evidence_ids)
        validated[aid] = v["kept"]
    if install_variants:
        for a in bp.actors:
            aid = a["id"]
            if aid in validated and validated[aid]:
                a["private_state_variants"] = [h.to_variant() for h in validated[aid]]

    shared_combos = _shared_combos(posterior_engine)
    gw_by_combo: dict = {}
    provenance: dict = {}
    for combo, _w in shared_combos:
        ck = __import__("json").dumps(combo, sort_keys=True)
        table = {}
        for aid, hyps in validated.items():
            # D2: pass the actor's feasible canonical options so a counted class can only
            # weight a state whose action tendency agrees with the class's counted action
            from swm.world_model_v2.lean_v2.state_completeness import feasible_options_for
            rows, unknown, prov = posterior_engine.weight_actor_states(
                aid, hyps, shared_world=combo,
                feasible_options=feasible_options_for(bp, aid))
            table[aid] = {"mid": {r.state_id: r.mid for r in rows},
                          "rng": {r.state_id: (r.lo, r.mid, r.hi) for r in rows},
                          "unknown": unknown, "prov": prov}
            provenance.setdefault(aid, []).append({"combo": combo, "unknown": unknown,
                                                   "weights": {r.state_id: r.mid
                                                               for r in rows},
                                                   "provenance": [r.provenance for r in rows]})
        gw_by_combo[ck] = table
    map_combo = shared_combos[0][0] if shared_combos else {}
    map_ck = __import__("json").dumps(map_combo, sort_keys=True)
    grounded_weights = gw_by_combo.get(map_ck, {})
    return validated, grounded_weights, gw_by_combo, shared_combos, provenance


# ---------------------------------------------------------------------- result assembly
def _grounded_prior_inputs(bp) -> dict:
    """A grounded rate feeds the recovery ONLY when it is explicitly about the YES option
    (deterministic text match) — an unrelated evidence number never becomes a prior."""
    yes_label = str((bp.resolution.get("options") or ["YES"])[0]).lower()
    for g in bp.grounded_rates:
        q = norm(g.get("quantity"), 120).lower()
        if yes_label and yes_label in q or "yes" in q or "unanim" in q:
            vr = g.get("value_range") or []
            if len(vr) == 2:
                mid = (float(vr[0]) + float(vr[1])) / 2.0
                if 0.0 <= mid <= 1.0:
                    return {"prior_mean": round(mid, 4),
                            "prior_source_class": str(g.get("source_class")
                                                      or "reference_class"),
                            "basis": norm(g.get("basis_quote"), 160)}
    return {}


def _outcome_prior(grounding: dict) -> GroundedPriorForecast:
    """The COUNTED outcome reference class → grounded prior. Never a label, never invented."""
    oc = (grounding or {}).get("outcome_reference_class") or {}
    prov = oc.get("provenance") or {}
    p = prov.get("rate_mean")
    if p is None or (prov.get("denominator") or 0) == 0:
        return GroundedPriorForecast(p=None, source="no_counted_outcome_class", n=0)
    return GroundedPriorForecast(
        p=round(float(p), 4), source="counted_outcome_reference_class",
        n=int(prov.get("denominator") or 0),
        interval=tuple(oc.get("interval") or (0.0, 1.0)),
        provenance={"quantity": oc.get("quantity"),
                    "hierarchy_level": prov.get("hierarchy_level"),
                    "numerator": prov.get("numerator"),
                    "denominator": prov.get("denominator"),
                    "fallback_reason": prov.get("level_fallback_reason")})


def _build_unresolved_ledger(eng) -> UnresolvedLedger:
    led = UnresolvedLedger()
    for reason, mass in (eng.unresolved_reasons or {}).items():
        led.add(classify_unresolved_reason(reason), mass, note=reason)
    if eng.truncated_mass > 0:
        led.add("unresolved_truncation", eng.truncated_mass, note="node-cap truncation")
    return led


def _resolve_institution_vote_terminal(bp, question, evidence_text, grounding, validated,
                                       gw_by_combo, shared_combos) -> dict:
    """Compose D7 (faithful representation) + D8 (grounded initial positions) + D14 (deliberative
    convergence + seat-weighted tally) into the institution-vote forecast. Returns the deliberative
    resolution dict (or a withheld marker when the faithful representation is not ready)."""
    from swm.world_model_v2.lean_v2.resolution_spec import INSTITUTION_VOTE, parse_resolution
    from swm.world_model_v2.lean_v2.representation import ensure_faithful_representation
    from swm.world_model_v2.lean_v2.institution_terminal import resolve_institution_terminal
    from swm.world_model_v2.lean_v2.state_completeness import feasible_options_for
    rspec = parse_resolution(
        bp.resolution.get("interpretation") or bp.resolution.get("yes_means") or "",
        question=question, terminal_kind_hint=INSTITUTION_VOTE)
    rep = ensure_faithful_representation(bp, rspec, evidence_text=evidence_text,
                                         grounding=grounding)
    if rep.verdict == "not_ready":
        return {"p_yes": None, "verdict": rep.verdict, "representation": rep.as_dict(),
                "note": "faithful representation not ready — deliberative resolution withheld"}
    feas = {}
    for u in rep.voter_units():
        for aid in list(getattr(u, "member_ids", None) or []) + [u.unit_id]:
            if bp.actor_by_id(aid):
                feas[aid] = feasible_options_for(bp, aid)
                break
    target = rep.target_option or str((bp.terminal.get("rule_params") or {}).get("option") or "")
    out = resolve_institution_terminal(
        bp, rep, grounding, states_by_actor=validated, gw_by_combo=gw_by_combo,
        shared_combos=shared_combos, feasible_options_by_actor=feas, target_option=target)
    out["representation"] = rep.as_dict()
    out["verdict"] = rep.verdict
    return out


def _assemble_result(question, bp, eng, ch_eng, unresolved_share, evidence_text, *,
                     grounding, obligations, lean_v2_prov, inst_terminal=None) -> SimulationResult:
    dist = eng.distribution()
    options = eng.options or ["YES", "NO"]
    led = _build_unresolved_ledger(eng)
    # abstention/absence mass is an EXECUTED institutional outcome, not genuine non-resolution
    genuine_unresolved = led.genuinely_unresolved()
    total = eng.yes_mass + eng.no_mass + eng.unresolved_mass + eng.truncated_mass
    genuine_share = genuine_unresolved / total if total else unresolved_share
    lean_v2_prov["unresolved"] = led.as_dict()

    status = "completed"
    if genuine_share >= 0.999:
        status = "unresolved"
    elif genuine_share > 0.001:
        status = "partially_resolved"

    # ---- the simulation-conditional forecast (P(yes | resolved mass)) ----
    resolved_mass = eng.yes_mass + eng.no_mass
    sim = SimulationConditionalForecast(
        p=eng.p_mid, resolved_mass=round(resolved_mass, 4),
        interval=(eng.p_low, eng.p_high) if eng.p_low is not None else None,
        weight_sensitive=eng.weight_sensitive,
        dependence_sensitive=eng.dependence_sensitive,
        provenance={"yes_mass": round(eng.yes_mass, 4), "no_mass": round(eng.no_mass, 4),
                    "dependence_range": (list(eng.dependence_range)
                                         if eng.dependence_range else None)})

    # ---- D7+D8+D14: an INSTITUTION VOTE is resolved by the deliberative sub-simulation over the
    # faithful roster and the REAL threshold — that IS the simulation-conditional forecast, and it
    # resolves fully (a probability), so it also sets the resolved distribution. The independent
    # per-node vote count no longer drives the institution answer.
    inst_p = (inst_terminal or {}).get("p_yes")
    inst_override = bp.terminal.get("kind") == "institution_vote" and inst_p is not None
    if inst_override:
        band = (inst_terminal or {}).get("band")
        yes_lab = str(options[0]) if options else "YES"
        no_lab = str(options[1]) if len(options) > 1 else "NO"
        dist = {yes_lab: round(inst_p, 4), no_lab: round(1.0 - inst_p, 4)}
        resolved_mass = 1.0
        genuine_share, status = 0.0, "completed"
        sim = SimulationConditionalForecast(
            p=round(inst_p, 4), resolved_mass=1.0,
            interval=(band[0], band[1]) if band else None,
            weight_sensitive=bool(band and band[0] < 0.5 < band[1]),
            dependence_sensitive=bool(band and (band[1] - band[0]) > DISAGREEMENT_SPREAD),
            provenance={"source": "deliberative_institution_vote",
                        "institution_type": inst_terminal.get("institution_type"),
                        "threshold": inst_terminal.get("threshold"),
                        "total_seats": inst_terminal.get("total_seats"),
                        "convergence_band": band,
                        "law": "D7 faithful roster + D8 grounded initial positions + D14 "
                               "convergence + seat-weighted absolute-threshold tally"})
    # material structural disagreement keeps BOTH models visible; the challenger becomes a
    # second simulation input (never hidden), reported separately
    structural_sensitivity = False
    structural_disagreement = None
    if ch_eng is not None and ch_eng.p_mid is not None and eng.p_mid is not None:
        spread = abs(ch_eng.p_mid - eng.p_mid)
        structural_sensitivity = spread > DISAGREEMENT_SPREAD
        structural_disagreement = {
            "primary": {"p": eng.p_mid, "distribution": eng.distribution()},
            "challenger": {"p": ch_eng.p_mid, "distribution": ch_eng.distribution()},
            "spread": round(spread, 4),
            "treatment": "kept separated by model; the more-resolved model's conditional is "
                         "the simulation input, the spread widens the interval"}
        if structural_sensitivity:
            # the simulation-conditional becomes the resolved-mass-weighted average of the two
            sim.p = round((eng.p_mid * resolved_mass + ch_eng.p_mid
                           * (ch_eng.yes_mass + ch_eng.no_mass))
                          / max(1e-9, resolved_mass + ch_eng.yes_mass + ch_eng.no_mass), 4)
            lo = min(eng.p_mid, ch_eng.p_mid, (sim.interval or [sim.p])[0]
                     if sim.interval else sim.p)
            hi = max(eng.p_mid, ch_eng.p_mid, (sim.interval or [sim.p, sim.p])[1]
                     if sim.interval else sim.p)
            sim.interval = (round(lo, 4), round(hi, 4))

    # ---- the grounded prior (counted) ----
    prior = _outcome_prior(grounding)

    # ---- combine through empirical reliability (never a fixed blend) ----
    feats = ForecastReliabilityFeatures(
        prior_n=prior.n, prior_specificity=(prior.provenance or {}).get("hierarchy_level", ""),
        resolved_mass=round(resolved_mass, 4),
        unknown_state_mass=round(led.by_cause.get("unresolved_unknown_state", 0.0), 4),
        evidence_coverage=min(1.0, len(getattr(bp, "grounded_rates", []) or []) / 3.0),
        structural_sensitivity=structural_sensitivity,
        weight_sensitive=eng.weight_sensitive, dependence_sensitive=eng.dependence_sensitive,
        prior_sim_divergence=(abs(prior.p - sim.p) if prior.p is not None
                              and sim.p is not None else 0.0),
        horizon_days=0)
    combiner = ForecastReliabilityCombiner()
    combo_report = combiner.combine(prior, sim, feats)
    lean_v2_prov["forecast_decomposition"] = {
        "grounded_prior": prior.as_dict(), "simulation_conditional": sim.as_dict(),
        "combined": combo_report.combined,
        "combined_interval": (list(combo_report.combined_interval)
                              if combo_report.combined_interval else None),
        "method": combo_report.method, "disagreement": combo_report.disagreement,
        "sim_weight": combo_report.sim_weight, "prior_weight": combo_report.prior_weight,
        "fixed_blend_used": combo_report.fixed_blend_used,
        "combiner_available": combo_report.combiner_available,
        "notes": combo_report.notes,
        "reliability_features": feats.as_dict(),
        # §separation labels — the prior and the simulation stay fully separate and both
        # are ALWAYS reported (the headline keys are filled in below once chosen)
        "grounded_prior_probability": prior.p,
        "simulation_conditional_probability": sim.p,
        "resolved_simulation_mass": round(resolved_mass, 4),
        "unresolved_mass_by_cause": dict(led.by_cause),
        "simulation_probability_bounds": (
            [eng.p_low_bounded, eng.p_high_bounded]
            if eng.p_low_bounded is not None else None),
        "residual_bound": eng.residual_bound,
        "prior_forecast": prior.p,
        "simulation_forecast": sim.p}
    lean_v2_prov["combiner"] = combiner.manifest()
    lean_v2_prov["action_calibration"] = load_action_reliability().as_dict()

    # ---- the headline probability ----
    # (a) a TRAINED, leakage-audited combiner (if fitted) is authoritative;
    # (b) otherwise the headline is the mass-based forecast-recovery blend: the resolved
    #     simulation mass keeps its simulated conditional, the unresolved mass takes the
    #     grounded prior — a DATA-DRIVEN split (by how much the simulation actually resolved),
    #     never a fixed blend, the same principled mechanism both other profiles use. The
    #     simulation forecast is preserved (it drives the resolved share); the prior is a
    #     separate, disclosed input for the unresolved share — never a wholesale replacement.
    if inst_override:
        # the deliberative institution vote fully resolves the terminal — it IS the headline,
        # not a wave-mass blend and not a combiner output (there is no unresolved mass to hand
        # to the prior; the faithful roster + real threshold already produced P(YES))
        headline, source = round(inst_p, 4), "deliberative_institution_vote"
    elif combo_report.combined is not None:
        headline, source = combo_report.combined, "combined_calibrated"
    elif sim.p is not None or prior.p is not None:
        rec = recover_forecast(
            distribution=dist, options=options,
            unresolved_mass=eng.unresolved_mass + eng.truncated_mass,
            prior_mean=prior.p, prior_source_class="reference_class" if prior.p is not None
            else "")
        if rec is not None and rec.probability is not None:
            headline = rec.probability
            source = ("mass_weighted:" + rec.probability_source
                      + ("+grounded_prior" if prior.p is not None else ""))
        else:
            headline, source = (sim.p if sim.p is not None else prior.p), \
                ("simulation_conditional" if sim.p is not None else "grounded_prior")
    else:
        headline, source = None, ""
    lean_v2_prov["forecast_decomposition"]["headline_forecast"] = headline
    lean_v2_prov["forecast_decomposition"]["headline_source"] = source
    resolution_report = {}
    if status != "completed":
        resolution_report = {
            "unresolved_share": round(genuine_share, 4),
            "by_cause": led.by_cause,
            "missing_mechanisms": [{"mechanism": k, "unresolved_mass": v}
                                   for k, v in sorted(eng.unresolved_reasons.items())],
            "resolved_conditional_distribution":
                {str(options[0]): sim.p,
                 str(options[1]): (None if sim.p is None else round(1 - sim.p, 4))},
            "honest_bounds": {"min_supported_yes_share": round(eng.yes_mass, 4),
                              "max_possible_yes_share":
                                  round(eng.yes_mass + genuine_unresolved, 4)},
            "per_node": eng.node_audit[:40]}

    res = SimulationResult(
        question=question, simulation_status=status, support_grade="exploratory",
        raw_distribution=dist, raw_probability=headline,
        probability_source=source,
        probability_conditional_on_resolved=sim.p,
        grounding_grade=("partially_grounded" if prior.n >= 4 else "exploratory"),
        confidence=("low" if resolved_mass > 0.5 and not eng.weight_sensitive
                    else "very_low"),
        unresolved_mass=round(genuine_share, 4),
        uncertainty_interval=(
            [eng.p_low_bounded, eng.p_high_bounded]
            if eng.p_low_bounded is not None and eng.residual_bound > 0
            else (list(combo_report.combined_interval)
                  if combo_report.combined_interval
                  else (list(sim.interval) if sim.interval else None))),
        weight_sensitive=bool(eng.weight_sensitive or eng.dependence_sensitive
                              or structural_sensitivity),
        resolution_report=resolution_report,
        structural_disagreement=structural_disagreement,
        limitations=[])
    # honest disclosures
    res.limitations.append(
        f"forecast decomposition — grounded prior {prior.p} (counted n={prior.n}); "
        f"simulation-conditional {sim.p} (resolved mass {resolved_mass:.2f}); "
        f"headline {headline} via {combo_report.method}. Prior and simulation are reported "
        f"separately and never blended by a fixed rule.")
    if eng.dependence_sensitive:
        res.limitations.append(
            f"dependence_sensitive: the actor-state joint dependence is unidentified; the "
            f"answer moves across {eng.dependence_range} between the independent and "
            f"comonotonic structures — no single correlation was assumed")
    if eng.weight_sensitive:
        res.limitations.append(
            f"weight_sensitive: within the counted reference-class intervals P({options[0]}) "
            f"spans [{eng.p_low}, {eng.p_high}]")
    if eng.residual_bound > 0:
        res.limitations.append(
            f"bounded omitted-state residual {eng.residual_bound:.3f}: the interval widens "
            f"to [{eng.p_low_bounded}, {eng.p_high_bounded}] — private-state omissions are "
            f"BOUNDS, never unknown-state worlds")
    if eng.mechanism_straddle:
        res.limitations.append(
            "mechanism straddle: the bounded numeric mechanism's min/max observations "
            "disagree with its decisive statistic about the threshold — disclosed as "
            "sensitivity, not hidden")
    for cause, mass in led.by_cause.items():
        if mass > 0.02:
            res.limitations.append(f"unresolved [{cause}]: {mass:.3f} — {led.treatment(cause)}")
    if not combo_report.combiner_available:
        res.limitations.append(
            "no leakage-audited prior↔simulation reliability combiner is fitted — the "
            "headline is the mass-based recovery blend (resolved mass keeps its simulated "
            "conditional; unresolved mass takes the grounded prior); the two inputs are "
            "reported separately and never blended by a fixed rule")
    return res


def _stopped_before_rollout(question, bp, pre, evidence_text, *, structural_fails,
                            grounding=None) -> SimulationResult:
    """The preflight stop: NO actor simulation was spent discovering unanswerability at the
    end. Returns the best defensible LABELED forecast available (COUNTED outcome prior when one
    exists; nothing grounded → honest p=None) with the exact blocking gap reported."""
    gp = _outcome_prior(grounding or {})
    rec = recover_forecast(distribution={}, options=bp.resolution.get("options") or None,
                           unresolved_mass=1.0, prior_mean=gp.p,
                           prior_source_class="reference_class" if gp.p is not None else "")
    blocking = "; ".join(f"{b['check']}: {b['note']}" for b in pre.blocking[:4]) \
        or "static analysis could not prove a terminal pathway"
    res = SimulationResult(
        question=question, simulation_status="under_modeled",
        support_grade="exploratory",
        under_modeled_subtypes=["under_modeled_nonhuman_mechanism"],
        under_modeled_components=[{"component": b["check"], "kind": "terminal_pathway",
                                   "why": b["note"], "sensitivity": "decisive"}
                                  for b in pre.blocking[:6]]
        or [{"component": "terminal_pathway", "kind": "terminal_pathway",
             "why": blocking, "sensitivity": "decisive"}],
        limitations=[
            f"ANSWERABILITY PREFLIGHT STOP ({pre.verdict}): {blocking}",
            "no actor simulation was run — the missing mechanism would have left every "
            "particle unresolved; repair was attempted once before stopping"])
    if pre.one_sided:
        res.limitations.append(f"mechanically one-sided world recorded: {pre.one_sided} — "
                               f"no fabricated opposite path")
    if structural_fails:
        res.limitations.append("unrepaired validator failures: "
                               + "; ".join(f["what"][:80] for f in structural_fails[:4]))
    if rec is not None:
        attach_recovery(res, rec, override_probability=True)
    return res
