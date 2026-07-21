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
from swm.world_model_v2.lean_v2.engine import EngineConfig, WaveEngine
from swm.world_model_v2.lean_v2.gateway import BudgetExhausted, LLMGateway
from swm.world_model_v2.lean_v2.preflight import run_preflight
from swm.world_model_v2.lean_v2.slice import backward_slice
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

    # ---------------- 5. slice + 6. consequence templates -------------------------------
    sl = ckpt.run_stage("actor_authority_slicing", lambda: backward_slice(bp))
    executor = TemplateExecutor(ckpt.run_stage(
        "consequence_template_construction", lambda: precompile_templates(bp, cache)), bp)
    lean_v2_prov["slice"] = sl.manifest()

    # ---------------- 7. three-valued answerability preflight ---------------------------
    pre = ckpt.run_stage("answerability_preflight", lambda: run_preflight(
        bp, as_of=as_of, horizon=horizon, consequence_templates=executor.templates))
    lean_v2_prov["preflight"] = pre.as_dict()
    if pre.verdict == "unanswerable" or (pre.verdict == "uncertain"
                                         and not pre.probe.get("reached_terminal")):
        return _finish(_stopped_before_rollout(question, bp, pre, evidence_text,
                                               structural_fails=fails))

    # ---------------- 8. weighted causal waves (the primary run) ------------------------
    engine_cfg = EngineConfig(
        max_workers=int(v2.get("max_workers") or 6),
        behavioral_replicate_index=0)
    decision_cache = DecisionEquivalenceCache()
    coalescer = WeightedBranchCoalescer(max_nodes=budget.max_weighted_nodes)
    primary = WaveEngine(bp=bp, kept_actors=sl.kept_actors, promotable=sl.promotable,
                         executor=executor, gateway=gateway, budget_ledger=ledger,
                         compile_cache=cache, config=engine_cfg, coalescer=coalescer,
                         decision_cache=decision_cache, structural_model="primary")
    try:
        eng = ckpt.run_stage("causal_waves_primary",
                             lambda: primary.run(as_of=as_of, horizon=horizon),
                             idempotent=False)
    except BudgetExhausted as e:
        eng = primary.result
        primary._finalize([])                    # account what exists; nothing is invented
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
                    structural_model="challenger")
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

    # ---------------- 11. terminal projection + forecast recovery ------------------------
    res = ckpt.run_stage("terminal_projection", lambda: _assemble_result(
        question, bp, eng, challenger_engine, unresolved_share, evidence_text,
        executor=executor, ledger=ledger))
    lean_v2_prov["consequences"] = consequences_manifest(executor)
    return _finish(res)


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


def _assemble_result(question, bp, eng, ch_eng, unresolved_share, evidence_text, *,
                     executor, ledger) -> SimulationResult:
    dist = eng.distribution()
    options = eng.options or ["YES", "NO"]
    resolution_report = None
    status = "completed"
    if unresolved_share >= 0.999:
        status = "unresolved"
    elif unresolved_share > 0.001:
        status = "partially_resolved"
    if status != "completed":
        resolution_report = {
            "unresolved_share": round(unresolved_share, 4),
            "missing_mechanisms": [{"mechanism": k, "unresolved_mass": v}
                                   for k, v in sorted(eng.unresolved_reasons.items())],
            "resolved_conditional_distribution":
                {str(options[0]): eng.p_mid, str(options[1]):
                 (None if eng.p_mid is None else round(1 - eng.p_mid, 4))},
            "honest_bounds": {"min_supported_yes_share": round(eng.yes_mass, 4),
                              "max_possible_yes_share":
                                  round(eng.yes_mass + eng.unresolved_mass
                                        + eng.truncated_mass, 4)},
            "per_node": eng.node_audit[:40]}

    prior = _grounded_prior_inputs(bp)
    rec = recover_forecast(distribution=dist, options=options,
                           unresolved_mass=eng.unresolved_mass + eng.truncated_mass,
                           prior_mean=prior.get("prior_mean"),
                           prior_source_class=prior.get("prior_source_class", ""))
    res = SimulationResult(
        question=question, simulation_status=status, support_grade="exploratory",
        raw_distribution=dist,
        resolution_report=resolution_report or {},
        structural_disagreement=None,
        limitations=[])
    if rec is not None:
        attach_recovery(res, rec, override_probability=True)
    # weight-range sensitivity widens (never narrows) the interval; crossing 0.5 marks it
    if eng.p_low is not None and eng.p_high is not None and res.raw_probability is not None:
        cur = res.uncertainty_interval or [res.raw_probability, res.raw_probability]
        res.uncertainty_interval = [round(min(cur[0], eng.p_low), 4),
                                    round(max(cur[1], eng.p_high), 4)]
        res.weight_sensitive = bool(res.weight_sensitive or eng.weight_sensitive)
        if eng.weight_sensitive:
            res.limitations.append(
                f"weight_sensitive: plausible private-state weight assignments within the "
                f"grounded ranges move P({options[0]}) across [{eng.p_low}, {eng.p_high}] "
                f"— the point weights are support-class midpoints, not measurements")
    # challenger: report both worlds; material disagreement becomes an equal-weight mixture
    if ch_eng is not None and ch_eng.p_mid is not None and eng.p_mid is not None:
        spread = abs(ch_eng.p_mid - eng.p_mid)
        res.structural_disagreement = {
            "primary": {"p": eng.p_mid, "distribution": eng.distribution()},
            "challenger": {"p": ch_eng.p_mid, "distribution": ch_eng.distribution()},
            "spread": round(spread, 4)}
        if spread > DISAGREEMENT_SPREAD and res.raw_probability is not None:
            mixed = round((eng.p_mid + ch_eng.p_mid) / 2.0, 4)
            res.limitations.append(
                f"structural disagreement: primary {eng.p_mid} vs challenger {ch_eng.p_mid} "
                f"(spread {spread:.3f}) — headline is the equal-weight mixture {mixed}; "
                f"the question required deeper computation than the consumer default")
            res.raw_probability = mixed
            res.probability_source = "mixed:" + (res.probability_source
                                                 or "completed_rollouts") + "+challenger"
            res.weight_sensitive = res.weight_sensitive or (
                min(eng.p_mid, ch_eng.p_mid) < 0.5 < max(eng.p_mid, ch_eng.p_mid))
            lo, hi = (res.uncertainty_interval or [mixed, mixed])
            res.uncertainty_interval = [round(min(lo, eng.p_mid, ch_eng.p_mid), 4),
                                        round(max(hi, eng.p_mid, ch_eng.p_mid), 4)]
    if eng.truncated_mass > 0:
        res.limitations.append(
            f"node-cap truncation: {eng.truncated_mass:.4f} probability mass exceeded the "
            f"bounded node population and is disclosed as truncated — never renormalized away")
    if prior:
        res.limitations.append(
            f"grounded reference rate in recovery: {prior.get('basis')} "
            f"({prior.get('prior_source_class')})")
    return res


def _stopped_before_rollout(question, bp, pre, evidence_text, *, structural_fails
                            ) -> SimulationResult:
    """The preflight stop: NO actor simulation was spent discovering unanswerability at the
    end. Returns the best defensible LABELED forecast available (grounded rate → labeled
    prior; nothing grounded → honest p=None) with the exact blocking gap reported."""
    prior = _grounded_prior_inputs(bp)
    rec = recover_forecast(distribution={}, options=bp.resolution.get("options") or None,
                           unresolved_mass=1.0, prior_mean=prior.get("prior_mean"),
                           prior_source_class=prior.get("prior_source_class", ""))
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
