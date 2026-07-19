"""World Model V2 — the ONE canonical, maximum-capacity, default-on runtime.

Before this module the facade ran the lightweight `pipeline.simulate` (no posterior), while the posterior
(Phase 3), populations/networks (Phase 9), and dynamic recompilation (Phase 11) lived in separate orphan
pipelines and nonlinear mechanisms (Phase 7) were CLI-only. `simulate_world()` unifies them: a single
question→terminal path that threads ONE `WorldExecutionPlan`, ONE evidence bundle, ONE posterior, ONE
persistent-state lineage, ONE Phase-11 recompilation loop, and ONE terminal projection through the single
`run_from_plan` / persistence rollout funnel (which already fires the Phase-4 actor-policy, Phase-6/7 registry,
and Phase-10 institution operators the compiled plan names).

DEFAULT-ON: the caller does not opt subsystems in. Every completed phase is available automatically; the
compiler decides causal relevance; a subsystem executes when relevant and its omission is RECORDED (with a
reason) when it is irrelevant or cannot be instantiated. Experimental subsystems execute with widened
uncertainty and a lowered support grade rather than being silently dropped.

For every run an ACTIVE-COMPONENT MANIFEST records, per phase: available / selected / executed / omitted /
reason / version / n_events / n_state_deltas / whether removal changes the terminal (filled by the ablation
harness). Honesty rule: a phase is only *causally integrated* if its removal changes execution/terminal on
scenarios where it should matter (or is verified irrelevant); this module wires the phases and records
activation truthfully — the ablation harness (`experiments/unified_ablations.py`) supplies the removal-effect
evidence and the validation doc reports which phases are shallow.
"""
from __future__ import annotations
import time as _time

from swm.world_model_v2.result import SimulationResult, ClarificationRequired, CompilerExecutionError

# Phase 7: importing this registers the nonlinear TransitionOperators (nonlinear_mechanism / _contagion /
# _state_step) into transitions._OPERATORS so they are RUNTIME-CALLABLE from the shared funnel (no longer
# CLI-only). They fire iff the compiled plan names them; each emits StateDeltas.
try:
    import swm.world_model_v2.nonlinear.operators as _nl_ops  # noqa: F401
    _NONLINEAR_OPERATORS_REGISTERED = True
except Exception:  # noqa: BLE001
    _NONLINEAR_OPERATORS_REGISTERED = False

RUNTIME_VERSION = "unified-1.0"
# Phases that are default-on and threaded through the one funnel.
_PHASES = ["phase1_compiler", "phase2_evidence", "phase3_posterior", "phase4_actor_policy",
           "phase6_registry", "phase7_nonlinear", "phase8_persistence", "phase9_populations",
           "phase9_networks", "phase10_institutions", "phase11_recompilation"]


def _mani(available=True, selected=False, executed=False, omitted=False, reason="", version="",
          n_events=0, n_state_deltas=0, causally_irrelevant=False):
    return {"available": available, "selected": selected, "executed": executed, "omitted": omitted,
            "reason": reason, "version": version, "n_events": n_events, "n_state_deltas": n_state_deltas,
            "causally_irrelevant": causally_irrelevant, "removal_changes_terminal": None}


def evidence_sufficiency_signal(bundle, posterior, *, as_of, evidence_dropped=False, retried=False) -> dict:
    """Explicit, testable evidence-sufficiency signal. `starved` is true when as_of was supplied (so
    evidence was expected) yet ZERO effective observations reached the posterior — i.e. the forecast is
    prior-driven, not evidence-driven. Pure function of counts; no side effects."""
    n_docs = len(bundle.documents) if bundle is not None else 0
    n_claims = len(bundle.included_claim_ids) if bundle is not None else 0
    n_eff = int(getattr(posterior, "n_effective_observations", 0) or 0) if posterior is not None else 0
    return {"as_of_supplied": bool(as_of), "n_documents": n_docs, "n_included_claims": n_claims,
            "n_effective_observations": n_eff, "retried": bool(retried),
            "starved": bool(as_of) and n_eff == 0 and not evidence_dropped}


def simulate_world(question: str, *, as_of: str, horizon: str = "", intervention: str = "",
                   user_context=None, prior_checkpoint=None, compute_budget=None, seed: int = 0,
                   llm=None, execution_policy: dict = None, trace_level: str = "standard",
                   config=None, prebuilt_bundle=None) -> SimulationResult:
    """THE canonical public V2 entry. One shared plan/world/queue/StateDelta/terminal path across all phases.

    The ordinary caller does NOT choose which phases run — the compiler selects causally-relevant subsystems.
    `execution_policy` may cap fidelity for the compute budget or force an ablation (removal of a named phase)
    for the causal-ablation harness; it is NOT how normal callers enable/disable phases.
    """
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig, gather_evidence
    from swm.world_model_v2.evidence_requirements import requirements_from_plan
    from swm.world_model_v2.evidence_recompile import recompile_with_evidence
    from swm.world_model_v2.evidence_materialize import attach_evidence_observations
    from swm.world_model_v2.phase3_latent_spec import tag_claims
    from swm.world_model_v2.phase3_posterior import infer_posterior
    from swm.world_model_v2.phase3_priors import build_outcome_rate_prior

    policy = execution_policy or {}
    drop = set(policy.get("drop_phases", []))            # ablation hook (harness only)
    cfg = config or OrchestratorConfig()
    t0 = _time.time()
    manifest = {p: _mani(available=True) for p in _PHASES}
    lineage = {"plan_hashes": [], "recompilations": []}
    costs = {"llm_calls": 0}

    def _iso(s):
        return s

    # ---------- Phase 1: universal compiler → the ONE shared plan ----------
    try:
        plan = compile_world(question, llm=llm, evidence="", as_of=as_of, horizon=horizon,
                             intervention=intervention, seed=seed)
    except ClarificationRequired as e:
        return SimulationResult(question=question, simulation_status="clarification_required",
                                clarification_reason=str(e), latency_s=round(_time.time() - t0, 3),
                                provenance={"runtime": RUNTIME_VERSION, "active_component_manifest": manifest})
    except CompilerExecutionError as e:
        return SimulationResult(question=question, simulation_status="execution_failed",
                                failure_taxonomy=e.taxonomy, latency_s=round(_time.time() - t0, 3),
                                provenance={"runtime": RUNTIME_VERSION, "active_component_manifest": manifest})
    manifest["phase1_compiler"].update(selected=True, executed=True, version="compiler",
                                       reason="always required")
    lineage["plan_hashes"].append(plan.plan_hash())

    # ---------- Phase 2: strict as-of evidence → bundle → evidence-conditioned compiler revision ----------
    bundle = None
    if "phase2_evidence" not in drop and as_of:
        try:
            if prebuilt_bundle is not None:
                # sealed-replay injection point: a FROZEN, time-locked bundle built by the Temporal Replay
                # Laboratory (possibly causally blinded) replaces live retrieval. Recorded, never silent.
                bundle = prebuilt_bundle
                manifest["phase2_evidence"].update(
                    selected=True, executed=True, version="phase2-1.0",
                    reason=f"injected_replay_bundle: {len(bundle.included_claim_ids)} as-of claims")
            else:
                reqs = requirements_from_plan(plan, as_of_iso=_iso(as_of), question=question)
                bundle = gather_evidence(question, as_of=as_of, requirements=reqs, llm=llm, config=cfg,
                                         plan_hash=plan.plan_hash(), seed=seed)
                if not bundle.included_claim_ids:
                    # Empty pull. Retrieval is a live query + LLM claim-extraction, both stochastic — an
                    # empty bundle here is the EXP-104 failure (the run silently forecast from priors). Retry
                    # ONCE with a fresh seed before letting the simulation run blind.
                    retry = gather_evidence(question, as_of=as_of, requirements=reqs, llm=llm, config=cfg,
                                            plan_hash=plan.plan_hash(), seed=seed + 1)
                    lineage["evidence_retry"] = {"first_pull_empty": True,
                                                 "retry_included_claims": len(retry.included_claim_ids)}
                    if retry.included_claim_ids:
                        bundle = retry
                manifest["phase2_evidence"].update(selected=True, executed=True, version="phase2-1.0",
                                                   reason=f"{len(bundle.included_claim_ids)} as-of claims")
            revised, diff = recompile_with_evidence(plan, bundle, llm=llm, horizon=horizon)
            plan = attach_evidence_observations(revised, bundle)
            lineage["plan_hashes"].append(plan.plan_hash())
        except Exception as e:  # noqa: BLE001 — evidence failure never blocks the forecast
            manifest["phase2_evidence"].update(omitted=True, reason=f"evidence_error: {type(e).__name__}")
    else:
        manifest["phase2_evidence"].update(omitted=True,
                                           reason=("dropped_by_policy" if "phase2_evidence" in drop
                                                   else "no as_of supplied"))

    # ---------- Phase 3: posterior over hidden state + structural hypotheses → materialize onto the plan ----
    posterior = None
    prior_spec = None
    if "phase3_posterior" not in drop and bundle is not None:
        try:
            tags = tag_claims(question, bundle, plan, llm=llm)
            prior_spec = build_outcome_rate_prior(plan, llm=llm)
            posterior = infer_posterior(plan, bundle, tags, seed=seed, prior_spec=prior_spec)
            if posterior.n_effective_observations > 0:
                plan.posterior_rate_particles = list(posterior.outcome_rate_particles)
                if posterior.structural_posterior:
                    plan.structural_posterior = dict(posterior.structural_posterior)
                manifest["phase3_posterior"].update(
                    selected=True, executed=True, version="phase3",
                    reason=f"{posterior.n_effective_observations} eff obs; "
                           f"prior {posterior.outcome_rate_prior_mean:.3f}→post {posterior.outcome_rate_mean:.3f}")
            else:
                manifest["phase3_posterior"].update(selected=True, executed=False, omitted=True,
                                                    reason="no admissible as-of evidence updated the posterior")
        except Exception as e:  # noqa: BLE001
            manifest["phase3_posterior"].update(omitted=True, reason=f"posterior_error: {type(e).__name__}")
    else:
        manifest["phase3_posterior"].update(omitted=True,
                                            reason=("dropped_by_policy" if "phase3_posterior" in drop
                                                    else "no evidence bundle"))

    # ---------- EVIDENCE-SUFFICIENCY GATE: a run must never simulate BLIND, silently ----------
    # If as_of was supplied (evidence is expected) but nothing reached the posterior, the forecast is
    # produced from PRIORS, not evidence — the EXP-104 failure where a rich structure hugged ~0.5 from
    # imagination. Record the sufficiency signal explicitly on the result and warn loudly so a starved run
    # is never a quiet coin flip. (This gates VISIBILITY, not execution — the forecast still returns, but
    # honestly labelled evidence-starved so downstream/telemetry can see it.)
    evidence_sufficiency = evidence_sufficiency_signal(
        bundle, posterior, as_of=as_of, evidence_dropped="phase2_evidence" in drop,
        retried="evidence_retry" in lineage)
    lineage["evidence_sufficiency"] = evidence_sufficiency
    if evidence_sufficiency["starved"]:
        import warnings as _warnings
        _warnings.warn(
            f"EVIDENCE-STARVED run: as_of supplied but 0 effective observations reached the posterior — "
            f"forecasting from priors, not evidence: {question[:70]!r}", stacklevel=1)

    # grounded fallback probability: the posterior mean if evidence updated it, else the (now grounded)
    # prior mean. Used ONLY as the last-resort guard when the rollout produces no bound outcome, so a
    # coherent question is NEVER returned with p=None.
    grounded_fallback_mean = None
    try:
        if posterior is not None and int(getattr(posterior, "n_effective_observations", 0) or 0) > 0:
            grounded_fallback_mean = float(getattr(posterior, "outcome_rate_mean"))
        elif prior_spec is not None:
            grounded_fallback_mean = float(prior_spec.mean)
    except Exception:  # noqa: BLE001
        grounded_fallback_mean = None

    # ---------- Phase 9: populations + multilayer networks — instantiate into the shared plan when declared --
    _thread_populations_networks(plan, manifest, drop)

    # ---------- Phase 10 completion: normalize declared institution rule kinds onto the EXECUTABLE set so a
    #            declared institution is not silently dropped to an empty (ornamental) RuleSystem. Records the
    #            report + completeness diagnostics. Does NOT invent institutions. ----
    if "phase10_institutions" not in drop:
        try:
            from swm.world_model_v2.integration_completion import (normalize_institution_rules,
                                                                   executable_rule_count, completeness_diagnostics)
            before = executable_rule_count(plan)
            norm = normalize_institution_rules(plan)
            after = executable_rule_count(plan)
            manifest["phase10_institutions"]["normalization"] = {
                "executable_rules_before": before, "executable_rules_after": after, **norm}
            lineage["completeness_diagnostics"] = completeness_diagnostics(plan)
        except Exception as e:  # noqa: BLE001
            manifest["phase10_institutions"]["normalization"] = {"error": type(e).__name__}

    # ---------- Fidelity layer (identity-preserving directive): full actor decomposition, grounded
    #            institutional parameters, and the Phase-1.5 scheduled-reality (dated public facts) ----
    if "fidelity" not in drop and llm is not None:
        try:
            from swm.world_model_v2.fidelity import fidelity_expand
            from swm.world_model_v2.scheduled_facts import extract_scheduled_facts, attach_scheduled_facts
            from swm.world_model_v2.resolution_criteria import (parse_resolution_criterion,
                                                                ground_actor_intentions)
            ev_text = bundle.render(max_chars=2400) if bundle is not None else ""
            # universal resolution-criterion parsing: the precise state that resolves YES anchors the
            # contract's rule, the fact-entailment judgments, and the intention grounding
            crit = parse_resolution_criterion(question, horizon=horizon, llm=llm)
            if crit:
                lineage["resolution_criterion"] = crit
                plan.provenance["resolution_criterion"] = crit
                try:
                    plan.outcome_contract.resolution_rule = str(crit["resolves_yes_iff"])[:300]
                except Exception:  # noqa: BLE001
                    pass
            crit_q = (f"{question}\nRESOLVES YES IFF: {crit.get('resolves_yes_iff')}"
                      if crit else question)
            lineage["fidelity_expansion"] = fidelity_expand(plan, crit_q, as_of=as_of,
                                                            evidence_text=ev_text, llm=llm)
            facts = extract_scheduled_facts(crit_q, as_of=as_of, horizon=horizon,
                                            evidence_text=ev_text, llm=llm)
            lineage["scheduled_reality"] = attach_scheduled_facts(plan, facts)
            lineage["scheduled_reality"]["facts"] = facts[:8]
            # canonical mode decomposition BEFORE intention grounding, so stances can be
            # MODE-SCOPED (stance(actor, mode)) and the pathway processes exist for the trajectory
            # layer + hazard consumption. K-pass self-consistency makes the mode set reproducible.
            from swm.world_model_v2.event_time import is_when_question as _is_when
            from swm.world_model_v2.mode_graph import (canonical_modes, declare_pathway_processes,
                                                       ground_process_states, mode_pathway)
            if _is_when(question):
                _modes, _cons = canonical_modes(
                    question=question, criterion=crit,
                    hypotheses=list(getattr(plan, "structural_hypotheses", []) or []),
                    options=list(getattr(plan.outcome_contract, "options", []) or []), llm=llm)
                plan._canonical_modes, plan._mode_consensus = _modes, _cons
                lineage["mode_graph"] = _cons
                _pws = sorted({mode_pathway(m) for m in _modes})
                _states = ground_process_states(question, crit, _pws, as_of=as_of,
                                                evidence_text=ev_text, llm=llm)
                lineage["pathway_processes"] = declare_pathway_processes(plan, _modes,
                                                                         grounding=_states)
            # per-actor evidence-grounded intentions (state, not policy guesses) — mode-scoped
            # against the canonical modes when they exist
            lineage["actor_intentions"] = ground_actor_intentions(
                plan, question, criterion=crit, evidence_text=ev_text, llm=llm,
                modes=getattr(plan, "_canonical_modes", None))
            # capability becomes a live, depletable capacity resource (world-dynamics layer)
            from swm.world_model_v2.world_dynamics import declare_actor_capacity
            lineage["actor_capacity"] = declare_actor_capacity(plan)
            # binary/other questions: the resolution's causal pathways are named by the grounded
            # stances themselves — declare their processes so actions move the residual chain too
            if not getattr(plan, "_declared_pathways", None):
                _st_pws = sorted({str(s.get("pathway")) for s in
                                  (getattr(plan, "_intention_stances", None) or [])
                                  if s.get("pathway") and s.get("pathway") != "any"})
                if _st_pws:
                    _pseudo = [{"id": pw, "pathway": pw} for pw in _st_pws]
                    _states = ground_process_states(question, crit, _st_pws, as_of=as_of,
                                                    evidence_text=ev_text, llm=llm)
                    lineage["pathway_processes"] = declare_pathway_processes(plan, _pseudo,
                                                                             grounding=_states)
        except Exception as e:  # noqa: BLE001 — fidelity must never block the forecast
            lineage["fidelity_expansion"] = {"error": f"{type(e).__name__}: {e}"[:140]}

    # ---------- Activation synthesis: relevance-gated completion of the execution chain ----------
    # For each phase the compiler's own causal analysis marks REQUIRED, complete the missing execution
    # linkage from DECLARED components (institutional_decision events, population/network consumers,
    # nonlinear state chains, registry pack events); for NOT-required phases, gate off ornamental
    # execution. Default-on; ablation policy can drop it whole or per phase.
    if "activation_synthesis" not in drop:
        try:
            from swm.world_model_v2.activation_synthesis import phase_requirements, synthesize_activation
            req = phase_requirements(plan)
            # respect per-phase ablation drops: a dropped phase must not be synthesized back in
            for ph in list(req):
                if ph in drop:
                    req[ph] = {"required": False, "why": "dropped_by_policy"}
            rep = synthesize_activation(plan, req)
            rep["_pre_synthesis_requirements"] = req         # the ONE relevance verdict, reused by assess
            lineage["activation_synthesis"] = rep
            # situation-dependent trajectory depth (never one decision at horizon-3d)
            if "fidelity" not in drop:
                from swm.world_model_v2.fidelity import deepen_trajectory
                cad = (lineage.get("fidelity_expansion") or {}).get("decision_cadence_days")
                lineage["trajectory_depth"] = deepen_trajectory(plan, req, cadence_days=cad)
            for ph, r in req.items():
                if ph in manifest:
                    manifest[ph]["relevance"] = {"required": r["required"], "why": r["why"]}
        except Exception as e:  # noqa: BLE001 — synthesis must never block the forecast
            lineage["activation_synthesis"] = {"error": f"{type(e).__name__}: {e}"[:160]}

    # ---------- Event-time conversion: the outcome of the simulation IS the answer ----------
    # READOUT, NOT RESOLVER. The answer mechanism needs a readout (translate the simulated world into the
    # question's format); it must not have a resolver (an extra decision/draw that declares the answer).
    # Here — after synthesis/depth/intentions so `_consumed_state` and the grounded intentions exist —
    # BOTH question shapes are rewired onto first-passage semantics: "when/how-long" questions become
    # timing contracts; binary deadline questions have their terminal resolver events REMOVED and the
    # answer becomes F(deadline) read from the same trajectories (entailing dated facts and institutional
    # decisions absorb at their real dates; the evidence-updated posterior parameterizes the residual
    # hazard chain instead of biasing a terminal coin). Universal: routing is purely linguistic +
    # structural, never scenario-specific. On conversion failure the resolver path remains (recorded).
    if "event_time" not in drop:
        try:
            from swm.world_model_v2.event_time import (convert_binary_to_event_time,
                                                       convert_to_event_time, is_when_question)
            crit = lineage.get("resolution_criterion") or {}
            _opts = [str(o) for o in (getattr(plan.outcome_contract, "options", None) or [])]
            if is_when_question(question):
                convert_to_event_time(plan, crit, lineage=lineage, llm=llm)
            elif len(_opts) > 2:
                # CATEGORICAL unification: "how/which/who" questions run the SAME first-passage
                # machinery — modes are the question's own options; the distribution is the
                # absorbed_by marginal with honest none-by-horizon mass (no forced pick)
                convert_to_event_time(plan, crit, lineage=lineage, llm=llm,
                                      categorical_options=_opts)
            else:
                fex = lineage.get("fidelity_expansion")
                cad = fex.get("decision_cadence_days") if isinstance(fex, dict) else None
                convert_binary_to_event_time(plan, crit, lineage=lineage, llm=llm, cadence_days=cad)
        except Exception as e:  # noqa: BLE001 — conversion must never block the forecast
            lineage["event_time"] = {"error": f"{type(e).__name__}: {e}"[:160]}

    # ---------- Phase 11: dynamic-recompilation loop over the as-of observations (same plan lineage) --------
    if "phase11_recompilation" not in drop and bundle is not None:
        _run_recompilation(plan, bundle, as_of, horizon, seed, llm, manifest, lineage, costs)
    else:
        manifest["phase11_recompilation"].update(
            omitted=True, reason=("dropped_by_policy" if "phase11_recompilation" in drop
                                  else "no observations"))

    # ---------- Terminal projection through the ONE funnel (Phase 8 persistence + P4/P6/P7/P10 operators) ----
    res = _project_terminal(question, plan, as_of, horizon, intervention, seed, llm, user_context,
                            prior_checkpoint, manifest, drop, t0)

    # ROLLOUT RETRY: the persistence-aware rollout is stochastic; an INTERMITTENT empty/failed rollout
    # (observed: BoJ once ran 47 calls with an empty operator census → no absorber → no forecast, and the
    # next run of the SAME plan ran the full 266-call rollout to 0.73) recovers on a re-roll. Retry ONCE
    # with a fresh seed before falling back — this recovers the real simulation instead of settling for the
    # prior. Records the first failure so a persistent (non-transient) cause stays diagnosable.
    def _no_forecast(r):
        return (not r.has_forecast()) or (r.raw_probability is None and r.calibrated_probability is None)

    if _no_forecast(res):
        census0 = ((res.provenance or {}).get("operator_delta_census") or {})
        lineage["rollout_retry"] = {"first_status": res.simulation_status,
                                    "first_taxonomy": res.failure_taxonomy,
                                    "first_census_ops": sorted(census0.keys())}
        res_retry = _project_terminal(question, plan, as_of, horizon, intervention, seed + 1, llm,
                                      user_context, prior_checkpoint, manifest, drop, t0)
        lineage["rollout_retry"]["retry_status"] = res_retry.simulation_status
        lineage["rollout_retry"]["recovered"] = not _no_forecast(res_retry)
        if not _no_forecast(res_retry):
            res = res_retry

    # surface evidence sufficiency on the result (provenance + a loud limitation when starved), so a
    # prior-driven forecast is never mistaken for an evidence-driven one.
    try:
        res.provenance["evidence_sufficiency"] = evidence_sufficiency
        if evidence_sufficiency.get("starved"):
            res.limitations = (list(res.limitations or []) + [
                "EVIDENCE-STARVED: 0 as-of observations reached the posterior — this forecast is "
                "prior-driven, not evidence-driven (retrieval returned nothing usable even after retry)"])
    except Exception:  # noqa: BLE001 — telemetry attach must never break the forecast
        pass

    # NO-NULL GUARANTEE: a COHERENT question must NEVER return without a probability — not on an empty
    # rollout (completed_with_degradation, nothing absorbed) NOR on an execution_failed exception that the
    # retry above did not recover. In either case fall back to the grounded posterior/prior mean, honestly
    # labelled EXECUTION-DEGRADED, and mark the run completed_with_degradation so it carries a forecast.
    # (Only a genuine clarification_required — an incoherent question — is exempt.) A last resort, not the
    # primary path: the retry recovers the real simulation first; this only guarantees we never emit None.
    try:
        coherent = res.simulation_status != "clarification_required"
        no_p = res.raw_probability is None and res.calibrated_probability is None
        if coherent and no_p:
            fb = grounded_fallback_mean if grounded_fallback_mean is not None else 0.5
            src = ("posterior" if (posterior is not None
                                   and int(getattr(posterior, "n_effective_observations", 0) or 0) > 0)
                   else "prior")
            orig_status = res.simulation_status
            res.raw_probability = round(float(fb), 4)
            res.simulation_status = "completed_with_degradation"   # now carries a forecast
            res.limitations = (list(res.limitations or []) + [
                f"EXECUTION-DEGRADED (was {orig_status!r}): the rollout produced no bound outcome even after "
                f"retry; forecast falls back to the grounded {src} mean ({fb:.3f}) rather than returning None"])
            res.provenance["execution_degraded_fallback"] = {"used": True, "value": round(float(fb), 4),
                                                             "source": src, "original_status": orig_status}
    except Exception:  # noqa: BLE001 — the guard must never itself break the forecast
        pass

    # ---------- MANDATORY PHASE SUPERVISION: one PhaseExecutionRecord per phase, every run ----------
    # The manifest is DERIVED from these records; a relevant phase that did not execute is a recorded
    # integration failure (blocked_*) that lowers support — a phase can never disappear silently.
    if res.has_forecast():
        try:
            from swm.world_model_v2 import phase_supervision as PS
            pre_req = (lineage.get("activation_synthesis") or {}).get("_pre_synthesis_requirements")
            recs = PS.assess(plan, has_as_of=bool(as_of), has_bundle=bundle is not None,
                             versions={p: manifest[p].get("version", "") for p in manifest},
                             req=pre_req)
            core_meta = {p: {"executed": bool(manifest[p].get("executed")),
                             "reason": manifest[p].get("reason", "")}
                         for p in ("phase1_compiler", "phase2_evidence", "phase3_posterior",
                                   "phase8_persistence", "phase11_recompilation")}
            sup = PS.finalize(recs, plan, res, phase_meta=core_meta)
            for ph, m in sup["manifest"].items():
                manifest[ph].update(m)
            res.provenance["phase_execution_records"] = {p: r.as_dict() for p, r in sup["records"].items()}
            res.provenance["phase_integration_failures"] = sup["integration_failures"]
            res.provenance["fully_integrated"] = sup["fully_integrated"]
            if sup["integration_failures"]:
                if res.support_grade in ("empirically_supported", "transfer_supported"):
                    res.support_grade = "exploratory"
                res.limitations = list(res.limitations) + [
                    f"integration failure: {f['phase']} {f['status']} ({f['reason'][:60]})"
                    for f in sup["integration_failures"][:4]]
        except Exception as e:  # noqa: BLE001 — supervision must not kill the forecast, but never hides
            res.provenance["phase_supervision_error"] = f"{type(e).__name__}: {e}"[:160]
            res.provenance["fully_integrated"] = False

    # ---------- attach unified provenance + manifest + Phase-12 incompatibility ----------
    res.provenance["runtime"] = RUNTIME_VERSION
    res.provenance["active_component_manifest"] = manifest
    res.provenance["plan_lineage"] = lineage
    res.provenance["evidence_bundle_hash"] = bundle.bundle_hash() if bundle is not None else ""
    res.provenance["posterior_consumed"] = bool(posterior and posterior.n_effective_observations > 0)
    res.provenance["calibration_compatibility"] = {
        "old_phase12_calibrator": "INCOMPATIBLE",
        "reason": "unified runtime changes the forecast distribution; the pre-unification Phase-12 calibrator "
                  "must be refit on the post-unification corpus before it may serve this runtime.",
        "refit_command": "PYTHONPATH=. python experiments/phase12_refit.py --regen"}
    res.latency_s = round(_time.time() - t0, 3)
    return res


def _thread_populations_networks(plan, manifest, drop):
    """Phase 9: instantiate population segments + multilayer-network layers into the shared plan when the
    compiler declared them. Distinct edge layers control distinct causal processes (delivery/exposure/trust/
    influence/authority/alliance). When the plan declares none, record a causally-irrelevant omission."""
    if "phase9_populations" in drop and "phase9_networks" in drop:
        for k in ("phase9_populations", "phase9_networks"):
            manifest[k].update(omitted=True, reason="dropped_by_policy")
        return
    pops = list(getattr(plan, "populations", []) or [])
    nets = list(getattr(plan, "networks", []) or [])
    if pops and "phase9_populations" not in drop:
        try:
            from swm.world_model_v2.phase9_population import PopulationSpec, population_latent_specs
            specs = []
            for p in pops:
                ps = p if isinstance(p, PopulationSpec) else None
                if ps is not None:
                    specs.extend(population_latent_specs(ps))
            plan.population_latent_specs = [s.as_dict() if hasattr(s, "as_dict") else s for s in specs]
            manifest["phase9_populations"].update(
                selected=True, executed=bool(specs), omitted=(not specs), version="phase9-pop",
                reason=(f"{len(pops)} population segment(s); {len(specs)} latent spec(s) attached" if specs
                        else "compiler declared populations but not as PopulationSpec — not yet instantiable"))
        except Exception as e:  # noqa: BLE001
            manifest["phase9_populations"].update(selected=True, omitted=True,
                                                  reason=f"pop_instantiation_error: {type(e).__name__}")
    else:
        manifest["phase9_populations"].update(omitted=True, causally_irrelevant=True,
                                              reason="no population segments declared by the compiler")
    manifest["phase9_networks"].update(
        selected=bool(nets and "phase9_networks" not in drop), omitted=not nets, causally_irrelevant=not nets,
        version=("phase9-net" if nets else ""),
        reason=(f"{len(nets)} network layer(s) declared" if nets
                else "no multilayer network declared by the compiler"))


def _run_recompilation(plan, bundle, as_of, horizon, seed, llm, manifest, lineage, costs):
    """Phase 11: run the RecompilationController over observations built from the as-of evidence, on the SAME
    plan lineage. Records recompilation traces; if a revision is adopted it mutates the plan in place so the
    terminal projects from the revised plan."""
    try:
        from swm.world_model_v2.phase11.controller import RecompilationController, ExecutionAdapter
        from swm.world_model_v2.phase11.contracts import RecompileObservation
        as_ts = float(getattr(bundle, "as_of", 0.0) or 0.0)
        obs = []
        for i, cid in enumerate(list(bundle.included_claim_ids)[:12]):
            claim = next((c for c in bundle.claims if c.get("claim_id") == cid), {})
            pub = claim.get("publication_time") or as_ts
            obs.append(RecompileObservation(
                observation_id=f"obs_{i}", observation_type=claim.get("claim_class", "actor_statement"),
                origin="external_evidence", event_time=float(pub or as_ts), ingestion_time=float(as_ts),
                evidence_ids=[cid], provenance={"observed_value": None}))
        ctrl = RecompilationController(llm=llm, seed=seed, max_recompiles=2)
        # a lightweight adapter: the controller monitors surprise/triggers; terminal is projected downstream
        cr = ctrl.run(plan=plan, worlds=[], weights=[], pending_events=[], observations=obs,
                      horizon_ts=float(as_ts) + 1.0, as_of=float(as_ts), execution=ExecutionAdapter())
        manifest["phase11_recompilation"].update(
            selected=True, executed=True, version="phase11",
            n_events=cr.n_observations,
            reason=f"{cr.n_recompiles} recompile(s) over {cr.n_observations} obs; eligible={cr.n_eligible}")
        lineage["recompilations"] = cr.traces[:5]
        costs["llm_calls"] += (cr.cost or {}).get("llm_calls", 0)
    except Exception as e:  # noqa: BLE001 — recompilation never blocks the forecast
        manifest["phase11_recompilation"].update(omitted=True, reason=f"recompile_error: {type(e).__name__}")


def _project_terminal(question, plan, as_of, horizon, intervention, seed, llm, user_context,
                      prior_checkpoint, manifest, drop, t0):
    """Terminal projection through the single funnel: Phase-8 persistence-aware rollout (which fires the
    Phase-4 actor-policy, Phase-6/7 registry, and Phase-10 institution operators the plan names). Phase 8 is
    default-on; when no history/checkpoint exists it runs the ordinary rollout with broad priors."""
    from swm.world_model_v2.phase8_pipeline import run_with_persistence
    persistence_dropped = "phase8_persistence" in drop
    try:
        if persistence_dropped:
            from swm.world_model_v2.materialize import run_from_plan
            from swm.world_model_v2.pipeline import result_from_run
            result, branches = run_from_plan(plan, llm=llm, seed=seed)
            res = result_from_run(question, plan, result, branches, intervention=intervention, t0=t0)
            manifest["phase8_persistence"].update(omitted=True, reason="dropped_by_policy")
        else:
            res, _p8meta = run_with_persistence(question, plan, llm=llm, context=user_context,
                                                actor_history=(prior_checkpoint or {}).get("actor_history") if prior_checkpoint else None,
                                                intervention=intervention, t0=t0, seed=seed)
            manifest["phase8_persistence"].update(
                selected=True, executed=True, version="phase8-1.0",
                reason=("checkpoint-conditioned" if prior_checkpoint else "no prior history — broad-prior rollout"))
    except CompilerExecutionError as e:
        return SimulationResult(question=question, simulation_status="execution_failed",
                                failure_taxonomy=e.taxonomy, plan_hash=plan.plan_hash(),
                                latency_s=round(_time.time() - t0, 3))
    except Exception as e:  # noqa: BLE001
        return SimulationResult(question=question, simulation_status="execution_failed",
                                failure_taxonomy="runtime_exception", limitations=[str(e)[:160]],
                                latency_s=round(_time.time() - t0, 3))
    # record P4/P6/P7/P10 activation from the executed plan (they fire iff the plan named their operators)
    _record_operator_phases(plan, res, manifest)
    return res


def _record_operator_phases(plan, res, manifest):
    """Fill P4/P6/P7/P10 manifest entries from the operators the compiled plan actually selected."""
    mechs = [str(m.get("operator", m.get("family", ""))) for m in (getattr(plan, "accepted_mechanisms", []) or [])
             if isinstance(m, dict)]
    text = " ".join(mechs).lower()
    def _set(key, needles, version):
        hit = any(n in text for n in needles)
        if hit:
            manifest[key].update(selected=True, executed=True, version=version, reason="operator in plan")
        else:
            manifest[key].update(omitted=True, causally_irrelevant=True,
                                 reason="no operator of this phase selected by the compiler")
    _set("phase4_actor_policy", ["decision", "actor", "policy"], "phase4")
    _set("phase6_registry", ["registry", "family", "mechanism"], "phase6")
    _set("phase7_nonlinear", ["nonlinear", "hazard", "threshold", "saturation", "hawkes", "diffusion"], "phase7")
    _set("phase10_institutions", ["institution", "rule", "vote", "approval"], "phase10")
    n_deltas = len(getattr(res, "sensitivity_contributors", []) or []) or None
    for k in ("phase4_actor_policy", "phase6_registry", "phase7_nonlinear", "phase10_institutions"):
        if manifest[k]["executed"] and n_deltas:
            manifest[k]["n_state_deltas"] = n_deltas
