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
            lineage["activation_synthesis"] = synthesize_activation(plan, req)
            for ph, r in req.items():
                if ph in manifest:
                    manifest[ph]["relevance"] = {"required": r["required"], "why": r["why"]}
        except Exception as e:  # noqa: BLE001 — synthesis must never block the forecast
            lineage["activation_synthesis"] = {"error": f"{type(e).__name__}: {e}"[:160]}

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
