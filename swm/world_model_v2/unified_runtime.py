"""World Model V2 — the ONE canonical, maximum-capacity, default-on runtime.

Before this module the facade ran the lightweight `pipeline.simulate` (no posterior), while the posterior
(Phase 3), populations/networks (Phase 9), and dynamic recompilation (Phase 11) lived in separate orphan
pipelines and nonlinear mechanisms (Phase 7) were CLI-only. `simulate_world()` unifies them: a single
question→terminal path that threads ONE evidence bundle and ONE persistent-state lineage per structural
model through the single `run_from_plan` / persistence rollout funnel (which already fires the Phase-4
actor-policy, Phase-6/7 registry, and Phase-10 institution operators the compiled plan names).

STRUCTURAL-MODEL UNCERTAINTY IS DEFAULT-ON. The runtime no longer begins with one `compile_world(...)`:
it begins with the ensemble compiler (swm.world_model_v2.ensemble_compiler) — several INDEPENDENT actual
LLM generation calls produce materially different candidate causal models; adversarial critics search for
missing actors/institutions/constraints/mechanisms; candidates compile SEPARATELY against one shared
immutable evidence bundle; conservative dedup collapses only genuine structural equivalence; every
plausible distinct model gets a REAL pilot through this same canonical funnel; promoted models each get
AT LEAST the full single-model particle budget (pilot particles reused as a deterministic prefix); and
the result reports per-model distributions, structural sensitivity, reversal conditions and structural
value-of-information. A perfectly executed simulation of the wrong causal model is still wrong — that is
why one valid schema is no longer enough. See swm/world_model_v2/structural_runtime.py.

Single-model compilation survives ONLY as the explicitly named ablation
`execution_policy={"structural_mode": "single_structural_model"}` (scientific ablations, frozen
historical artifact compatibility, isolated compiler unit tests). An ordinary caller cannot silently
reach it, and a missing LLM backend fails loudly instead of producing a deterministic fallback model.

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

RUNTIME_VERSION = "unified-2.0-structural-ensemble"
#: modes for execution_policy["structural_mode"]; "ensemble" is the ONLY default. The single-model mode
#: is an explicit ablation/baseline label — tests enforce that ordinary calls never reach it silently.
STRUCTURAL_MODES = ("ensemble", "single_structural_model")
#: The two PERMANENT execution profiles. `full_fidelity` is the research-grade PR-#127 runtime,
#: byte-for-byte the pre-profile behavior; `lean_adaptive` preserves the same causal architecture while
#: removing duplicated computation (shared run artifacts, actor-state cohorts, decision-equivalence
#: caching, reversal-triggered structural models, progressive particles). The product is world_model_v2;
#: profiles are internal execution strategies, never product names.
#:
#: §25 DEFAULT SWITCH (evaluated on the COMPLETE five-question paired baseline, EXP-107 vs
#: EXP-108; evidence in experiments/results/exp109_comparison.json + docs/LEAN_FINAL_REPORT.md):
#: all seven switch conditions passed — safety invariants; controlled cache parity (byte-identical,
#: 70→4 decision calls, 0 invalidations); no catastrophic paired-forecast degradation (lean Brier
#: 0.337 vs 0.440, correct side 2/5 both); prediction changes explainable per row via
#: probability_source; material savings (19× calls / 14.5× tokens / 12× cost / 17.7× time);
#: unstable questions escalated to full budgets; full_fidelity remains available. `lean_adaptive`
#: is therefore the default; `full_fidelity` is the explicit research-grade option.
#: `lean_v2` is the first-principles CONSUMER execution path (one blueprint call, answerability
#: preflight, terminal-causal slicing, weighted world-state coalescing, conditional challenger,
#: grounded counted-reference-class weighting, consumer compute budget). On the EXP-112 five-
#: question evaluation it reached mean Brier 0.074 vs Lean V1 0.337 vs full_fidelity 0.440 (best
#: on 4/5), at ~18 calls / ~105 s / ~$0.019 per question (docs/LEAN_V2_ACCURACY_REPORT.md).
#: By owner decision `lean_v2` is now the DEFAULT; `lean_adaptive` (Lean V1) and `full_fidelity`
#: remain explicitly selectable. (The report's §16 caveats — the reliability combiner is not yet
#: trained and the accuracy is prior-dominated — are the standing follow-up work, not blockers
#: on the default per the owner's call.)
EXECUTION_PROFILES = ("full_fidelity", "lean_adaptive", "lean_v2")
DEFAULT_EXECUTION_PROFILE = "lean_v2"


def resolve_execution_profile(execution_profile=None) -> str:
    """One authority for profile resolution: explicit argument > SWM_EXECUTION_PROFILE env override >
    module default. Unknown names fail loudly — a profile typo must never silently run the wrong
    execution strategy."""
    import os as _os
    prof = str(execution_profile or _os.environ.get("SWM_EXECUTION_PROFILE", "").strip()
               or DEFAULT_EXECUTION_PROFILE)
    if prof not in EXECUTION_PROFILES:
        raise ValueError(f"unknown execution_profile {prof!r}; one of {EXECUTION_PROFILES}")
    return prof
# Phases that are default-on and threaded through the one funnel.
_PHASES = ["phase1_compiler", "phase2_evidence", "phase3_posterior", "phase4_actor_policy",
           "phase6_registry", "phase7_nonlinear", "phase8_persistence", "phase9_populations",
           "phase9_networks", "phase10_institutions", "phase11_recompilation"]


def _bundle_text(b, max_chars: int) -> str:
    """Evidence text for prompt context — tolerant of both bundle generations (the V2
    bundle has no render()). Module-level: the per-plan conditioning helper (shared by the
    single-model ablation and every structural-ensemble candidate) uses it."""
    if b is None:
        return ""
    if hasattr(b, "render"):
        try:
            return b.render(max_chars=max_chars)
        except Exception:  # noqa: BLE001
            pass
    rows = []
    for c in (getattr(b, "claims", None) or [])[:20]:
        if isinstance(c, dict):
            rows.append(f"- {str(c.get('text', c.get('claim', '')))[:200]} "
                        f"[{str(c.get('source', ''))[:40]}]")
    return "\n".join(rows)[:max_chars]


def _mani(available=True, selected=False, executed=False, omitted=False, reason="", version="",
          n_events=0, n_state_deltas=0, causally_irrelevant=False):
    return {"available": available, "selected": selected, "executed": executed, "omitted": omitted,
            "reason": reason, "version": version, "n_events": n_events, "n_state_deltas": n_state_deltas,
            "causally_irrelevant": causally_irrelevant, "removal_changes_terminal": None}


def evidence_sufficiency_signal(bundle, posterior, *, as_of, evidence_dropped=False, retried=False) -> dict:
    """Explicit, testable evidence-sufficiency signal. `starved` is true when as_of was supplied (so
    evidence was expected) yet ZERO effective observations reached the posterior — i.e. the forecast is
    prior-driven, not evidence-driven. Pure function of counts; no side effects. Tolerant of both
    bundle generations (the V2 replay bundle carries `claims`, not `documents`/`included_claim_ids`)."""
    if bundle is None:
        n_docs, n_claims = 0, 0
    else:
        docs = getattr(bundle, "documents", None)
        claims = getattr(bundle, "included_claim_ids", None)
        if claims is None:
            claims = getattr(bundle, "claims", None) or []
        if docs is None:
            docs = claims
        n_docs, n_claims = len(docs), len(claims)
    n_eff = int(getattr(posterior, "n_effective_observations", 0) or 0) if posterior is not None else 0
    return {"as_of_supplied": bool(as_of), "n_documents": n_docs, "n_included_claims": n_claims,
            "n_effective_observations": n_eff, "retried": bool(retried),
            "starved": bool(as_of) and n_eff == 0 and not evidence_dropped}


def _used_probability(res):
    """The probability a scorer reads from a result: calibrated if present, else the raw projection."""
    p = getattr(res, "calibrated_probability", None)
    return p if p is not None else getattr(res, "raw_probability", None)


def simulate_world_stable(question: str, *, n_runs: int = 3, **kwargs):
    """MEAN-OF-K (opt-in): run simulate_world K times with varied seeds and aggregate to a stable forecast.
    A single run's temperature-0.2 compile draws a different world each time and the probability can swing by
    ~0.6 (measured), so a lone number is noise-dominated — the BTF-1 finding and FutureSearch's #1 disclosed
    lever (mean of multiple runs). The default single-run `simulate_world` is unchanged; callers opt in here.

    Returns the run whose probability is closest to the mean, with its forecast REPLACED by the K-run mean and
    `provenance.mean_of_k = {n_runs, per_run, mean, sd, spread}` attached. Runs that produced no forecast are
    excluded from the mean (but counted). Cost ≈ K × a single run."""
    k = max(1, int(n_runs))
    base_seed = kwargs.pop("seed", 0)
    runs = [simulate_world(question, seed=base_seed + i, **kwargs) for i in range(k)]
    scored = [(r, _used_probability(r)) for r in runs]
    valid = [(r, p) for r, p in scored if p is not None]
    if not valid:
        return runs[0]
    ps = [p for _, p in valid]
    mean_p = sum(ps) / len(ps)
    sd = (sum((p - mean_p) ** 2 for p in ps) / len(ps)) ** 0.5
    rep = min(valid, key=lambda rp: abs(rp[1] - mean_p))[0]     # representative = closest-to-mean run
    rep.raw_probability = round(mean_p, 4)
    rep.calibrated_probability = round(mean_p, 4)               # the scored value IS the K-run mean
    rep.provenance = {**(rep.provenance or {}),
                      "mean_of_k": {"n_runs": k, "n_valid": len(valid),
                                    "per_run": [round(p, 4) for p in ps], "mean": round(mean_p, 4),
                                    "sd": round(sd, 4), "spread": round(max(ps) - min(ps), 4)}}
    rep.limitations = (list(rep.limitations or [])
                       + [f"mean-of-{k}: forecast is the mean of {len(valid)} runs "
                          f"(per-run {[round(p, 3) for p in ps]}, sd {sd:.3f})"])
    return rep


def simulate_world(question: str, *, as_of: str, horizon: str = "", intervention: str = "",
                   user_context=None, prior_checkpoint=None, compute_budget=None, seed: int = 0,
                   llm=None, execution_policy: dict = None, trace_level: str = "standard",
                   config=None, prebuilt_bundle=None, evidence: str = "",
                   execution_profile: str = None) -> SimulationResult:
    """THE canonical public V2 entry. One shared evidence bundle; one funnel; DEFAULT structural-model
    ensemble (several independently generated causal models, each fully simulated).

    `evidence` (caller-supplied as-of text, e.g. a frozen benchmark background) conditions the
    decomposition directly; the Phase-2 retrieval bundle still supersedes it downstream when built.

    `execution_profile` selects an execution STRATEGY, never a different product: `full_fidelity`
    (default) is the research-grade PR-#127 runtime unchanged; `lean_adaptive` runs the same causal
    architecture through the lean runtime (shared artifacts, cohorts, decision-equivalence caching,
    reversal-triggered structural models, progressive particles). See EXECUTION_PROFILES.

    The ordinary caller does NOT choose which phases run — the compiler selects causally-relevant
    subsystems — and does NOT enable the structural ensemble: it is the default. `execution_policy` may
    cap fidelity for the compute budget, force an ablation (removal of a named phase), or select the
    explicit `single_structural_model` ablation baseline; it is NOT how normal callers enable/disable
    behavior."""
    profile = resolve_execution_profile(execution_profile)
    policy = execution_policy or {}
    mode = str(policy.get("structural_mode", "ensemble"))
    if mode not in STRUCTURAL_MODES:
        raise ValueError(f"unknown structural_mode {mode!r}; one of {STRUCTURAL_MODES} "
                         f"(single_structural_model is an explicit ablation, never a default)")
    if profile == "lean_adaptive":
        from swm.world_model_v2.lean_runtime import simulate_world_lean
        res = simulate_world_lean(
            question, as_of=as_of, horizon=horizon, intervention=intervention,
            user_context=user_context, prior_checkpoint=prior_checkpoint,
            compute_budget=compute_budget, seed=seed, llm=llm, execution_policy=policy,
            trace_level=trace_level, config=config, prebuilt_bundle=prebuilt_bundle,
            evidence=evidence)
        res.provenance = {**(res.provenance or {}), "execution_profile": "lean_adaptive"}
        return res
    if profile == "lean_v2":
        from swm.world_model_v2.lean_v2.runtime import simulate_world_lean_v2
        res = simulate_world_lean_v2(
            question, as_of=as_of, horizon=horizon, intervention=intervention,
            user_context=user_context, prior_checkpoint=prior_checkpoint,
            compute_budget=compute_budget, seed=seed, llm=llm, execution_policy=policy,
            trace_level=trace_level, config=config, prebuilt_bundle=prebuilt_bundle,
            evidence=evidence)
        res.provenance = {**(res.provenance or {}), "execution_profile": "lean_v2"}
        return res
    if mode == "single_structural_model":
        res = _simulate_single_structural_model(
            question, as_of=as_of, horizon=horizon, intervention=intervention,
            user_context=user_context, prior_checkpoint=prior_checkpoint,
            compute_budget=compute_budget, seed=seed, llm=llm, execution_policy=policy,
            trace_level=trace_level, config=config, prebuilt_bundle=prebuilt_bundle,
            evidence=evidence)
    else:
        from swm.world_model_v2.structural_runtime import simulate_structural_ensemble
        res = simulate_structural_ensemble(
            question, as_of=as_of, horizon=horizon, intervention=intervention,
            user_context=user_context, prior_checkpoint=prior_checkpoint,
            compute_budget=compute_budget, seed=seed, llm=llm, execution_policy=policy,
            trace_level=trace_level, config=config, prebuilt_bundle=prebuilt_bundle,
            evidence=evidence)
    res.provenance = {**(res.provenance or {}), "execution_profile": "full_fidelity"}
    return res


def _simulate_single_structural_model(question: str, *, as_of: str, horizon: str = "",
                                      intervention: str = "", user_context=None, prior_checkpoint=None,
                                      compute_budget=None, seed: int = 0, llm=None,
                                      execution_policy: dict = None, trace_level: str = "standard",
                                      config=None, prebuilt_bundle=None,
                                      evidence: str = "") -> SimulationResult:
    """The EXPLICIT single-structural-model ablation/baseline: exactly the pre-ensemble canonical path
    (one compile_world plan, one funnel). Retained for scientific ablations, frozen historical artifact
    compatibility and isolated compiler tests — never the default; reaching it requires
    execution_policy={"structural_mode": "single_structural_model"}."""
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.evidence_orchestrator import (OrchestratorConfig,
                                                          gather_evidence_with_escalation)
    from swm.world_model_v2.evidence_requirements import requirements_from_plan
    from swm.world_model_v2.evidence_recompile import recompile_with_evidence
    from swm.world_model_v2.evidence_materialize import attach_evidence_observations

    policy = execution_policy or {}
    drop = set(policy.get("drop_phases", []))            # ablation hook (harness only)
    cfg = config or OrchestratorConfig()
    t0 = _time.time()
    manifest = {p: _mani(available=True) for p in _PHASES}
    lineage = {"plan_hashes": [], "recompilations": []}
    costs = {"llm_calls": 0}

    # ---------- Individual-reaction route: a personal question about how a specific person will
    # react, with the person's context supplied by the caller (user_context["individual"]),
    # runs the SAME qualitative actor architecture directly — the person is automatically
    # Tier 1 and never needs to pre-exist in a compiled world. Named public figures without
    # supplied context continue through compilation, where the selector's
    # reaction_is_the_question rule makes them Tier 1.
    individual = _route_individual_reaction(question, user_context, llm, as_of, seed, t0)
    if individual is not None:
        return individual

    def _iso(s):
        return s

    # ---------- Phase 1: universal compiler → the ONE plan (explicit single-model ablation) ----------
    # `evidence` (caller-supplied as-of text, e.g. a frozen benchmark background) conditions the
    # decomposition directly; the Phase-2 retrieval bundle still supersedes it downstream when built.
    try:
        plan = compile_world(question, llm=llm, evidence=evidence, as_of=as_of, horizon=horizon,
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
                                       reason="always required (single_structural_model ablation)")
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
                rrule = str(getattr(plan.outcome_contract, "resolution_rule", "") or "")
                # ONE evidence-retry authority (escalation on a thin first pull) shared with the
                # ensemble runtime — see evidence_orchestrator.gather_evidence_with_escalation
                bundle, retry_rec = gather_evidence_with_escalation(
                    question, as_of=as_of, requirements=reqs, llm=llm, config=cfg,
                    plan_hash=plan.plan_hash(), seed=seed, resolution_rule=rrule)
                if retry_rec:
                    lineage["evidence_retry"] = retry_rec
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

    # ---------- Phase 3 + conditioning phases (shared with the ensemble runtime, per-plan) ----------
    posterior = _phase3_block(question, plan, bundle, llm, seed, manifest, drop)
    # evidence-sufficiency gate: a starved run is recorded and warned about, never a quiet coin flip
    evidence_sufficiency = _evidence_sufficiency_block(question, bundle, posterior,
                                                       as_of=as_of, drop=drop, lineage=lineage)
    _condition_plan(question, plan, bundle, as_of, horizon, seed, llm,
                    manifest, lineage, costs, drop,
                    user_context=user_context, intervention=intervention, evidence=evidence)

    # ---------- Terminal projection through the ONE funnel (Phase 8 persistence + P4/P6/P7/P10 operators) ----
    # guarded: one recorded rollout retry on an intermittent empty rollout, then the honesty guards
    res = _project_terminal_guarded(question, plan, as_of, horizon, intervention, seed, llm,
                                    user_context, prior_checkpoint, manifest, drop, t0, lineage)
    _apply_result_guards(res, posterior=posterior,
                         prior_spec=getattr(plan, "_outcome_prior_spec", None),
                         evidence_sufficiency=evidence_sufficiency, lineage=lineage)

    _attach_supervision(res, plan, as_of, bundle, manifest, lineage)

    # ---------- attach unified provenance + manifest + Phase-12 incompatibility ----------
    res.provenance["runtime"] = RUNTIME_VERSION
    res.provenance["structural_mode"] = "single_structural_model"
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


# ------------------------------------------------------------------ shared per-plan phase helpers
# These are the SAME phase steps for both structural modes: the ensemble runtime applies them to EVERY
# structural model's own plan (no mutable state shared across models); the single-model ablation applies
# them to its one plan. Extracting them is what guarantees a pilot/full ensemble simulation runs the full
# causal runtime, not a simplified copy.

def _apply_evidence_to_plan(question, plan, bundle, llm, horizon, manifest, lineage):
    """Evidence-conditioned compiler revision + observation attachment for ONE plan (retrieval NOT
    included — the shared bundle is gathered once at ensemble level under one as-of boundary)."""
    from swm.world_model_v2.evidence_recompile import recompile_with_evidence
    from swm.world_model_v2.evidence_materialize import attach_evidence_observations
    try:
        revised, _diff = recompile_with_evidence(plan, bundle, llm=llm, horizon=horizon)
        revised = attach_evidence_observations(revised, bundle)
        lineage["plan_hashes"].append(revised.plan_hash())
        manifest["phase2_evidence"].update(selected=True, executed=True, version="phase2-1.0",
                                           reason=f"{len(bundle.included_claim_ids)} as-of claims (shared bundle)")
        return revised
    except Exception as e:  # noqa: BLE001 — evidence failure never blocks the forecast
        manifest["phase2_evidence"].update(omitted=True, reason=f"evidence_error: {type(e).__name__}")
        return plan


def _phase3_block(question, plan, bundle, llm, seed, manifest, drop):
    """Phase 3: evidence-updated posterior over hidden state + structural hypotheses for ONE plan.
    Every structural model receives ITS OWN posterior — latent definitions differ across models, so a
    posterior is never copied between plans."""
    from swm.world_model_v2.phase3_latent_spec import tag_claims
    from swm.world_model_v2.phase3_posterior import infer_posterior
    from swm.world_model_v2.phase3_priors import build_outcome_rate_prior
    posterior = None
    prior_spec = None
    if "phase3_posterior" not in drop and bundle is not None:
        try:
            tags = tag_claims(question, bundle, plan, llm=llm)
            prior_spec = build_outcome_rate_prior(plan, llm=llm)
            # stash the prior spec on the plan: the no-silent-None guard and provenance surfacing
            # need the grounded prior mean even when the posterior saw zero observations. Its FULL
            # provenance (grounded/recurrence/reference-class/lean, evidence quality, retained
            # effective N) also rides the plan so the §NAP ledger row for the residual outcome
            # process names the actual prior behind the posterior — the acknowledged remaining
            # assumption is specific, not generic.
            try:
                plan._outcome_prior_spec = prior_spec
                plan._outcome_prior_provenance = {
                    "source_class": getattr(prior_spec, "source_class", None),
                    "reference_class": getattr(prior_spec, "reference_class", None),
                    "transport_risk": getattr(prior_spec, "transport_risk", None),
                    "retained_effective_n": getattr(prior_spec, "retained_effective_n", None),
                    "mean": round(float(prior_spec.mean), 4),
                    **{k: v for k, v in (getattr(prior_spec, "provenance", None) or {}).items()
                       if k in ("evidence_quality", "is_recurrence", "estimated_base_rate",
                                "widening", "rule")}}
            except Exception:  # noqa: BLE001
                pass
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
    return posterior

def _evidence_sufficiency_block(question, bundle, posterior, *, as_of, drop, lineage) -> dict:
    """EVIDENCE-SUFFICIENCY GATE: a run must never simulate BLIND, silently. If as_of was supplied
    (evidence is expected) but nothing reached the posterior, the forecast is produced from PRIORS,
    not evidence — the EXP-104 failure where a rich structure hugged ~0.5 from imagination. Records
    the sufficiency signal in lineage and warns loudly so a starved run is never a quiet coin flip.
    (Gates VISIBILITY, not execution — the forecast still returns, honestly labelled evidence-starved.)
    Shared by BOTH structural modes: the ensemble runtime calls it per structural model."""
    evidence_sufficiency = evidence_sufficiency_signal(
        bundle, posterior, as_of=as_of, evidence_dropped="phase2_evidence" in drop,
        retried="evidence_retry" in lineage)
    lineage["evidence_sufficiency"] = evidence_sufficiency
    if evidence_sufficiency["starved"]:
        import warnings as _warnings
        _warnings.warn(
            f"EVIDENCE-STARVED run: as_of supplied but 0 effective observations reached the posterior — "
            f"forecasting from priors, not evidence: {question[:70]!r}", stacklevel=1)
    return evidence_sufficiency


def _grounded_fallback_mean(posterior, prior_spec):
    """The grounded fallback probability: the posterior mean if evidence updated it, else the (now
    grounded) prior mean. Used ONLY by the last-resort no-silent-None guard when the rollout produces
    no bound outcome. Returns None when neither source exists."""
    try:
        if posterior is not None and int(getattr(posterior, "n_effective_observations", 0) or 0) > 0:
            return float(getattr(posterior, "outcome_rate_mean"))
        if prior_spec is not None:
            return float(prior_spec.mean)
    except Exception:  # noqa: BLE001
        return None
    return None


def _condition_plan(question, plan, bundle, as_of, horizon, seed, llm, manifest, lineage, costs, drop,
                    user_context=None, intervention="", structural_model_id="", evidence=""):
    """Phases 9/10 + fidelity + activation synthesis + event-time conversion + Phase-11 recompilation for
    ONE plan. Mutates the plan in place (each structural model owns its plan object exclusively)."""
    # ---------- Phase 9: populations + multilayer networks — instantiate into the plan when declared --
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
            ev_text = _bundle_text(bundle, 2400) or str(evidence or "")[:2400]
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
            # MODE-SCOPED (stance(actor, mode)) and the typed process records exist for the
            # trajectory layer. K-pass self-consistency makes the mode set reproducible. §NAP:
            # process grounding is QUALITATIVE — typed {state, waiting_on, basis} records, never
            # a 0-1 progress bar; no capacity resource is invented.
            from swm.world_model_v2.event_time import is_when_question as _is_when
            from swm.world_model_v2.mode_graph import (canonical_modes, declare_typed_processes,
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
                lineage["pathway_processes"] = declare_typed_processes(plan, _modes,
                                                                       grounding=_states)
            # per-actor evidence-grounded intentions (state, not policy guesses) — mode-scoped
            # against the canonical modes when they exist
            lineage["actor_intentions"] = ground_actor_intentions(
                plan, question, criterion=crit, evidence_text=ev_text, llm=llm,
                modes=getattr(plan, "_canonical_modes", None))
            # binary/other questions: the resolution's causal pathways are named by the grounded
            # stances themselves — record their QUALITATIVE typed process state too
            if not getattr(plan, "_declared_pathways", None):
                _st_pws = sorted({str(s.get("pathway")) for s in
                                  (getattr(plan, "_intention_stances", None) or [])
                                  if s.get("pathway") and s.get("pathway") != "any"})
                if _st_pws:
                    _pseudo = [{"id": pw, "pathway": pw} for pw in _st_pws]
                    _states = ground_process_states(question, crit, _st_pws, as_of=as_of,
                                                    evidence_text=ev_text, llm=llm)
                    lineage["pathway_processes"] = declare_typed_processes(plan, _pseudo,
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
            for ph, r in req.items():
                if ph in manifest:
                    manifest[ph]["relevance"] = {"required": r["required"], "why": r["why"]}
        except Exception as e:  # noqa: BLE001 — synthesis must never block the forecast
            lineage["activation_synthesis"] = {"error": f"{type(e).__name__}: {e}"[:160]}

    # ---------- SCENARIO TEMPORAL MODEL: the default-on LLM temporal compilation stage ----------
    # Replaces the quarantined periodic scheduler (legacy_ablations): the LLM generates the
    # scenario's real temporal structure (channels, actor attention, institutional stages,
    # continuous processes, deadlines, sourced recurrences, decision-trigger sources), two
    # independent critics check it, and the runtime executes decisions only on real triggers.
    if "temporal_model" not in drop:
        try:
            from swm.world_model_v2.temporal_compiler import (attach_temporal_model,
                                                              compile_temporal_model)
            ev_text = _bundle_text(bundle, 2000) or str(evidence or "")[:2000]
            tmodel = compile_temporal_model(plan, llm=llm, question=question,
                                            evidence_text=ev_text, user_context=user_context,
                                            intervention=intervention, seed=seed,
                                            structural_model_id=structural_model_id)
            lineage["temporal_model"] = attach_temporal_model(plan, tmodel)
            costs["llm_calls"] += len(tmodel.compilation_trace)
        except Exception as e:  # noqa: BLE001 — compilation failure is a LOUD degradation
            lineage["temporal_model"] = {"error": f"{type(e).__name__}: {e}"[:200],
                                         "degraded": "temporal_compilation_failed"}

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
                convert_binary_to_event_time(plan, crit, lineage=lineage, llm=llm)
        except Exception as e:  # noqa: BLE001 — conversion must never block the forecast
            lineage["event_time"] = {"error": f"{type(e).__name__}: {e}"[:160]}

    # ---------- Phase 11: dynamic-recompilation loop over the as-of observations (same plan lineage) --------
    if "phase11_recompilation" not in drop and bundle is not None:
        _run_recompilation(plan, bundle, as_of, horizon, seed, llm, manifest, lineage, costs)
    else:
        manifest["phase11_recompilation"].update(
            omitted=True, reason=("dropped_by_policy" if "phase11_recompilation" in drop
                                  else "no observations"))


def _no_forecast(r) -> bool:
    """True when a result carries NO usable probability/distribution — the condition that triggers the
    rollout retry and, if that fails too, the no-silent-None guard. Honest §NAP refusals are NOT
    'no forecast' in the retry sense: an `unresolved`/`partially_resolved` status is a deliberate,
    structured epistemic outcome (bounds + named missing mechanisms), not an accidental empty rollout —
    retrying or overwriting it would fabricate certainty the model refused to fabricate."""
    if getattr(r, "simulation_status", "") in ("unresolved", "partially_resolved"):
        return False
    return (not r.has_forecast()) or (r.raw_probability is None and r.calibrated_probability is None)


def _project_terminal_guarded(question, plan, as_of, horizon, intervention, seed, llm, user_context,
                              prior_checkpoint, manifest, drop, t0, lineage):
    """ROLLOUT RETRY around the terminal projection: the persistence-aware rollout is stochastic; an
    INTERMITTENT empty/failed rollout (observed: BoJ once ran 47 calls with an empty operator census →
    no absorber → no forecast, and the next run of the SAME plan ran the full 266-call rollout to 0.73)
    recovers on a re-roll. Retry ONCE with a fresh seed before falling back — this recovers the real
    simulation instead of settling for the prior. Records the first failure so a persistent
    (non-transient) cause stays diagnosable. Honest unresolved results are never retried (see
    `_no_forecast`)."""
    res = _project_terminal(question, plan, as_of, horizon, intervention, seed, llm, user_context,
                            prior_checkpoint, manifest, drop, t0)
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
    return res


def _manifest_row(res, bucket: str, row: dict):
    """Append one late numeric-provenance row to the result's `numeric_causal_inputs` manifest
    (the projection-time merge has already run when the guards fire)."""
    try:
        rr = res.resolution_report if isinstance(getattr(res, "resolution_report", None), dict) else None
        if rr is None:
            return
        man = rr.setdefault("numeric_causal_inputs", {})
        man.setdefault(bucket, []).append(dict(row, n_occurrences=1))
        man["n_inputs"] = int(man.get("n_inputs", 0) or 0) + 1
        if bucket == "rejected":
            man["n_rejected"] = int(man.get("n_rejected", 0) or 0) + 1
    except Exception:  # noqa: BLE001
        pass


def _apply_result_guards(res, *, posterior=None, prior_spec=None, evidence_sufficiency=None,
                         lineage=None, plan=None):
    """Post-projection honesty guards shared by BOTH structural modes:

    1. surface evidence sufficiency on the result (provenance + a loud limitation when starved), so a
       prior-driven forecast is never mistaken for an evidence-driven one;
    2. NO-SILENT-NONE GUARANTEE (§NAP-composed): a COHERENT question must never return a silent
       nothing — not on an empty rollout NOR on an execution_failed exception the retry did not
       recover. The fallback ladder is provenance-gated:
         * evidence-updated POSTERIOR mean exists → EXECUTION-DEGRADED forecast from it (a
           deterministic derivation of ledger-registered evidence — recorded in the manifest);
         * no posterior, generic prior EXPLICITLY allowed (§28 door: SWM_ALLOW_GENERIC_PRIOR /
           policy) → a deliberately PRIOR-DRIVEN forecast, loudly labelled as such;
         * no posterior, prior NOT allowed → an explicit `unresolved` result (§NAP): the missing
           mechanism is named, the grounded prior mean rides ONLY as a labelled non-headline
           diagnostic (`prior_driven_reference`, registered REJECTED in the manifest), and no
           probability is manufactured.
       Honest §NAP refusals (`unresolved` / `partially_resolved`) pass through untouched: they are
       structured epistemic outcomes, not silent Nones."""
    try:
        if evidence_sufficiency is not None:
            res.provenance["evidence_sufficiency"] = evidence_sufficiency
            if evidence_sufficiency.get("starved"):
                res.limitations = (list(res.limitations or []) + [
                    "EVIDENCE-STARVED: 0 as-of observations reached the posterior — this forecast is "
                    "prior-driven, not evidence-driven (retrieval returned nothing usable even after "
                    "retry)"])
    except Exception:  # noqa: BLE001 — telemetry attach must never break the forecast
        pass
    try:
        no_p = res.raw_probability is None and res.calibrated_probability is None
        if res.simulation_status in ("unresolved", "partially_resolved") and no_p:
            # FORECAST-AVAILABILITY CONTRACT: an honest refusal keeps its STATUS (never retried,
            # never re-labeled completed) — but if a defensible probability source exists it is
            # served, labeled. Recovery here covers refusals constructed upstream of
            # result_from_run's own recovery pass.
            from swm.world_model_v2.forecast_recovery import attach_recovery, recover_forecast
            fb0 = _grounded_fallback_mean(posterior, prior_spec)
            has_post = (posterior is not None
                        and int(getattr(posterior, "n_effective_observations", 0) or 0) > 0)
            if fb0 is not None:
                rec0 = recover_forecast(
                    distribution=dict(res.raw_distribution or {}), options=None,
                    unresolved_mass=None,
                    posterior_mean=(float(fb0) if has_post else None),
                    posterior_n_eff=(1 if has_post else 0),
                    prior_mean=(None if has_post else float(fb0)),
                    prior_source_class=str(getattr(prior_spec, "source_class", "") or ""))
                attach_recovery(res, rec0, override_probability=True)
            return res
        if res.simulation_status == "clarification_required" or not no_p:
            return res
        orig_status = res.simulation_status
        has_posterior = (posterior is not None
                         and int(getattr(posterior, "n_effective_observations", 0) or 0) > 0)
        fb = _grounded_fallback_mean(posterior, prior_spec)
        from swm.world_model_v2.fallback import generic_prior_allowed
        if has_posterior and fb is not None:
            res.raw_probability = round(float(fb), 4)
            res.simulation_status = "completed_with_degradation"  # now carries a forecast
            res.limitations = (list(res.limitations or []) + [
                f"EXECUTION-DEGRADED (was {orig_status!r}): the rollout produced no bound outcome "
                f"even after retry; forecast falls back to the evidence-updated posterior mean "
                f"({fb:.3f}) rather than returning None"])
            res.provenance["execution_degraded_fallback"] = {
                "used": True, "value": round(float(fb), 4), "source": "posterior",
                "original_status": orig_status}
            _manifest_row(res, "approved_and_consumed", {
                "name": "execution_degraded_fallback_mean", "value": round(float(fb), 4),
                "units": "probability", "causal_role": "evidence-updated posterior mean served as "
                "the execution-degraded fallback forecast", "source_class": "derived_deterministic",
                "consumer": "unified_runtime._apply_result_guards",
                "evidence_id": "phase3_posterior", "production_eligible": True, "consumed": True})
        elif fb is not None and generic_prior_allowed():
            res.raw_probability = round(float(fb), 4)
            res.simulation_status = "completed_with_degradation"
            res.limitations = (list(res.limitations or []) + [
                f"PRIOR-DRIVEN forecast (was {orig_status!r}): the rollout produced no bound outcome; "
                f"the generic-prior door is EXPLICITLY open (SWM_ALLOW_GENERIC_PRIOR/policy) so the "
                f"grounded prior mean ({fb:.3f}) is served, labelled — this is NOT a simulated "
                f"structural outcome"])
            res.provenance["execution_degraded_fallback"] = {
                "used": True, "value": round(float(fb), 4), "source": "prior_explicitly_allowed",
                "original_status": orig_status}
        else:
            # FORECAST AVAILABILITY ≠ GROUNDING (forecast_recovery contract): the honest status
            # is UNRESOLVED with the missing mechanism named — and the best defensible
            # probability still ships, labeled by its source and grade. The grounded prior mean
            # serves as `grounded_reference_prior` (grade exploratory) or, when its provenance
            # is lean/llm-estimated, as `exploratory_model_estimate` (grade ungrounded). Nothing
            # here can produce a neutral-default 0.5; with NO prior of any kind the probability
            # honestly stays None.
            rr = res.resolution_report if isinstance(getattr(res, "resolution_report", None), dict) \
                else {}
            missing = list(rr.get("missing_mechanisms") or [])
            missing.append({"mechanism": "rollout_execution",
                            "why": f"terminal projection produced no bound outcome even after retry "
                                   f"(was {orig_status!r}); no evidence-updated posterior exists to "
                                   f"ground a degraded fallback",
                            "missing": "recoverable rollout or evidence-updated posterior"})
            rr.update({"unresolved_share": rr.get("unresolved_share") or 1.0,
                       "missing_mechanisms": missing,
                       "note": rr.get("note") or
                       "the headline probability (when present) is a labeled prior-driven "
                       "estimate — see grounding_grade/probability_source; unresolved execution "
                       "is fully disclosed"})
            res.resolution_report = rr
            res.simulation_status = "unresolved"
            if getattr(res, "recommendation_status", "") not in ("", "not_requested"):
                res.recommendation_status = "withheld"      # unresolved mass still gates actions
            if res.support_grade not in ("empirically_supported", "transfer_supported",
                                         "exploratory", "highly_speculative"):
                res.support_grade = "highly_speculative"
            if fb is not None:
                from swm.world_model_v2.forecast_recovery import attach_recovery, recover_forecast
                src_class = str(getattr(prior_spec, "source_class", "") or "")
                rec = recover_forecast(distribution={}, options=None, unresolved_mass=1.0,
                                       posterior_mean=None, posterior_n_eff=0,
                                       prior_mean=float(fb), prior_source_class=src_class)
                attach_recovery(res, rec, override_probability=True)
                res.provenance["prior_driven_reference"] = {
                    "value": round(float(fb), 4), "source": "grounded_prior_mean",
                    "headline": res.raw_probability is not None,
                    "note": "served as the labeled headline probability under the "
                            "forecast-availability contract (see forecast_recovery provenance); "
                            "grounding_grade discloses its weakness"}
                _manifest_row(res, "approved_and_consumed", {
                    "name": "prior_driven_reference", "value": round(float(fb), 4),
                    "units": "probability", "causal_role": "grounded prior mean served as the "
                    "labeled headline probability for an unresolved execution",
                    "source_class": src_class or "llm_estimated",
                    "consumer": "unified_runtime._apply_result_guards",
                    "production_eligible": True, "consumed": True})
                res.limitations = (list(res.limitations or []) + [
                    "Execution unresolved under the current model (missing mechanisms named in "
                    "resolution_report); the headline probability is a labeled "
                    f"{res.probability_source or 'prior'}-driven estimate "
                    f"(grounding_grade={res.grounding_grade or 'exploratory'}), not a simulated "
                    "structural outcome"])
            else:
                res.limitations = (list(res.limitations or []) + [
                    "Outcome unresolved under the current model and NO defensible probability "
                    "source exists (no resolved mass, no evidence-updated posterior, no grounded "
                    "prior) — probability honestly stays None; no neutral default was invented"])
    except Exception:  # noqa: BLE001 — the guard must never itself break the forecast
        pass
    return res


def _attach_supervision(res, plan, as_of, bundle, manifest, lineage):
    """MANDATORY PHASE SUPERVISION: one PhaseExecutionRecord per phase, every run. The manifest is
    DERIVED from these records; a relevant phase that did not execute is a recorded integration failure
    (blocked_*) that lowers support — a phase can never disappear silently."""
    if not res.has_forecast():
        return
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


def _route_individual_reaction(question, user_context, llm, as_of, seed, t0):
    """SINGLE-FRAME individual-reaction route — used ONLY by the explicit single_structural_model
    ablation. The DEFAULT ensemble path routes personal reactions through
    structural_runtime._route_individual_reaction_ensemble, which simulates several plausible causal
    models of the reaction (relationship / attention / interpretation / obligations …) rather than one
    unchallenged frame. Returns a SimulationResult, or None (ordinary compiled route). The caller's
    context — who the person is, the relationship, the history, optionally the exact stimulus — is
    preserved verbatim into the person's actor-local world; the target is automatically Tier 1; the
    answer is the counted (and externally calibrated or `unvalidated`) distribution over the person's
    simulated observable responses, with the full per-sample artifact and the actor-policy report in
    provenance."""
    from swm.world_model_v2.actor_selection import is_individual_reaction_question
    person = (user_context or {}).get("individual") if isinstance(user_context, dict) else None
    if not isinstance(person, dict) or not is_individual_reaction_question(question):
        return None
    report = {"requested_actor_policy_mode": "hybrid_relevant_actor_policy",
              "actual_actor_policy_mode": "persistent_qualitative_llm_policy",
              "route": "individual_reaction", "reason": "reaction_is_the_question",
              "degraded": False, "construction_error": ""}
    if llm is None:
        return SimulationResult(
            question=question, simulation_status="execution_failed",
            failure_taxonomy="missing_required_operator",
            limitations=["individual-reaction route requires an LLM backend; none supplied"],
            provenance={"actor_policy_report": {**report, "degraded": True,
                                                "actual_actor_policy_mode": "none",
                                                "construction_error": "no_llm_backend"}},
            latency_s=round(_time.time() - t0, 3))
    try:
        from swm.world_model_v2.compiler import parse_time
        from swm.world_model_v2.individual_reaction import simulate_individual_reaction
        now = parse_time(as_of) if as_of else _time.time()
        artifact = simulate_individual_reaction(
            person_id=str(person.get("person_id", person.get("name", "the_person"))),
            stimulus=str(person.get("stimulus", question))[:800],
            context=person, llm=llm,
            n_hypotheses=int(person.get("n_hypotheses", 3) or 3),
            samples_per_hypothesis=int(person.get("samples_per_hypothesis", 2) or 2),
            seed=seed, as_of=now)
        dist = dict(artifact["raw_qualitative_simulation_distribution"])
        calibrated = (dict(artifact["calibrated_distribution"])
                      if artifact["calibration_status"] == "calibrated" else None)
        fallbacks = int(artifact.get("n_excluded_numeric_fallbacks", 0))
        return SimulationResult(
            question=question, simulation_status="completed", support_grade="exploratory",
            raw_distribution=dist, calibrated_distribution=calibrated,
            limitations=([] if artifact["calibration_status"] == "calibrated" else
                         ["reaction distribution is counted from qualitative simulations and "
                          "is unvalidated (no fitted calibrator for this person/role)"]),
            provenance={"runtime": RUNTIME_VERSION, "route": "individual_reaction",
                        "structural_mode": "single_structural_model",
                        "actor_policy_report": {
                            **report, "actors_routed_qualitatively": [artifact["person_id"]],
                            "actors_routed_numerically": [], "fallbacks": fallbacks,
                            "fallback_reasons": ([{"actor_and_reason":
                                                   f"{artifact['person_id']}:llm_failed", "n": fallbacks}]
                                                 if fallbacks else [])},
                        "individual_reaction": artifact},
            cost_usd=0.0, latency_s=round(_time.time() - t0, 3))
    except Exception as e:  # noqa: BLE001 — LOUD failure, never a silent fall-through
        return SimulationResult(
            question=question, simulation_status="execution_failed",
            failure_taxonomy="runtime_exception",
            limitations=[f"individual-reaction route failed: {type(e).__name__}: {e}"[:200]],
            provenance={"actor_policy_report": {**report, "degraded": True,
                                                "construction_error": f"{type(e).__name__}: {e}"[:200]}},
            latency_s=round(_time.time() - t0, 3))


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
                      prior_checkpoint, manifest, drop, t0, particle_plan: dict = None,
                      particle_scope=None):
    """Terminal projection through the single funnel: Phase-8 persistence-aware rollout (which fires the
    Phase-4 actor-policy, Phase-6/7 registry, and Phase-10 institution operators the plan names). Phase 8 is
    default-on; when no history/checkpoint exists it runs the ordinary rollout with broad priors.

    `particle_plan` (structural-ensemble progressive simulation): {"n_total", "start", "stop"} rolls a
    deterministic index-keyed slice through the SAME canonical funnel; used with a shared handle by the
    ensemble runtime (see structural_runtime) so pilot particles become a reusable prefix of the full run."""
    from swm.world_model_v2.phase8_pipeline import run_with_persistence
    persistence_dropped = "phase8_persistence" in drop
    # COMPUTE knob (§26): an explicit particle budget prioritizes Monte-Carlo resolution under
    # a caller's compute budget — it never truncates causal chains, drops actors, or shortens
    # the horizon (those produce temporally_truncated records instead). Recorded on the plan.
    n_particles = None
    policy = drop and {} or {}
    if isinstance(user_context, dict) and isinstance(user_context.get("_execution_policy"),
                                                     dict):
        policy = user_context["_execution_policy"]
    np_req = policy.get("n_particles")
    if isinstance(np_req, (int, float)) and np_req >= 1:
        n_particles = int(np_req)
        plan.compute_plan["n_particles"] = n_particles
        plan.provenance = {**(plan.provenance or {}),
                           "n_particles_override": {"value": n_particles,
                                                    "reason": "caller compute budget (§26); "
                                                              "MC resolution only"}}
    try:
        if persistence_dropped:
            from swm.world_model_v2.materialize import run_from_plan
            from swm.world_model_v2.pipeline import result_from_run
            from swm.world_model_v2.phase8_pipeline import (
                _surface_actor_policy_degradation, _surface_consequence_degradation)
            result, branches = run_from_plan(plan, llm=llm, seed=seed,
                                             n_particles=n_particles)
            res = result_from_run(question, plan, result, branches, intervention=intervention, t0=t0)
            res.provenance = {**(res.provenance or {}),
                              "actor_policy_report": result.get("actor_policy_report", {}),
                              "consequence_report": result.get("consequence_report", {})}
            if result.get("actor_decision_distributions"):
                res.provenance["actor_decision_distributions"] = \
                    result["actor_decision_distributions"]
            _surface_actor_policy_degradation(res, result.get("actor_policy_report", {}))
            _surface_consequence_degradation(res, result.get("consequence_report", {}))
            manifest["phase8_persistence"].update(omitted=True, reason="dropped_by_policy")
        else:
            res, _p8meta = run_with_persistence(question, plan, llm=llm, context=user_context,
                                                actor_history=(prior_checkpoint or {}).get("actor_history") if prior_checkpoint else None,
                                                intervention=intervention, t0=t0, seed=seed,
                                                n_particles=n_particles)
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
