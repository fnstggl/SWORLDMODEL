"""Posterior-conditioned simulation entry — Phase 3.

The full production path, extending the Phase-2 evidence pipeline with a REAL numeric posterior over hidden
world-state that is CAUSALLY CONSUMED by the simulator:

  question → compile → typed evidence requirements → live retrieval → verified as-of bundle (Phase 2)
    → LLM QUALITATIVE claim tags (semantic mapping only) → registered observation models → dependence-
      corrected likelihood → PARTICLE POSTERIOR over outcome-rate + STRUCTURAL posterior over competing
      causal structures (Phase 3) → posterior particles materialized onto the resolve_outcome event and the
      structural strata → rollout draws each terminal from the POSTERIOR (not a broad prior) → terminal
      distribution with propagated posterior uncertainty → SimulationResult.

Plane discipline (the five planes the spec demands be kept distinct):
  CODE        the modules here + phase3_observation/phase3_posterior
  EVIDENCE    the Phase-2 immutable bundle (verified, as-of, dependence/contradiction-annotated)
  POSTERIOR   PosteriorResult — numeric, likelihood-updated, reproducible; NOT written by the LLM
  WORLD-STATE revised.posterior_rate_particles / revised.structural_posterior on the plan
  EXECUTION   the resolve_outcome event payload the GenericOutcomeOperator actually reads per particle

A posterior that stops at the POSTERIOR plane is scaffolding; a posterior stored on WORLD-STATE but never
read is ornamental. This module carries it all the way into EXECUTION and records where it crossed each
boundary (artifacts["planes"]).

No-abstention contract preserved: weak/contradictory/absent evidence WIDENS the posterior and lowers the
support grade; it never blocks the forecast.
"""
from __future__ import annotations

import hashlib
import json
import time as _time

from swm.world_model_v2.compiler import compile_world
from swm.world_model_v2.evidence_materialize import attach_evidence_observations
from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig, gather_evidence
from swm.world_model_v2.evidence_recompile import recompile_with_evidence
from swm.world_model_v2.evidence_requirements import requirements_from_plan
from swm.world_model_v2.phase3_latent_spec import outcome_rate_spec, structural_spec, tag_claims
from swm.world_model_v2.phase3_posterior import infer_posterior
from swm.world_model_v2.phase3_priors import build_outcome_rate_prior
from swm.world_model_v2.pipeline import result_from_run
from swm.world_model_v2.result import ClarificationRequired, CompilerExecutionError, SimulationResult


def _posterior_hash(posterior) -> str:
    """Deterministic content hash of the consumed posterior (rate particles rounded + structural mass), so a
    reviewer can prove the SAME evidence reproduces the SAME posterior and terminal."""
    payload = json.dumps({
        "rate": [[round(r, 5), round(w, 6)] for r, w in posterior.outcome_rate_particles],
        "structural": posterior.structural_posterior,
        "prior_mean": posterior.outcome_rate_prior_mean}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def simulate_with_posterior(question: str, *, llm, as_of: str, horizon: str, intervention: str = "",
                            seed: int = 0, config: OrchestratorConfig | None = None,
                            user_documents: list | None = None, dataset_path: str = "",
                            prior_bundle_path: str = "", store=None, n_rate_particles: int = 400,
                            use_dependence: bool = True, use_structural: bool = True,
                            consume_posterior: bool = True, reference_data: dict = None,
                            use_reference_prior: bool = True, bundle=None, tags=None, plan=None) -> tuple:
    """Run the posterior-conditioned simulation. Returns (SimulationResult, artifacts).

    `consume_posterior=False` is the ABLATION arm: the posterior is still computed and reported, but NOT
    materialized onto the plan, so the terminal falls back to the broad lean-Beta prior. Comparing the two
    arms isolates the causal effect of the posterior on the terminal (Part P: posterior-ignored ablation).

    `use_dependence` / `use_structural` toggle the dependence correction and structural updating for their
    own ablations. Deterministic under `seed`."""
    t0 = _time.time()
    planes = {}

    # ---- CODE + compile (no-abstention). A pre-built `plan` may be supplied so the ablation/reproducibility
    #      arms share the EXACT same compiled world — isolating the (stochastic) LLM compile step so the numeric
    #      posterior pipeline is byte-reproducible given fixed plan+bundle+tags+seed. ----
    if plan is None:
        try:
            plan = compile_world(question, llm=llm, evidence="", as_of=as_of, horizon=horizon,
                                 intervention=intervention, seed=seed)
        except ClarificationRequired as e:
            return (SimulationResult(question=question, simulation_status="clarification_required",
                                     clarification_reason=str(e), latency_s=round(_time.time() - t0, 3)),
                    {"stage": "compile"})
        except CompilerExecutionError as e:
            return (SimulationResult(question=question, simulation_status="execution_failed",
                                     failure_taxonomy=e.taxonomy, latency_s=round(_time.time() - t0, 3)),
                    {"stage": "compile"})

    # ---- EVIDENCE: typed requirements → live retrieval → verified as-of bundle (Phase 2). A pre-gathered
    #      `bundle` may be supplied to (a) run the posterior-consumed and posterior-ignored ABLATION arms on
    #      the SAME evidence, and (b) avoid a second live retrieval. Reproducibility of the posterior given a
    #      fixed bundle is then exact. ----
    if bundle is None:
        reqs = requirements_from_plan(plan, as_of_iso=_iso(as_of), question=question)
        bundle = gather_evidence(question, as_of=as_of, requirements=reqs, llm=llm,
                                 user_documents=user_documents, dataset_path=dataset_path,
                                 prior_bundle_path=prior_bundle_path, config=config,
                                 plan_hash=plan.plan_hash(), seed=seed, store=store)
    planes["evidence"] = {"bundle_hash": bundle.bundle_hash(),
                          "n_included_claims": len(bundle.included_claim_ids),
                          "n_documents": len(bundle.documents)}

    # ---- LLM SEMANTIC MAPPING ONLY: qualitative claim tags (no numbers). Pre-computed `tags` may be supplied
    #      to reuse the EXACT semantic mapping across arms — isolating the (stochastic) LLM tagging step so the
    #      NUMERIC posterior pipeline is exactly reproducible given fixed evidence + tags. ----
    if tags is None:
        tags = tag_claims(question, bundle, plan, llm=llm)

    # ---- PRIOR: reference-class prior with provenance + transport-risk inflation (Part B). LLM names the
    #      reference class + qualitative transport risk; the base rate comes from `reference_data` (DATA) when
    #      supplied, else it falls back to the honestly-labeled generic lean prior. ----
    prior_spec = build_outcome_rate_prior(plan, llm=llm, reference_data=reference_data) \
        if use_reference_prior else None

    # ---- POSTERIOR: registered observation models → dependence-corrected likelihood → particle posterior ----
    # Inferred on the ORIGINAL compiled plan's lean + structural priors (the PRE-evidence prior), so the
    # evidence is assimilated exactly once (not double-counted with any Phase-2 heuristic lean shift).
    posterior = infer_posterior(plan, bundle, tags, n_rate_particles=n_rate_particles, seed=seed,
                                use_dependence=use_dependence, use_structural=use_structural,
                                prior_spec=prior_spec)
    latent_specs = _latent_specs(question, plan, bundle)
    planes["posterior"] = {"posterior_hash": _posterior_hash(posterior),
                           "prior_mean": posterior.outcome_rate_prior_mean,
                           "posterior_mean": posterior.outcome_rate_mean,
                           "posterior_sd": posterior.outcome_rate_sd,
                           "n_effective_observations": posterior.n_effective_observations,
                           "n_claims_collapsed": posterior.n_claims_collapsed,
                           "rate_ess": posterior.rate_ess,
                           "structural_prior": posterior.structural_prior,
                           "structural_posterior": posterior.structural_posterior,
                           "warnings": posterior.warnings}

    # ---- Phase-2 structural grounding (kept for continuity + audit): evidence-conditioned recompile ----
    revised, diff = recompile_with_evidence(plan, bundle, llm=llm, horizon=horizon)
    revised = attach_evidence_observations(revised, bundle)

    # ---- WORLD-STATE materialization: place the posterior where a mechanism will read it ----
    consumed = False
    if consume_posterior and posterior.n_effective_observations > 0:
        revised.posterior_rate_particles = list(posterior.outcome_rate_particles)
        if use_structural and posterior.structural_posterior:
            revised.structural_posterior = dict(posterior.structural_posterior)
        consumed = True
        planes["world_state"] = {"posterior_rate_particles_on_plan": len(posterior.outcome_rate_particles),
                                 "structural_posterior_on_plan": bool(getattr(revised, "structural_posterior",
                                                                              None))}
    else:
        planes["world_state"] = {"posterior_rate_particles_on_plan": 0,
                                 "reason": "consume_posterior=False (ablation)" if not consume_posterior
                                 else "no effective observation updated the posterior — prior-only path"}

    # ---- EXECUTION: rollout draws each terminal from the posterior particles (via run_from_plan injection) ----
    from swm.world_model_v2.materialize import run_from_plan
    try:
        result, branches = run_from_plan(revised, llm=llm, seed=seed)
    except CompilerExecutionError as e:
        return (SimulationResult(question=question, simulation_status="execution_failed",
                                 failure_taxonomy=e.taxonomy, plan_hash=revised.plan_hash(),
                                 latency_s=round(_time.time() - t0, 3)),
                {"bundle": bundle, "posterior": posterior, "stage": "rollout"})
    except Exception as e:  # noqa: BLE001
        return (SimulationResult(question=question, simulation_status="execution_failed",
                                 failure_taxonomy="runtime_exception", limitations=[str(e)[:120]],
                                 latency_s=round(_time.time() - t0, 3)),
                {"bundle": bundle, "posterior": posterior, "stage": "rollout"})
    planes["execution"] = {"rate_source": _rate_source(branches),
                           "structural_source": result.get("structural_source", "n/a"),
                           "n_deltas": result.get("n_deltas")}

    # ---- result + posterior decomposition ----
    res = result_from_run(question, revised, result, branches, intervention=intervention, t0=t0)
    res.posterior_inference = {
        "consumed_by_simulator": consumed,
        "outcome_rate": {"prior_mean": posterior.outcome_rate_prior_mean,
                         "posterior_mean": posterior.outcome_rate_mean,
                         "posterior_sd": posterior.outcome_rate_sd,
                         "shift": round(posterior.outcome_rate_mean - posterior.outcome_rate_prior_mean, 5),
                         "representation": "continuous_probabilistic (bounded [0,1] particle set)",
                         "rate_ess": posterior.rate_ess, "n_particles": n_rate_particles},
        "structural": {"representation": "discrete_structural",
                       "prior": posterior.structural_prior,
                       "posterior": posterior.structural_posterior},
        "n_effective_observations": posterior.n_effective_observations,
        "n_claims_collapsed": posterior.n_claims_collapsed,
        "prior_provenance": posterior.prior_provenance,
        "assimilation_ledger": posterior.assimilation_ledger,
        "diagnostics": posterior.diagnostics,
        "warnings": posterior.warnings,
        "posterior_hash": _posterior_hash(posterior),
        "latent_specs": [s.as_dict() for s in latent_specs]}
    res.provenance["evidence_bundle_hash"] = bundle.bundle_hash()
    res.provenance["posterior_hash"] = _posterior_hash(posterior)
    res.provenance["posterior_consumed"] = consumed
    res.provenance["n_included_claims"] = len(bundle.included_claim_ids)
    res.evidence_quality = f"{len(bundle.included_claim_ids)} included claims → " \
                           f"{posterior.n_effective_observations} effective (dependence-collapsed) observations"
    if consumed:
        res.limitations.append(
            f"posterior-conditioned: outcome-rate {posterior.outcome_rate_prior_mean:.3f}→"
            f"{posterior.outcome_rate_mean:.3f} (±{posterior.outcome_rate_sd:.3f}) from "
            f"{posterior.n_effective_observations} effective observations")
    elif posterior.n_effective_observations == 0:
        res.limitations.append("no admissible as-of evidence updated the posterior — broad-prior forecast")
    res.limitations.extend(posterior.warnings)

    artifacts = {"bundle": bundle, "posterior": posterior, "tags": tags, "plan_diff": diff,
                 "latent_specs": latent_specs, "planes": planes, "plan": plan,
                 "pre_plan_hash": plan.plan_hash(), "post_plan_hash": revised.plan_hash(),
                 "posterior_hash": _posterior_hash(posterior), "posterior_consumed": consumed}
    return res, artifacts


def _latent_specs(question: str, plan, bundle) -> list:
    """The typed latent-variable specs materialized for this scenario (Part A). Both are evidence-linked and
    causally consumed — the anti-ornamental invariant. `.measurable()` is asserted before they are reported."""
    specs = [outcome_rate_spec(question, list(bundle.included_claim_ids))]
    ss = structural_spec(plan)
    if ss is not None:
        specs.append(ss)
    return [s for s in specs if s.measurable()]                # ornamental (unconsumed) specs are dropped


def _rate_source(branches) -> str:
    """Read back what the resolver actually used (posterior vs prior_beta) from the terminal StateDeltas —
    proof the posterior was CONSUMED in execution, not merely stored."""
    for b in branches:
        for d in b.log:
            if d.event_type == "resolve_outcome":
                src = (d.uncertainty or {}).get("rate_source")
                if src:
                    return src
    return "unknown"


def _iso(as_of: str) -> str:
    from swm.world_model_v2.state import parse_time
    return _time.strftime("%Y-%m-%d", _time.gmtime(parse_time(as_of)))
