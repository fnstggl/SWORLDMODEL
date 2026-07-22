"""QUARANTINED LEGACY ENTRY — Phase 9 is served by `unified_runtime.simulate_world`; do not call this
module's entry directly for new work (see docs/WMV2_CANONICAL_PATH.md).

Integrated population + multilayer-network simulation — Phase 9 (Parts G, L, T, U + no-abstention).

The end-to-end path across all five planes:

  scenario (populations + candidate graph + typed evidence)
   → POSTERIOR: population compositional posterior (Phase-3 conjugate) + per-edge existence posteriors
     (Phase-3 log-odds) + SBM community posterior + graph structural posterior
   → WORLD-STATE: posterior-weighted particles (composition × sampled graph)
   → EXECUTION: typed multilayer mechanisms (diffusion/exposure/authority) produce StateDeltas + a terminal
     adoption distribution with population + graph uncertainty propagated
   → RESULT: terminal distribution, uncertainty decomposition, support grade, forensic provenance.

No-abstention contract (Phase 1, preserved): weak/absent graph or survey evidence WIDENS posteriors and lowers
the support grade — it never blocks the forecast. Only a genuine engineering failure raises (taxonomy-labeled).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from swm.world_model_v2.phase3_posterior import infer_compositional_posterior
from swm.world_model_v2.phase9_execution import materialize_worlds, simulate_multilayer
from swm.world_model_v2.phase9_network import (graph_structural_posterior, infer_communities,
                                               infer_network_edges)
from swm.world_model_v2.phase9_population import materialize_population_particles


def simulate_with_populations_networks(question: str, *, llm=None, as_of: str, horizon: str,
                                       intervention: str = "", seed: int = 0, config=None,
                                       user_documents=None, prior_world=None, n_particles: int = 40):
    """UNIVERSAL production entry (Part 2): caller supplies ONLY a question + as-of + horizon (+ optional user
    facts). The path compiles, gathers Phase-2 evidence, AUTOMATICALLY DISCOVERS the population/network slice,
    constructs typed observations from the evidence, infers all posteriors through Phase 3, materializes and
    executes — no segments/edges/hypotheses/susceptibility/seeds supplied by the caller. No-abstention preserved.

    Returns (Phase9Result, artifacts) where artifacts carries the discovery plan, evidence bundle, observation
    set and plan hashes for a full forensic trace."""
    import time as _time
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.evidence_orchestrator import gather_evidence
    from swm.world_model_v2.evidence_requirements import requirements_from_plan
    from swm.world_model_v2.phase9_discovery import construct_observations, discover
    from swm.world_model_v2.result import ClarificationRequired, CompilerExecutionError
    t0 = _time.time()
    # ---- compile (no-abstention) ----
    try:
        plan = compile_world(question, llm=llm, evidence="", as_of=as_of, horizon=horizon,
                             intervention=intervention, seed=seed)
    except ClarificationRequired as e:
        return (Phase9Result(simulation_status="clarification_required",
                             limitations=[str(e)[:160]]), {"stage": "compile"})
    except CompilerExecutionError as e:
        return (Phase9Result(simulation_status="execution_failed", provenance={"failure_taxonomy": e.taxonomy}),
                {"stage": "compile"})
    # ---- Phase-2 evidence ----
    bundle = None
    try:
        iso = _time.strftime("%Y-%m-%d", _time.gmtime(__import__("swm.world_model_v2.state", fromlist=["parse_time"]).parse_time(as_of)))
        reqs = requirements_from_plan(plan, as_of_iso=iso, question=question)
        bundle = gather_evidence(question, as_of=as_of, requirements=reqs, llm=llm,
                                 user_documents=user_documents, config=config, plan_hash=plan.plan_hash(),
                                 seed=seed)
    except Exception as e:  # noqa: BLE001 — evidence retrieval is best-effort; weak evidence must not block
        bundle = None
    # ---- AUTOMATIC discovery + typed observation construction ----
    discovery = discover(question, plan, bundle, llm=llm)
    survey_obs, edge_obs = construct_observations(discovery, bundle, llm=llm)
    # susceptibility: a BROAD prior per segment (0.3) — not caller-supplied, not LLM-minted; a documented
    # weakly-informative default the evidence could later refine.
    susc = {s: 0.3 for s in discovery.population_segments} or {"all": 0.3}
    res = simulate_populations_networks(
        segments=discovery.population_segments or None, survey_observations=survey_obs,
        candidate_edges=discovery.candidate_edges, edge_observations=edge_obs,
        structural_hypotheses=discovery.structural_hypotheses,
        segment_susceptibility=susc, seeds=discovery.seeds or None, contagion="simple",
        n_particles=n_particles, seed=seed)
    if res.simulation_status == "completed" and (not discovery.relevant):
        res.simulation_status = "completed_with_degradation"
        res.limitations.append("populations/networks judged low-relevance for this question")
    res.provenance["discovery_hash"] = _hash(discovery.as_dict())
    res.provenance["plan_hash"] = plan.plan_hash()
    res.provenance["evidence_bundle_hash"] = bundle.bundle_hash() if bundle else ""
    res.provenance["universal_path"] = True
    res.forensic["discovery"] = discovery.as_dict()
    res.forensic["n_edge_observations"] = len(edge_obs)
    res.forensic["latency_s"] = round(_time.time() - t0, 3)
    return res, {"discovery": discovery, "bundle": bundle, "edge_observations": edge_obs,
                 "survey_observations": survey_obs, "plan_hash": plan.plan_hash(),
                 "evidence_bundle_hash": bundle.bundle_hash() if bundle else ""}


@dataclass
class Phase9Result:
    simulation_status: str = "completed"
    support_grade: str = "exploratory"
    terminal: dict = field(default_factory=dict)
    population_posterior: dict = field(default_factory=dict)
    graph_posterior: dict = field(default_factory=dict)
    structural_posterior: dict = field(default_factory=dict)
    community_posterior: dict = field(default_factory=dict)
    uncertainty_decomposition: dict = field(default_factory=dict)
    limitations: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    forensic: dict = field(default_factory=dict)

    def as_dict(self):
        return self.__dict__.copy()


def _support_grade(edges, n_survey_obs, n_effective_pop) -> str:
    """Weak evidence lowers the grade, never blocks (no-abstention). Grade from the fraction of the graph that
    is OBSERVED vs hypothesized and the amount of survey data."""
    if not edges:
        obs_frac = 0.0
    else:
        obs_frac = sum(1 for e in edges if e.observed_status == "observed") / len(edges)
    if obs_frac >= 0.5 and n_survey_obs >= 1:
        return "transfer_supported"
    if obs_frac >= 0.2 or n_survey_obs >= 1:
        return "exploratory"
    return "highly_speculative"


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def simulate_populations_networks(*, segments=None, prior_alpha=None, survey_observations=None,
                                  candidate_edges=None, edge_observations=None, layer_priors=None,
                                  structural_hypotheses=None, segment_susceptibility=None, seeds=None,
                                  contagion="simple", community_K=2, n_particles=40, seed=0) -> Phase9Result:
    """Run the integrated Phase-9 posterior→materialize→execute path. All numeric posteriors come from Phase-3
    inference; the caller supplies typed evidence (survey counts + edge observations), never probabilities."""
    res = Phase9Result()
    survey_observations = survey_observations or []
    edge_observations = edge_observations or []
    candidate_edges = candidate_edges or []

    # ---- POSTERIOR: population composition ----
    pop_particles, comp = [], None
    if segments:
        a0 = prior_alpha or [1.0] * len(segments)
        comp = infer_compositional_posterior(segments, a0, survey_observations, seed=seed,
                                             n_particles=max(60, n_particles))
        res.population_posterior = {"segments": segments, "mean": {s: round(comp.posterior_mean[i], 4)
                                    for i, s in enumerate(segments)},
                                    "sd": {s: round(comp.posterior_sd[i], 4) for i, s in enumerate(segments)},
                                    "n_effective_observations": comp.n_effective_observations,
                                    "assimilation_ledger": comp.assimilation_ledger}
        from swm.world_model_v2.phase9_population import infer_segment_rates
        # a uniform behavior rate posterior per segment (broad) unless caller gives susceptibilities
        trait_rates = {"susceptibility": infer_segment_rates(
            {s: (int((segment_susceptibility or {}).get(s, 0.3) * 20), 20) for s in segments})}
        pop_particles = materialize_population_particles(comp, trait_rates, n=n_particles, seed=seed)

    # ---- POSTERIOR: per-edge existence + communities + structure ----
    edges = infer_network_edges(candidate_edges, edge_observations, layer_priors=layer_priors)
    res.graph_posterior = {"n_candidate_edges": len(candidate_edges), "n_edges_inferred": len(edges),
                           "edges": [e.as_dict() for e in edges[:40]],
                           "n_observed": sum(1 for e in edges if e.observed_status == "observed"),
                           "n_hypothesized": sum(1 for e in edges if e.observed_status == "hypothesized")}
    node_ids = sorted({e.src for e in edges} | {e.dst for e in edges})
    # concrete adjacency from the posterior-mean graph (edges above 0.5) for SBM + structure
    adj = {(e.src, e.dst) for e in edges if e.existence_p >= 0.5}
    if len(node_ids) >= community_K and adj:
        comm = infer_communities(node_ids, adj, community_K, seed=seed)
        res.community_posterior = {"K": community_K, "block_matrix": comm["block_matrix"],
                                   "membership_sample": dict(list(comm["membership"].items())[:8])}
    if structural_hypotheses and len(node_ids) >= 2:
        sp = graph_structural_posterior(node_ids, adj, structural_hypotheses, seed=seed)
        res.structural_posterior = sp

    # ---- WORLD-STATE + EXECUTION ----
    if not edges:
        # no graph at all → still simulate (no-abstention): a population-only aggregate with wide uncertainty
        res.simulation_status = "completed_with_degradation"
        res.limitations.append("no candidate network — population-only aggregate; graph uncertainty maximal")
        if pop_particles:
            from swm.world_model_v2.phase9_population import poststratified_estimate
            est = poststratified_estimate(pop_particles, "susceptibility")
            res.terminal = {"terminal_mean": est["mean"], "terminal_sd": est["sd"], "n_particles": len(pop_particles),
                            "n_deltas": 0, "basis": "population_only_no_graph"}
    else:
        term = simulate_multilayer(pop_particles or None, edges, communities=res.community_posterior,
                                   segment_susceptibility=segment_susceptibility, seeds=seeds,
                                   contagion=contagion, n_particles=n_particles, seed=seed)
        res.terminal = term
        res.simulation_status = "completed"

    # ---- grade + decomposition + provenance ----
    n_eff_pop = comp.n_effective_observations if comp else 0
    res.support_grade = _support_grade(edges, len(survey_observations), n_eff_pop)
    if res.support_grade in ("exploratory", "highly_speculative"):
        res.limitations.append(f"support grade {res.support_grade}: "
                               f"{res.graph_posterior['n_hypothesized']} hypothesized edges, "
                               f"{n_eff_pop} effective survey observations")
    res.uncertainty_decomposition = {
        "terminal_sd": res.terminal.get("terminal_sd"),
        "graph_edges_observed_frac": round(res.graph_posterior["n_observed"] /
                                           max(1, len(edges)), 3) if edges else 0.0,
        "population_effective_obs": n_eff_pop}
    res.provenance = {
        "population_posterior_hash": _hash(res.population_posterior.get("mean", {})),
        "graph_posterior_hash": _hash([e.as_dict() for e in edges]),
        "structural_posterior_hash": _hash(res.structural_posterior.get("posterior", {})),
        "terminal_hash": _hash(res.terminal), "seed": seed,
        "llm_role": "may propose segments/nodes/edges/hypotheses; NO probabilities minted here"}
    res.forensic = {
        "n_agents": len(node_ids), "n_edges_realized_mean": res.terminal.get("mean_edges_per_world"),
        "n_deltas": res.terminal.get("n_deltas"),
        "observed_status_breakdown": {"observed": res.graph_posterior["n_observed"],
                                      "hypothesized": res.graph_posterior["n_hypothesized"]}}
    return res
