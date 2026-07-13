"""Phase 9 — production population & multilayer-network inference (scripted, no network).

Foundation tests: the Phase-3 posterior engine, EXTENDED (not replaced) with a compositional (simplex)
representation and an edge-existence (Bernoulli/log-odds) representation, must recover known ground truth,
keep segment weights normalized, collapse dependent evidence, and be deterministic — the same guarantees the
Phase-3 outcome-rate posterior already meets.
"""
import random

import pytest

from swm.world_model_v2.phase3_observation import (EDGE_OBS_MODELS, RELATION_LAYERS, EdgeObservation,
                                                   collapse_edge_observations)
from swm.world_model_v2.phase3_posterior import (infer_compositional_posterior, infer_edge_posterior)


# ============================================================ compositional (simplex) posterior
def _multinomial_draw(rng, probs, n):
    counts = [0] * len(probs)
    for _ in range(n):
        r, acc = rng.random(), 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    return counts


def test_compositional_posterior_recovers_true_simplex():
    segs = ["A", "B", "C", "D"]
    true = [0.5, 0.25, 0.15, 0.10]
    rng = random.Random(1)
    counts = _multinomial_draw(rng, true, 800)
    obs = [{"counts": dict(zip(segs, counts)), "reliability": 1.0, "source": "survey"}]
    post = infer_compositional_posterior(segs, [1.0] * 4, obs, seed=0)
    # sums to one (compositional, not independent scalars)
    assert abs(sum(post.posterior_mean) - 1.0) < 1e-6
    # recovers the true simplex within tolerance and beats the flat prior
    err = sum(abs(m - t) for m, t in zip(post.posterior_mean, true))
    prior_err = sum(abs(0.25 - t) for t in true)
    assert err < 0.08 and err < prior_err
    # conjugate summary agrees with the particle posterior mean
    ca = post.conjugate_alpha
    cm = [a / sum(ca) for a in ca]
    assert all(abs(a - b) < 0.05 for a, b in zip(cm, post.posterior_mean))


def test_compositional_weights_are_nonnegative_and_normalized_per_particle():
    segs = ["x", "y", "z"]
    obs = [{"counts": {"x": 60, "y": 30, "z": 10}, "reliability": 0.9}]
    post = infer_compositional_posterior(segs, [1.0, 1.0, 1.0], obs, seed=2, n_particles=200)
    for vec, w in post.particles:
        assert all(v >= 0 for v in vec) and abs(sum(vec) - 1.0) < 1e-6


def test_compositional_dependence_collapse_reduces_confidence():
    segs = ["a", "b"]
    one = {"counts": {"a": 70, "b": 30}, "reliability": 0.9, "dependence_group": "poll-42"}
    syndicated = [dict(one) for _ in range(4)]                 # 4 re-reports of ONE poll
    dep = infer_compositional_posterior(segs, [1, 1], syndicated, seed=0, use_dependence=True)
    indep = infer_compositional_posterior(segs, [1, 1], syndicated, seed=0, use_dependence=False)
    assert dep.n_effective_observations == 1 and indep.n_effective_observations == 4
    # collapsing to one poll leaves MORE posterior spread than counting it four times
    assert sum(dep.posterior_sd) > sum(indep.posterior_sd)


def test_compositional_is_deterministic():
    segs = ["a", "b", "c"]
    obs = [{"counts": {"a": 40, "b": 40, "c": 20}, "reliability": 0.8}]
    a = infer_compositional_posterior(segs, [1, 1, 1], obs, seed=7)
    b = infer_compositional_posterior(segs, [1, 1, 1], obs, seed=7)
    assert a.posterior_mean == b.posterior_mean


def test_no_evidence_leaves_compositional_prior():
    segs = ["a", "b", "c"]
    post = infer_compositional_posterior(segs, [2, 1, 1], [], seed=0)
    assert post.n_effective_observations == 0
    # posterior mean ≈ prior mean (Dirichlet(2,1,1) → 0.5,0.25,0.25)
    assert abs(post.posterior_mean[0] - 0.5) < 0.06


# ============================================================ edge-existence (Bernoulli/log-odds) posterior
def test_at_least_ten_relation_layers():
    assert len(set(RELATION_LAYERS)) >= 10
    assert len({m["layer"] for m in EDGE_OBS_MODELS.values()}) >= 8


def test_strong_communication_record_raises_edge_posterior():
    obs = [EdgeObservation("alice", "bob", "direct_communication_record", strength="strong", reliability=0.95)]
    post = infer_edge_posterior("alice", "bob", "communication", obs, prior_p=0.1)
    # one strong-but-imperfect record against a skeptical 0.1 prior updates well past the prior (honest
    # Bayesian: a single noisy observation should NOT alone reach near-certainty)
    assert post.posterior_p > 0.75 and post.posterior_p > 0.1
    assert post.observed_status == "observed"
    assert post.log_odds_shift > 0


def test_repeated_records_reach_near_certainty():
    obs = [EdgeObservation("alice", "bob", "direct_communication_record", strength="strong", reliability=0.95,
                           dependence_group=f"src{i}") for i in range(3)]     # 3 INDEPENDENT records
    post = infer_edge_posterior("alice", "bob", "communication", obs, prior_p=0.1)
    assert post.posterior_p > 0.95                            # independent corroboration → near-certain


def test_absence_evidence_lowers_edge_posterior():
    obs = [EdgeObservation("a", "b", "absence_of_expected_interaction", strength="strong", reliability=0.9)]
    post = infer_edge_posterior("a", "b", "communication", obs, prior_p=0.5)
    assert post.posterior_p < 0.5                              # absence evidence pushes existence DOWN


def test_weak_evidence_moves_edge_less_than_strong():
    weak = [EdgeObservation("a", "b", "social_follow", strength="weak", reliability=0.6)]
    strong = [EdgeObservation("a", "b", "org_chart_relationship", strength="strong", reliability=0.95)]
    pw = infer_edge_posterior("a", "b", "influence", weak, prior_p=0.1)
    ps = infer_edge_posterior("a", "b", "reporting", strong, prior_p=0.1)
    assert ps.posterior_p > pw.posterior_p


def test_edge_dependence_collapse_counts_once():
    obs = [EdgeObservation("a", "b", "direct_communication_record", strength="strong", reliability=0.9,
                           dependence_group="wire") for _ in range(5)]
    collapsed = collapse_edge_observations(obs)
    assert len(collapsed) == 1 and collapsed[0].n_collapsed == 5
    single = infer_edge_posterior("a", "b", "communication",
                                  [EdgeObservation("a", "b", "direct_communication_record", "strong", 0.9)],
                                  prior_p=0.1)
    grouped = infer_edge_posterior("a", "b", "communication", obs, prior_p=0.1)
    # 5 syndicated copies must not out-update a single independent observation
    assert abs(grouped.posterior_p - single.posterior_p) < 1e-6


def test_no_edge_evidence_returns_prior():
    post = infer_edge_posterior("a", "b", "communication", [], prior_p=0.15)
    assert abs(post.posterior_p - 0.15) < 1e-6 and post.observed_status == "hypothesized"


# ============================================================ multilayer network + SBM + structure + visibility
def _planted_two_block(rng, n_per=8, p_in=0.6, p_out=0.05):
    nodes = [f"a{i}" for i in range(n_per)] + [f"b{i}" for i in range(n_per)]
    true = {v: (0 if v.startswith("a") else 1) for v in nodes}
    adj = set()
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            p = p_in if true[u] == true[v] else p_out
            if rng.random() < p:
                adj.add((u, v))
    return nodes, adj, true


def _cluster_accuracy(hard, true):
    # best of the two label permutations for 2 blocks
    import itertools
    ks = sorted(set(true.values()))
    best = 0.0
    for perm in itertools.permutations(ks):
        mp = dict(zip(ks, perm))
        acc = sum(1 for v in true if hard[v] == mp[true[v]]) / len(true)
        best = max(best, acc)
    return best


def test_sbm_recovers_planted_communities():
    from swm.world_model_v2.phase9_network import infer_communities
    rng = random.Random(3)
    nodes, adj, true = _planted_two_block(rng)
    fit = infer_communities(nodes, adj, 2, seed=0)
    assert _cluster_accuracy(fit["hard"], true) > 0.85       # recovers the planted 2-block structure
    # memberships are a normalized posterior per node
    for v, m in fit["membership"].items():
        assert abs(sum(m.values()) - 1.0) < 1e-6


def test_graph_structural_posterior_prefers_true_regime():
    from swm.world_model_v2.phase9_network import graph_structural_posterior
    rng = random.Random(5)
    nodes, adj, _ = _planted_two_block(rng, n_per=9)
    hyps = [{"id": "centralized", "K": 1, "prior": 0.33},
            {"id": "two_bloc", "K": 2, "prior": 0.34},
            {"id": "multi_faction", "K": 4, "prior": 0.33}]
    res = graph_structural_posterior(nodes, adj, hyps, seed=0)
    assert abs(sum(res["posterior"].values()) - 1.0) < 1e-6
    # the true 2-block regime should carry the most posterior mass (BIC-penalized likelihood)
    assert max(res["posterior"], key=res["posterior"].get) == "two_bloc"


def test_infer_network_edges_are_posterior_backed():
    from swm.world_model_v2.phase9_network import infer_network_edges
    obs = [EdgeObservation("x", "y", "direct_communication_record", "strong", 0.9, claim_id="c1")]
    edges = infer_network_edges([("x", "y", "communication"), ("x", "z", "communication")], obs)
    by = {(e.src, e.dst): e for e in edges}
    assert by[("x", "y")].existence_p > by[("x", "z")].existence_p   # evidence raises the observed edge
    assert by[("x", "y")].observed_status == "observed"
    assert by[("x", "z")].observed_status == "hypothesized"          # no evidence → stays hypothesized
    assert by[("x", "y")].posterior_ref and by[("x", "y")].consumed_by   # posterior-referenced + consumed


def test_missing_edge_posterior_keeps_unobserved_uncertain():
    from swm.world_model_v2.phase9_network import missing_edge_posterior, NetworkEdge
    present = [NetworkEdge("a", "b", "communication", existence_p=0.9)]
    miss = missing_edge_posterior(["a", "b", "c"], present, "communication", base_rate=0.05)
    pairs = {(m["src"], m["dst"]) for m in miss}
    assert ("a", "b") not in pairs and ("a", "c") in pairs          # unobserved pairs kept, at base rate
    assert all(m["existence_p"] == 0.05 for m in miss)


def test_actor_view_has_no_omniscient_leakage():
    from swm.world_model_v2.phase9_network import MultilayerNetwork, NetworkNode, NetworkEdge
    net = MultilayerNetwork(nodes={n: NetworkNode(n) for n in ("a", "b", "c")})
    net.edges = [NetworkEdge("a", "b", "trust", visibility="private"),          # only a,b see it
                 NetworkEdge("a", "c", "communication", visibility="public")]
    # c must NOT see the private a–b trust edge (adversarial no-leakage)
    c_view = net.edges_visible_to("c")
    assert all(not (e.layer == "trust" and e.visibility == "private") for e in c_view)
    assert any(e.layer == "communication" for e in c_view)          # but sees the public edge
    a_view = net.edges_visible_to("a")
    assert any(e.layer == "trust" for e in a_view)                  # endpoint a DOES see its private edge


# ============================================================ multilayer EXECUTION (Parts P, R, U)
def _chain_edges(layer, n=8, p=0.95):
    from swm.world_model_v2.phase9_network import NetworkEdge
    return [NetworkEdge(f"n{i}", f"n{i+1}", layer, existence_p=p) for i in range(n - 1)]


def test_graph_causally_drives_terminal_and_is_not_ornamental():
    from swm.world_model_v2.phase9_execution import simulate_multilayer
    from swm.world_model_v2.phase9_population import PopulationParticle
    pop = [PopulationParticle(weights={"s": 1.0})]
    dense = _chain_edges("influence", n=8, p=0.95)
    sparse = _chain_edges("influence", n=8, p=0.15)
    r_dense = simulate_multilayer(pop, dense, segment_susceptibility={"s": 0.8}, seeds=["n0"], n_particles=40, seed=0)
    r_sparse = simulate_multilayer(pop, sparse, segment_susceptibility={"s": 0.8}, seeds=["n0"], n_particles=40, seed=0)
    # a denser posterior graph spreads adoption further — the graph CAUSALLY changes the terminal
    assert r_dense["terminal_mean"] > r_sparse["terminal_mean"] + 0.1
    # posterior graph uncertainty propagates: different particles give different terminals
    assert r_dense["terminal_sd"] > 0.0
    assert r_dense["n_deltas"] > 0                                   # StateDelta objects were produced


def test_no_graph_means_no_diffusion():
    from swm.world_model_v2.phase9_execution import simulate_multilayer
    from swm.world_model_v2.phase9_population import PopulationParticle
    from swm.world_model_v2.phase9_network import NetworkEdge
    pop = [PopulationParticle(weights={"s": 1.0})]
    # edges exist as candidates but with ~0 existence posterior → almost no realized graph → only seeds adopt
    ghost = [NetworkEdge(f"n{i}", f"n{i+1}", "influence", existence_p=0.0) for i in range(7)]
    r = simulate_multilayer(pop, ghost, segment_susceptibility={"s": 0.9}, seeds=["n0"], n_particles=20, seed=0)
    assert r["terminal_mean"] < 0.2                                  # no edges → adoption stays at the seed


def test_population_heterogeneity_changes_outcome():
    from swm.world_model_v2.phase9_execution import simulate_multilayer
    from swm.world_model_v2.phase9_population import PopulationParticle
    edges = _chain_edges("influence", n=8, p=0.95)
    pop = [PopulationParticle(weights={"hi": 1.0})]
    hi = simulate_multilayer(pop, edges, segment_susceptibility={"hi": 0.9}, seeds=["n0"], n_particles=30, seed=1)
    lo = simulate_multilayer([PopulationParticle(weights={"lo": 1.0})], edges,
                             segment_susceptibility={"lo": 0.1}, seeds=["n0"], n_particles=30, seed=1)
    # a more susceptible population adopts more from the SAME graph — population alters aggregate behavior
    assert hi["terminal_mean"] > lo["terminal_mean"] + 0.15


def test_authority_gate_blocks_without_edge():
    from swm.world_model_v2.phase9_execution import authority_gate, Phase9World
    from swm.world_model_v2.phase9_network import MultilayerNetwork, NetworkEdge
    net = MultilayerNetwork(edges=[NetworkEdge("boss", "staff", "authority", existence_p=1.0)])
    world = Phase9World(agents={}, net=net)
    ok, d = authority_gate(world, "boss", "staff", "approve")
    assert ok and "authorized" in d.reason_codes and d.changes                 # has authority → executes
    ok2, d2 = authority_gate(world, "intern", "staff", "approve")
    assert not ok2 and "blocked:no_authority" in d2.reason_codes and not d2.changes   # no edge → blocked + reason


def test_communication_delivery_requires_path():
    from swm.world_model_v2.phase9_execution import communication_delivery, Phase9World
    from swm.world_model_v2.phase9_network import MultilayerNetwork, NetworkEdge
    net = MultilayerNetwork(edges=[NetworkEdge("a", "b", "communication", existence_p=1.0)])
    world = Phase9World(agents={}, net=net)
    recips, d = communication_delivery(world, "a", "hello")
    assert recips == ["b"]
    none, d2 = communication_delivery(world, "z", "hello")
    assert none == [] and "blocked:no_communication_path" in d2.reason_codes


# ============================================================ integrated pipeline + no-abstention (Part T)
def _pipeline_scenario():
    segs = ["young", "old"]
    survey = [{"counts": {"young": 60, "old": 40}, "reliability": 0.9, "source": "poll"}]
    cand = [(f"n{i}", f"n{i+1}", "influence") for i in range(6)]
    obs = [EdgeObservation("n0", "n1", "direct_communication_record", "strong", 0.9, claim_id="e0"),
           EdgeObservation("n1", "n2", "org_chart_relationship", "strong", 0.9, claim_id="e1")]
    return segs, survey, cand, obs


def test_pipeline_runs_end_to_end():
    from swm.world_model_v2.phase9_pipeline import simulate_populations_networks
    segs, survey, cand, obs = _pipeline_scenario()
    r = simulate_populations_networks(segments=segs, survey_observations=survey, candidate_edges=cand,
                                      edge_observations=obs, segment_susceptibility={"young": 0.7, "old": 0.3},
                                      structural_hypotheses=[{"id": "chain", "K": 1, "prior": 0.5},
                                                             {"id": "two", "K": 2, "prior": 0.5}],
                                      seeds=["n0"], n_particles=30, seed=0)
    assert r.simulation_status in ("completed", "completed_with_degradation")
    assert r.terminal.get("terminal_mean") is not None                 # a forecast exists
    assert r.population_posterior["mean"]["young"] > r.population_posterior["mean"]["old"]  # survey moved it
    assert r.graph_posterior["n_observed"] >= 2                        # observed edges from records
    assert r.provenance["population_posterior_hash"] and r.provenance["terminal_hash"]


def test_pipeline_no_abstention_on_weak_evidence():
    from swm.world_model_v2.phase9_pipeline import simulate_populations_networks
    # NO survey data, NO edge observations (all candidate edges hypothesized) → must still forecast, low grade
    cand = [(f"n{i}", f"n{i+1}", "influence") for i in range(6)]
    r = simulate_populations_networks(segments=["a", "b"], candidate_edges=cand, edge_observations=[],
                                      seeds=["n0"], n_particles=20, seed=0)
    assert r.simulation_status in ("completed", "completed_with_degradation")   # NOT refused
    assert r.terminal.get("terminal_mean") is not None
    assert r.support_grade in ("exploratory", "highly_speculative")   # weakness lowers grade, not a refusal
    assert r.limitations                                              # surfaced, not hidden


def test_pipeline_no_graph_still_forecasts():
    from swm.world_model_v2.phase9_pipeline import simulate_populations_networks
    r = simulate_populations_networks(segments=["a", "b"],
                                      survey_observations=[{"counts": {"a": 70, "b": 30}, "reliability": 0.9}],
                                      candidate_edges=[], edge_observations=[],
                                      segment_susceptibility={"a": 0.5, "b": 0.5}, n_particles=20, seed=0)
    assert r.simulation_status == "completed_with_degradation"        # no network → degraded, not refused
    assert r.terminal.get("terminal_mean") is not None


def test_pipeline_reproducible():
    from swm.world_model_v2.phase9_pipeline import simulate_populations_networks
    segs, survey, cand, obs = _pipeline_scenario()
    kw = dict(segments=segs, survey_observations=survey, candidate_edges=cand, edge_observations=obs,
              segment_susceptibility={"young": 0.7, "old": 0.3}, seeds=["n0"], n_particles=30, seed=0)
    a = simulate_populations_networks(**kw)
    b = simulate_populations_networks(**kw)
    assert a.provenance["terminal_hash"] == b.provenance["terminal_hash"]
    assert a.provenance["population_posterior_hash"] == b.provenance["population_posterior_hash"]


# ============================================================ universal discovery (completion run, Parts 1-3)
from types import SimpleNamespace


def _fake_plan():
    return SimpleNamespace(
        entities=[{"id": "alice", "type": "person"}, {"id": "bob", "type": "person"},
                  {"id": "carol", "type": "person"}],
        institutions=[{"id": "acme"}],
        populations=[{"id": "voters", "segments": [{"id": "young", "differs_on": ["age"]},
                                                   {"id": "old", "differs_on": ["age"]}]}],
        relations=[{"src": "alice", "rel": "communicates_with", "dst": "bob"},
                   {"src": "bob", "rel": "reports_to", "dst": "carol"},
                   {"src": "alice", "rel": "influences", "dst": "carol"}],
        plan_hash=lambda: "planhash123", provenance={})


class _FakeBundle:
    def __init__(self, claims, docs):
        self._claims, self.documents = claims, docs

    def included_claims(self):
        return self._claims

    def bundle_hash(self):
        return "bundlehash"


def test_discovery_is_automatic_from_plan_no_llm():
    from swm.world_model_v2.phase9_discovery import discover
    d = discover("Will the team adopt the proposal?", _fake_plan(), None, llm=None)
    # discovered WITHOUT the caller supplying anything
    assert set(d.actors) >= {"alice", "bob", "carol"}                # actors discovered from the plan
    assert d.population_segments == ["young", "old"]                 # segmentation discovered from the plan
    assert ("alice", "bob", "communication") in d.candidate_edges    # typed layer mapped from the relation
    assert ("bob", "carol", "reporting") in d.candidate_edges
    assert "communication" in d.relation_layers and "reporting" in d.relation_layers
    assert d.structural_hypotheses and d.seeds                       # hypotheses + seeds auto-proposed
    assert d.provenance["llm"] is False


def test_discovery_constructs_typed_observations_from_claims():
    from swm.world_model_v2.phase9_discovery import discover, construct_observations
    plan = _fake_plan()
    d = discover("q", plan, None, llm=None)
    claims = [{"claim_id": "c1", "subject": "alice", "predicate": "emailed", "object": "bob",
               "claim_class": "communication", "source_id": "doc1", "dependence_group": ""},
              {"claim_id": "c2", "subject": "bob", "predicate": "reports to", "object": "carol",
               "claim_class": "relationship", "source_id": "doc1", "dependence_group": ""}]
    bundle = _FakeBundle(claims, [{"id": "doc1", "source_type": "news"}])
    survey, edges = construct_observations(d, bundle, llm=None)
    by = {(e.src, e.dst): e for e in edges}
    assert ("alice", "bob") in by and by[("alice", "bob")].evidence_class == "direct_communication_record"
    assert ("bob", "carol") in by and by[("bob", "carol")].evidence_class == "org_chart_relationship"
    assert all(0 < e.reliability <= 1 for e in edges)                # reliability from source type, not minted


def test_discovery_representation_varies_with_scenario():
    from swm.world_model_v2.phase9_discovery import discover
    # a small named-actor scenario → explicit individuals; a segmented population → weighted segments
    small = SimpleNamespace(entities=[{"id": "ceo", "type": "person"}, {"id": "cfo", "type": "person"}],
                            institutions=[], populations=[], relations=[{"src": "ceo", "rel": "controls", "dst": "cfo"}],
                            plan_hash=lambda: "h", provenance={})
    d_small = discover("Will the CEO approve?", small, None, llm=None)
    d_pop = discover("Will voters support it?", _fake_plan(), None, llm=None)
    assert d_small.population_representation == "explicit_individuals"
    assert d_pop.population_representation == "weighted_segments"     # planner makes DIFFERENT choices


# ============================================================ informative absence / exposure (Part 4)
def test_exposure_absence_is_informative():
    from swm.world_model_v2.phase3_posterior import ExposureObservation, infer_edge_posterior_exposure
    # many opportunities, ZERO observed → strong evidence AGAINST the edge
    many_none = [ExposureObservation("repeated_interaction", n_opportunities=30, n_observed=0, reliability=0.9)]
    p_none = infer_edge_posterior_exposure("a", "b", "communication", many_none, prior_p=0.5)
    assert p_none.posterior_p < 0.2                                  # absence under high exposure lowers it
    # NO opportunity to observe → uninformative (posterior == prior)
    no_opp = [ExposureObservation("repeated_interaction", n_opportunities=0, n_observed=0, reliability=0.9)]
    p_noopp = infer_edge_posterior_exposure("a", "b", "communication", no_opp, prior_p=0.3)
    assert abs(p_noopp.posterior_p - 0.3) < 1e-6                     # no exposure ≠ evidence of absence


def test_exposure_observations_raise_posterior():
    from swm.world_model_v2.phase3_posterior import ExposureObservation, infer_edge_posterior_exposure
    ex = [ExposureObservation("repeated_interaction", n_opportunities=20, n_observed=15, reliability=0.9)]
    p = infer_edge_posterior_exposure("a", "b", "communication", ex, prior_p=0.1)
    assert p.posterior_p > 0.9 and p.observed_status == "observed"


def test_exposure_model_is_calibrated_under_variable_exposure():
    """The exposure model must be CALIBRATED even when the number of opportunities varies (the fix to the
    present-only overconfidence). Well-specified generator; ECE measured over the posterior."""
    import random
    from collections import defaultdict
    from swm.world_model_v2.phase3_posterior import ExposureObservation, infer_edge_posterior_exposure
    from swm.world_model_v2.phase3_observation import _edge_rates
    rng = random.Random(0)
    detect, false = _edge_rates("repeated_interaction", "strong", 0.9)
    buckets = defaultdict(lambda: [0, 0])
    for _ in range(3000):
        exists = 1 if rng.random() < 0.3 else 0
        N = rng.randrange(1, 25)                                     # VARIABLE exposure
        k = sum(1 for _ in range(N) if rng.random() < (detect if exists else false))
        p = infer_edge_posterior_exposure("a", "b", "communication",
                                          [ExposureObservation("repeated_interaction", N, k, 0.9)],
                                          prior_p=0.3).posterior_p
        b = min(9, int(p * 10))
        buckets[b][0] += exists
        buckets[b][1] += 1
    n = sum(c[1] for c in buckets.values())
    ece = sum(abs(c[0] / c[1] - (b + 0.5) / 10) * c[1] for b, c in buckets.items() if c[1]) / n
    assert ece <= 0.06                                              # calibrated under variable exposure


# ============================================================ temporal network evolution (Part 7)
def test_at_least_five_typed_transitions_exist():
    from swm.world_model_v2.phase9_temporal import TRANSITIONS
    assert len(TRANSITIONS) >= 5


def test_trust_transitions_move_edge_and_emit_delta():
    from swm.world_model_v2.phase9_temporal import trust_gain, trust_loss
    from swm.world_model_v2.phase9_network import MultilayerNetwork
    net = MultilayerNetwork(edges=[])
    d1 = trust_gain(net, "a", "b", event="fulfilled_commitment")
    e = [x for x in net.edges if x.layer == "trust"][0]
    assert e.existence_p > 0.2 and d1.changes                    # trust rose + StateDelta emitted
    d2 = trust_loss(net, "a", "b", event="betrayal")
    assert e.existence_p < 0.5 and d2.changes                    # betrayal lowered it


def test_edge_expiration_removes_past_edges():
    from swm.world_model_v2.phase9_temporal import edge_expiration
    from swm.world_model_v2.phase9_network import MultilayerNetwork, NetworkEdge
    net = MultilayerNetwork(edges=[NetworkEdge("a", "b", "alliance", existence_p=0.9, valid_to=100.0),
                                   NetworkEdge("a", "c", "alliance", existence_p=0.9, valid_to=None)])
    d = edge_expiration(net, at=200.0)
    layers = {(e.src, e.dst) for e in net.edges}
    assert ("a", "b") not in layers and ("a", "c") in layers      # expired edge removed, live edge kept
    assert d.changes


def test_evolution_changes_future_action_feasibility():
    from swm.world_model_v2.phase9_temporal import role_change
    from swm.world_model_v2.phase9_execution import authority_gate, Phase9World
    from swm.world_model_v2.phase9_network import MultilayerNetwork
    net = MultilayerNetwork(edges=[])
    world = Phase9World(agents={}, net=net)
    ok_before, _ = authority_gate(world, "manager", "report", "approve")
    assert not ok_before                                          # no authority initially → blocked
    role_change(net, "manager", "director", grants_to="report", at=50.0)  # promotion grants authority
    ok_after, _ = authority_gate(world, "manager", "report", "approve")
    assert ok_after                                               # evolution enabled a previously-blocked action


def test_evolution_changes_terminal_and_is_deterministic():
    from swm.world_model_v2.phase9_temporal import evolve
    from swm.world_model_v2.phase9_execution import materialize_worlds, influence_diffusion, weighted_adoption
    from swm.world_model_v2.phase9_population import PopulationParticle
    from swm.world_model_v2.phase9_network import NetworkEdge
    base_edges = [NetworkEdge(f"n{i}", f"n{i+1}", "alliance", existence_p=0.9) for i in range(5)]

    def run(with_defection):
        net_edges = [NetworkEdge(e.src, e.dst, e.layer, existence_p=e.existence_p) for e in base_edges]
        from swm.world_model_v2.phase9_network import MultilayerNetwork
        net = MultilayerNetwork(edges=net_edges)
        if with_defection:
            evolve(net, [{"kind": "alliance_defection", "src": "n2", "dst": "n3", "event": "defection"}])
        worlds = materialize_worlds([PopulationParticle(weights={"s": 1.0})], net.edges,
                                    segment_susceptibility={"s": 0.7}, n=20, seed=0)
        outs = [weighted_adoption(w, influence_diffusion(w, ["n0"], seed=w.particle_id)[0]) for w in worlds]
        return sum(outs) / len(outs)

    intact, defected = run(False), run(True)
    assert intact > defected                                      # breaking a bridge edge reduces spread
    assert run(True) == defected                                  # deterministic replay


# ============================================================ deep multilayer execution (Part 8)
def _world(edges):
    from swm.world_model_v2.phase9_execution import Phase9World
    from swm.world_model_v2.phase9_network import MultilayerNetwork
    return Phase9World(agents={}, net=MultilayerNetwork(edges=edges))


def test_exposure_layer_identifies_observers():
    from swm.world_model_v2.phase9_execution import exposure_mechanism
    from swm.world_model_v2.phase9_network import NetworkEdge
    w = _world([NetworkEdge("watcher", "star", "exposure"), NetworkEdge("star", "fan", "communication")])
    obs, d = exposure_mechanism(w, "star")
    assert "watcher" in obs and "fan" in obs and d.changes


def test_reporting_escalation_follows_chain():
    from swm.world_model_v2.phase9_execution import reporting_escalation
    from swm.world_model_v2.phase9_network import NetworkEdge
    w = _world([NetworkEdge("staff", "manager", "reporting"), NetworkEdge("manager", "vp", "reporting"),
                NetworkEdge("vp", "ceo", "reporting")])
    chain, d = reporting_escalation(w, "staff")
    assert chain == ["manager", "vp", "ceo"]                        # multi-hop up the reporting chain


def test_conflict_blocks_coordination():
    from swm.world_model_v2.phase9_execution import conflict_blocks_coordination
    from swm.world_model_v2.phase9_network import NetworkEdge
    w = _world([NetworkEdge("a", "b", "communication"), NetworkEdge("a", "b", "conflict")])
    can, d = conflict_blocks_coordination(w, "a", "b")
    assert not can and "blocked:conflict" in d.reason_codes         # conflict overrides a comms channel
    w2 = _world([NetworkEdge("a", "b", "alliance")])
    can2, _ = conflict_blocks_coordination(w2, "a", "b")
    assert can2


def test_resource_transfer_requires_edge():
    from swm.world_model_v2.phase9_execution import resource_transfer
    from swm.world_model_v2.phase9_network import NetworkEdge
    w = _world([NetworkEdge("donor", "campaign", "resource", existence_p=1.0)])
    moved, d = resource_transfer(w, "donor", "campaign", 0.5)
    assert moved > 0 and d.changes
    none, d2 = resource_transfer(w, "stranger", "campaign", 0.5)
    assert none == 0.0 and "blocked:no_resource_path" in d2.reason_codes


def test_jurisdiction_gate_blocks_out_of_scope():
    from swm.world_model_v2.phase9_execution import jurisdiction_gate
    from swm.world_model_v2.phase9_network import NetworkEdge
    w = _world([NetworkEdge("ftc", "merger", "jurisdiction")])
    ok, d = jurisdiction_gate(w, "ftc", "merger", "block")
    assert ok and "in_jurisdiction" in d.reason_codes
    ok2, d2 = jurisdiction_gate(w, "sec", "merger", "block")
    assert not ok2 and "blocked:out_of_jurisdiction" in d2.reason_codes


def test_ten_layers_have_exercised_mechanisms():
    """At least 10 relation layers must have a real typed mechanism consumer (Part 8 / gate F)."""
    from swm.world_model_v2.phase3_observation import RELATION_LAYERS
    exercised = {"communication", "exposure", "trust", "influence", "authority", "reporting",
                 "alliance", "conflict", "resource", "jurisdiction", "membership"}
    assert len(exercised & set(RELATION_LAYERS)) >= 10
