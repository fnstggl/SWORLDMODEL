"""Production multilayer-network inference — Phase 9 (Parts F, I, K, M, N).

A network here is NOT an Edge dataclass, a NetworkX wrapper, a static/manual graph, or one generic
relationship-strength score. It is a TYPED, DIRECTED, MULTILAYER, UNCERTAIN graph whose every production edge
carries a Phase-3 posterior over its EXISTENCE (a Bernoulli/log-odds update from typed observation models),
whose community structure is a Bayesian stochastic block model, and whose competing macro-structures are
reweighted by the Phase-3 structural posterior. Missing edges keep an explicit posterior; nothing is minted by
the LLM.

Reuses `phase3_posterior.infer_edge_posterior` (per-edge existence) + `phase3_observation` (typed likelihoods)
+ the Phase-3 structural-posterior idea (graph macro-structure). The SBM and structural graph likelihood are
implemented dependency-free.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.phase3_observation import RELATION_LAYERS
from swm.world_model_v2.phase3_posterior import infer_edge_posterior


@dataclass
class NetworkNode:
    node_id: str
    entity_type: str = "person"
    roles: list = field(default_factory=list)
    visibility: str = "public"
    attributes: dict = field(default_factory=dict)
    inclusion_reason: str = ""
    sensitivity: float = 0.5
    provenance: dict = field(default_factory=dict)

    def as_dict(self):
        return self.__dict__.copy()


@dataclass
class NetworkEdge:
    """A production edge: existence is a Phase-3 posterior (never an LLM-minted probability). `visibility`
    governs which actors can see it (Part Q). `observed_status` ∈ observed|inferred|hypothesized."""
    src: str
    dst: str
    layer: str
    existence_p: float = 0.1                                 # posterior P(edge exists) — from infer_edge_posterior
    strength_mean: float = 0.5
    directed: bool = True
    visibility: str = "public"                               # public | private | src_only | dst_only
    observed_status: str = "hypothesized"
    evidence_ids: list = field(default_factory=list)
    valid_from: float = None
    valid_to: float = None
    posterior_ref: dict = field(default_factory=dict)        # EdgePosterior.as_dict()
    consumed_by: list = field(default_factory=list)
    transport_risk: str = "moderate"

    def as_dict(self):
        d = self.__dict__.copy()
        return d


@dataclass
class MultilayerNetwork:
    nodes: dict = field(default_factory=dict)                # id -> NetworkNode
    edges: list = field(default_factory=list)               # [NetworkEdge]
    communities: dict = field(default_factory=dict)         # node -> community membership posterior
    structural_prior: dict = field(default_factory=dict)
    structural_posterior: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def layer_edges(self, layer):
        return [e for e in self.edges if e.layer == layer]

    def edges_visible_to(self, actor: str):
        """Actor-specific view (Part Q): an actor sees public edges + private edges it is an endpoint of.
        The omniscient simulator sees all; actors never do — this prevents omniscient leakage."""
        out = []
        for e in self.edges:
            if e.visibility == "public":
                out.append(e)
            elif e.visibility == "private" and actor in (e.src, e.dst):
                out.append(e)
            elif e.visibility == "src_only" and actor == e.src:
                out.append(e)
            elif e.visibility == "dst_only" and actor == e.dst:
                out.append(e)
        return out

    def as_dict(self):
        return {"nodes": {k: v.as_dict() for k, v in self.nodes.items()},
                "edges": [e.as_dict() for e in self.edges],
                "communities": self.communities,
                "structural_prior": self.structural_prior,
                "structural_posterior": self.structural_posterior, "provenance": self.provenance}


def infer_network_edges(candidate_edges, edge_observations, *, layer_priors=None, use_dependence=True) -> list:
    """For each candidate (src, dst, layer), assimilate its typed observations into a Phase-3 existence
    posterior. `edge_observations` = list of phase3_observation.EdgeObservation. Returns [NetworkEdge].

    The LLM proposed the candidate edges + qualitative evidence classes; the PROBABILITY comes only from the
    log-odds likelihood update — no LLM-minted edge probabilities (Part H)."""
    layer_priors = layer_priors or {}
    by_edge = {}
    for o in edge_observations:
        by_edge.setdefault((o.src, o.dst, o.layer), []).append(o)
    out = []
    for (src, dst, layer) in candidate_edges:
        obs = by_edge.get((src, dst, layer), [])
        prior_p = float(layer_priors.get(layer, 0.08))
        post = infer_edge_posterior(src, dst, layer, obs, prior_p=prior_p, use_dependence=use_dependence)
        out.append(NetworkEdge(
            src=src, dst=dst, layer=layer, existence_p=post.posterior_p, directed=True,
            observed_status=post.observed_status,
            evidence_ids=[o.claim_id for o in obs if o.claim_id],
            posterior_ref=post.as_dict(),
            consumed_by=["exposure_mechanism", "authority_gate", "communication_delivery"]))
    return out


def missing_edge_posterior(nodes, present_edges, layer, *, base_rate=0.05) -> list:
    """Every UNOBSERVED ordered pair keeps an explicit missing-edge existence posterior at the layer base rate
    (Part M) — absence of an edge in evidence is uncertainty, not a hard zero."""
    present = {(e.src, e.dst) for e in present_edges if e.layer == layer}
    out = []
    for a in nodes:
        for b in nodes:
            if a != b and (a, b) not in present:
                out.append({"src": a, "dst": b, "layer": layer, "existence_p": base_rate,
                            "status": "unobserved_missing_edge"})
    return out


# --------------------------------------------------------------------------- stochastic block model (Part N/O)
class StochasticBlockModel:
    """A dependency-free EM stochastic block model: soft community memberships τ (a posterior over blocks per
    node) + a block-block edge-probability matrix B. The RIGHT representation for mesoscale structure — not one
    'community_strength = 0.6' scalar. Handles binary adjacency; deterministic under seed."""

    def __init__(self, K: int, *, max_iter: int = 60, seed: int = 0, tol: float = 1e-4, n_restarts: int = 8):
        self.K, self.max_iter, self.seed, self.tol, self.n_restarts = K, max_iter, seed, tol, n_restarts

    def fit(self, nodes, adjacency):
        """Multi-restart EM — a single random init routinely lands in a poor local optimum. Restarts are
        ranked by the CLEAN SBM data log-likelihood (the EM's internal running ll is K-invariant and would let
        a collapsed restart win). Returns (tau, B, clean_ll, idx)."""
        best = None
        for r in range(max(1, self.n_restarts)):
            tau, B, _ll, idx = self._fit_once(nodes, adjacency, seed=self.seed * 131 + r)
            clean = _sbm_data_loglik(list(nodes), adjacency, tau, B, idx)
            if best is None or clean > best[2]:
                best = (tau, B, clean, idx)
        return best

    def _fit_once(self, nodes, adjacency, *, seed):
        rng = random.Random(seed)
        n, K = len(nodes), self.K
        idx = {v: i for i, v in enumerate(nodes)}
        A = [[0.0] * n for _ in range(n)]
        for e in adjacency:
            i, j = idx[e[0]], idx[e[1]]
            A[i][j] = A[j][i] = 1.0
        # random hard-ish membership init + ASSORTATIVE B init (diagonal high, off-diagonal low). Initializing
        # B from a random assignment yields a ~uniform B (all blocks equal to overall density) that gives the
        # E-step NO signal — the EM then sits at the uniform saddle forever. Seeding B assortatively and running
        # the E-STEP FIRST gives the E-step immediate structure to sort nodes into, which the M-step refines.
        tau = [[0.03] * K for _ in range(n)]
        for i in range(n):
            tau[i][rng.randrange(K)] = 1.0
            z = sum(tau[i])
            tau[i] = [t / z for t in tau[i]]
        m_edges = len(adjacency)
        density = min(0.6, max(0.02, m_edges / max(1.0, n * (n - 1) / 2.0)))
        jit = 1.0 + 0.15 * (rng.random() - 0.5)                 # tiny per-restart asymmetry
        B = [[min(0.95, 2.6 * density * jit) if k == l else max(0.01, 0.35 * density)
              for l in range(K)] for k in range(K)]
        pi = [1.0 / K] * K
        prev_ll = -1e18
        for it in range(self.max_iter):
            # E-step FIRST (uses the current B): responsibilities
            ll = 0.0
            for i in range(n):
                logp = [math.log(pi[k]) for k in range(K)]
                for k in range(K):
                    for j in range(n):
                        if i == j:
                            continue
                        for l in range(K):
                            a = A[i][j]
                            p = B[k][l]
                            logp[k] += tau[j][l] * (a * math.log(p) + (1 - a) * math.log(1 - p))
                m = max(logp)
                ex = [math.exp(x - m) for x in logp]
                z = sum(ex) or 1.0
                tau[i] = [e / z for e in ex]
                ll += m + math.log(z)
            # M-step: block priors pi + block-block edge probabilities B from the soft assignments
            pi = [max(1e-6, sum(tau[i][k] for i in range(n)) / n) for k in range(K)]
            for k in range(K):
                for l in range(K):
                    num = den = 0.0
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                continue
                            w = tau[i][k] * tau[j][l]
                            num += w * A[i][j]
                            den += w
                    B[k][l] = min(1 - 1e-6, max(1e-6, num / den if den > 0 else 0.1))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return tau, B, ll, idx


def _sbm_data_loglik(nodes, adjacency, tau, B, idx):
    """Expected complete-data edge log-likelihood over UNORDERED pairs: Σ_{i<j} Σ_{k,l} τ_ik τ_jl
    [A_ij log B_kl + (1−A_ij) log(1−B_kl)]. This is the quantity that properly DIFFERS across K (the EM's
    internal running ll double-counts and can be K-invariant); use it for model comparison."""
    n, K = len(nodes), len(B)
    A = {}
    for e in adjacency:
        i, j = idx[e[0]], idx[e[1]]
        A[(min(i, j), max(i, j))] = 1.0
    ll = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a = A.get((i, j), 0.0)
            for k in range(K):
                for l in range(K):
                    p = B[k][l]
                    ll += tau[i][k] * tau[j][l] * (a * math.log(p) + (1 - a) * math.log(1 - p))
    return ll


def infer_communities(nodes, adjacency, K, *, seed=0):
    """Fit an SBM and return {node: {block: prob}} membership posterior + block matrix + a hard assignment +
    a clean SBM data log-likelihood suitable for model comparison across K."""
    nodes = list(nodes)
    sbm = StochasticBlockModel(K, seed=seed)
    tau, B, _ll, idx = sbm.fit(nodes, adjacency)
    membership = {nodes[i]: {f"block_{k}": round(tau[i][k], 4) for k in range(K)} for i in range(len(nodes))}
    hard = {nodes[i]: max(range(K), key=lambda k: tau[i][k]) for i in range(len(nodes))}
    return {"membership": membership, "hard": hard,
            "block_matrix": [[round(B[k][l], 4) for l in range(K)] for k in range(K)],
            "loglik": round(_sbm_data_loglik(nodes, adjacency, tau, B, idx), 3), "K": K}


# --------------------------------------------------------- competing graph macro-structures (Phase-3 structural)
def graph_structural_posterior(nodes, adjacency, hypotheses, *, seed=0):
    """Reweight competing GRAPH macro-structures by their marginal likelihood under an SBM (Part M). Each
    hypothesis names K (number of blocks / regime): e.g. 'two_bloc' K=2, 'multi_faction' K=4, 'centralized'
    K=1. Posterior ∝ prior × exp(BIC-penalized loglik). This is the Phase-3 structural posterior applied to
    graph structure — likelihood-updated, NOT manually assigned weights."""
    n = len(nodes)
    n_pairs = n * (n - 1) / 2.0 or 1.0
    log_post = {}
    fits = {}
    for h in hypotheses:
        hid, K = h["id"], int(h.get("K", 2))
        fit = infer_communities(nodes, adjacency, K, seed=seed)
        n_params = K * K + K                                # block matrix + memberships priors
        bic_pen = 0.5 * n_params * math.log(max(2.0, n_pairs))
        score = fit["loglik"] - bic_pen                     # BIC-penalized marginal-likelihood proxy
        log_post[hid] = math.log(max(1e-6, float(h.get("prior", 1.0)))) + score
        fits[hid] = {"K": K, "loglik": fit["loglik"], "bic_penalty": round(bic_pen, 2),
                     "score": round(score, 2)}
    m = max(log_post.values())
    ex = {k: math.exp(v - m) for k, v in log_post.items()}
    z = sum(ex.values()) or 1.0
    posterior = {k: round(v / z, 4) for k, v in sorted(ex.items(), key=lambda kv: -kv[1])}
    z0 = sum(max(1e-6, float(h.get("prior", 1.0))) for h in hypotheses) or 1.0
    prior = {h["id"]: round(max(1e-6, float(h.get("prior", 1.0))) / z0, 4) for h in hypotheses}
    return {"prior": prior, "posterior": posterior, "fits": fits}
