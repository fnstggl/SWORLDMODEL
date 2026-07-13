"""Multilayer network + population EXECUTION — Phase 9 (Parts P, Q, R, U).

The five-plane completion: posterior population + graph particles are MATERIALIZED into worlds and CONSUMED by
typed multilayer mechanisms that produce StateDelta objects and terminal outcomes. Each relation layer has
distinct causal semantics (communication delivers, exposure lets an agent observe, trust weights credibility,
influence changes behavior, authority gates actions). Different posterior graph particles produce causally
DIFFERENT worlds — otherwise the graph posterior would be ornamental. Actors see only their own view (no
omniscient leakage); actions blocked by missing relations emit explicit reason codes.

A concrete, self-contained substrate exercises all of this: information/behavior diffusion through a multilayer
graph over a heterogeneous population, where the terminal (weighted adoption) depends on BOTH the sampled graph
structure and the population composition. Reuses `state.StateDelta`-style deltas.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from swm.world_model_v2.phase9_network import MultilayerNetwork, NetworkEdge


@dataclass
class Phase9Delta:
    """A machine-readable StateDelta for a Phase-9 mechanism (mirrors transitions.StateDelta shape)."""
    event_type: str
    operator: str
    reason_codes: list = field(default_factory=list)
    changes: list = field(default_factory=list)             # [{path, before, after}]
    uncertainty: dict = field(default_factory=dict)

    def change(self, path, before, after):
        self.changes.append({"path": path, "before": before, "after": after})
        return self

    def as_dict(self):
        return self.__dict__.copy()


@dataclass
class Phase9World:
    """One materialized posterior particle: agents (with segment + susceptibility), a SAMPLED graph (edges
    drawn from their existence posteriors), and community memberships. Different particles → different worlds."""
    agents: dict                                            # agent_id -> {segment, susceptibility, weight}
    net: MultilayerNetwork
    communities: dict = field(default_factory=dict)
    particle_id: int = 0


def _sample_graph(edges, rng):
    """Draw a concrete graph from the per-edge existence posteriors (Bernoulli) — one posterior particle."""
    kept = []
    for e in edges:
        if rng.random() < e.existence_p:
            kept.append(e)
    return kept


def materialize_worlds(pop_particles, edge_posteriors, *, communities=None, n=30, seed=0,
                       segment_susceptibility=None) -> list:
    """Materialize `n` posterior-weighted worlds: each pairs a population particle (segment weights) with a
    graph sampled from the edge existence posteriors. Agents are representative draws per segment. Different
    worlds differ in BOTH composition and realized graph (Part U causal materialization)."""
    rng = random.Random(seed * 6151 + 1)
    seg_susc = segment_susceptibility or {}
    worlds = []
    node_ids = sorted({e.src for e in edge_posteriors} | {e.dst for e in edge_posteriors})
    for pid in range(n):
        pp = pop_particles[pid % len(pop_particles)] if pop_particles else None
        # assign each node to a segment by the particle's composition; susceptibility from the segment
        agents = {}
        for nid in node_ids:
            if pp is not None:
                r, acc, seg = rng.random(), 0.0, list(pp.weights)[0]
                for s, w in pp.weights.items():
                    acc += w
                    if r <= acc:
                        seg = s
                        break
                w_i = pp.weights.get(seg, 1.0 / max(1, len(node_ids)))
            else:
                seg, w_i = "all", 1.0
            agents[nid] = {"segment": seg, "susceptibility": float(seg_susc.get(seg, 0.3)), "weight": w_i}
        kept = _sample_graph(edge_posteriors, rng)
        net = MultilayerNetwork(nodes={n: None for n in node_ids}, edges=kept)
        worlds.append(Phase9World(agents=agents, net=net, communities=communities or {}, particle_id=pid))
    return worlds


# ------------------------------------------------------------------------------- typed multilayer mechanisms
def communication_delivery(world: Phase9World, src: str, msg: str) -> tuple:
    """Deliver a message only along COMMUNICATION edges from src, respecting actor visibility. Returns
    (recipients, delta). No communication edge → no delivery (an action-feasibility constraint, Part R)."""
    d = Phase9Delta("communication_delivery", "phase9_comm", reason_codes=[f"msg={msg[:20]}"])
    recips = [e.dst for e in world.net.layer_edges("communication") if e.src == src]
    if not recips:
        d.reason_codes.append("blocked:no_communication_path")
        return [], d
    for r in recips:
        d.change(f"delivered[{r}]", False, True)
    return recips, d


def authority_gate(world: Phase9World, actor: str, target: str, action: str) -> tuple:
    """An authority-requiring action is feasible only if actor has an AUTHORITY/REPORTING edge to target
    (Part R). Blocked actions emit an explicit reason code — never a silent no-op."""
    has = any(e.src == actor and e.dst == target for e in
              world.net.layer_edges("authority") + world.net.layer_edges("reporting"))
    d = Phase9Delta("authority_action", "phase9_authority",
                    reason_codes=[f"action={action}", "authorized" if has else "blocked:no_authority"])
    if has:
        d.change(f"action[{action}:{target}]", "pending", "executed")
    return has, d


def trust_weighted_credibility(world: Phase9World, receiver: str, source: str, base: float = 0.5) -> float:
    """Credibility of `source` to `receiver` is raised by a TRUST edge (Part P). Used to weight belief update."""
    trusts = any(e.src == receiver and e.dst == source for e in world.net.layer_edges("trust"))
    return min(1.0, base + 0.4) if trusts else base


def influence_diffusion(world: Phase9World, seeds, *, contagion: str = "simple", max_rounds: int = 6,
                        seed: int = 0) -> tuple:
    """Behavior diffusion over the INFLUENCE (+ communication) layers, gated by per-agent susceptibility.
    `simple` contagion: adopt if ANY influencing neighbor adopted (prob = susceptibility). `complex` contagion:
    adopt only if a FRACTION ≥ threshold of influencing neighbors adopted. Returns (adopted set, deltas)."""
    rng = random.Random(seed * 3 + 7)
    adopted = set(seeds)
    deltas = []
    influ_edges = world.net.layer_edges("influence") + world.net.layer_edges("communication")
    in_nbrs = {}
    for e in influ_edges:
        in_nbrs.setdefault(e.dst, set()).add(e.src)
    for rnd in range(max_rounds):
        new = set()
        for node, nbrs in in_nbrs.items():
            if node in adopted:
                continue
            active = nbrs & adopted
            if not active:
                continue
            susc = world.agents.get(node, {}).get("susceptibility", 0.3)
            if contagion == "complex":
                frac = len(active) / max(1, len(nbrs))
                if frac >= 0.5 and rng.random() < susc:
                    new.add(node)
            else:
                p = 1 - (1 - susc) ** len(active)           # simple: independent exposure per active neighbor
                if rng.random() < p:
                    new.add(node)
        if not new:
            break
        for node in new:
            d = Phase9Delta("adoption", "phase9_diffusion",
                            reason_codes=[f"round={rnd}", f"contagion={contagion}"])
            d.change(f"adopted[{node}]", False, True)
            deltas.append(d)
        adopted |= new
    return adopted, deltas


def weighted_adoption(world: Phase9World, adopted) -> float:
    """Population-weighted terminal outcome: fraction of the (weighted) population that adopted."""
    tot = sum(a["weight"] for a in world.agents.values()) or 1.0
    return sum(world.agents[n]["weight"] for n in adopted if n in world.agents) / tot


def simulate_multilayer(pop_particles, edge_posteriors, *, communities=None, segment_susceptibility=None,
                        seeds=None, contagion="simple", n_particles=40, seed=0) -> dict:
    """Materialize posterior worlds and run diffusion in each → a terminal adoption DISTRIBUTION with the
    population + graph uncertainty propagated. Different particles give different terminals (the anti-ornamental
    property). Returns the distribution, mean/sd, delta counts, and a per-particle trace."""
    worlds = materialize_worlds(pop_particles, edge_posteriors, communities=communities, n=n_particles,
                                seed=seed, segment_susceptibility=segment_susceptibility)
    node_ids = sorted({e.src for e in edge_posteriors} | {e.dst for e in edge_posteriors})
    seeds = seeds or (node_ids[:1] if node_ids else [])
    outcomes, n_deltas, n_edges = [], 0, []
    for w in worlds:
        adopted, deltas = influence_diffusion(w, seeds, contagion=contagion, seed=seed + w.particle_id)
        outcomes.append(weighted_adoption(w, adopted))
        n_deltas += len(deltas)
        n_edges.append(len(w.net.edges))
    m = sum(outcomes) / len(outcomes) if outcomes else 0.0
    sd = (sum((o - m) ** 2 for o in outcomes) / len(outcomes)) ** 0.5 if outcomes else 0.0
    return {"terminal_mean": round(m, 5), "terminal_sd": round(sd, 5),
            "terminal_lo": round(min(outcomes), 5) if outcomes else None,
            "terminal_hi": round(max(outcomes), 5) if outcomes else None,
            "n_particles": n_particles, "mean_edges_per_world": round(sum(n_edges) / len(n_edges), 1),
            "n_deltas": n_deltas, "contagion": contagion, "seeds": seeds}
