"""Typed weighted graph — the network *state* (audit C.5).

This is the data structure only: nodes (people/accounts/communities), typed weighted directed
edges (follows, replies-to, co-membership), and per-node activation/exposure state. The diffusion
*dynamics* (independent cascade / linear threshold / Hawkes) live in
`swm/transition/diffusion.py` and operate on this object.

Kept separate so that (a) a world can carry a graph without committing to a diffusion model, and
(b) diffusion can be swapped/ablated without touching the state. As the audit warns, do not build
elaborate network state until a wedge has a genuinely networked outcome — this is the minimal,
honest substrate for when one does.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    node_id: str
    activation: float = 0.0        # current activation/infection level in [0,1]
    exposure: float = 0.0          # cumulative exposure received this run
    attrs: dict[str, float] = field(default_factory=dict)


@dataclass
class Graph:
    """Directed weighted multigraph with a small, explicit API. Edge weight = influence strength."""
    nodes: dict[str, Node] = field(default_factory=dict)
    # out_edges[u] = list of (v, weight, etype)
    out_edges: dict[str, list[tuple[str, float, str]]] = field(default_factory=dict)

    def add_node(self, node_id: str, **attrs: float) -> Node:
        n = self.nodes.get(node_id)
        if n is None:
            n = Node(node_id, attrs=dict(attrs))
            self.nodes[node_id] = n
            self.out_edges.setdefault(node_id, [])
        else:
            n.attrs.update(attrs)
        return n

    def add_edge(self, u: str, v: str, weight: float = 1.0, etype: str = "follows") -> None:
        self.add_node(u)
        self.add_node(v)
        self.out_edges[u].append((v, weight, etype))

    def neighbors(self, u: str) -> list[tuple[str, float, str]]:
        return self.out_edges.get(u, [])

    def out_degree(self, u: str) -> int:
        return len(self.out_edges.get(u, []))

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return sum(len(v) for v in self.out_edges.values())

    def reset_activation(self) -> None:
        for n in self.nodes.values():
            n.activation = 0.0
            n.exposure = 0.0

    def seed(self, node_ids: list[str], level: float = 1.0) -> None:
        for nid in node_ids:
            self.add_node(nid).activation = level

    def total_activation(self) -> float:
        return sum(n.activation for n in self.nodes.values())
