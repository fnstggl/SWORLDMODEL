"""Relationship & network state — Phase 1.4. ONE typed graph that decisions, diffusion and institutions share.

Core relation types are reusable, not closed: new types go through `register_relation` and must declare
directionality, valid entity types, state schema and causal uses. Every edge carries a strength DISTRIBUTION,
visibility, trust, channel, transmission delay, and provenance — a relationship that materially affects
execution is never just prose. This graph feeds: agent decision (who do I trust/report to), belief update
(source credibility), diffusion (who observes whom, with what delay), and institutions (authority edges).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.state import F, Provenance, StateField

CORE_RELATIONS = {
    "observes":          {"directed": True,  "uses": ["information", "diffusion"]},
    "trusts":            {"directed": True,  "uses": ["belief_update", "decision"]},
    "influences":        {"directed": True,  "uses": ["decision", "diffusion"]},
    "reports_to":        {"directed": True,  "uses": ["institution", "decision"]},
    "controls":          {"directed": True,  "uses": ["institution", "resources"]},
    "funds":             {"directed": True,  "uses": ["resources"]},
    "endorses":          {"directed": True,  "uses": ["information", "decision"]},
    "opposes":           {"directed": True,  "uses": ["decision"]},
    "communicates_with": {"directed": False, "uses": ["information", "diffusion"]},
    "depends_on":        {"directed": True,  "uses": ["resources", "institution"]},
    "belongs_to":        {"directed": True,  "uses": ["institution", "population"]},
}
_RELATIONS = dict(CORE_RELATIONS)


def register_relation(name: str, *, directed: bool, entity_types=("person", "institution"),
                      uses=(), schema: dict = None):
    """Typed relation extension: a novel relationship that affects execution must be registered, not prose."""
    if not name:
        raise ValueError("relation needs a name")
    _RELATIONS[name] = {"directed": directed, "entity_types": tuple(entity_types),
                        "uses": list(uses), "schema": schema or {}}
    return name


@dataclass
class RelationEdge:
    src: str
    rel: str
    dst: str
    strength: StateField = field(default_factory=lambda: F(0.5, dist={"mean": 0.5, "sd": 0.2},
                                                           status="assumed"))
    visibility: str = "public"            # public | private | src_only | dst_only
    trust: float = None                   # channel trust (belief update input)
    channel: str = ""                     # communication channel, if any
    delay_hours: float = 0.0              # transmission delay for information along this edge
    prov: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        if self.rel not in _RELATIONS:
            raise KeyError(f"unregistered relation {self.rel!r} — register_relation() first "
                           f"(known: {sorted(_RELATIONS)[:8]}…)")


@dataclass
class RelationGraph:
    edges: list = field(default_factory=list)

    def add(self, src, rel, dst, **kw) -> RelationEdge:
        e = RelationEdge(src=src, rel=rel, dst=dst, **kw)
        self.edges.append(e)
        return e

    def out_edges(self, src, rel=None):
        return [e for e in self.edges if e.src == src and (rel is None or e.rel == rel)]

    def in_edges(self, dst, rel=None):
        out = [e for e in self.edges if e.dst == dst and (rel is None or e.rel == rel)]
        # undirected relations match either endpoint
        out += [e for e in self.edges if e.src == dst and not _RELATIONS[e.rel]["directed"]
                and (rel is None or e.rel == rel) and e not in out]
        return out

    def edge(self, src, rel, dst):
        for e in self.edges:
            if e.src == src and e.rel == rel and e.dst == dst:
                return e
        return None

    def observers_of(self, eid):
        """Who observes `eid` (directed 'observes' edges in) — the diffusion/exposure frontier."""
        return [e.src for e in self.edges if e.rel == "observes" and e.dst == eid]
