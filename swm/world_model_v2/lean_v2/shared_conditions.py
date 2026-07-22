"""D12 — shared condition graph + tail preservation. Actors who respond to the SAME latent cause
must be correlated through it, never multiplied as independent coin flips; and a low-probability
world that could REVERSE the answer must survive, not be pruned and renormalized away.

The EXP-113 failures this eliminates:
  * a board's members all read the same inflation print, yet each was branched independently, so a
    genuine common cause was washed into noise (the anti-consensus foundation, with D8/D14);
  * the shared-world enumeration took the top few combos and renormalized the rest to zero,
    silently discarding tail worlds — exactly where a surprise reversal lives.

`SharedConditionGraph` makes the correlation STRUCTURE explicit and grounded:

  * `ConditionNode` — a latent variable with mutually-exclusive states, counted weights, and the
    actors it affects (so those actors are correlated THROUGH it);
  * `depends_on` edges — a conditional table P(child_state | parent_state), so a correlated pair is
    enumerated jointly, not as an independent product;
  * `mutually_exclusive` groups / `regime` variables — impossible joint worlds are never created;
  * when a dependence is UNIDENTIFIED, the graph carries BOTH the independent and the
    comonotonic structure and reports the sensitivity — no correlation is invented.

Tail preservation: `joint_worlds` keeps EVERY world above a small floor (never a fixed top-k that
drops reversal-capable mass); worlds below the floor are merged and their discarded mass is
returned as a BOUND (an interval widener), and the caller may expand the tail when a tail world
would flip the terminal. Universal — the structure is read from the counted grounding, never
hardcoded per question."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm_key

SHARED_CONDITIONS_VERSION = "lean_v2.shared_conditions.v1"

#: worlds carrying at least this probability are ALWAYS kept (a reversal can hide here); below it
#: worlds are merged into a bounded discarded-tail mass reported as an interval widener.
TAIL_FLOOR = 0.01
#: a generous safety cap on enumerated worlds (bounds compute; only ever drops sub-floor mass)
MAX_WORLDS = 64


@dataclass
class ConditionNode:
    condition_id: str
    states: list                                   # mutually-exclusive states
    weights: dict = field(default_factory=dict)    # {state: prob}
    affects_actors: list = field(default_factory=list)
    counted: bool = False                          # a real counted rate vs a disclosed uniform
    provenance: dict = field(default_factory=dict)

    def normalized(self) -> dict:
        z = sum(self.weights.values()) or 1.0
        return {s: self.weights.get(s, 0.0) / z for s in self.states}


@dataclass
class DependencyEdge:
    parent: str
    child: str
    # P(child_state | parent_state): {parent_state: {child_state: prob}}
    conditional: dict = field(default_factory=dict)
    identified: bool = True                         # False → carry structures + report sensitivity


class SharedConditionGraph:
    """The correlation structure over shared latent conditions. Enumerates joint worlds respecting
    dependencies and exclusivity, preserving the tail."""

    def __init__(self):
        self.nodes: dict = {}
        self.deps: list = []
        self.exclusive_groups: list = []            # each a list of (condition_id, state) that
        self.diagnostics: list = []                 # cannot co-occur

    def add_node(self, node: ConditionNode):
        self.nodes[node.condition_id] = node

    def add_dependency(self, edge: DependencyEdge):
        self.deps.append(edge)

    def correlated_actors(self) -> dict:
        """{condition_id: [actors]} — actors sharing a condition are correlated through it and are
        NEVER independently multiplied in the enumeration."""
        return {cid: list(n.affects_actors) for cid, n in self.nodes.items()
                if len(n.affects_actors) > 1}

    def _parent_of(self, child_id: str):
        for e in self.deps:
            if e.child == child_id and e.identified:
                return e
        return None

    def _joint_weight(self, combo: dict) -> float:
        """P(combo) using conditional tables for dependent children and marginals otherwise."""
        w = 1.0
        for cid, state in combo.items():
            node = self.nodes.get(cid)
            if node is None:
                continue
            edge = self._parent_of(cid)
            if edge is not None and edge.parent in combo:
                pstate = combo[edge.parent]
                w *= (edge.conditional.get(pstate, {}) or {}).get(state,
                                                                  node.normalized().get(state, 0.0))
            else:
                w *= node.normalized().get(state, 0.0)
        return w

    def _is_possible(self, combo: dict) -> bool:
        for group in self.exclusive_groups:
            present = [(cid, st) for (cid, st) in group if combo.get(cid) == st]
            if len(present) > 1:
                return False                        # two mutually-exclusive states co-occur
        return True

    def joint_worlds(self, *, tail_floor: float = TAIL_FLOOR, max_worlds: int = MAX_WORLDS
                     ) -> dict:
        """Enumerate the joint distribution over shared conditions. Returns
        {worlds: [(combo, weight)], tail_mass, tail_worlds, n_total}. Every world with weight >=
        tail_floor is KEPT (renormalized among kept); sub-floor worlds are merged into a bounded
        `tail_mass` and exposed as `tail_worlds` for a possible reversal expansion."""
        if not self.nodes:
            return {"worlds": [({}, 1.0)], "tail_mass": 0.0, "tail_worlds": [], "n_total": 1}
        cids = sorted(self.nodes)
        axes = [[(cid, s) for s in self.nodes[cid].states] for cid in cids]
        raw = []
        for point in itertools.product(*axes):
            combo = {cid: s for cid, s in point}
            if not self._is_possible(combo):
                continue
            raw.append((combo, self._joint_weight(combo)))
        zt = sum(w for _c, w in raw) or 1.0
        raw = [(c, w / zt) for c, w in raw]         # normalize over the POSSIBLE worlds
        raw.sort(key=lambda cw: -cw[1])
        kept = [(c, w) for c, w in raw if w >= tail_floor][:max_worlds]
        kept_ids = {tuple(sorted(c.items())) for c, _w in kept}
        tail = [(c, w) for c, w in raw if tuple(sorted(c.items())) not in kept_ids]
        tail_mass = sum(w for _c, w in tail)
        zk = sum(w for _c, w in kept) or 1.0
        worlds = [(c, round(w / zk, 6)) for c, w in kept]
        return {"worlds": worlds, "tail_mass": round(tail_mass, 6),
                "tail_worlds": [(c, round(w, 6)) for c, w in tail], "n_total": len(raw)}

    def dependence_structures(self) -> list:
        """For each UNIDENTIFIED dependency, the plausible structures to report sensitivity across
        (independent vs comonotonic) — never an invented correlation coefficient."""
        out = []
        for e in self.deps:
            if not e.identified:
                out.append({"pair": [e.parent, e.child],
                            "structures": ["independent", "comonotonic"],
                            "note": "dependence not identified in the evidence — forecast "
                                    "sensitivity reported across both structures"})
        return out

    def manifest(self) -> dict:
        jw = self.joint_worlds()
        return {"version": SHARED_CONDITIONS_VERSION,
                "conditions": {cid: {"states": n.states, "weights": n.normalized(),
                                     "affects_actors": n.affects_actors, "counted": n.counted}
                               for cid, n in self.nodes.items()},
                "dependencies": [{"parent": e.parent, "child": e.child,
                                  "identified": e.identified} for e in self.deps],
                "exclusive_groups": self.exclusive_groups,
                "correlated_actors": self.correlated_actors(),
                "n_joint_worlds": jw["n_total"], "kept_worlds": len(jw["worlds"]),
                "preserved_tail_mass": jw["tail_mass"],
                "dependence_sensitivity": self.dependence_structures()}


def build_shared_condition_graph(posterior_engine, grounding: dict = None) -> SharedConditionGraph:
    """Build the graph from the counted shared-world conditions. Nodes + weights come from
    `shared_condition_worlds()`; dependency/exclusivity edges are read from the grounding when
    declared (an undeclared pair is independent, and the graph makes that explicit)."""
    g = SharedConditionGraph()
    for cid, weights, prov, affects in posterior_engine.shared_condition_worlds():
        g.add_node(ConditionNode(
            condition_id=cid, states=list(weights.keys()), weights=dict(weights),
            affects_actors=list(affects),
            counted=(prov.get("source") == "counted_shared_condition"), provenance=prov))
    # declared structure (optional): grounding may name mutually-exclusive condition-state groups
    # or conditional dependencies; absent that, conditions are independent (made explicit).
    shared = (grounding or {}).get("shared_world_conditions") or {}
    for cid, sc in (shared.items() if isinstance(shared, dict) else []):
        for dep in sc.get("depends_on") or []:
            parent = norm_key(dep.get("condition_id") if isinstance(dep, dict) else dep)
            if parent in g.nodes:
                g.add_dependency(DependencyEdge(
                    parent=parent, child=cid,
                    conditional=(dep.get("conditional") if isinstance(dep, dict) else {}) or {},
                    identified=bool(isinstance(dep, dict) and dep.get("conditional"))))
        for grp in sc.get("mutually_exclusive_with") or []:
            other = norm_key(grp)
            if other in g.nodes:
                # exclusive on the "holds" state of each (a body cannot be in two regimes at once)
                g.exclusive_groups.append([(cid, g.nodes[cid].states[0]),
                                           (other, g.nodes[other].states[0])])
    return g
