"""Exact weighted world-state coalescing — WeightedWorldNode / EquivalenceKey / Coalescer.

Actor-decision caching removes duplicate CALLS; coalescing removes duplicate EXECUTION: after
each causal wave, branches whose ENTIRE terminal-relevant dynamic state is identical merge into
one weighted node that executes once. The merge is conservative and EXACT:

  * the equivalence key covers actor qualitative states, delivered observations, working
    memory, institution state, population state, resources, commitments, relationships, the
    event queue, clock/deadline state, windows, world state, structural-model id,
    terminal/unresolved/truncation state — AND the future transition law;
  * the future transition law is equal BY CONSTRUCTION under the content-addressed randomness
    rule: every stochastic draw in lean_v2 is seeded by the DECISION-CONTEXT/STATE signature
    (never by a particle index), so two nodes with equal keys face identical future stochastic
    streams — there is no independent per-particle stream to merge away. If a caller ever
    attaches an INDEPENDENT stream (`node.independent_stream_tag`), that tag enters the key
    and unequal tags refuse to merge — independent uncertainty is retained, never averaged;
  * merged weights ADD; ancestry and source-node ids are preserved for audit;
  * at every merge and split: incoming total weight == outgoing total weight, asserted within
    deterministic numerical tolerance;
  * later divergence (different observations, decisions, draws, consequences) splits the
    state again — a merged node is just a weighted world, fully splittable.

Weights are never LLM intuition: they enter from explicit sampling laws (variant weight RANGES
mapped deterministically from grounded support classes), existing particle weights, grounded
reference frequencies, or caller-provided distributions — and the run reports sensitivity
across the ranges instead of pretending point weights are exact."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

from swm.world_model_v2.lean_context import canonicalize

WEIGHT_TOL = 1e-9


@dataclass
class WeightedWorldNode:
    node_id: str
    weight: float                                   # current-scenario probability mass
    day: str = ""                                   # world clock (day granularity)
    actor_states: dict = field(default_factory=dict)    # actor_id -> qualitative state dict
    actor_variant: dict = field(default_factory=dict)   # actor_id -> variant_id (sampling law)
    working_memory: dict = field(default_factory=dict)  # actor_id -> [content...]
    delivered: dict = field(default_factory=dict)       # actor_id -> [fact records]
    pending_observations: dict = field(default_factory=dict)
    institution_state: dict = field(default_factory=dict)
    population_state: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    commitments: dict = field(default_factory=dict)
    relationships: dict = field(default_factory=dict)
    event_queue: list = field(default_factory=list)     # [{day, etype, ...}] sorted by caller
    emitted_events: list = field(default_factory=list)
    windows: dict = field(default_factory=dict)
    world_state: dict = field(default_factory=dict)
    authority_overrides: dict = field(default_factory=dict)
    structural_model: str = "primary"
    latent_shocks: dict = field(default_factory=dict)   # unresolved latent variables
    pending_stochastic: list = field(default_factory=list)  # pending stochastic event families
    independent_stream_tag: str = ""                # non-empty => independent randomness source
    prior_decisions: dict = field(default_factory=dict)  # actor_id -> last decision snapshot
    asked_missing_facts: dict = field(default_factory=dict)  # actor_id -> [fact fingerprints]
    terminal: dict = field(default_factory=dict)    # {resolved, outcome, day, detail}
    unresolved_reason: str = ""
    truncated: bool = False
    ancestry: list = field(default_factory=list)    # source node ids across merges/splits
    weight_range: tuple = None                      # (lo, hi) mass under variant-range sweeps

    def key(self) -> str:
        """The WorldStateEquivalenceKey — every terminal-relevant dynamic field, canonical."""
        payload = {
            "day": self.day,
            "actor_states": canonicalize(self.actor_states),
            "actor_variant": canonicalize(self.actor_variant),
            "working_memory": canonicalize(self.working_memory),
            "delivered": canonicalize(self.delivered),
            "pending_observations": canonicalize(self.pending_observations),
            "institution_state": canonicalize(self.institution_state),
            "population_state": canonicalize(self.population_state),
            "resources": canonicalize(self.resources),
            "commitments": canonicalize(self.commitments),
            "relationships": canonicalize(self.relationships),
            "event_queue": canonicalize(sorted(
                (dict(e) for e in self.event_queue),
                key=lambda e: (str(e.get("day")), str(e.get("etype")), str(e.get("source"))))),
            "windows": canonicalize(self.windows),
            "world_state": canonicalize(self.world_state),
            "authority_overrides": canonicalize(self.authority_overrides),
            "structural_model": self.structural_model,
            "latent_shocks": canonicalize(self.latent_shocks),
            "pending_stochastic": canonicalize(self.pending_stochastic),
            "independent_stream_tag": self.independent_stream_tag,
            "prior_decisions": canonicalize(self.prior_decisions),
            "asked_missing_facts": canonicalize(self.asked_missing_facts),
            "terminal": canonicalize(self.terminal),
            "unresolved_reason": self.unresolved_reason,
            "truncated": self.truncated,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def clone(self, *, new_id: str, weight: float) -> "WeightedWorldNode":
        c = copy.deepcopy(self)
        c.node_id = new_id
        c.weight = float(weight)
        c.ancestry = list(self.ancestry) + [self.node_id]
        return c


class WeightedBranchCoalescer:
    """Merge-after-wave with exact mass conservation and full audit."""

    def __init__(self, *, max_nodes: int = 4096):
        self.max_nodes = int(max_nodes)
        self.merge_log: list = []
        self.split_log: list = []
        self.truncated_mass = 0.0
        self.truncated_nodes = 0
        self.executed_unique_nodes = 0

    @staticmethod
    def total_weight(nodes: list) -> float:
        return float(sum(n.weight for n in nodes))

    def coalesce(self, nodes: list) -> list:
        incoming = self.total_weight(nodes)
        by_key: dict = {}
        order: list = []
        for n in nodes:
            k = n.key()
            if k not in by_key:
                by_key[k] = n
                order.append(k)
            else:
                keeper = by_key[k]
                self.merge_log.append({
                    "kept": keeper.node_id, "absorbed": n.node_id,
                    "weight_added": round(n.weight, 12), "key": k[:16],
                    "source_particle_ids": sorted(set(keeper.ancestry + [keeper.node_id]
                                                      + n.ancestry + [n.node_id]))[:16]})
                keeper.weight += n.weight
                keeper.ancestry = list(dict.fromkeys(keeper.ancestry + [n.node_id]
                                                     + n.ancestry))
                if keeper.weight_range and n.weight_range:
                    keeper.weight_range = (keeper.weight_range[0] + n.weight_range[0],
                                           keeper.weight_range[1] + n.weight_range[1])
        out = [by_key[k] for k in order]
        outgoing = self.total_weight(out)
        assert abs(incoming - outgoing) <= WEIGHT_TOL * max(1.0, abs(incoming)), \
            f"coalesce mass leak: in={incoming!r} out={outgoing!r}"
        out = self._enforce_cap(out)
        return out

    def split(self, node: WeightedWorldNode, parts: list) -> list:
        """Split one node into weighted variants: parts = [(suffix, fraction, mutate_fn)].
        Fractions must sum to 1 (deterministic sampling law, never LLM intuition)."""
        fr = sum(f for _, f, _ in parts)
        assert abs(fr - 1.0) <= 1e-9, f"split fractions must sum to 1, got {fr!r}"
        out = []
        for suffix, fraction, mutate in parts:
            child = node.clone(new_id=f"{node.node_id}.{suffix}",
                               weight=node.weight * fraction)
            if node.weight_range:
                child.weight_range = (node.weight_range[0] * fraction,
                                      node.weight_range[1] * fraction)
            if mutate is not None:
                mutate(child)
            out.append(child)
        self.split_log.append({"node": node.node_id,
                               "into": [c.node_id for c in out],
                               "fractions": [round(f, 6) for _, f, _ in parts]})
        incoming, outgoing = node.weight, self.total_weight(out)
        assert abs(incoming - outgoing) <= WEIGHT_TOL * max(1.0, abs(incoming)), \
            f"split mass leak: in={incoming!r} out={outgoing!r}"
        return out

    def _enforce_cap(self, nodes: list) -> list:
        """Bounded node population: beyond the cap, the LOWEST-weight nodes become explicit,
        DISCLOSED truncated mass (never silently dropped, never renormalized away)."""
        if len(nodes) <= self.max_nodes:
            return nodes
        nodes = sorted(nodes, key=lambda n: -n.weight)
        kept, cut = nodes[:self.max_nodes], nodes[self.max_nodes:]
        for n in cut:
            self.truncated_mass += n.weight
            self.truncated_nodes += 1
        return kept

    def manifest(self) -> dict:
        return {"merges": len(self.merge_log), "splits": len(self.split_log),
                "merge_log": self.merge_log[-40:], "split_log": self.split_log[-20:],
                "truncated_mass": round(self.truncated_mass, 6),
                "truncated_nodes": self.truncated_nodes,
                "max_nodes": self.max_nodes,
                "executed_unique_nodes": self.executed_unique_nodes,
                "randomness_rule": "content-addressed (state/context-seeded) draws — equal "
                                   "keys share the future transition law by construction; "
                                   "independent streams carry a tag that refuses the merge"}
