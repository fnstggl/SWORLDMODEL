"""General per-person evidence fusion — the connective tissue of the world model.

Every regime so far inferred a person's latent profile from ONE kind of evidence: demographics→values
(EXP-028), writing→persona (EXP-025), a single message→variables (EXP-021). A *general* social world
model must fuse whatever is available about a person — structured attributes, their observed
responses/choices, their free text — into the ONE value/persona profile that every simulation
conditions on, with provenance and confidence that grow as evidence accumulates. That is this module.

`PersonEvidence` holds any subset of:
  - attributes : known structured facts (demographics / profile)          -> value prior
  - responses  : observed (item, answer) choices the person has made      -> value refinement
  - texts      : the person's writing                                     -> deep persona traits

`EvidenceFusion` fuses them into a value vector (+ a full `VariableMap`), leakage-safe:
  - attributes map to a value PRIOR via an injectable attribute→value function;
  - each observed response pulls the estimate toward the value-centroid of the (training) people who
    answered that item the same way — so a person's OTHER choices sharpen who they are;
  - the response weight grows with how many responses we've seen (the same depth_factor used for text
    depth), so more evidence => more refinement, saturating — a person with one observed choice is barely
    moved, a person with ten is placed firmly;
  - texts add deep persona traits (the EXP-025 engine).

The fusion is domain-agnostic: items/answers are opaque ids, and the attribute→value map is injected, so
the same component serves surveys, product choices, votes, or any behavior with observed prior responses.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from swm.variables.deep_inference import DeepInferenceEngine, depth_factor
from swm.variables.demographic_values import VALUE_DIMS, value_vector as _demo_value_vector
from swm.variables.variable_map import VariableMap

_RESP_TAU = 4.0          # responses; scale at which observed choices saturate their refinement weight


@dataclass
class PersonEvidence:
    person_id: str = ""
    attributes: dict = field(default_factory=dict)      # known structured facts
    responses: list = field(default_factory=list)       # [(item_id, answer_idx), ...] observed choices
    texts: list = field(default_factory=list)           # [text, ...] the person's writing (as-of order)


@dataclass
class EvidenceFusion:
    """Fuse attributes + observed responses + text into one value profile / VariableMap."""
    attr_value_fn: object = staticmethod(_demo_value_vector)   # attributes -> value vector (injectable)
    engine: DeepInferenceEngine = field(default_factory=DeepInferenceEngine)
    resp_tau: float = _RESP_TAU
    dims: list = field(default_factory=lambda: list(VALUE_DIMS))
    _centroid: dict = field(default_factory=dict)       # item -> answer -> [mean value vec, count]

    # ---- fit the response->value index on a training population ----
    def fit(self, people) -> "EvidenceFusion":
        """people: iterable of PersonEvidence. Learns, per (item, answer), the mean attribute-value
        profile of the training people who gave that answer — the centroid a response pulls toward."""
        acc = defaultdict(lambda: defaultdict(lambda: [[0.0] * len(self.dims), 0]))
        for ev in people:
            v = self._attr_vec(ev.attributes)
            for item, ans in ev.responses:
                cell = acc[item][ans]
                for j in range(len(self.dims)):
                    cell[0][j] += v[j]
                cell[1] += 1
        self._centroid = {item: {ans: [[s / c[1] for s in c[0]], c[1]] for ans, c in ans_map.items()}
                          for item, ans_map in acc.items()}
        return self

    def _attr_vec(self, attributes):
        return list(self.attr_value_fn(attributes)) if attributes else [0.5] * len(self.dims)

    # ---- the general fusion: evidence -> value vector ----
    def value_profile(self, ev: PersonEvidence, *, exclude_item=None, min_cell=3):
        """Fuse attribute prior + response refinement into a value vector. Leakage-safe: `exclude_item`
        drops the target item from the person's own context; centroids come only from fit() (training)."""
        v_attr = self._attr_vec(ev.attributes)
        obs = []
        for item, ans in ev.responses:
            if item == exclude_item:
                continue
            cell = self._centroid.get(item, {}).get(ans)
            if cell and cell[1] >= min_cell:
                obs.append(cell[0])
        depth = len(obs)
        if depth == 0:
            return v_attr, {"depth": 0, "response_weight": 0.0}
        v_resp = [sum(o[j] for o in obs) / depth for j in range(len(self.dims))]
        w = depth_factor(depth, self.resp_tau)
        fused = [(1 - w) * v_attr[j] + w * v_resp[j] for j in range(len(self.dims))]
        return fused, {"depth": depth, "response_weight": round(w, 3)}

    # ---- full VariableMap (values with provenance/confidence + deep persona from text) ----
    def to_variable_map(self, ev: PersonEvidence, *, exclude_item=None) -> VariableMap:
        vec, meta = self.value_profile(ev, exclude_item=exclude_item)
        vm = VariableMap(entity_id=ev.person_id)
        has_attr = bool(ev.attributes)
        for j, d in enumerate(self.dims):
            if d not in vm.vars and d not in ("economic_left",):   # economic_left is signed; keep in meta
                pass
            # value dims are not all in the behavioral schema; store in meta for downstream value-similarity
        vm.meta["value_vector"] = vec
        vm.meta["value_dims"] = list(self.dims)
        vm.meta["evidence"] = {"n_responses": meta["depth"], "response_weight": meta["response_weight"],
                               "has_attributes": has_attr, "n_texts": len(ev.texts)}
        # deep persona from the person's writing (routes into the schema PERSONA traits)
        if ev.texts:
            persona = self.engine.infer_persona(list(ev.texts))
            for name, payload in persona.items():
                vm.set(name, payload["value"], provenance="llm", confidence=payload["confidence"],
                       evidence=payload.get("evidence", "deep inference"))
        return vm
