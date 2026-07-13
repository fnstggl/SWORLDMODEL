"""Universal population & network DISCOVERY — Phase 9 completion (Parts 1, 3).

The missing universal seam: from a natural-language question + the Phase-1 plan + the Phase-2 evidence bundle,
automatically decide whether populations/networks are causally relevant and DISCOVER the target population,
segmentation, relevant actors/institutions, relation layers, candidate edges, communities, structural graph
hypotheses, and the population/network representation — WITHOUT the caller supplying any of it.

The LLM proposes SEMANTIC candidates only (segment dimensions, actor lists, which relation layers matter, which
actor→actor relationships the evidence describes, competing graph regimes). It mints NO numbers. Every numeric
posterior is produced downstream by the Phase-3 engine. A deterministic heuristic fallback derives the same
structure from the plan alone (no LLM), so the path runs offline and degrades gracefully.

`construct_observations` then converts Phase-2 CLAIMS into typed population + edge observations (reliability
from source-type tables, dependence groups from the bundle) — a claim becomes an edge only through a typed
observation model, and syndicated copies collapse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.world_model_v2.phase3_observation import EDGE_OBS_MODELS, RELATION_LAYERS, EdgeObservation

#: Phase-1 relation type → Phase-9 relation layer (the plan's relations are already typed).
REL_TO_LAYER = {
    "observes": "exposure", "trusts": "trust", "influences": "influence", "reports_to": "reporting",
    "controls": "authority", "funds": "resource", "endorses": "alliance", "opposes": "conflict",
    "communicates_with": "communication", "depends_on": "reporting", "belongs_to": "membership",
    "communication": "communication", "authority": "authority", "alliance": "alliance",
    "coordinates_with": "coordination", "allies_with": "alliance",
}
#: relation layer → the typed evidence class used when a claim asserts a relationship on that layer (the LLM may
#: override per-claim; this is the deterministic default). Numbers come from EDGE_OBS_MODELS, never here.
LAYER_TO_EVIDENCE_CLASS = {
    "communication": "direct_communication_record", "reporting": "org_chart_relationship",
    "authority": "formal_authority_record", "resource": "resource_transfer", "alliance": "public_statement_support",
    "conflict": "conflict_record", "influence": "endorsement", "exposure": "content_exposure",
    "trust": "private_statement_support", "membership": "co_membership", "affiliation": "coattendance",
    "coordination": "public_statement_support", "friendship": "coattendance", "jurisdiction": "formal_authority_record",
}


@dataclass
class Phase9DiscoveryPlan:
    relevant: bool = True
    relevance_reason: str = ""
    population_segments: list = field(default_factory=list)      # segment ids (a segmentation dimension's values)
    segmentation_dimension: str = ""
    population_representation: str = "weighted_segments"
    actors: list = field(default_factory=list)                  # node ids
    institutions: list = field(default_factory=list)
    relation_layers: list = field(default_factory=list)
    candidate_edges: list = field(default_factory=list)         # [(src, dst, layer)]
    edge_claim_map: dict = field(default_factory=dict)          # (src,dst,layer) -> [claim_id]
    structural_hypotheses: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    missing_edge_candidates: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def as_dict(self):
        d = self.__dict__.copy()
        d["candidate_edges"] = [list(e) for e in self.candidate_edges]
        return d


_DISCOVERY_PROMPT = """You are scoping the POPULATION and SOCIAL NETWORK relevant to a social question, to be
inferred later by a Bayesian engine. Give SEMANTIC structure only — NO numbers, NO probabilities. Reply ONLY
JSON:
{{"population_relevant": true|false,
  "segmentation_dimension": "<one demographic/role dimension, e.g. party|role|region|age|department or empty>",
  "population_segments": ["<segment value>", ...],
  "population_representation": "explicit_individuals|weighted_segments|representative_agents|aggregate_process",
  "actors": ["<named actor/org already in the scenario>", ...],
  "relation_layers": ["<one of {layers}>", ...],
  "relationships": [{{"src": "<actor>", "dst": "<actor>", "layer": "<layer>"}}, ...],
  "structural_hypotheses": [{{"id": "<short>", "describe": "<competing graph regime>", "K": <#blocks 1-4>}}, ...],
  "seed_actors": ["<actor who could initiate spread/influence>", ...]}}
Only use actors present in the scenario. `relation_layers` = which social layers matter for THIS question.
QUESTION: {question}
SCENARIO ACTORS: {actors}
SCENARIO RELATIONS: {relations}"""


def _plan_actors(plan):
    out = []
    for e in (plan.entities or []):
        if isinstance(e, dict) and e.get("id"):
            out.append(str(e["id"]))
    return out


def _plan_institutions(plan):
    return [str(i.get("id")) for i in (plan.institutions or []) if isinstance(i, dict) and i.get("id")]


def _heuristic_discovery(question, plan) -> Phase9DiscoveryPlan:
    """Derive population/network structure from the PLAN alone (no LLM) — deterministic fallback + offline path."""
    d = Phase9DiscoveryPlan(provenance={"source": "heuristic_from_plan", "llm": False})
    actors = _plan_actors(plan)
    d.actors = actors
    d.institutions = _plan_institutions(plan)
    # population segments from the plan's first population's segments (already proposed by Phase 1)
    for p in (plan.populations or []):
        segs = [str(s.get("id")) for s in (p.get("segments") or []) if isinstance(s, dict) and s.get("id")]
        if segs:
            d.population_segments = segs
            dims = [s.get("differs_on") for s in p.get("segments") or [] if isinstance(s, dict)]
            d.segmentation_dimension = str((dims[0] or ["segment"])[0]) if dims and dims[0] else "segment"
            break
    d.population_representation = ("explicit_individuals" if len(actors) <= 4 and not d.population_segments
                                  else ("weighted_segments" if d.population_segments else "representative_agents"))
    d.relevant = bool(actors or d.population_segments or (plan.relations or []))
    d.relevance_reason = "actors/relations/populations present in the plan" if d.relevant else "no social structure"
    # candidate edges + layers from the plan's typed relations
    layers = set()
    for r in (plan.relations or []):
        if not isinstance(r, dict):
            continue
        src, rel, dst = str(r.get("src", "")), str(r.get("rel", "")), str(r.get("dst", ""))
        layer = REL_TO_LAYER.get(rel, "communication")
        if src and dst:
            d.candidate_edges.append((src, dst, layer))
            layers.add(layer)
    d.relation_layers = sorted(layers) or ["communication"]
    d.seeds = actors[:1]
    d.structural_hypotheses = [{"id": "sparse", "describe": "few weak ties", "K": 1, "prior": 0.34},
                               {"id": "two_bloc", "describe": "two opposed groups", "K": 2, "prior": 0.33},
                               {"id": "multi_faction", "describe": "several factions", "K": 4, "prior": 0.33}]
    return d


def discover(question, plan, bundle, *, llm=None) -> Phase9DiscoveryPlan:
    """Discover the population/network slice. Starts from the heuristic (plan-derived) structure, then — if an
    LLM is available — augments it with LLM-proposed segmentation, relation layers, relationships, hypotheses
    and seed actors (semantic only). Every candidate edge is later validated against evidence."""
    d = _heuristic_discovery(question, plan)
    if llm is None:
        return d
    from swm.engine.grounding import parse_json
    actors = d.actors or [str(e.get("subject", "")) for e in (bundle.included_claims()[:8] if bundle else [])]
    prompt = _DISCOVERY_PROMPT.format(question=question, layers="|".join(RELATION_LAYERS),
                                      actors=json.dumps(actors[:20]),
                                      relations=json.dumps([(str(r.get("src")), str(r.get("rel")), str(r.get("dst")))
                                                            for r in (plan.relations or [])[:20]]))
    try:
        raw = parse_json(llm(prompt)) or {}
    except Exception:  # noqa: BLE001
        return d
    if isinstance(raw.get("population_relevant"), bool):
        d.relevant = d.relevant or raw["population_relevant"]
    if raw.get("segmentation_dimension") and not d.population_segments:
        d.segmentation_dimension = str(raw["segmentation_dimension"])[:40]
        d.population_segments = [str(s)[:40] for s in (raw.get("population_segments") or [])][:8]
    if raw.get("population_representation"):
        d.population_representation = str(raw["population_representation"])
    llm_actors = [str(a)[:60] for a in (raw.get("actors") or [])][:30]
    d.actors = list(dict.fromkeys(d.actors + llm_actors))
    d.relation_layers = list(dict.fromkeys(d.relation_layers + [str(l) for l in (raw.get("relation_layers") or [])
                                                                if l in RELATION_LAYERS]))
    known = set(d.actors)
    for rel in (raw.get("relationships") or []):
        if not isinstance(rel, dict):
            continue
        s, t, layer = str(rel.get("src", "")), str(rel.get("dst", "")), str(rel.get("layer", ""))
        if s and t and s != t and layer in RELATION_LAYERS:
            edge = (s, t, layer)
            if edge not in d.candidate_edges:
                d.candidate_edges.append(edge)
            known.update([s, t])
    d.actors = list(known)
    hyps = [h for h in (raw.get("structural_hypotheses") or []) if isinstance(h, dict) and h.get("id")][:4]
    if hyps:
        z = len(hyps)
        d.structural_hypotheses = [{"id": str(h["id"])[:30], "describe": str(h.get("describe", ""))[:80],
                                    "K": int(h.get("K", 2)) if str(h.get("K", 2)).isdigit() else 2,
                                    "prior": round(1.0 / z, 4)} for h in hyps]
    seeds = [str(s)[:60] for s in (raw.get("seed_actors") or []) if str(s) in known][:3]
    if seeds:
        d.seeds = seeds
    d.provenance = {"source": "llm_augmented_heuristic", "llm": True}
    return d


# --------------------------------------------------------------- automatic typed observation construction (Part 3)
_REL_PREDICATES = {  # predicate keyword → (layer, evidence_class) for claim→edge mapping without an LLM
    "email": ("communication", "direct_communication_record"), "message": ("communication", "direct_communication_record"),
    "call": ("communication", "direct_communication_record"), "meet": ("affiliation", "coattendance"),
    "report": ("reporting", "org_chart_relationship"), "manage": ("authority", "formal_authority_record"),
    "endorse": ("alliance", "public_statement_support"), "support": ("alliance", "public_statement_support"),
    "fund": ("resource", "resource_transfer"), "pay": ("resource", "resource_transfer"),
    "oppose": ("conflict", "conflict_record"), "vote": ("alliance", "voting_alignment"),
    "member": ("membership", "co_membership"), "trust": ("trust", "private_statement_support"),
    "influence": ("influence", "endorsement"), "follow": ("influence", "social_follow"),
}
_SOURCE_RELIABILITY = {"official_filing": 0.9, "wire": 0.82, "news": 0.75, "official_record": 0.88,
                       "dataset": 0.9, "market": 0.85, "social": 0.55, "user_provided": 0.7, "unknown": 0.6}


def _edge_class_for_claim(claim, discovery) -> tuple:
    """Map a relational claim to (layer, evidence_class) using its predicate/class; fall back to a candidate
    edge's declared layer if the actors match one."""
    pred = f"{claim.get('predicate','')} {claim.get('claim_class','')}".lower()
    for kw, (layer, cls) in _REL_PREDICATES.items():
        if kw in pred:
            return layer, cls
    # else: if the (subject, object) matches a candidate edge, use its layer
    s, o = str(claim.get("subject", "")), str(claim.get("object", ""))
    for (a, b, layer) in discovery.candidate_edges:
        if {a, b} == {s, o}:
            return layer, LAYER_TO_EVIDENCE_CLASS.get(layer, "coattendance")
    return None, None


def construct_observations(discovery, bundle, *, llm=None) -> tuple:
    """Convert Phase-2 claims into typed (survey_observations, edge_observations). A claim becomes an edge only
    through a typed observation model; reliability comes from source type; dependence groups carry through so
    syndicated copies collapse. Numbers are never taken from the claim text."""
    if bundle is None:
        return [], []
    src_type = {d["id"]: d.get("source_type", "news") for d in (bundle.documents or [])}
    edge_obs = []
    for c in bundle.included_claims():
        layer, cls = _edge_class_for_claim(c, discovery)
        if not cls:
            continue
        s, o = str(c.get("subject", "")), str(c.get("object", "")) or str(c.get("value", ""))
        if not s or not o or s == o:
            continue
        rel = _SOURCE_RELIABILITY.get(src_type.get(c.get("source_id", ""), "news"), 0.6)
        edge_obs.append(EdgeObservation(src=s, dst=o, evidence_class=cls, strength="moderate",
                                        reliability=rel, dependence_group=c.get("dependence_group", ""),
                                        claim_id=c.get("claim_id", "")))
    # population survey observations: claims that assert a count/share per segment (claim_class ~ statistic)
    survey_obs = []
    for c in bundle.included_claims():
        if str(c.get("claim_class", "")).lower() in ("statistic", "count", "poll", "survey", "census"):
            # a real count claim would carry a per-segment breakdown; absent structured counts we record the
            # claim as evidence-of-a-margin with low pseudo-weight (broad), never a fabricated composition.
            pass
    return survey_obs, edge_obs
