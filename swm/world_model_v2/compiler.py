"""The world-slice compiler — Phase 5. Question + intervention → typed, validated WorldExecutionPlan.

NO scenario-level routing lives here: there is no `if election → …`. The LLM proposes a decomposition
(entities, populations, institutions, relations, quantities, information, latents, events, hazards,
mechanisms BY REGISTRY ID, readout) and the compiler VALIDATES every element into universal typed objects:
 - mechanisms must exist in the registry (unknown → candidate_experimental_mechanisms, marked, not executed);
 - the outcome contract must define a terminal readout before rollout may proceed;
 - proposed event types register through the event-type registry;
 - latents become LatentVariableRecords (distributions, never silent point values);
 - type/coherence checks reject malformed plans → the compiler ABSTAINS with the precise reason.
The fidelity planner then assigns explicit representation vs marginalization, particle/agent counts, and
temporal resolution from sensitivity/uncertainty/cost — deterministic rules over the LLM's ADVISORY hints.
Dynamic recompilation preserves history: `recompile()` produces a new plan version with provenance intact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from swm.world_model_v2.contracts import ContractError, FAMILIES, OutcomeContract
from swm.world_model_v2.events import register_event_type
from swm.world_model_v2.mechanisms import known_mechanisms
from swm.world_model_v2.state import parse_time


class CompileAbstention(Exception):
    """The world slice cannot be responsibly parameterized/typed — say so precisely; never wing it."""


_DECOMPOSE_PROMPT = """You are the WORLD-SLICE COMPILER's proposal stage for a structured social world model.
You propose a decomposition; a validator will type-check it. You may ONLY reference mechanisms from the
registry below; if a needed mechanism is missing, name it under "missing_mechanisms" (it will be marked
experimental, never silently executed). Do not force the scenario into one mechanism — assign mechanisms
component by component. Populate ONLY causally relevant fields.

QUESTION: {q}
INTERVENTION (optional): {intervention}
AS-OF: {as_of}   HORIZON: {horizon}
GROUNDED EVIDENCE:
{evidence}

MECHANISM REGISTRY: {mechanisms}
OUTCOME FAMILIES: {families}

Return ONLY JSON:
{{"outcome": {{"family": "...", "options": [...], "resolution_rule": "...", "readout_var": "<entity.field or quantity name whose terminal value answers the question>"}},
 "entities": [{{"id": "...", "type": "person|institution", "fields": {{"<universal field>": "<value or ?>"}}}}],
 "populations": [{{"id": "...", "segments": [{{"id": "...", "weight": <0..1>, "differs_on": ["<causally relevant dimension>", ...]}}]}}],
 "relations": [{{"src": "...", "rel": "<registered relation>", "dst": "..."}}],
 "institutions": [{{"id": "...", "rules": [{{"kind": "...", "params": {{}}}}]}}],
 "quantities": [{{"name": "...", "qtype": "...", "value": <num or null>, "sd": <num or null>}}],
 "latents": [{{"path": "<entity.field>", "why": "...", "lo": <num>, "hi": <num>}}],
 "scheduled_events": [{{"etype": "...", "at": "<RFC3339 or YYYY-MM-DD>", "participants": [...], "payload": {{}}}}],
 "hazards": [{{"etype": "...", "rate_per_day": <num>, "participants": [...]}}],
 "mechanisms": ["<registry id>", ...],
 "missing_mechanisms": [{{"name": "...", "why": "..."}}],
 "sensitivity": {{"<component id>": <0..1>}},
 "domain": "<one short tag, e.g. organizational_decision | election | messaging | market | diffusion>",
 "population_kind": "<e.g. online_social | organizational | electorate | lab | none>",
 "time_scale": "<hours | days | weeks | months>",
 "available_data": ["<evidence kinds present, e.g. polls | activity_log | org_chart | none>", ...],
 "rationale": "<one sentence per major inclusion/omission>"}}"""


@dataclass
class WorldExecutionPlan:
    question: str
    outcome_contract: OutcomeContract
    as_of: float
    horizon_ts: float
    entities: list = field(default_factory=list)
    populations: list = field(default_factory=list)
    institutions: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    quantities: list = field(default_factory=list)
    latents: list = field(default_factory=list)             # [LatentVariableRecord]
    scheduled_events: list = field(default_factory=list)
    stochastic_hazards: list = field(default_factory=list)
    accepted_mechanisms: list = field(default_factory=list)          # [{mech_id, why, sensitivity, ...}]
    candidate_experimental_mechanisms: list = field(default_factory=list)
    rejected_mechanisms: list = field(default_factory=list)          # [{id, rejection_reason}]
    fidelity_plan: dict = field(default_factory=dict)
    uncertainty_plan: dict = field(default_factory=dict)
    compute_plan: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    version: int = 1
    parent_version: int = 0


def _fidelity_plan(proposal: dict, n_budget: int = 30) -> dict:
    """Deterministic fidelity rules over the LLM's ADVISORY sensitivity hints: explicit representation for
    high-sensitivity components; marginalize low-sensitivity ones; particles from latent count; horizon-aware
    granularity. More agents/fields NEVER auto-count as fidelity."""
    sens = proposal.get("sensitivity") or {}
    explicit, marginalized = [], []
    for comp, s in sens.items():
        (explicit if (isinstance(s, (int, float)) and s >= 0.35) else marginalized).append(comp)
    n_latents = len(proposal.get("latents") or [])
    particles = max(10, min(60, n_budget + 5 * n_latents))
    return {"explicit": explicit, "marginalized": marginalized,
            "n_particles": particles,
            "agent_samples": min(24, 4 + 2 * len(proposal.get("entities") or [])),
            "note": "deterministic rules over advisory LLM sensitivity; components <0.35 marginalize"}


def compile_world(question: str, *, llm, evidence="", as_of: str, horizon: str,
                  intervention: str = "", n_budget: int = 30) -> WorldExecutionPlan:
    """The one compiler for every scenario class. Raises CompileAbstention with the precise reason when the
    slice can't be typed/parameterized responsibly.

    `evidence` may be a typed EvidenceBundle (Phase 2 — preferred: as-of-gated, hash-recorded, rendered
    with per-item dates/sources) or a legacy string (flagged in provenance as unaudited)."""
    from swm.engine.grounding import parse_json
    from swm.world_model_v2.init_state import LatentVariableRecord
    registry = known_mechanisms()
    bundle_hash, evidence_basis = "", "legacy_string_unaudited"
    if hasattr(evidence, "render") and hasattr(evidence, "bundle_hash"):   # EvidenceBundle
        bundle_hash = evidence.bundle_hash()
        evidence_basis = "typed_bundle"
        evidence_text = evidence.render(max_chars=4000)
    else:
        evidence_text = str(evidence or "")[:4000]
    prompt = _DECOMPOSE_PROMPT.format(
        q=question, intervention=intervention or "(none)", as_of=as_of, horizon=horizon,
        evidence=evidence_text or "(none)",
        mechanisms=json.dumps({k: v.causal_role for k, v in registry.items()}),
        families=json.dumps(list(FAMILIES)))
    raw = parse_json(llm(prompt)) or {}
    if not raw or not isinstance(raw.get("outcome"), dict):
        raise CompileAbstention("decomposition unparseable — no typed outcome proposed")

    # ---- outcome contract (readout REQUIRED pre-rollout) ----
    o = raw["outcome"]
    readout_var = str(o.get("readout_var", "")).strip()
    if not readout_var:
        raise CompileAbstention("no terminal readout variable proposed — cannot project an answer from "
                                "terminal states, so the simulation must not proceed")
    def make_readout(var):
        def read(world):
            if var in world.quantities:
                return world.quantities[var].value
            eid, _, fpath = var.partition(".")
            ent = world.entities.get(eid)
            if ent is None:
                return None
            fname, _, key = fpath.partition("[")
            return ent.value(fname, key=key.rstrip("]") or None)
        return read
    try:
        contract = OutcomeContract(family=str(o.get("family", "")), options=list(o.get("options") or []),
                                   resolution_rule=str(o.get("resolution_rule", ""))[:300],
                                   readout=make_readout(readout_var), readout_var=readout_var,
                                   horizon_ts=parse_time(horizon)).validate()
    except (ContractError, ValueError) as e:
        raise CompileAbstention(f"outcome contract invalid: {e}")

    # ---- mechanisms: registry-vetted AND executable; unknown → experimental candidates ----
    # A1: an "accepted" mechanism that resolves to no executable operator is a silent no-op factory —
    # reject it loudly instead (the empty-operator kernels were 3 of 9 entries in the audited registry).
    from swm.world_model_v2.transitions import _OPERATORS
    accepted, rejected = [], []
    for mid in raw.get("mechanisms") or []:
        if mid in registry:
            m = registry[mid]
            if not m.operator or m.operator not in _OPERATORS:
                rejected.append({"id": mid, "rejection_reason":
                                 f"registry entry names no executable operator ({m.operator!r}) — "
                                 f"refusing a mechanism that cannot cause any transition"})
                continue
            if _OPERATORS[m.operator]["experimental"]:
                rejected.append({"id": mid, "rejection_reason":
                                 f"operator {m.operator!r} is experimental (unvalidated) — excluded from "
                                 f"production compilation"})
                continue
            accepted.append({"mech_id": mid, "ontology_type": m.ontology_type, "causal_role": m.causal_role,
                             "parameter_source": m.parameter_source, "temporal_scale": m.temporal_scale,
                             "calibration_status": m.calibration_status, "operator": m.operator,
                             "sensitivity": (raw.get("sensitivity") or {}).get(mid, 0.5)})
        else:
            rejected.append({"id": mid, "rejection_reason": "not in registry — proposed id unknown"})
    experimental = [{"name": str(m.get("name", ""))[:60], "why": str(m.get("why", ""))[:200],
                     "status": "experimental — NOT executed until validated"}
                    for m in (raw.get("missing_mechanisms") or []) if isinstance(m, dict)]
    if not accepted:
        raise CompileAbstention("no executable registry mechanism applies — the needed mechanisms are "
                                f"missing or unexecutable; rejected: {[r['id'] for r in rejected]}; marked "
                                f"experimental: {[m['name'] for m in experimental]}")

    # ---- A2: production-registry applicability scoring (selection provenance) ----
    # The scenario descriptor comes from the proposal; families are scored with real applicability rules
    # (domain/population/timescale/state/data/evidence-quality). Scores gate production-family selection
    # and are recorded in the plan for audit; lean-registry mechanisms above remain the executable floor.
    scenario = {"domain": str(raw.get("domain", "") or ""),
                "population_kind": str(raw.get("population_kind", "") or ""),
                "time_scale": str(raw.get("time_scale", "") or ""),
                "available_state": (["network"] if raw.get("relations") else [])
                + (["entities"] if raw.get("entities") else [])
                + (["populations"] if raw.get("populations") else [])
                + (["institutions"] if raw.get("institutions") else [])
                + (["quantities"] if raw.get("quantities") else []),
                "available_data": list(raw.get("available_data") or []),
                "institutional": bool(raw.get("institutions"))}
    try:
        from swm.world_model_v2.registry import load_registry
        from swm.world_model_v2.registry.applicability import rank_mechanisms
        mechanism_selection = rank_mechanisms(load_registry(), scenario)
    except Exception as e:                                    # registry unavailable → logged, not fatal
        mechanism_selection = {"selected": [], "rejected": [],
                               "note": f"production registry unavailable: {e}"}

    # ---- latents: always distributions ----
    latents = []
    for l in raw.get("latents") or []:
        try:
            lo, hi = float(l.get("lo", 0.0)), float(l.get("hi", 1.0))
        except (TypeError, ValueError):
            lo, hi = 0.0, 1.0
        latents.append(LatentVariableRecord(
            path=str(l.get("path", "")), method="prior", confidence=0.3,
            candidates={"mean": (lo + hi) / 2, "sd": (hi - lo) / 4, "lo": lo, "hi": hi},
            evidence=[str(l.get("why", ""))[:120]]))

    # ---- events: proposed types register through the registry door ----
    sched = []
    for ev in raw.get("scheduled_events") or []:
        et = str(ev.get("etype", ""))
        from swm.world_model_v2.events import event_type_registered
        if not event_type_registered(et):
            register_event_type(et, scheduling="scheduled", validated=False,
                                parameter_source="compiler-proposed")
        try:
            ts = parse_time(ev.get("at"))
        except ValueError:
            continue
        sched.append({"etype": et, "ts": ts, "participants": list(ev.get("participants") or []),
                      "payload": dict(ev.get("payload") or {})})
    hazards = []
    for h in (raw.get("hazards") or []):
        if not isinstance(h, dict):
            continue
        het = str(h.get("etype", "distraction"))
        from swm.world_model_v2.events import event_type_registered
        if not event_type_registered(het):                   # symmetric with scheduled events
            register_event_type(het, scheduling="hazard", validated=False,
                                parameter_source="compiler-proposed")
        hazards.append({"etype": het,
                        "rate_per_day": max(0.0, float(h.get("rate_per_day", 0.0) or 0.0)),
                        "participants": list(h.get("participants") or [])})

    plan = WorldExecutionPlan(
        question=question, outcome_contract=contract, as_of=parse_time(as_of),
        horizon_ts=parse_time(horizon),
        entities=list(raw.get("entities") or []), populations=list(raw.get("populations") or []),
        institutions=list(raw.get("institutions") or []), relations=list(raw.get("relations") or []),
        quantities=list(raw.get("quantities") or []), latents=latents,
        scheduled_events=sched, stochastic_hazards=hazards,
        accepted_mechanisms=accepted, candidate_experimental_mechanisms=experimental,
        rejected_mechanisms=rejected,
        fidelity_plan=_fidelity_plan(raw, n_budget),
        uncertainty_plan={"latents": len(latents), "hazards": len(hazards)},
        compute_plan={"n_particles": _fidelity_plan(raw, n_budget)["n_particles"]},
        provenance={"prompt_hash": hashlib.sha1(prompt.encode()).hexdigest()[:12],
                    "rationale": str(raw.get("rationale", ""))[:400],
                    "scenario": scenario,
                    "evidence_basis": evidence_basis, "evidence_bundle_hash": bundle_hash,
                    "production_registry_selection": mechanism_selection})
    return plan


def recompile(plan: WorldExecutionPlan, *, llm, new_evidence: str, reason: str) -> WorldExecutionPlan:
    """Dynamic recompilation preserving history: a NEW versioned plan whose provenance chains to the old."""
    import time as _t
    new = compile_world(plan.question, llm=llm, evidence=new_evidence,
                        as_of=_t.strftime("%Y-%m-%d", _t.gmtime(plan.as_of)),
                        horizon=_t.strftime("%Y-%m-%d", _t.gmtime(plan.horizon_ts)))
    new.version = plan.version + 1
    new.parent_version = plan.version
    new.provenance["recompiled_because"] = reason[:200]
    return new
