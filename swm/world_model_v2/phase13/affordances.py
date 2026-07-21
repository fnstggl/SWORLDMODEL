"""Phase 13 action generation from affordances (Part 5) — the world proposes, not only the LLM.

Candidate actions come from inspecting the ACTUAL world: the decision-maker's authority and resources,
controllable quantities, institutional procedures that name them as holders, network edges (who they can
contact), the information ledger (what they can disclose/investigate), pending scheduled events they own
(delay/accelerate/cancel), and explicit decision points (defer/wait). A constrained LLM proposer is ONE
additional source; every proposal — LLM or otherwise — passes the same deterministic validation and the
FeasibilityEngine before simulation. The LLM can propose semantics; it cannot edit hidden state, set
another actor's beliefs, set outcomes, or mint authority (those proposals are rejected with typed
reasons, recorded).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.world_model_v2.phase13.ontology import (ActionSchema, do_nothing, defer, dedupe,
                                                 operation_registered, operation_spec)

_FORBIDDEN_LLM_OPS = ("set_belief", "set_outcome", "set_hidden_state", "grant_authority",
                      "create_resource")


@dataclass
class GenerationReport:
    candidates: list = field(default_factory=list)     # [ActionSchema]
    sources: dict = field(default_factory=dict)        # source -> count
    rejected: list = field(default_factory=list)       # [{"proposal":..., "reason":...}]
    deduped: list = field(default_factory=list)

    def as_dict(self):
        return {"n_candidates": len(self.candidates), "sources": self.sources,
                "rejected": self.rejected[:20], "n_deduped": len(self.deduped)}


def generate_actions(world, problem, *, llm=None, max_llm_proposals: int = 8,
                     parameter_grid: int = 5) -> GenerationReport:
    """Build the candidate space from affordances + user candidates + (optionally) a constrained LLM."""
    rep = GenerationReport()
    maker = problem.decision_maker
    now = float(world.clock.now) if world is not None else 0.0

    def add(a: ActionSchema, source: str):
        a.provenance = source
        rep.candidates.append(a)
        rep.sources[source] = rep.sources.get(source, 0) + 1

    # --- mandatory reference actions (Part 9): do nothing / defer / gather information -------------
    add(do_nothing(maker), "baseline")
    if problem.decision_points:
        from swm.world_model_v2.state import parse_time
        try:
            add(defer(maker, parse_time(problem.decision_points[0])), "baseline")
        except (ValueError, TypeError):
            pass
    if problem.information_gathering_allowed:
        add(ActionSchema(action_id="gather_information", actor=maker,
                         operation="gather_information",
                         params={"about": "outcome-relevant hidden state"}), "baseline")

    # --- user-supplied candidates (always admitted to feasibility) ---------------------------------
    for a in problem.candidate_actions or []:
        add(a, "user")

    if not problem.generated_action_permission or world is None:
        rep.candidates, rep.deduped = dedupe(rep.candidates)
        return rep

    actor = (world.entities or {}).get(maker)

    # --- authority-derived operations --------------------------------------------------------------
    for cap in problem.authority or []:
        op, _, obj = str(cap).partition(":")
        if operation_registered(op):
            spec = operation_spec(op)
            base = ActionSchema(action_id=f"{op}{'_' + obj if obj else ''}", actor=maker,
                                operation=op, object=obj, authority_basis=cap, timing_ts=now)
            add(base, "authority")
            # timing variants for time-family and communication operations: alternatives are
            # anchored to REAL scenario times (§23) — acting just before/after a scheduled
            # event, a deadline, or an institutional decision already in this world's queue —
            # never an arbitrary "mid-horizon" point
            if problem.horizon and spec["family"] in ("information", "negotiation", "time"):
                from swm.world_model_v2.state import parse_time
                try:
                    hz = parse_time(problem.horizon)
                except ValueError:
                    hz = None
                anchors = []
                tmodel = getattr(world, "temporal_model", None)
                for d in (getattr(tmodel, "deadlines", None) or []):
                    t = (d.get("timing") or {}).get("ts")
                    if isinstance(t, (int, float)) and now < t and (hz is None or t <= hz):
                        anchors.append((float(t), f"before_deadline:{d.get('label', '')[:40]}"))
                for f in (getattr(tmodel, "scheduled_facts", None) or []):
                    t = f.get("ts")
                    if isinstance(t, (int, float)) and now < t and (hz is None or t <= hz):
                        anchors.append((float(t),
                                        f"after_{str(f.get('fact', 'scheduled_fact'))[:32]}"))
                for dp in (problem.decision_points or []):
                    try:
                        t = parse_time(dp)
                    except (ValueError, TypeError):
                        continue
                    if now < t and (hz is None or t <= hz):
                        anchors.append((float(t), "at_decision_point"))
                for i, (a_ts, a_lbl) in enumerate(sorted(set(anchors))[:3]):
                    add(ActionSchema(action_id=f"{op}{'_' + obj if obj else ''}_at_{i}",
                                     actor=maker, operation=op, object=obj, authority_basis=cap,
                                     timing_ts=a_ts,
                                     params={"timing": a_lbl,
                                             "timing_anchor": "real_scenario_time"}),
                        "authority")

    # --- controllable quantities → set_parameter grids ---------------------------------------------
    for qname, amount in (problem.controllable_resources or {}).items():
        q = (world.quantities or {}).get(qname)
        if q is not None and isinstance(getattr(q, "value", None), (int, float)):
            lo, hi = 0.0, float(amount)
            for i in range(parameter_grid):
                lvl = lo + (hi - lo) * i / max(1, parameter_grid - 1)
                add(ActionSchema(action_id=f"set_{qname}_{round(lvl, 3)}", actor=maker,
                                 operation="set_parameter", object=qname,
                                 params={"value": lvl, "amount": lvl},
                                 required_resources={qname: lvl} if lvl > 0 else {},
                                 timing_ts=now), "controllable_quantity")

    # --- resource transfers along funding/control edges --------------------------------------------
    net = getattr(world, "network", None)
    edges = list(getattr(net, "edges", []) or [])
    for e in edges:
        if e.src != maker:
            continue
        if e.rel in ("communicates_with", "trusts", "influences", "observes"):
            add(ActionSchema(action_id=f"contact_{e.dst}", actor=maker, operation="contact",
                             object=e.dst, recipients=[e.dst], timing_ts=now,
                             params={"channel": e.channel or e.rel}), "network")
        if e.rel in ("controls", "funds"):
            for rname, amount in (problem.controllable_resources or {}).items():
                if amount and float(amount) > 0:
                    add(ActionSchema(action_id=f"transfer_{rname}_to_{e.dst}", actor=maker,
                                     operation="transfer", object=e.dst,
                                     params={"resource": rname, "amount": float(amount) / 2},
                                     required_resources={rname: float(amount) / 2},
                                     recipients=[e.dst], timing_ts=now), "network")

    # --- institutional procedures naming the maker as a holder -------------------------------------
    for iid, inst in (world.institutions or {}).items():
        for rule in getattr(inst, "rules", []) or []:
            holders = (rule.params or {}).get("holders", [])
            for op in (rule.params or {}).get("actions", []):
                if maker in holders and operation_registered(op):
                    add(ActionSchema(action_id=f"{op}_{iid}", actor=maker, operation=op,
                                     object=iid, institutional_permission=iid,
                                     authority_basis=f"institution:{iid}", timing_ts=now),
                        "institution")

    # --- pending scheduled events the maker participates in → delay / accelerate / cancel ----------
    # (plan-scheduled events are visible via world quantities only after queue build; the canonical
    #  seam is the plan, so api.py passes plan.scheduled_events through problem.meta if available)
    for ev in (getattr(problem, "output", {}) or {}).get("_plan_scheduled_events", []) or []:
        if maker in (ev.get("participants") or []):
            eid = ev.get("etype", "event")
            for op, shift in (("delay", +3 * 86400.0), ("accelerate", -3 * 86400.0)):
                add(ActionSchema(action_id=f"{op}_{eid}", actor=maker, operation=op, object=eid,
                                 params={"shift_s": shift, "etype": eid}, timing_ts=now),
                    "scheduled_event")

    # --- information ledger: disclose privately-held items -----------------------------------------
    info = getattr(world, "information", None)
    if info is not None:
        for item in list(getattr(info, "items", {}).values())[:6]:
            if getattr(item, "kind", "") == "private" and getattr(item, "source", "") == maker:
                add(ActionSchema(action_id=f"disclose_{item.item_id}", actor=maker,
                                 operation="disclose", object=item.item_id, timing_ts=now,
                                 params={"item_id": item.item_id}), "information")

    # --- constrained LLM proposer (ONE source among many; deterministically validated) -------------
    if llm is not None:
        rep_llm = _llm_proposals(world, problem, llm, max_llm_proposals)
        for a in rep_llm["accepted"]:
            add(a, "llm_proposer")
        rep.rejected.extend(rep_llm["rejected"])

    rep.candidates, rep.deduped = dedupe(rep.candidates)
    return rep


def _llm_proposals(world, problem, llm, k: int) -> dict:
    """Ask the LLM for ≤k typed proposals; validate DETERMINISTICALLY. The LLM may propose semantic
    candidates/parameters/branches; it may not touch hidden state, outcomes, beliefs, or authority."""
    from swm.world_model_v2.phase13.ontology import OPERATION_FAMILIES, operations_in_family
    ops = {f: operations_in_family(f)[:8] for f in OPERATION_FAMILIES}
    prompt = (
        "You propose candidate ACTIONS for a decision-maker in a causal world simulation.\n"
        f"DECISION-MAKER: {problem.decision_maker}   CONTEXT: {problem.context[:400]}\n"
        f"AUTHORITY: {problem.authority}   RESOURCES: {problem.controllable_resources}\n"
        f"ENTITIES: {sorted((world.entities or {}).keys())[:12]}\n"
        f"OPERATIONS (choose only from these): {json.dumps(ops)}\n"
        f"Propose up to {k} DIVERSE actions as a JSON list. Each: {{\"operation\":..., \"object\":..., "
        "\"params\":{...}, \"recipients\":[...], \"why\":...}}. You may NOT set beliefs, outcomes, "
        "hidden state, or grant authority. Return ONLY the JSON list.")
    accepted, rejected = [], []
    try:
        raw = llm(prompt) if callable(llm) else ""
        s = str(raw)
        arr = json.loads(s[s.find("["):s.rfind("]") + 1])
    except Exception as e:  # noqa: BLE001 — proposer failure is recorded, generation continues
        return {"accepted": [], "rejected": [{"proposal": "llm_call", "reason": f"{type(e).__name__}"}]}
    for i, p in enumerate(arr[:k]):
        if not isinstance(p, dict):
            rejected.append({"proposal": str(p)[:80], "reason": "not an object"})
            continue
        op = str(p.get("operation", ""))
        if op in _FORBIDDEN_LLM_OPS:
            rejected.append({"proposal": op, "reason": "forbidden operation class for LLM proposals"})
            continue
        if not operation_registered(op):
            rejected.append({"proposal": op, "reason": "unregistered operation"})
            continue
        params = p.get("params") if isinstance(p.get("params"), dict) else {}
        if any(str(key).startswith("_") for key in params):
            rejected.append({"proposal": op, "reason": "private-parameter injection rejected"})
            continue
        accepted.append(ActionSchema(
            action_id=f"llm_{op}_{i}", actor=problem.decision_maker, operation=op,
            object=str(p.get("object", ""))[:60], params=params,
            recipients=[str(r) for r in (p.get("recipients") or [])][:6],
            timing_ts=float(world.clock.now) if world is not None else 0.0,
            meta={"why": str(p.get("why", ""))[:200]}))
    return {"accepted": accepted, "rejected": rejected}
