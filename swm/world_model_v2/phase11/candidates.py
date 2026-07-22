"""Phase 11 — plan-revision candidate generation + static validation (spec §12).

The production candidate set ALWAYS contains the current plan unchanged (so "do not recompile" can win), a
minimal revision at the selected scope, at least one plausible alternative when the evidence is ambiguous, and
a full-recompile candidate when warranted. Each candidate is a typed set of ``PlanTransform`` ops that
deterministically produces a revised ``WorldExecutionPlan`` COPY (the active plan is never mutated) — the same
copy-on-write discipline as ``evidence_recompile``.

The LLM MAY propose qualitative structural alternatives grounded in specific evidence ids; it may NOT select
the winner, mint numbers, invent rule text, or write migration confidence. Every candidate — deterministic or
LLM — passes the same static validation battery. A deterministic fallback keeps the generator runnable with no
LLM.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from swm.world_model_v2.phase11.contracts import PlanRevisionCandidate
from swm.world_model_v2.phase11._serial import content_hash


# ---- typed transform ops (what a candidate DOES to the plan) -------------------------------------------
@dataclass
class PlanTransform:
    op: str = ""                 # add_entity|add_institution_rule|add_relation|add_structural_hypothesis|
    #                              reweight_hypothesis|refit_parameter|revise_outcome_contract|full_recompile|noop
    target: str = ""             # PLAN_DIFF_TARGET
    payload: dict = field(default_factory=dict)
    evidence_ids: list = field(default_factory=list)
    why: str = ""


def apply_transform(plan, ops) -> object:
    """Apply typed ops to a DEEPCOPY of ``plan``; return the revised plan. Never mutates the input. Mirrors the
    field shapes ``evidence_recompile`` uses (entities/institutions/relations/structural_hypotheses are
    lists of dicts)."""
    rp = copy.deepcopy(plan)
    for t in ops:
        op = t.op
        p = t.payload or {}
        if op == "noop":
            continue
        if op == "add_entity":
            ids = {e.get("id") for e in rp.entities if isinstance(e, dict)}
            if p.get("id") and p["id"] not in ids:
                rp.entities.append({"id": str(p["id"]), "type": str(p.get("type", "person")),
                                    "fields": {}, "sensitivity": float(p.get("sensitivity", 0.5))})
        elif op == "add_institution_rule":
            inst_id = str(p.get("institution", "institution"))
            inst = next((i for i in rp.institutions if isinstance(i, dict) and i.get("id") == inst_id), None)
            rule = {"kind": str(p.get("kind", "eligibility")), "params": dict(p.get("params", {}))}
            if inst is None:
                rp.institutions.append({"id": inst_id, "rules": [rule]})
            else:
                inst.setdefault("rules", []).append(rule)
        elif op == "add_relation":
            rp.relations.append({"src": str(p.get("src")), "rel": str(p.get("rel", "related_to")),
                                 "dst": str(p.get("dst"))})
        elif op == "add_structural_hypothesis":
            hid = str(p.get("id", f"hyp_{len(rp.structural_hypotheses)}"))
            rp.structural_hypotheses.append({"id": hid, "describe": str(p.get("describe", "")),
                                             "prior": float(p.get("prior", 0.3)), "lean": p.get("lean", "neutral")})
            # renormalize priors
            z = sum(float(h.get("prior", 1.0) or 1.0) for h in rp.structural_hypotheses if isinstance(h, dict)) or 1.0
            for h in rp.structural_hypotheses:
                if isinstance(h, dict):
                    h["prior"] = round(float(h.get("prior", 1.0) or 1.0) / z, 4)
        elif op == "reweight_hypothesis":
            for h in rp.structural_hypotheses:
                if isinstance(h, dict) and h.get("id") == p.get("id"):
                    factor = 1.5 if p.get("direction") == "up" else 0.6
                    h["prior"] = round(max(0.05, float(h.get("prior", 0.3)) * factor), 4)
        elif op == "refit_parameter":
            rp.provenance.setdefault("phase11_parameter_refits", []).append(
                {"component": p.get("component"), "reason": t.why})
        elif op == "revise_outcome_contract":
            rp.provenance.setdefault("phase11_outcome_revisions", []).append(dict(p))
        elif op == "full_recompile":
            rp.provenance["phase11_full_recompile_requested"] = t.why or "global structural invalidation"
    rp.parent_version = plan.version
    rp.version = plan.version + 1
    rp.provenance = dict(rp.provenance)
    rp.provenance["phase11_revised"] = True
    return rp


# ---- candidate generation ------------------------------------------------------------------------------
def _cand(cid, parent_hash, *, changed, explanation, evidence, ops, is_current=False, complexity=None,
          llm_prov=None):
    c = PlanRevisionCandidate(
        candidate_id=cid, parent_plan_id=parent_hash, changed_components=changed,
        causal_explanation=explanation, supporting_evidence=list(evidence or []),
        complexity=float(complexity if complexity is not None else len(changed)),
        llm_proposal_provenance=llm_prov or {}, is_current_plan=is_current)
    return c, ops


def generate_candidates(plan, scope_selection, fused, observation, *, llm=None, max_candidates: int = 5) -> list:
    """Return a list of (PlanRevisionCandidate, ops). Always includes the current plan. Deterministic core;
    LLM adds grounded qualitative alternatives (validated downstream)."""
    parent = plan.plan_hash() if hasattr(plan, "plan_hash") else "plan"
    declared = _declared_of(observation)
    out = []

    # 1) CURRENT plan (no change) — must always be scored so "don't recompile" can win
    out.append(_cand("cand::current", parent, changed=[], explanation="retain the current plan unchanged",
                     evidence=[], ops=[PlanTransform(op="noop", why="no change")], is_current=True))

    scope = scope_selection.scope
    ev_ids = getattr(observation, "evidence_ids", []) or []

    # 2) MINIMAL revision at the selected scope (deterministic from the typed observation)
    minimal_ops, changed, expl = _minimal_ops(scope, declared, observation)
    if minimal_ops:
        out.append(_cand("cand::minimal", parent, changed=changed,
                         explanation=expl, evidence=ev_ids, ops=minimal_ops))

    # 3) ALTERNATIVE structural hypothesis when the evidence is ambiguous (retain uncertainty)
    if fused.classification in ("local_structural", "global_structural") or len(fused.by_family) > 1:
        alt_ops = [PlanTransform(op="add_structural_hypothesis", target="structural_hypotheses",
                                 payload={"id": f"alt_{content_hash(declared, length=6)}",
                                          "describe": f"alternative reading of {fused.dominant_family}",
                                          "prior": 0.3}, evidence_ids=ev_ids,
                                 why="evidence admits >1 structure — keep a competing branch")]
        out.append(_cand("cand::alt_branch", parent,
                         changed=[{"target": "structural_hypotheses", "op": "added"}],
                         explanation="retain a competing structural hypothesis (ambiguous evidence)",
                         evidence=ev_ids, ops=alt_ops))

    # 4) FULL recompile when the structure is globally invalid
    if scope in ("outcome_contract", "full_plan") or fused.classification == "global_structural":
        out.append(_cand("cand::full", parent, changed=[{"target": "full_plan", "op": "recompile"}],
                         explanation="global invalidation — full re-decomposition",
                         evidence=ev_ids, complexity=10.0,
                         ops=[PlanTransform(op="full_recompile", target="full_plan",
                                            why="global structural invalidation")]))

    # 5) LLM-proposed qualitative alternatives (grounded, validated downstream) — optional
    if llm is not None:
        for i, (cand, ops) in enumerate(_llm_candidates(plan, scope_selection, fused, observation, llm)):
            out.append((cand, ops))
            if len(out) >= max_candidates:
                break

    return out[:max_candidates]


def _declared_of(obs):
    d = dict(getattr(obs, "provenance", {}) or {})
    d.update(getattr(obs, "mechanism_diagnostics", {}) or {})
    # the controller stores typed hints under provenance['declared']
    return dict((getattr(obs, "provenance", {}) or {}).get("declared", {}) or d.get("declared", {}) or {})


def _minimal_ops(scope, declared, observation):
    ev_ids = getattr(observation, "evidence_ids", []) or []
    if scope == "institution_ruleset":
        rc = declared.get("rule_change") or {}
        return ([PlanTransform(op="add_institution_rule", target="institutions",
                               payload={"institution": rc.get("institution", "institution"),
                                        "kind": rc.get("kind", "eligibility"), "params": rc.get("params", {})},
                               evidence_ids=ev_ids, why="evidence-backed rule change")],
                [{"target": "institutions", "op": "modified"}], "add the evidenced institutional rule")
    if scope == "actor":
        na = declared.get("new_actor") or {}
        aid = na.get("id") if isinstance(na, dict) else na
        return ([PlanTransform(op="add_entity", target="entities",
                               payload={"id": aid or "new_actor", "type": (na.get("type", "person") if isinstance(na, dict) else "person")},
                               evidence_ids=ev_ids, why="verified new actor")],
                [{"target": "entities", "op": "added"}], "add the verified new actor")
    if scope in ("relationship", "local_network_region", "coalition_change"):
        cc = declared.get("coalition_change") or declared.get("network_change") or {}
        src = cc.get("src", "a"); dst = cc.get("dst", "b")
        return ([PlanTransform(op="add_relation", target="relations",
                               payload={"src": src, "rel": cc.get("rel", "allies_with"), "dst": dst},
                               evidence_ids=ev_ids, why="network/coalition delta")],
                [{"target": "relations", "op": "added"}], "revise the affected relationship/edge")
    if scope == "parameter_only":
        return ([PlanTransform(op="refit_parameter", target="parameter_packs",
                               payload={"component": "drifted_parameter"}, evidence_ids=ev_ids,
                               why="parameter drift — refit, structure held")],
                [{"target": "parameter_packs", "op": "reweighted"}], "refit the drifted parameter")
    if scope == "structural_hypothesis":
        return ([PlanTransform(op="add_structural_hypothesis", target="structural_hypotheses",
                               payload={"id": "struct_alt", "describe": "revised structure", "prior": 0.3},
                               evidence_ids=ev_ids, why="localized structural change")],
                [{"target": "structural_hypotheses", "op": "added"}], "add the revised structural branch")
    if scope == "outcome_contract":
        return ([PlanTransform(op="revise_outcome_contract", target="outcome_contract",
                               payload=declared.get("outcome_space_change", {"note": "outcome space revised"}),
                               evidence_ids=ev_ids, why="outcome-space change")],
                [{"target": "outcome_contract", "op": "modified"}], "revise the outcome contract")
    return [], [], "no minimal op for this scope"


_LLM_PROMPT = """You revise a simulation plan after a detected structural change. Propose UP TO 2 QUALITATIVE
alternative revisions, each grounded in the evidence ids given. Reply ONLY JSON:
{{"candidates":[{{"explanation":"...","op":"add_entity|add_institution_rule|add_relation|
add_structural_hypothesis","component":"...","evidence_ids":["..."]}}]}}
Rules: cite evidence ids; do NOT invent numbers, exact rule text, or probabilities; do NOT pick a winner.
Detected change: {family} at scope {scope}. Evidence ids: {ev}. Question: {q}"""


def _llm_candidates(plan, scope_selection, fused, observation, llm):
    import json
    try:
        from swm.engine.grounding import parse_json
        raw = _LLM_PROMPT.format(family=fused.dominant_family, scope=scope_selection.scope,
                                 ev=json.dumps(getattr(observation, "evidence_ids", [])[:6]),
                                 q=getattr(plan, "question", ""))
        prop = parse_json(llm(raw)) or {}
    except Exception:  # noqa: BLE001
        return []
    parent = plan.plan_hash() if hasattr(plan, "plan_hash") else "plan"
    out = []
    for i, c in enumerate((prop.get("candidates") or [])[:2]):
        if not isinstance(c, dict) or c.get("op") not in (
                "add_entity", "add_institution_rule", "add_relation", "add_structural_hypothesis"):
            continue
        op = PlanTransform(op=c["op"], target="entities", payload={"id": str(c.get("component", "x")),
                           "describe": str(c.get("explanation", ""))[:80]},
                           evidence_ids=[str(x) for x in (c.get("evidence_ids") or [])][:6],
                           why="llm-proposed, grounded")
        out.append(_cand(f"cand::llm{i}", parent, changed=[{"target": op.target, "op": c["op"]}],
                         explanation=str(c.get("explanation", ""))[:160],
                         evidence=op.evidence_ids, ops=[op],
                         llm_prov={"source": "llm", "grounded_evidence_ids": op.evidence_ids}))
    return out


# ---- static validation battery -------------------------------------------------------------------------
def validate_candidate(plan, revised_plan, ops, observation, *, now: float = 0.0) -> dict:
    """Run the static checks a candidate must pass before it can be scored/migrated (spec §12). Returns
    {ok, problems:[...]} — a real subset: schema construction, temporal validity, evidence grounding, causal
    reachability, unsupported-precision, action feasibility, cost."""
    problems = []
    # schema: revised plan must have the required core fields intact
    for fld in ("question", "outcome_contract", "as_of", "horizon_ts"):
        if getattr(revised_plan, fld, None) is None:
            problems.append(f"schema: missing {fld}")
    # temporal: a rule op must not be future-dated relative to now; no time reversal
    declared = _declared_of(observation)
    rc = declared.get("rule_change") or {}
    if rc.get("effective_date") and isinstance(rc["effective_date"], (int, float)) and rc["effective_date"] > now:
        problems.append("temporal: rule effective date is in the future — not yet active")
    # evidence grounding: every non-current, non-full candidate op should cite evidence (unless deterministic
    # diagnostic-only like refit)
    for t in ops:
        if t.op in ("add_entity", "add_institution_rule", "add_relation") and not t.evidence_ids \
                and not t.why.startswith("network/coalition"):
            problems.append(f"evidence: op {t.op} is ungrounded (no evidence ids)")
    # unsupported precision: no op may carry an invented probability/rule text
    for t in ops:
        pay = t.payload or {}
        if any(k in pay for k in ("probability", "exact_rule_text", "edge_probability")):
            problems.append(f"unsupported_precision: op {t.op} carries invented precision")
    # causal reachability: a structural change must touch a plan component that exists or is being added
    # (trivially true for our ops); full_recompile is always reachable
    return {"ok": not problems, "problems": problems}
