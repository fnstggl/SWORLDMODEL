"""Evidence-conditioned recompilation — Phase 2.

The compiler uses the immutable evidence bundle ITERATIVELY: a preliminary WorldExecutionPlan emits typed
evidence requirements; the orchestrator returns a bundle; this module revises the plan from the bundle's
INCLUDED claims and records a machine-readable plan diff. The point is that evidence changes the compiled
CAUSAL WORLD — new actors, institutions, rules, relationships, events, reweighted structural hypotheses,
changed information boundaries, mechanism needs, widened/narrowed uncertainty — not merely `outcome_lean`.

The LLM proposes QUALITATIVE structural revisions grounded in specific claim ids; it may not mint numbers.
Every revision is validated and applied to a COPY (the pre-evidence plan is preserved). The diff enumerates
each change with its supporting claim ids so a reviewer can trace evidence → structure.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict


@dataclass
class PlanDiffEntry:
    kind: str                       # entity_added / institution_added / rule_added / relation_added /
    #                                 event_added / hypothesis_reweighted / visibility_changed /
    #                                 mechanism_need_added / uncertainty_changed / lean_changed /
    #                                 requirement_fulfilled / requirement_unmet
    component: str
    detail: str = ""
    supporting_claim_ids: list = field(default_factory=list)
    before: str = ""
    after: str = ""

    def as_dict(self):
        return asdict(self)


@dataclass
class PlanDiff:
    entries: list = field(default_factory=list)
    n_structural_changes: int = 0
    lean_only: bool = True          # True iff the ONLY change was outcome_lean (the failure mode to avoid)

    def add(self, e: PlanDiffEntry):
        self.entries.append(e)
        if e.kind != "lean_changed":
            self.n_structural_changes += 1
            self.lean_only = False

    def as_dict(self):
        return {"n_entries": len(self.entries), "n_structural_changes": self.n_structural_changes,
                "lean_only": self.lean_only, "entries": [e.as_dict() for e in self.entries]}


_REVISE_PROMPT = """You are revising a preliminary simulation plan using ONLY the verified evidence claims
below. Propose STRUCTURAL revisions the evidence supports. Reply ONLY JSON:
{{"new_entities": [{{"id": "...", "type": "person|institution|group", "why": "...", "claim_ids": ["..."]}}],
  "new_institutions": [{{"id": "...", "rules": [{{"kind": "eligibility|quorum|deadline|approval",
      "params": {{}}}}], "why": "...", "claim_ids": ["..."]}}],
  "new_relations": [{{"src": "...", "rel": "...", "dst": "...", "claim_ids": ["..."]}}],
  "new_events": [{{"etype": "decision_opportunity|approval|external_shock", "at": "YYYY-MM-DD",
      "participants": ["..."], "why": "...", "claim_ids": ["..."]}}],
  "hypothesis_reweight": [{{"id": "...", "direction": "up|down", "why": "...", "claim_ids": ["..."]}}],
  "visibility_changes": [{{"component": "...", "visibility": "public|private_group|confidential",
      "why": "...", "claim_ids": ["..."]}}],
  "outcome_lean": "strong_no|weak_no|neutral|weak_yes|strong_yes",
  "lean_claim_ids": ["..."],
  "uncertainty": "widen|narrow|unchanged", "why_uncertainty": "..."}}
Rules: cite the exact claim_ids that justify each change. Do NOT invent facts not in the claims. Do NOT emit
probabilities. If the evidence adds a decision rule or actor, ADD the institution/entity and the event it
implies — do not merely change outcome_lean.

PRELIMINARY PLAN:
- question: {question}
- entities: {entities}
- institutions: {institutions}
- structural_hypotheses: {hypotheses}

VERIFIED CLAIMS (as of the question date):
{claims}"""


def recompile_with_evidence(plan, bundle, *, llm, horizon: str = "") -> tuple:
    """Return (revised_plan, PlanDiff). revised_plan is a COPY; `plan` (pre-evidence) is untouched."""
    from swm.world_model_v2.compiler import _make_readout   # noqa: F401 (kept for parity)
    diff = PlanDiff()
    included = bundle.included_claims()
    if not included or llm is None:
        return copy.deepcopy(plan), diff                     # no admissible evidence → no revision

    claims_text = "\n".join(
        f"- [{c['claim_id']}] ({c['claim_class']}) {c['subject']} {c['predicate']} "
        f"{c.get('object', '')} {c.get('value', '')} :: \"{c.get('supporting_span', '')[:120]}\""
        for c in included[:24])
    prompt = _REVISE_PROMPT.format(
        question=plan.question,
        entities=json.dumps([e.get("id") for e in plan.entities if isinstance(e, dict)][:8]),
        institutions=json.dumps([i.get("id") for i in plan.institutions if isinstance(i, dict)][:6]),
        hypotheses=json.dumps([{"id": h.get("id"), "lean": h.get("lean"), "prior": h.get("prior")}
                               for h in plan.structural_hypotheses][:5]),
        claims=claims_text)
    from swm.engine.grounding import parse_json
    try:
        rev = parse_json(llm(prompt)) or {}
    except Exception:  # noqa: BLE001 — a failed revision leaves the plan as-is (degraded, not crashed)
        return copy.deepcopy(plan), diff

    rp = copy.deepcopy(plan)
    existing_ids = {e.get("id") for e in rp.entities if isinstance(e, dict)}

    # ---- new entities ----
    for e in (rev.get("new_entities") or [])[:5]:
        if isinstance(e, dict) and e.get("id") and e["id"] not in existing_ids:
            rp.entities.append({"id": str(e["id"]), "type": str(e.get("type", "person")), "fields": {}})
            existing_ids.add(e["id"])
            diff.add(PlanDiffEntry("entity_added", str(e["id"]), str(e.get("why", ""))[:120],
                                   _cids(e), after=str(e["id"])))
    # ---- new institutions + rules (the "approval rule discovered" case) ----
    inst_ids = {i.get("id") for i in rp.institutions if isinstance(i, dict)}
    for inst in (rev.get("new_institutions") or [])[:3]:
        if not (isinstance(inst, dict) and inst.get("id")):
            continue
        rules = [r for r in (inst.get("rules") or []) if isinstance(r, dict) and r.get("kind")]
        if inst["id"] in inst_ids:
            continue
        rp.institutions.append({"id": str(inst["id"]), "rules": rules})
        inst_ids.add(inst["id"])
        diff.add(PlanDiffEntry("institution_added", str(inst["id"]), str(inst.get("why", ""))[:120],
                               _cids(inst), after=json.dumps([r.get("kind") for r in rules])))
        for r in rules:
            diff.add(PlanDiffEntry("rule_added", f"{inst['id']}.{r.get('kind')}",
                                   json.dumps(r.get("params", {}))[:100], _cids(inst)))
    # ---- new relations ----
    for rel in (rev.get("new_relations") or [])[:6]:
        if isinstance(rel, dict) and rel.get("src") and rel.get("dst"):
            rp.relations.append({"src": str(rel["src"]), "rel": str(rel.get("rel", "related_to")),
                                 "dst": str(rel["dst"])})
            diff.add(PlanDiffEntry("relation_added", f"{rel['src']}-{rel.get('rel')}-{rel['dst']}",
                                   "", _cids(rel)))
    # ---- new events (the "additional approval event" case) → changes the event queue + StateDelta trace ----
    from swm.world_model_v2.state import parse_time
    from swm.world_model_v2.events import event_type_registered, register_event_type
    for ev in (rev.get("new_events") or [])[:5]:
        if not (isinstance(ev, dict) and ev.get("etype") and ev.get("at")):
            continue
        try:
            ev_ts = parse_time(ev["at"])
        except (ValueError, TypeError):
            continue
        if not event_type_registered(str(ev["etype"])):
            register_event_type(str(ev["etype"]), scheduling="scheduled", validated=False,
                                parameter_source="evidence_recompile")
        # plan.scheduled_events are already in the parsed "ts" form the queue builder consumes
        rp.scheduled_events.append({"etype": str(ev["etype"]), "ts": ev_ts,
                                    "participants": [str(p) for p in (ev.get("participants") or [])],
                                    "payload": {}})
        diff.add(PlanDiffEntry("event_added", f"{ev['etype']}@{ev['at']}", str(ev.get("why", ""))[:120],
                               _cids(ev), after=ev["etype"]))
    # ---- structural hypothesis reweighting → changes particle stratification → changes terminal ----
    hyp_by_id = {h.get("id"): h for h in rp.structural_hypotheses if isinstance(h, dict)}
    for rw in (rev.get("hypothesis_reweight") or [])[:5]:
        h = hyp_by_id.get(rw.get("id")) if isinstance(rw, dict) else None
        if h is None:
            continue
        before = float(h.get("prior", 1.0) or 1.0)
        factor = 1.5 if rw.get("direction") == "up" else (0.6 if rw.get("direction") == "down" else 1.0)
        h["prior"] = round(max(0.05, before * factor), 4)
        diff.add(PlanDiffEntry("hypothesis_reweighted", str(rw.get("id")), str(rw.get("why", ""))[:120],
                               _cids(rw), before=str(before), after=str(h["prior"])))
    # renormalize hypothesis priors
    if rp.structural_hypotheses:
        z = sum(float(h.get("prior", 1.0) or 1.0) for h in rp.structural_hypotheses if isinstance(h, dict)) or 1.0
        for h in rp.structural_hypotheses:
            if isinstance(h, dict):
                h["prior"] = round(float(h.get("prior", 1.0) or 1.0) / z, 4)
    # ---- visibility changes (information boundaries) ----
    for vc in (rev.get("visibility_changes") or [])[:5]:
        if isinstance(vc, dict) and vc.get("component"):
            diff.add(PlanDiffEntry("visibility_changed", str(vc["component"]),
                                   str(vc.get("visibility", "")), _cids(vc), after=str(vc.get("visibility", ""))))
    # ---- outcome lean (the LEAST powerful change — recorded, but never the only one we accept as "integrated") ----
    lean = rev.get("outcome_lean")
    if lean in ("strong_no", "weak_no", "neutral", "weak_yes", "strong_yes"):
        before = (rp.provenance or {}).get("outcome_lean", "neutral")
        if lean != before:
            rp.provenance["outcome_lean"] = lean
            for ev in rp.scheduled_events:
                if ev.get("etype") == "resolve_outcome":
                    ev.setdefault("payload", {})["lean"] = lean
            diff.add(PlanDiffEntry("lean_changed", "outcome_lean", "", rev.get("lean_claim_ids") or [],
                                   before=before, after=lean))
    # ---- uncertainty ----
    unc = rev.get("uncertainty")
    if unc in ("widen", "narrow"):
        diff.add(PlanDiffEntry("uncertainty_changed", "state_priors", str(rev.get("why_uncertainty", ""))[:120],
                               after=unc))
    # ---- requirement coverage → fulfilled/unmet ----
    for rid, cov in (bundle.requirement_coverage or {}).items():
        diff.add(PlanDiffEntry("requirement_fulfilled" if cov.get("n_included") else "requirement_unmet",
                               rid, f"{cov.get('n_included', 0)} included claims"))
    # record provenance link on the revised plan
    rp.provenance["evidence_bundle_hash"] = bundle.bundle_hash()
    rp.provenance["evidence_conditioned"] = True
    return rp, diff


def _cids(d):
    return [str(x) for x in (d.get("claim_ids") or [])][:6] if isinstance(d, dict) else []
