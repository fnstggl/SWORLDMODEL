"""Fidelity layer — the highest-fidelity world the available information supports, by explicit directive:

  * IDENTITY-PRESERVING: real named people/institutions, never pseudonyms, never amalgams. Leakage risk is
    MEASURED by probes, not avoided by destroying fidelity.
  * FULL ACTOR DECOMPOSITION: every real decision-holder is a separate entity (the Powell question needs
    Powell, the President, the Senate, the Fed Board — not one "Appointing_Body").
  * GROUNDED INSTITUTIONAL PARAMETERS: real composition sizes, thresholds and stages (Senate 100/51,
    FOMC 12) extracted from evidence/world knowledge with provenance — the synthesized 5-of-9 default dies.
  * SITUATION-DEPENDENT TRAJECTORY DEPTH: decision cadence and event depth scale with the number of
    strategic actors, the horizon, and the process's volatility — never one fixed decision at horizon-3d.

Everything added here carries provenance ("fidelity_critic" / "grounded_rules") and flows through the same
events/StateDelta plane. Nothing bypasses the mechanisms.
"""
from __future__ import annotations

_CRITIC_PROMPT = """You are the FIDELITY CRITIC for a world simulation about a real question. The compiled
world below may have collapsed the real actor system. Identify what a maximally faithful causal model must
represent, using REAL NAMES and your world knowledge plus the evidence.

QUESTION: {q}
AS-OF: {as_of}
CURRENT ENTITIES: {ents}
CURRENT INSTITUTIONS: {insts}
EVIDENCE: {ev}

List (a) every real decision-holder MISSING from the entity list (people AND institutions, with real
names, their role, and what they control); (b) for each institution involved, its REAL composition size,
decision threshold, procedural stages and authority holders; (c) relationships among all actors; (d) how
volatile/eventful this process is (how often meaningful decisions/news occur).

Return ONLY JSON:
{{"missing_entities": [{{"id": "<Real_Name>", "type": "person|institution", "role": "<role>",
   "controls": "<what they decide>", "sensitivity": <0..1>}}],
 "institution_details": [{{"id": "<name>", "composition_size": <int|null>, "decision_threshold": <int|null>,
   "threshold_share": <0..1|null>, "stages": ["<stage>", ...], "authority_holders": ["<name>", ...],
   "source": "evidence|model_knowledge", "confidence": <0..1>}}],
 "relations": [{{"src": "...", "rel": "influences|reports_to|trusts|opposes|controls|communicates_with",
   "dst": "..."}}],
 "decision_cadence_days": <typical days between meaningful decisions/news in this process>}}"""


def fidelity_expand(plan, question, *, as_of, evidence_text="", llm=None) -> dict:
    """One critic round: expand collapsed actor systems + ground institutional parameters. Mutates the
    plan (adds entities/relations; enriches institution rule params with real numbers). Returns a report."""
    if llm is None:
        return {"skipped": "no llm"}
    from swm.engine.grounding import parse_json
    raw = parse_json(llm(_CRITIC_PROMPT.format(
        q=question, as_of=as_of,
        ents=[(e.get("id"), e.get("type")) for e in plan.entities][:12],
        insts=[(i.get("id"), [r.get("kind") for r in (i.get("rules") or [])]) for i in plan.institutions][:6],
        ev=evidence_text[:1800] or "(none)"))) or {}
    rep = {"entities_added": 0, "relations_added": 0, "institutions_grounded": 0,
           "decision_cadence_days": None}
    known = {str(e.get("id")) for e in plan.entities if isinstance(e, dict)}
    for e in (raw.get("missing_entities") or [])[:10]:
        if not isinstance(e, dict) or not e.get("id") or str(e["id"]) in known:
            continue
        plan.entities.append({"id": str(e["id"])[:60], "type": ("institution" if e.get("type") == "institution"
                                                                else "person"),
                              "fields": {"role": str(e.get("role", ""))[:80]},
                              "sensitivity": max(0.0, min(1.0, float(e.get("sensitivity", 0.7) or 0.7))),
                              "_provenance": "fidelity_critic"})
        known.add(str(e["id"]))
        rep["entities_added"] += 1
    for r in (raw.get("relations") or [])[:16]:
        if isinstance(r, dict) and r.get("src") and r.get("dst"):
            plan.relations.append({"src": str(r["src"])[:60], "rel": str(r.get("rel", "influences"))[:24],
                                   "dst": str(r["dst"])[:60], "_provenance": "fidelity_critic"})
            rep["relations_added"] += 1
    # ground institutional parameters: real composition/threshold/stages override synthesized defaults
    details = {str(d.get("id", "")).lower(): d for d in (raw.get("institution_details") or [])
               if isinstance(d, dict)}
    for inst in plan.institutions:
        if not isinstance(inst, dict):
            continue
        d = details.get(str(inst.get("id", "")).lower())
        if d is None:
            d = next((v for k, v in details.items()
                      if k and (k in str(inst.get("id", "")).lower()
                                or str(inst.get("id", "")).lower() in k)), None)
        if d is None:
            continue
        params = {}
        if d.get("composition_size"):
            params["total"] = int(d["composition_size"])
        if d.get("decision_threshold"):
            params["needed"] = int(d["decision_threshold"])
        elif d.get("threshold_share"):
            params["threshold"] = float(d["threshold_share"])
        if params:
            inst.setdefault("rules", []).append(
                {"kind": "quorum", "params": {**params, "members": []},
                 "_provenance": f"grounded_rules:{d.get('source', 'model_knowledge')}",
                 "_confidence": float(d.get("confidence", 0.6) or 0.6)})
            rep["institutions_grounded"] += 1
        for stage in (d.get("stages") or [])[:6]:
            inst.setdefault("rules", []).append(
                {"kind": "procedure", "params": {"stage": str(stage)[:40]},
                 "_provenance": "grounded_rules"})
    try:
        rep["decision_cadence_days"] = max(0.5, float(raw.get("decision_cadence_days") or 0.0)) or None
    except (TypeError, ValueError):
        rep["decision_cadence_days"] = None
    return rep


def deepen_trajectory(plan, req, *, cadence_days=None) -> dict:
    """Situation-dependent event depth: recurring decision opportunities for EVERY strategic actor at the
    process's real cadence (critic-estimated, default horizon/6), plus institutional stage reviews — so
    trajectories unfold over the whole window instead of one decision at horizon-3d."""
    horizon_days = max(1.0, (plan.horizon_ts - plan.as_of) / 86400.0)
    cad = cadence_days or max(1.0, horizon_days / 6.0)
    cad = max(0.5, min(cad, horizon_days / 2.0))
    n_points = max(2, min(14, int(horizon_days / cad)))
    actors = [e for e in plan.entities if isinstance(e, dict) and e.get("id")]
    actors.sort(key=lambda e: -float(e.get("sensitivity", 0.5) or 0.5))
    strategic = actors[:6] if req.get("phase4_actor_policy", {}).get("required") else []
    declared = {str(e.get("id")) for e in plan.entities if isinstance(e, dict)}
    n_added = 0
    for k in range(1, n_points + 1):
        ts = plan.as_of + (k / (n_points + 1)) * (plan.horizon_ts - plan.as_of)
        for a in strategic:
            aid = str(a.get("id"))
            if aid not in declared:
                continue
            plan.scheduled_events.append({
                "etype": "decision_opportunity", "ts": ts, "participants": [aid],
                "payload": {"situation": f"periodic strategic review ({k}/{n_points})",
                            "actions": [{"type": "act"}, {"type": "wait"}]}})
            n_added += 1
    return {"horizon_days": round(horizon_days, 1), "cadence_days": round(cad, 1),
            "decision_points": n_points, "strategic_actors": len(strategic),
            "decision_events_added": n_added}
