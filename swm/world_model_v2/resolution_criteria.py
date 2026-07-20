"""Universal resolution-criterion parsing + per-actor evidence-grounded intentions.

Both are QUESTION-GENERAL: they operate on any compiled plan from its own text/evidence/entities, with no
scenario branching.

RESOLUTION CRITERION — the Powell trace exposed the failure class: "out as Fed Chair" (steps down as chair
= YES) was arbitrated against news saying he "stays at the Fed" (a different predicate). The parser turns
the question into a precise machine-readable criterion (subject, predicate, target state, deadline,
disambiguations of near-miss states), which then anchors (a) the outcome contract's resolution rule,
(b) the scheduled-facts extractor's entailment judgments, (c) the fidelity critic.

ACTOR INTENTIONS — a strategic actor's PUBLIC STATED intention (with quote + date) is world state. The
LLM only ever CLASSIFIES (commitment level, pathway, target mode, graded control, capability,
reliability, basis kind) — it never mints effect sizes, and NOTHING here converts the classification
into a number (§NAP: the old _STANCE_WEIGHT 0.95/0.75/0.4/… aggregation and the `actor_intentions`
share quantity are gone). Grounded stances are written TWO places:
  1. plan._intention_stances — the auditable qualitative record (reports, conversion provenance);
  2. the entity's `stances` field (registered extension) — projected into the actor's own view so the
     actor's OWN situated cognition is conditioned on its stated commitments (the behavior channel).
Plus TYPED commitments on the entity's `commitments` field. A commitment is BINDING only when its
basis is itself a binding instrument (law / treaty / contract / formal institutional rule) — a public
statement, however firm, conditions the actor's reasoning but never hard-blocks feasibility. The
`prohibits` list of a binding commitment is the LITERAL content of the instrument (what its text
actually forbids, quoted) — never a table of hand-authored action magnitudes over a 0.5 threshold
(§NAP: actions_advancing_pathway is quarantined). Ambiguous contradiction stays an actor choice.
"""
from __future__ import annotations

from swm.world_model_v2.state import register_entity_extension

# the grounded stance record is a TYPED extension field so materialization keeps it addressable and
# the ActorViewBuilder can project it into the actor's own view (never simulator-only state)
register_entity_extension("grounded_stances", fields={
    "stances": "evidence-grounded stance records (mode-scoped, graded control, classification only)",
}, entity_types=("person", "institution"))

_CRITERION_PROMPT = """Parse this forecasting question into a PRECISE resolution criterion.
QUESTION: {q}
HORIZON: {horizon}
Return ONLY JSON:
{{"subject": "<who/what>", "predicate": "<the exact state that must hold>",
 "deadline": "<YYYY-MM-DD or null>",
 "resolves_yes_iff": "<one precise sentence>",
 "absorbing_event": "<the concrete EVENT whose first occurrence settles the question>",
 "event_polarity": "occurrence_resolves_yes|occurrence_resolves_no — occurrence_resolves_yes when the \
event happening by the deadline makes the answer YES (a does-X-happen question); occurrence_resolves_no \
when YES means the current state PERSISTS to the deadline and the event breaking it makes the answer NO \
(a remains/still-in-state question)",
 "near_miss_states": [{{"state": "<a state that LOOKS like yes/no but is NOT>", "resolves": "yes|no"}}],
 "notes": "<ambiguities>"}}"""

_INTENTIONS_PROMPT = """For each STRATEGIC ACTOR below, state their PUBLICLY STATED or strongly evidenced
intentions relevant to this question, using the evidence and your knowledge of the real people/institutions.
Only include intentions you can ground. CLASSIFY qualitatively — do NOT invent numeric strengths,
weights, or probabilities anywhere.
An actor may hold SEVERAL stances at once toward different end-states (pursuing their own preferred
outcome while committed to preventing a rival's) — list each separately with its target_mode.

QUESTION: {q}
RESOLUTION CRITERION: {crit}
CANDIDATE END-STATES (modes): {modes}
ACTORS: {actors}
EVIDENCE: {ev}

Return ONLY JSON:
{{"intentions": [{{"actor": "<entity id>", "stated_intention": "<one sentence>",
  "basis_quote": "<short quote or knowledge basis>", "source": "evidence|model_knowledge",
  "basis_kind": "law|treaty|contract|formal_institutional_rule|public_statement|reported_leaning|inference — what KIND of thing the basis is; only a literally binding instrument (law/treaty/contract/formal rule) is 'binding'",
  "commitment_level": "committed_to_prevent|conditionally_opposed|weakly_opposed|neutral|inclined_toward|actively_pursuing|formally_committed — the actor's stance toward the target end-state (universal: works for a deal, a resignation, a rate cut, a bill, a launch)",
  "target_mode": "<the specific end-state id from the CANDIDATE END-STATES this stance concerns, or null when it concerns the resolution as a whole>",
  "pathway": "cooperative_agreement|unilateral_action|institutional_procedure|operational_execution|competitive_interaction|any — which causal pathway this stance concerns (a refusal to negotiate concerns the cooperative path; a vow to fight on concerns the unilateral path; a whip count concerns the institutional path)",
  "control": "sole_authority|veto|agenda_setting|partial_implementation|coalition_member|operational_capability|informal_influence|none — the actor's REAL degree of control over that pathway (a president may want a bill but lack the votes; a legislature may pass one but lack implementation capacity)",
  "capability": "high|medium|low — can the actor practically act on this stance (means, position, resources)?",
  "reliability": "high|medium|low — how reliable the basis is (law/treaty=high, direct public statement=high, reported leaning=medium, inference=low)",
  "explicit_prohibitions": ["<ONLY for a binding basis_kind: the concrete acts the instrument's LITERAL text forbids, each a short verb phrase quoted or closely paraphrased from the instrument; [] otherwise>"],
  "entails_direction": "yes|no|neutral — under the resolution criterion",
  "date": "<YYYY-MM-DD or null>"}}]}}"""

#: qualitative vocabularies — classification targets only; no entry maps to any number (§NAP)
_CONTROLS = ("sole_authority", "veto", "agenda_setting", "partial_implementation", "coalition_member",
             "operational_capability", "informal_influence", "none")
_LEVELS3 = ("high", "medium", "low")
_BINDING_BASIS_KINDS = ("law", "treaty", "contract", "formal_institutional_rule")
_BASIS_KINDS = _BINDING_BASIS_KINDS + ("public_statement", "reported_leaning", "inference")
_STANCE_LEVELS_ALL = (
    "committed_to_prevent", "conditionally_opposed", "weakly_opposed", "neutral",
    "inclined_toward", "actively_pursuing", "formally_committed",
    # legacy agreement-specific labels (older packs/transcripts) map onto the universal set
    "categorical_refusal", "conditional_refusal", "weak_opposition",
    "openness_to_agreement", "formal_commitment_toward_agreement")


def parse_resolution_criterion(question, *, horizon, llm=None) -> dict:
    if llm is None:
        return {}
    from swm.engine.grounding import parse_json
    raw = parse_json(llm(_CRITERION_PROMPT.format(q=question, horizon=horizon))) or {}
    if not raw.get("resolves_yes_iff"):
        return {}
    out = {k: raw.get(k) for k in ("subject", "predicate", "deadline", "resolves_yes_iff",
                                   "absorbing_event", "near_miss_states", "notes")}
    pol = str(raw.get("event_polarity") or "").strip().lower()
    # tolerate the parser echoing the guidance text — keep only a clean polarity token
    for tok in ("occurrence_resolves_yes", "occurrence_resolves_no"):
        if pol.startswith(tok):
            out["event_polarity"] = tok
            break
    return out


def _binding_prohibitions(stance: dict) -> list:
    """§NAP: a commitment is binding ONLY when its basis is itself a literally binding instrument
    (law/treaty/contract/formal institutional rule), and its prohibition set is the instrument's
    OWN literal content (`explicit_prohibitions`, quoted from the basis) — never a hand-authored
    action-magnitude table over an arbitrary threshold. A firm public statement conditions the
    actor's own reasoning (the stance rides in its view) but never hard-blocks feasibility.
    Ambiguity → no prohibition: an ambiguous contradiction remains the actor's choice."""
    if str(stance.get("basis_kind", "")) not in _BINDING_BASIS_KINDS:
        return []
    return [str(p)[:60] for p in (stance.get("explicit_prohibitions") or [])
            if isinstance(p, str) and p.strip()][:12]


def ground_actor_intentions(plan, question, *, criterion=None, evidence_text="", llm=None,
                            modes: list = None) -> dict:
    """Extract grounded stances for declared strategic actors; write them onto entity fields
    (stances + typed commitments) so the actor's own cognition consumes them. QUALITATIVE ONLY
    (§NAP): no share quantity, no consumed-state weight, no numeric strength anywhere. Universal —
    uses only the plan's own entities and modes."""
    if llm is None:
        return {"skipped": "no llm"}
    from swm.engine.grounding import parse_json
    actors = [str(e.get("id")) for e in plan.entities if isinstance(e, dict) and e.get("id")][:8]
    if not actors:
        return {"skipped": "no actors"}
    modes = list(modes or getattr(plan, "_canonical_modes", None) or [])
    mode_ids = [str(m.get("id")) for m in modes if isinstance(m, dict) and m.get("id")]
    raw = parse_json(llm(_INTENTIONS_PROMPT.format(
        q=question, crit=(criterion or {}).get("resolves_yes_iff", "(as stated)"),
        modes=(mode_ids or "(none elicited — use null target_mode)"),
        actors=actors, ev=evidence_text[:1600] or "(none)"))) or {}
    ents = {str(e.get("id")): e for e in plan.entities if isinstance(e, dict)}
    n_grounded = 0
    kept, stances = [], []
    per_actor_records = {}
    from swm.world_model_v2.mode_graph import PATHWAYS
    for it in (raw.get("intentions") or []):
        if not isinstance(it, dict) or str(it.get("actor")) not in ents:
            continue
        actor_id = str(it["actor"])
        e = ents[actor_id]
        level = str(it.get("commitment_level", "neutral")).strip().lower()
        if level not in _STANCE_LEVELS_ALL:
            level = "neutral"
        reliability = str(it.get("reliability", "medium")).strip().lower()
        if reliability not in _LEVELS3:
            reliability = "medium"
        capability = str(it.get("capability", "high")).strip().lower()
        if capability not in _LEVELS3:
            capability = "high"
        basis_kind = str(it.get("basis_kind", "")).strip().lower()
        if basis_kind not in _BASIS_KINDS:
            basis_kind = "inference"
        pathway = str(it.get("pathway", "any")).strip().lower()
        if pathway not in PATHWAYS and pathway != "any":
            pathway = "any"
        control = str(it.get("control", "")).strip().lower()
        if control not in _CONTROLS:
            control = None                                   # graded-control absent → legacy boolean
        target_mode = str(it.get("target_mode") or "").strip() or None
        if target_mode and mode_ids and target_mode not in mode_ids:
            target_mode = None                               # unknown mode id → resolution as a whole
        stance = {"actor": actor_id, "commitment_level": level,
                  "reliability": reliability, "capability": capability,
                  "basis_kind": basis_kind,
                  "pathway": pathway, "target_mode": target_mode,
                  "quote": str(it.get("basis_quote", ""))[:160],
                  "statement": str(it.get("stated_intention", ""))[:160],
                  "source": str(it.get("source", "model_knowledge")),
                  "explicit_prohibitions": [str(p)[:60] for p in
                                            (it.get("explicit_prohibitions") or [])
                                            if isinstance(p, str)][:12],
                  "entails": str(it.get("entails_direction", "neutral")).lower()}
        if control:
            stance["control"] = control
        else:
            stance["controls_pathway"] = bool(it.get("controls_pathway"))
        stances.append(stance)
        per_actor_records.setdefault(actor_id, []).append(stance)
        # legacy single-intention record kept for older consumers/transcripts
        e["_intention"] = {"quote": stance["quote"], "source": stance["source"],
                           "commitment_level": level, "reliability": reliability,
                           "pathway": pathway, "controls_pathway": bool(it.get("controls_pathway")),
                           "entails": stance["entails"]}
        n_grounded += 1
        kept.append({k: it.get(k) for k in ("actor", "stated_intention", "basis_quote", "source",
                                            "basis_kind", "commitment_level", "target_mode",
                                            "pathway", "control", "capability", "reliability",
                                            "entails_direction")})
    # write per-actor structured stances + typed commitments onto the entity so materialization
    # carries them into the world and the ActorView projects them into the actor's OWN cognition
    for actor_id, recs in per_actor_records.items():
        e = ents[actor_id]
        fields = e.setdefault("fields", {})
        fields["stances"] = recs
        commitments = []
        for st in recs:
            prohibits = _binding_prohibitions(st)
            commitments.append({"id": f"grounded_stance:{st['commitment_level']}:{st['pathway']}"
                                      + (f":{st['target_mode']}" if st.get("target_mode") else ""),
                                "kind": "stated_intention", "statement": st["statement"],
                                "quote": st["quote"], "source": st["source"],
                                "basis_kind": st["basis_kind"],
                                "binding": bool(prohibits), "prohibits": prohibits,
                                "commitment_level": st["commitment_level"],
                                "pathway": st["pathway"], "target_mode": st.get("target_mode")})
        fields["commitments"] = commitments
    # the full qualitative record — auditable provenance for reports and conversion (§NAP: no
    # aggregate share quantity and no consumed-state weight are derived from these)
    plan._intention_stances = stances
    return {"n_actors": len(actors), "n_grounded": n_grounded,
            "n_mode_scoped": sum(1 for s in stances if s.get("target_mode")),
            "n_binding_instruments": sum(1 for s in stances
                                         if s.get("basis_kind") in _BINDING_BASIS_KINDS),
            "intentions": kept[:12]}
