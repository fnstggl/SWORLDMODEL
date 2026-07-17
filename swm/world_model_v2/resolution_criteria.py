"""Universal resolution-criterion parsing + per-actor evidence-grounded intentions.

Both are QUESTION-GENERAL: they operate on any compiled plan from its own text/evidence/entities, with no
scenario branching.

RESOLUTION CRITERION — the Powell trace exposed the failure class: "out as Fed Chair" (steps down as chair
= YES) was arbitrated against news saying he "stays at the Fed" (a different predicate). The parser turns
the question into a precise machine-readable criterion (subject, predicate, target state, deadline,
disambiguations of near-miss states), which then anchors (a) the outcome contract's resolution rule,
(b) the scheduled-facts extractor's entailment judgments, (c) the fidelity critic.

ACTOR INTENTIONS — a strategic actor's PUBLIC STATED intention (with quote + date) is world state, not a
Tier-7 mixture's guess. The LLM only ever CLASSIFIES (commitment level, pathway, target mode, graded
control, capability, reliability) — it never mints effect sizes. Stances are MODE-SCOPED when the
canonical mode set exists (`target_mode`): Russia can pursue ITS victory while committed to preventing
Ukraine's while conditionally open to a ceasefire. Grounded stances are written THREE places:
  1. plan._intention_stances — consumed by event_time/mode_graph to build per-mode hazard-ratio
     DISTRIBUTIONS under each mode's decision structure;
  2. the entity's `stances` field (registered extension) — projected into the Phase-4 ActorView so the
     actor's ACTION POLICY is conditioned on their own stated commitments (the behavior channel);
  3. the entity's `commitments` field as TYPED commitment dicts — a high-reliability categorical
     stance becomes a BINDING commitment that prohibits the actions most contrary to it (the
     feasibility channel), until evidence revises it.
Plus the `actor_intentions` aggregate quantity (confidence-weighted yes-share) for the bounded state
channel — evidence-grounded, provenance-labeled, never a probability override.
"""
from __future__ import annotations

from swm.world_model_v2.state import register_entity_extension

# the grounded stance record is a TYPED extension field so materialization keeps it addressable and
# the Phase-4 ActorViewBuilder can project it into the actor's own view (never simulator-only state)
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
Only include intentions you can ground. CLASSIFY qualitatively — do NOT invent numeric strengths.
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
  "commitment_level": "committed_to_prevent|conditionally_opposed|weakly_opposed|neutral|inclined_toward|actively_pursuing|formally_committed — the actor's stance toward the target end-state (universal: works for a deal, a resignation, a rate cut, a bill, a launch)",
  "target_mode": "<the specific end-state id from the CANDIDATE END-STATES this stance concerns, or null when it concerns the resolution as a whole>",
  "pathway": "cooperative_agreement|unilateral_action|institutional_procedure|operational_execution|competitive_interaction|any — which causal pathway this stance concerns (a refusal to negotiate concerns the cooperative path; a vow to fight on concerns the unilateral path; a whip count concerns the institutional path)",
  "control": "sole_authority|veto|agenda_setting|partial_implementation|coalition_member|operational_capability|informal_influence|none — the actor's REAL degree of control over that pathway (a president may want a bill but lack the votes; a legislature may pass one but lack implementation capacity)",
  "capability": "high|medium|low — can the actor practically act on this stance (means, position, resources)?",
  "reliability": "high|medium|low — how reliable/binding the basis is (law/treaty=high, direct public statement=high, reported leaning=medium, inference=low)",
  "entails_direction": "yes|no|neutral — under the resolution criterion",
  "date": "<YYYY-MM-DD or null>"}}]}}"""

# Stance-weight map used ONLY to aggregate the direction-share quantity (`actor_intentions`, consumed
# through the bounded state channel). These are documented aggregation weights, not effect sizes —
# the EFFECT of stances on timing hazards lives in event_time.INTENTION_HR_PRIORS (distributions,
# sampled per particle, replaceable by a fitted pack).
_STANCE_WEIGHT = {"committed_to_prevent": 0.95, "conditionally_opposed": 0.75, "weakly_opposed": 0.4,
                  "neutral": 0.0, "inclined_toward": 0.6, "actively_pursuing": 0.85,
                  "formally_committed": 0.95,
                  # legacy agreement-specific labels (older packs/transcripts) map onto the universal set
                  "categorical_refusal": 0.95, "conditional_refusal": 0.75, "weak_opposition": 0.4,
                  "openness_to_agreement": 0.6, "formal_commitment_toward_agreement": 0.95}
_CONTROLS = ("sole_authority", "veto", "agenda_setting", "partial_implementation", "coalition_member",
             "operational_capability", "informal_influence", "none")
_LEVELS3 = ("high", "medium", "low")


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
    """A HIGH-reliability categorical stance is a public commitment device: the actions most contrary
    to it become infeasible-until-revised (the FeasibilityEngine's binding-commitment contract).
    Only prevent-side stances prohibit — an actor formally committed TOWARD an outcome is pushed by
    the utility channel, not hard-blocked from hesitating. Universal: prohibitions are derived from
    the phase4 action ontology's pathway effects, never from scenario keywords."""
    if str(stance.get("reliability")) != "high" or \
            str(stance.get("commitment_level")) not in ("committed_to_prevent", "categorical_refusal"):
        return []
    try:
        from swm.world_model_v2.phase4_policy import actions_advancing_pathway
        return actions_advancing_pathway(str(stance.get("pathway", "any")), min_effect=0.5)
    except Exception:  # noqa: BLE001 — grounding must never block on the policy layer
        return []


def ground_actor_intentions(plan, question, *, criterion=None, evidence_text="", llm=None,
                            modes: list = None) -> dict:
    """Extract grounded stances for declared strategic actors; write them onto entity fields (stances +
    typed commitments) so the Phase-4 policy consumes them; aggregate an `actor_intentions` quantity
    into the plan for mechanism consumption. Mode-scoped when a canonical mode set exists. Universal —
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
    n_grounded, num, den = 0, 0.0, 0.0
    kept, stances = [], []
    per_actor_records = {}
    from swm.world_model_v2.mode_graph import PATHWAYS
    for it in (raw.get("intentions") or []):
        if not isinstance(it, dict) or str(it.get("actor")) not in ents:
            continue
        actor_id = str(it["actor"])
        e = ents[actor_id]
        level = str(it.get("commitment_level", "neutral")).strip().lower()
        if level not in _STANCE_WEIGHT:
            level = "neutral"
        reliability = str(it.get("reliability", "medium")).strip().lower()
        if reliability not in _LEVELS3:
            reliability = "medium"
        capability = str(it.get("capability", "high")).strip().lower()
        if capability not in _LEVELS3:
            capability = "high"
        pathway = str(it.get("pathway", "any")).strip().lower()
        if pathway not in PATHWAYS and pathway != "any":
            pathway = "any"
        control = str(it.get("control", "")).strip().lower()
        if control not in _CONTROLS:
            control = None                                   # graded-control absent → legacy boolean
        target_mode = str(it.get("target_mode") or "").strip() or None
        if target_mode and mode_ids and target_mode not in mode_ids:
            target_mode = None                               # unknown mode id → resolution as a whole
        strength = _STANCE_WEIGHT[level]
        stance = {"actor": actor_id, "commitment_level": level,
                  "reliability": reliability, "capability": capability,
                  "pathway": pathway, "target_mode": target_mode,
                  "quote": str(it.get("basis_quote", ""))[:160],
                  "statement": str(it.get("stated_intention", ""))[:160],
                  "source": str(it.get("source", "model_knowledge")),
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
        d = stance["entails"]
        if d in ("yes", "no"):
            num += strength * (1.0 if d == "yes" else 0.0)
            den += strength
        kept.append({k: it.get(k) for k in ("actor", "stated_intention", "basis_quote", "source",
                                            "commitment_level", "target_mode", "pathway", "control",
                                            "capability", "reliability", "entails_direction")})
    # write per-actor structured stances + typed commitments onto the entity so materialization
    # carries them into the world and the ActorView projects them into the POLICY (behavior channel)
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
                                "binding": bool(prohibits), "prohibits": prohibits,
                                "commitment_level": st["commitment_level"],
                                "pathway": st["pathway"], "target_mode": st.get("target_mode")})
        fields["commitments"] = commitments
    # the full qualitative record — consumed by event_time/mode_graph to build per-mode hazard-ratio
    # DISTRIBUTIONS under each mode's decision structure (never a point coefficient invented here)
    plan._intention_stances = stances
    if den > 0:
        share = num / den
        plan.quantities.append({"name": "actor_intentions", "qtype": "actor_intentions",
                                "value": round(share, 4), "sd": None})
        if not hasattr(plan, "_consumed_state"):
            plan._consumed_state = []
        if not any(m.get("var") == "actor_intentions" for m in plan._consumed_state):
            plan._consumed_state.append({"var": "actor_intentions", "weight": 0.2})
    return {"n_actors": len(actors), "n_grounded": n_grounded,
            "n_mode_scoped": sum(1 for s in stances if s.get("target_mode")),
            "intention_yes_share": (round(num / den, 3) if den > 0 else None), "intentions": kept[:12]}
