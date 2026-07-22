"""Question-specific causal relevant-actor selection and dynamic promotion.

Tiers (docs/ARCHITECTURE_QUALITATIVE_ACTORS.md §6):
    1 — primary consequential actor  → persistent qualitative LLM cognition
    2 — secondary potentially consequential → same engine, promoted attention
    3 — routine actor → numeric policy (hybrid mode)

Selection is CAUSAL and question-specific, computed from the compiled plan's own structure —
never from fame, never from scenario keywords: direct decision authority (institution
``decision_right`` holders), scheduled decision events, veto/blocking rules, pathway
principals, stance-carrying capability actors, resource/implementation control, persuasive
access to principals through declared relations, and *reaction-is-the-question* (a question
asking how a specific person will react makes that person Tier 1 automatically — no stance,
capacity, or network precondition). Every assignment records its reasons. The Phase-4L
additive heuristic (`llm_actor.persona_relevance`) survives only as a fallback for bare
worlds. Promotion is dynamic: an event participant outside the tier map is re-scored at event
time from live world state, and promotions are recorded on the world."""
from __future__ import annotations

import re

#: patterns that make a named individual's reaction THE question (single-individual mode)
_REACTION_PATTERNS = (
    r"how (?:will|would|is|might) (?P<name>[\w .'-]+?) (?:react|respond|reply|interpret|take|feel)",
    r"(?:will|would) (?P<name>[\w .'-]+?) (?:accept|agree|reply|respond|approve|reject|come|attend|forgive|say yes)",
    r"what (?:will|would) (?P<name>[\w .'-]+?) (?:do|say|think)",
)


def is_individual_reaction_question(question: str) -> bool:
    """Pattern-only detection (no entity list needed): does the question ask how a specific
    person — named, or referenced as 'my manager' / 'this person' / a pronoun — will react to
    a stimulus? The public API uses this to route personal questions through the qualitative
    individual-reaction path when the caller supplies the person's context."""
    q = (question or "").lower()
    return any(re.search(pat, q) for pat in _REACTION_PATTERNS)


def reaction_target(question: str, entity_ids) -> str | None:
    """The entity whose reaction the question itself asks about, if any."""
    q = (question or "").lower()
    norm = {str(e).lower().replace("_", " "): str(e) for e in entity_ids}
    for pat in _REACTION_PATTERNS:
        m = re.search(pat, q)
        if not m:
            continue
        name = m.group("name").strip().rstrip("?.,")
        for low, eid in norm.items():
            if name == low or name in low or low in name:
                return eid
    return None


class RelevantActorSelector:
    """Causal tier assignment from the compiled plan + dynamic promotion from live worlds."""

    version = "selector-1.0"

    def select(self, plan, question: str = "") -> dict:
        """{actor_id: {"tier": 1|2|3, "reasons": [...], "selector": version}} for every declared
        person/institution-adjacent actor. Falls back to the Phase-4L heuristic score only when
        the plan declares no causal structure at all."""
        entities = [e for e in (getattr(plan, "entities", None) or []) if isinstance(e, dict)]
        ids = [str(e.get("id")) for e in entities if e.get("id")]
        reasons: dict = {aid: [] for aid in ids}

        def note(aid, why):
            if aid in reasons:
                reasons[aid].append(why)

        # reaction-is-the-question → automatic Tier 1 (single-individual mode)
        target = reaction_target(question or getattr(plan, "question", ""), ids)
        if target:
            note(target, "reaction_is_the_question")

        # direct decision authority + veto/blocking power from executable institution rules
        for inst in (getattr(plan, "institutions", None) or []):
            if not isinstance(inst, dict):
                continue
            for rule in inst.get("rules") or []:
                if not isinstance(rule, dict):
                    continue
                params = rule.get("params") or {}
                kind = str(rule.get("kind", ""))
                for holder in params.get("holders") or []:
                    if kind == "decision_right":
                        note(str(holder), f"direct_decision_authority:{inst.get('id')}")
                    elif kind in ("veto", "quorum", "approval"):
                        note(str(holder), f"veto_or_blocking_power:{inst.get('id')}")

        # scheduled decision events — the compiler's own causal claim that this actor decides
        for ev in (getattr(plan, "scheduled_events", None) or []):
            if isinstance(ev, dict) and ev.get("etype") == "decision_opportunity":
                for p in ev.get("participants") or []:
                    note(str(p), "scheduled_decision_event")
        for dec in (getattr(plan, "actor_decisions", None) or []):
            if isinstance(dec, dict) and dec.get("actor"):
                note(str(dec["actor"]), "compiler_actor_decision")

        # grounded intention stances: pathway pursuit/prevention with declared capability
        principals = set()
        for st in (getattr(plan, "_intention_stances", None) or []):
            if not isinstance(st, dict):
                continue
            aid = str(st.get("actor"))
            cap = str(st.get("capability", "high")).lower()
            note(aid, f"pathway_stance:{st.get('pathway')}:{st.get('commitment_level')}")
            if cap in ("high", "medium"):
                note(aid, f"implementation_control:capability_{cap}")
                principals.add(aid)

        # declared pathway principals (shared-process approvers)
        for q in (getattr(plan, "quantities", None) or []):
            if isinstance(q, dict) and str(q.get("name", "")).startswith("pathway_principals:"):
                for p in str(q.get("value", "") or "").split("|"):
                    if p:
                        note(p, f"pathway_principal:{q['name'].split(':', 1)[1]}")
                        principals.add(p)

        # resource control declared on the entity itself
        for e in entities:
            res = (e.get("fields") or {}).get("resources") or {}
            if isinstance(res, dict) and res:
                note(str(e.get("id")), f"resource_control:{','.join(sorted(res))[:60]}")

        # persuasive access: declared relations touching a principal/authority holder
        strong = {aid for aid, why in reasons.items()
                  if any(w.startswith(("direct_decision_authority", "pathway_principal",
                                       "implementation_control", "reaction_is_the_question"))
                         for w in why)}
        for rel in (getattr(plan, "relations", None) or []):
            if not isinstance(rel, dict):
                continue
            src, dst = str(rel.get("src")), str(rel.get("dst"))
            if dst in strong and src in reasons and src not in strong:
                note(src, f"persuasive_access_to:{dst}")
            if src in strong and dst in reasons and dst not in strong:
                note(dst, f"access_from_principal:{src}")

        out = {}
        for aid in ids:
            why = reasons[aid]
            if any(w.startswith(("reaction_is_the_question", "direct_decision_authority",
                                 "scheduled_decision_event", "compiler_actor_decision",
                                 "pathway_principal", "implementation_control")) for w in why):
                tier = 1
            elif any(w.startswith(("veto_or_blocking_power", "pathway_stance",
                                   "persuasive_access_to", "resource_control")) for w in why):
                tier = 2
            else:
                tier = 3
                why = why or ["no causal signal in the compiled plan"]
            out[aid] = {"tier": tier, "reasons": why, "selector": self.version}
        return out

    # ---- dynamic promotion ---------------------------------------------------------
    def promote_if_consequential(self, world, actor_id: str, decision: dict | None = None) -> dict | None:
        """Re-score an unmapped actor at event time from LIVE world state; record promotions.
        Returns a tier assignment (tier ≤ 2 ⇒ promoted to qualitative cognition) or None."""
        ent = (world.entities or {}).get(actor_id)
        if ent is None:
            return None
        why = []
        stances = ent.value("stances", default=None)
        if isinstance(stances, list) and stances:
            why.append("acquired_grounded_stances")
        if isinstance(ent.value("resources", key="capacity", default=None), (int, float)):
            why.append("acquired_capacity_resource")
        for inst in (world.institutions or {}).values():
            for rule in getattr(inst, "rules", []):
                if rule.kind == "decision_right" and actor_id in (rule.params.get("holders") or []):
                    why.append(f"direct_decision_authority:{inst.institution_id}")
        if decision and (decision.get("candidate_actions") or decision.get("situation")):
            why.append("named_in_live_decision_event")
        # being named in an event is corroboration, never sufficient by itself — promotion
        # requires at least one live CAUSAL signal (stances, capacity, authority)
        if not [w for w in why if w != "named_in_live_decision_event"]:
            return None
        tier = 1 if any(w.startswith("direct_decision_authority") for w in why) or \
            ("acquired_grounded_stances" in why and "named_in_live_decision_event" in why) else 2
        assignment = {"tier": tier, "reasons": why + ["dynamically_promoted_at_event_time"],
                      "selector": self.version}
        world.uncertainty_meta.setdefault("actor_tier_promotions", []).append(
            {"actor": actor_id, "at": world.clock.now, **assignment})
        return assignment

    # ---- fallback ------------------------------------------------------------------
    @staticmethod
    def heuristic_fallback(view, decision: dict | None = None) -> dict:
        """The Phase-4L additive heuristic, demoted to fallback for bare worlds."""
        from swm.world_model_v2.llm_actor import persona_relevance
        score, why = persona_relevance(view, decision)
        return {"tier": 1 if score >= 0.5 else 3,
                "reasons": [f"heuristic_fallback_score:{score}"] + why,
                "selector": "heuristic-fallback"}
