"""Universal resolution-criterion parsing + per-actor evidence-grounded intentions.

Both are QUESTION-GENERAL: they operate on any compiled plan from its own text/evidence/entities, with no
scenario branching.

RESOLUTION CRITERION — the Powell trace exposed the failure class: "out as Fed Chair" (steps down as chair
= YES) was arbitrated against news saying he "stays at the Fed" (a different predicate). The parser turns
the question into a precise machine-readable criterion (subject, predicate, target state, deadline,
disambiguations of near-miss states), which then anchors (a) the outcome contract's resolution rule,
(b) the scheduled-facts extractor's entailment judgments, (c) the fidelity critic.

ACTOR INTENTIONS — a strategic actor's PUBLIC STATED intention (with quote + date + commitment strength) is
world state, not a Tier-7 mixture's guess. Intentions are written onto the entity (goals/commitments fields
the ActorView exposes) and aggregated into a typed quantity (`actor_intentions`: confidence-weighted share
of stated intentions entailing YES under the parsed criterion) consumed by the outcome mechanisms through
the bounded state channel — evidence-grounded, provenance-labeled, never a probability override.
"""
from __future__ import annotations

_CRITERION_PROMPT = """Parse this forecasting question into a PRECISE resolution criterion.
QUESTION: {q}
HORIZON: {horizon}
Return ONLY JSON:
{{"subject": "<who/what>", "predicate": "<the exact state that must hold>",
 "deadline": "<YYYY-MM-DD or null>",
 "resolves_yes_iff": "<one precise sentence>",
 "near_miss_states": [{{"state": "<a state that LOOKS like yes/no but is NOT>", "resolves": "yes|no"}}],
 "notes": "<ambiguities>"}}"""

_INTENTIONS_PROMPT = """For each STRATEGIC ACTOR below, state their PUBLICLY STATED or strongly evidenced
intention relevant to this question, using the evidence and your knowledge of the real people/institutions.
Only include intentions you can ground. CLASSIFY qualitatively — do NOT invent numeric strengths.

QUESTION: {q}
RESOLUTION CRITERION: {crit}
ACTORS: {actors}
EVIDENCE: {ev}

Return ONLY JSON:
{{"intentions": [{{"actor": "<entity id>", "stated_intention": "<one sentence>",
  "basis_quote": "<short quote or knowledge basis>", "source": "evidence|model_knowledge",
  "commitment_level": "committed_to_prevent|conditionally_opposed|weakly_opposed|neutral|inclined_toward|actively_pursuing|formally_committed — the actor's stance toward the RESOLVING STATE of the criterion (universal: works for a deal, a resignation, a rate cut, a bill, a launch)",
  "pathway": "cooperative_agreement|unilateral_action|institutional_procedure|any — which causal pathway to the resolving state this stance concerns (a refusal to negotiate concerns the cooperative path; a vow to fight on concerns the unilateral path; a whip count concerns the institutional path)",
  "controls_pathway": true_or_false_lowercase — does this actor control or hold a veto over that pathway,
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
_PATHWAYS = ("cooperative_agreement", "unilateral_action", "institutional_procedure", "any")


def parse_resolution_criterion(question, *, horizon, llm=None) -> dict:
    if llm is None:
        return {}
    from swm.engine.grounding import parse_json
    raw = parse_json(llm(_CRITERION_PROMPT.format(q=question, horizon=horizon))) or {}
    return {k: raw.get(k) for k in ("subject", "predicate", "deadline", "resolves_yes_iff",
                                    "near_miss_states", "notes")} if raw.get("resolves_yes_iff") else {}


def ground_actor_intentions(plan, question, *, criterion=None, evidence_text="", llm=None) -> dict:
    """Extract grounded intentions for declared strategic actors; write them onto entity fields; aggregate
    an `actor_intentions` quantity into the plan for mechanism consumption. Universal — uses only the
    plan's own entities."""
    if llm is None:
        return {"skipped": "no llm"}
    from swm.engine.grounding import parse_json
    actors = [str(e.get("id")) for e in plan.entities if isinstance(e, dict) and e.get("id")][:8]
    if not actors:
        return {"skipped": "no actors"}
    raw = parse_json(llm(_INTENTIONS_PROMPT.format(
        q=question, crit=(criterion or {}).get("resolves_yes_iff", "(as stated)"),
        actors=actors, ev=evidence_text[:1600] or "(none)"))) or {}
    ents = {str(e.get("id")): e for e in plan.entities if isinstance(e, dict)}
    n_grounded, num, den = 0, 0.0, 0.0
    kept, stances = [], []
    for it in (raw.get("intentions") or []):
        if not isinstance(it, dict) or str(it.get("actor")) not in ents:
            continue
        e = ents[str(it["actor"])]
        level = str(it.get("commitment_level", "neutral")).strip().lower()
        if level not in _STANCE_WEIGHT:
            level = "neutral"
        reliability = str(it.get("reliability", "medium")).strip().lower()
        if reliability not in ("high", "medium", "low"):
            reliability = "medium"
        pathway = str(it.get("pathway", "any")).strip().lower()
        if pathway not in _PATHWAYS:
            pathway = "any"
        strength = _STANCE_WEIGHT[level]
        f = e.setdefault("fields", {})
        # ActorView (phase4_policy) consumes `commitments` as a LIST into action scoring — the
        # stated intention conditions the actor's POLICY, not just the hazard layer
        if not isinstance(f.get("commitments"), list):
            f["commitments"] = [f["commitments"]] if f.get("commitments") else []
        f["commitments"].append(str(it.get("stated_intention", ""))[:160])
        e["_intention"] = {"quote": str(it.get("basis_quote", ""))[:160],
                           "source": str(it.get("source", "model_knowledge")),
                           "commitment_level": level, "reliability": reliability,
                           "pathway": pathway, "controls_pathway": bool(it.get("controls_pathway")),
                           "entails": str(it.get("entails_direction", "neutral"))}
        n_grounded += 1
        d = str(it.get("entails_direction", "neutral")).lower()
        if d in ("yes", "no"):
            num += strength * (1.0 if d == "yes" else 0.0)
            den += strength
        stances.append({"actor": str(it["actor"]), "commitment_level": level,
                        "reliability": reliability, "pathway": pathway,
                        "controls_pathway": bool(it.get("controls_pathway")), "entails": d})
        kept.append({k: it.get(k) for k in ("actor", "stated_intention", "basis_quote", "source",
                                            "commitment_level", "pathway", "controls_pathway",
                                            "reliability", "entails_direction")})
    # the full qualitative record — consumed by event_time to build per-mode hazard-ratio
    # DISTRIBUTIONS (never a point coefficient invented here)
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
            "intention_yes_share": (round(num / den, 3) if den > 0 else None), "intentions": kept[:8]}
