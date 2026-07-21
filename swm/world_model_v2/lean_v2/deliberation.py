"""Selective additional deliberation — real reflection, never a reflex loop.

Real people refine decisions by reconsidering tradeoffs, recalling memories, noticing missed
implications, mentally simulating reactions, resolving goal conflicts. Lean V2 preserves that
WITHOUT the automatic multi-stage pipeline, by distinguishing deterministically:

  INFORMATION limitation — a genuinely necessary fact is unavailable. The actor's wait /
  gather-information / delegate choice IS the decision: it stands, a gathering action is
  created when feasible, reconsideration is scheduled for when the fact could arrive, and the
  actor is NEVER re-asked to reason over the same known absence (per-actor asked-once ledger).
  No escalation happens merely because a missing fact was mentioned.

  DELIBERATION limitation — the actor has enough information but remains conflicted between
  materially different options. ONE bounded additional deliberation call is allowed when the
  decision is terminal-relevant AND a deterministic trigger fires (internal conflict named,
  overlooked available material fact, anticipation of another actor's reaction, competing
  commitments, near-tie between materially different options). The second call receives the
  SAME information boundary, the first decision + reasoning summary, and the specific
  unresolved tradeoff — never invented new facts. After it, the actor may keep, change,
  defer, gather, delegate, or remain uncertain.

Cap: one normal decision call + at most one materially justified deliberation call. The full
staged pipeline remains reachable only for malformed/invalid/boundary-violating/terminally
incoherent responses. Every trigger, change and added call is recorded."""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm

#: deterministic deliberation triggers (any one, on a terminal-relevant decision)
TRIGGERS = ("internal_conflict_named", "overlooked_available_fact",
            "anticipates_other_actor_reaction", "competing_commitments",
            "near_tie_between_material_options")


@dataclass
class DeliberationRecord:
    actor_id: str
    context_hash: str
    trigger: str = ""
    ran: bool = False
    changed_action: bool = False
    first_action: str = ""
    final_action: str = ""
    added_calls: int = 0
    added_latency_s: float = 0.0
    note: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("actor_id", "context_hash", "trigger", "ran", "changed_action",
                 "first_action", "final_action", "added_calls", "added_latency_s", "note")}


def classify_limitation(decision: dict, *, available_fact_ids: set) -> tuple:
    """(kind, detail): 'information' | 'deliberation' | 'none'. Pure code over the parsed
    one-call response — no model judges the model."""
    interp = decision.get("interpretation") or {}
    missing = norm(interp.get("missing_decisive_fact"), 200)
    act = str((decision.get("decision") or {}).get("act_or_wait") or "").lower()
    if missing and act in ("wait", "gather_information", "delegate"):
        return "information", missing
    conflict = norm(interp.get("unresolved_ambiguity"), 200)
    considered = [norm(c, 120) for c in (decision.get("considered_actions") or [])]
    chosen = norm((decision.get("decision") or {}).get("chosen_action"), 120)
    screened = {norm((s or {}).get("option"), 120)
                for s in decision.get("screened_out") or [] if isinstance(s, dict)}
    if conflict and act == "act":
        return "deliberation", f"internal_conflict_named: {conflict}"
    if chosen and chosen in screened:
        return "deliberation", f"near_tie_between_material_options: chose an option it also " \
                               f"screened out ('{chosen}')"
    noticed = {str(n.get("obs_id")) for n in
               ((decision.get("attention") or {}).get("noticed") or [])
               if isinstance(n, dict)}
    unjudged = available_fact_ids - noticed - {
        str(m.get("obs_id")) for m in
        ((decision.get("attention") or {}).get("ignored") or []) if isinstance(m, dict)}
    if unjudged and len(considered) > 1 and act == "act":
        return "deliberation", f"overlooked_available_fact: {sorted(unjudged)[:3]}"
    if any("react" in c.lower() or "response" in c.lower() for c in considered) \
            and conflict:
        return "deliberation", "anticipates_other_actor_reaction"
    return "none", ""


_DELIBERATION_PROMPT = """You are the SAME person continuing the SAME moment of thought — no new outside
information exists. As of {day}.

{snapshot}

Your first decision was: {first_action}
Your reasoning summary was: {summary}
The unresolved issue requiring further reflection: {tradeoff}

Reflect once, seriously: weigh the tradeoff, recall anything relevant you already know, mentally
simulate how others would react, resolve conflicts between your goals. Then either KEEP your decision,
CHANGE it, DEFER, GATHER information, or DELEGATE — whichever the reflection genuinely supports.

Reply ONLY JSON:
{{"reflection_summary": "...", "decision": {{"chosen_action": "...",
  "act_or_wait": "act|wait|gather_information|delegate|do_nothing", "target": "",
  "timing": "immediate", "intended_effect": "..."}},
 "changed": true|false, "residual_uncertainty": "<or empty>",
 "actor_state_update": {{"beliefs": [], "goals": []}}}}"""


def run_deliberation(*, actor_id: str, context_hash: str, trigger_detail: str, day: str,
                     snapshot: str, first_decision: dict, gateway, budget_ledger) -> tuple:
    """The ONE bounded second call. Returns (revised_decision_dict|None, DeliberationRecord)."""
    import time as _t

    from swm.engine.grounding import parse_json
    rec = DeliberationRecord(actor_id=actor_id, context_hash=context_hash,
                             trigger=trigger_detail[:160])
    dec = first_decision.get("decision") or {}
    rec.first_action = norm(dec.get("chosen_action") or dec.get("act_or_wait"), 120)
    ok, why = budget_ledger.can_afford(what=f"deliberation:{actor_id}", est_calls=1,
                                       deliberation=True)
    if not ok:
        rec.note = f"skipped: {why}"
        return None, rec
    prompt = _DELIBERATION_PROMPT.format(
        day=day, snapshot=snapshot[:5000],
        first_action=rec.first_action or "(wait)",
        summary=norm(first_decision.get("decision_summary"), 300) or "(none given)",
        tradeoff=trigger_detail[:300])
    t = _t.time()
    try:
        text = gateway.call("actor_decision", prompt)
    except Exception as e:  # noqa: BLE001 — a failed reflection keeps the first decision
        rec.note = f"provider_failure:{type(e).__name__} — first decision stands"
        return None, rec
    rec.ran = True
    rec.added_calls = 1
    rec.added_latency_s = round(_t.time() - t, 3)
    budget_ledger.record_deliberation()
    r = parse_json(text)
    if not isinstance(r, dict) or not isinstance(r.get("decision"), dict):
        rec.note = "unparseable reflection — first decision stands"
        return None, rec
    rec.final_action = norm(r["decision"].get("chosen_action")
                            or r["decision"].get("act_or_wait"), 120)
    rec.changed_action = bool(r.get("changed")) or (rec.final_action != rec.first_action)
    rec.note = norm(r.get("reflection_summary"), 200)
    return r, rec
