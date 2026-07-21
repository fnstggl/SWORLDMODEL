"""The terminal-causal backward slice — who can actually change the answer.

Builds the dependency graph BACKWARD from the terminal predicate:

    terminal result ← terminal-writing mechanisms/actions ← required state changes
    ← actions capable of producing them ← actors/institutions with authority over those actions
    ← observations capable of changing those decisions (event emissions, channels)

An actor enters the detailed simulation only when a path in this graph connects them to the
terminal result — through action, information, communication, authority, approval, resource
control, constraint, institutional role, response, or dynamically activated involvement
(BROAD relevance: writing the terminal directly is NOT required). Deterministic removals and
merges are RECORDED, never silent:

    * duplicate names for the same person/institution (alias-normalized identity);
    * an institution and a named person that are duplicate runtime representations of the
      EXACT same decision right (single-member, single-rule) — merged, distinct roles kept;
    * ceremonial actors with no discretion;
    * evidence-mention-only actors with no action, no trigger, no observation channel into a
      surviving pathway.

When uncertain, the actor is RETAINED (recorded as retained_uncertain). Dynamic promotion
stays alive: the engine re-adds a pruned actor the moment an event targets them with
terminal-relevant effect."""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint, norm_key


@dataclass
class SliceResult:
    kept_actors: list = field(default_factory=list)
    pruned: list = field(default_factory=list)         # [{actor_id, reason}]
    merged: list = field(default_factory=list)         # [{kept, removed, reason}]
    retained_uncertain: list = field(default_factory=list)
    promotable: list = field(default_factory=list)     # pruned but re-addable dynamically
    edges: list = field(default_factory=list)          # audit: the backward graph edges

    def manifest(self) -> dict:
        return {"kept_actors": list(self.kept_actors),
                "n_pruned": len(self.pruned), "pruned": self.pruned,
                "n_merged": len(self.merged), "merged": self.merged,
                "retained_uncertain": self.retained_uncertain,
                "promotable": self.promotable,
                "n_edges": len(self.edges)}


def merge_duplicate_actors(bp: ConsumerWorldBlueprint) -> list:
    """Alias-normalized identity merge (duplicate representations of the same person), plus the
    institution-as-person case: a single-member 'single'-rule institution IS that member's
    decision right — the member keeps the right; the institution stays as structure only."""
    merged = []
    by_key: dict = {}
    for a in list(bp.actors):
        keys = {norm_key(a.get("id")), norm_key(a.get("name"))} \
            | {norm_key(x) for x in (a.get("aliases") or [])}
        keys.discard("")
        hit = next((k for k in keys if k in by_key), None)
        if hit is None:
            for k in keys:
                by_key[k] = a
            continue
        keeper = by_key[hit]
        merged.append({"kept": keeper.get("id"), "removed": a.get("id"),
                       "reason": f"duplicate identity via alias '{hit}'"})
        keeper["aliases"] = sorted({*(keeper.get("aliases") or []), a.get("name") or "",
                                    *(a.get("aliases") or []), a.get("id") or ""} - {""})
        keeper["authority"] = sorted({*(keeper.get("authority") or []),
                                      *(a.get("authority") or [])})
        if not keeper.get("private_state_variants"):
            keeper["private_state_variants"] = a.get("private_state_variants") or []
        removed_id = a.get("id")
        bp.actors.remove(a)
        for inst in bp.institutions:
            inst["members"] = [keeper.get("id") if m == removed_id else m
                               for m in (inst.get("members") or [])]
            inst["members"] = list(dict.fromkeys(inst["members"]))
        for t in bp.action_templates:
            t["actor_ids"] = list(dict.fromkeys(keeper.get("id") if x == removed_id else x
                                                for x in (t.get("actor_ids") or [])))
        for d in bp.decision_triggers:
            if d.get("actor_id") == removed_id:
                d["actor_id"] = keeper.get("id")
    return merged


def backward_slice(bp: ConsumerWorldBlueprint) -> SliceResult:
    out = SliceResult()
    out.merged = merge_duplicate_actors(bp)
    actor_ids = {a.get("id") for a in bp.actors}

    # --- layer 0: the terminal writers -------------------------------------------------
    term = bp.terminal
    writer_actions = set(term.get("written_by_action_ids") or [])
    for t in bp.action_templates:
        if t.get("writes_terminal") or any(e.get("kind") == "record_vote"
                                           for e in t.get("effects") or []):
            writer_actions.add(t.get("action_id"))
    for m in bp.mechanisms:
        if m.get("writes_terminal"):
            out.edges.append({"from": "terminal", "to": f"mechanism:{m.get('id')}",
                              "kind": "terminal_writer"})
    relevant_actions = set(writer_actions)
    relevant_actors: set = set()
    inst = bp.institution_by_id(term.get("institution_id")) \
        if term.get("kind") == "institution_vote" else None
    if inst is not None:
        relevant_actors |= set(inst.get("members") or [])
        for m in inst.get("members") or []:
            out.edges.append({"from": "terminal", "to": f"actor:{m}",
                              "kind": "institution_member"})

    # --- backward closure over actions/events/observations ----------------------------
    changed = True
    guard = 0
    while changed and guard < 24:
        guard += 1
        changed = False
        for t in bp.action_templates:
            tid = t.get("action_id")
            takers = set(t.get("actor_ids") or [])
            targets = set(t.get("targets") or [])
            emits = {e.get("etype") for e in (t.get("emits_events") or [])}
            if tid in relevant_actions:
                new = takers - relevant_actors
                if new:
                    relevant_actors |= new
                    changed = True
                    for a in new:
                        out.edges.append({"from": f"action:{tid}", "to": f"actor:{a}",
                                          "kind": "authority_over_terminal_path"})
                continue
            # an action becomes relevant when it targets a relevant actor/institution
            # (communication/approval/constraint/response channel) or emits an event a
            # relevant actor observes / is triggered by
            hits_target = bool(targets & relevant_actors) \
                or (inst is not None and inst.get("id") in targets)
            observed = False
            for e in bp.event_types:
                if e.get("etype") in emits:
                    obs = set(e.get("observers") or [])
                    if "public" in obs or obs & relevant_actors:
                        observed = True
            triggering = any(d.get("etype") in emits and d.get("actor_id") in relevant_actors
                             for d in bp.decision_triggers)
            if hits_target or observed or triggering:
                relevant_actions.add(tid)
                changed = True
                out.edges.append({"from": "relevant_set", "to": f"action:{tid}",
                                  "kind": "influences_relevant_actor"})

    # --- classify every actor -----------------------------------------------------------
    triggered = {d.get("actor_id") for d in bp.decision_triggers}
    for a in bp.actors:
        aid = a.get("id")
        discretion = str(a.get("discretion") or "")
        if aid in relevant_actors:
            out.kept_actors.append(aid)
            continue
        if discretion == "ceremonial":
            out.pruned.append({"actor_id": aid, "reason": "ceremonial: no discretion over "
                                                          "any surviving terminal pathway"})
            out.promotable.append(aid)
            continue
        takes_action = any(aid in (t.get("actor_ids") or []) for t in bp.action_templates
                           if t.get("action_id") in relevant_actions)
        if takes_action or aid in triggered:
            out.kept_actors.append(aid)
            out.edges.append({"from": "trigger_or_action", "to": f"actor:{aid}",
                              "kind": "kept"})
            continue
        if discretion == "decisive":
            # relevance unproven but the compiler called them decisive — RETAIN, record
            out.kept_actors.append(aid)
            out.retained_uncertain.append(
                {"actor_id": aid, "reason": "compiler marked decisive; backward slice found "
                                            "no path — retained (when uncertain, retain)"})
            continue
        out.pruned.append({"actor_id": aid,
                           "reason": "no action, trigger, target or observation channel "
                                     "reaches a surviving terminal pathway "
                                     "(evidence-mention-only / non-causal)"})
        out.promotable.append(aid)
    out.kept_actors = list(dict.fromkeys(out.kept_actors))
    return out
