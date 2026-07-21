"""Parameterized consequence templates — precompile the mechanical scaffold, never the meaning.

The feasible action language is known once the blueprint exists, so every standard action's
MECHANICAL consequence structure compiles ONCE at scenario build (deterministically — zero LLM
calls): authority requirements, targets, direct mechanical effects (record_vote / send_message /
schedule_meeting / institution_stage / transfer_authority / open_window / close_window /
set_state), event emissions, validation rule, causal-boundary contract.

What is NEVER precompiled generically (and never executed mechanically here): how persuasive a
message is, how a recipient interprets its content, whether a threat changes a mind, the social
reaction to unusual wording. A send_message template precompiles DELIVERY mechanics only — the
content lands in the recipient's observation bundle and the recipient's next decision is their
own real actor-simulated choice.

During rollout: actor chooses action id → bind current content/target/timing → validate →
execute in the current world node. The consequence LLM is called ONLY for a genuinely novel
action no template can represent (one compile call, content-keyed cache, failures never
cached), and only if the bound content changes a mechanical contract beyond the template's
parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import (MECHANICAL_EFFECT_KINDS,
                                                  ConsumerWorldBlueprint, norm)


@dataclass
class ConsequenceTemplate:
    action_id: str
    description: str = ""
    actor_ids: list = field(default_factory=list)
    authority_required: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    effects: list = field(default_factory=list)          # [{kind, params}] mechanical only
    emits_events: list = field(default_factory=list)     # [{etype, observers}]
    writes_terminal: bool = False
    validation: str = ""
    unknown_effect_kinds: list = field(default_factory=list)
    source: str = "blueprint_precompiled"                # | novel_compiled

    def as_dict(self) -> dict:
        return {"action_id": self.action_id, "description": self.description,
                "actor_ids": list(self.actor_ids),
                "authority_required": list(self.authority_required),
                "targets": list(self.targets), "effects": list(self.effects),
                "emits_events": list(self.emits_events),
                "writes_terminal": self.writes_terminal, "validation": self.validation,
                "unknown_effect_kinds": list(self.unknown_effect_kinds),
                "source": self.source}


def precompile_templates(bp: ConsumerWorldBlueprint, cache=None) -> dict:
    """Blueprint action language -> template registry. Pure code; cached across runs as an
    immutable artifact when a cache is supplied."""
    def _build() -> dict:
        out = {}
        for t in bp.action_templates:
            aid = str(t.get("action_id") or "").strip()
            if not aid:
                continue
            effects, unknown = [], []
            for e in t.get("effects") or []:
                kind = str(e.get("kind") or "")
                if kind in MECHANICAL_EFFECT_KINDS:
                    effects.append({"kind": kind, "params": dict(e.get("params") or {})})
                else:
                    unknown.append(kind)
            out[aid] = ConsequenceTemplate(
                action_id=aid, description=norm(t.get("description"), 240),
                actor_ids=list(t.get("actor_ids") or []),
                authority_required=list(t.get("authority_required") or []),
                targets=list(t.get("targets") or []),
                effects=effects,
                emits_events=[{"etype": str(e.get("etype") or ""),
                               "observers": list(e.get("observers") or [])}
                              for e in (t.get("emits_events") or []) if isinstance(e, dict)],
                writes_terminal=bool(t.get("writes_terminal")
                                     or any(e["kind"] == "record_vote" for e in effects)),
                validation=norm(t.get("validation"), 200),
                unknown_effect_kinds=unknown).as_dict()
        return out
    if cache is not None:
        deps = {"blueprint_hash": bp.raw_response_hash,
                "n_actions": len(bp.action_templates)}
        raw, _hit = cache.get_or_compile("consequence_templates", deps, _build)
    else:
        raw = _build()
    return {aid: ConsequenceTemplate(**{**row, "effects": list(row["effects"]),
                                        "emits_events": list(row["emits_events"])})
            for aid, row in raw.items()}


class TemplateExecutor:
    """Binds + validates + executes one chosen action inside one world node. Mechanical only;
    every recipient interpretation is deferred to that recipient's own decision context."""

    def __init__(self, templates: dict, bp: ConsumerWorldBlueprint):
        self.templates = templates
        self.bp = bp
        self.hits = 0
        self.novel_requests = 0
        self.rejections: list = []

    def find(self, chosen_action: str) -> ConsequenceTemplate | None:
        """Deterministic resolution: exact id, else normalized description/id containment."""
        c = norm(chosen_action, 120).lower()
        if not c:
            return None
        t = self.templates.get(chosen_action)
        if t is not None:
            return t
        for tmpl in self.templates.values():
            if tmpl.action_id.lower() in c or c in tmpl.action_id.lower() \
                    or (tmpl.description and tmpl.description.lower()[:60] in c):
                return tmpl
        return None

    def validate(self, tmpl: ConsequenceTemplate, *, actor_id: str, actor_authority: list,
                 binding: dict) -> tuple:
        if tmpl.actor_ids and actor_id not in tmpl.actor_ids:
            return False, f"actor {actor_id} is not empowered for {tmpl.action_id}"
        req = {norm(a, 60).lower() for a in tmpl.authority_required}
        held = {norm(a, 60).lower() for a in (actor_authority or [])}
        if req and not req <= held:
            return False, f"missing authority {sorted(req - held)[:3]}"
        for e in tmpl.effects:
            if e["kind"] == "record_vote":
                opts = [str(o) for o in (e["params"].get("options") or [])]
                if opts and binding.get("vote_option") and binding["vote_option"] not in opts:
                    return False, (f"vote option '{binding['vote_option']}' outside the "
                                   f"mechanical options {opts}")
        return True, ""

    def execute(self, tmpl: ConsequenceTemplate, *, node, actor_id: str, binding: dict,
                day: str) -> dict:
        """Apply mechanical effects to the node's state. Returns the execution record. The
        node is the caller's OWN copy — branch isolation is the caller's responsibility."""
        rec = {"action_id": tmpl.action_id, "actor_id": actor_id, "day": day,
               "effects_applied": [], "events_emitted": []}
        for e in tmpl.effects:
            kind, p = e["kind"], dict(e["params"] or {})
            if kind == "record_vote":
                # the vote is recorded on the TERMINAL institution when the actor is one of its
                # members (the authoritative tally the terminal reads) — a blueprint
                # institution_id that disagrees with the terminal's must never split the tally
                term_inst = self.bp.terminal.get("institution_id") or ""
                term_members = set((self.bp.institution_by_id(term_inst) or {}).get("members")
                                   or [])
                inst = term_inst if actor_id in term_members else \
                    (p.get("institution_id") or term_inst)
                option = binding.get("vote_option") or (p.get("options") or [""])[0]
                node.institution_state.setdefault(inst, {}).setdefault("votes", {})[
                    actor_id] = str(option)
                rec["effects_applied"].append({"kind": kind, "institution": inst,
                                               "vote": str(option)})
            elif kind == "send_message":
                content = norm(binding.get("content") or p.get("value") or
                               binding.get("intended_effect"), 400)
                targets = binding.get("targets") or tmpl.targets
                for tgt in targets:
                    node.pending_observations.setdefault(str(tgt), []).append(
                        {"channel": "message", "source": actor_id, "content": content,
                         "day": day})
                rec["effects_applied"].append({"kind": kind, "targets": list(targets),
                                               "delivery": "mechanical only — recipient "
                                                           "interpretation stays actor-simulated"})
            elif kind == "schedule_meeting":
                node.event_queue.append({"day": str(binding.get("when") or p.get("value")
                                                    or day), "etype": p.get("key")
                                         or "meeting", "source": actor_id})
                rec["effects_applied"].append({"kind": kind})
            elif kind == "institution_stage":
                inst = p.get("institution_id") or ""
                node.institution_state.setdefault(inst, {})["stage"] = \
                    str(p.get("stage") or binding.get("stage") or "advanced")
                rec["effects_applied"].append({"kind": kind, "institution": inst})
            elif kind == "transfer_authority":
                node.authority_overrides[str(p.get("key") or "")] = \
                    str(binding.get("targets", [""])[0] if binding.get("targets") else
                        (tmpl.targets or [""])[0])
                rec["effects_applied"].append({"kind": kind})
            elif kind in ("open_window", "close_window"):
                node.windows[str(p.get("key") or tmpl.action_id)] = (kind == "open_window")
                rec["effects_applied"].append({"kind": kind})
            elif kind == "set_state":
                node.world_state[str(p.get("key") or tmpl.action_id)] = \
                    norm(binding.get("value") or p.get("value"), 200)
                rec["effects_applied"].append({"kind": kind, "key": p.get("key")})
        for em in tmpl.emits_events:
            node.emitted_events.append({"etype": em["etype"], "observers":
                                        list(em["observers"]), "source": actor_id, "day": day})
            rec["events_emitted"].append(em["etype"])
        self.hits += 1
        return rec


_NOVEL_PROMPT = """An actor in a causal simulation chose an action no precompiled template represents.
Compile its MECHANICAL consequence structure only — no persuasion outcomes, no interpretation, no
social reactions (those remain simulated by the affected actors themselves).

Actor: {actor_id}  Day: {day}
Chosen action: {chosen}
Intended effect (actor's words): {intended}
Known mechanical effect kinds: record_vote, send_message, schedule_meeting, institution_stage,
transfer_authority, open_window, close_window, set_state.

Reply ONLY JSON:
{{"action_id": "novel_<snake_case>", "description": "...", "actor_ids": ["{actor_id}"],
 "authority_required": [], "targets": ["<recipient ids>"],
 "effects": [{{"kind": "<one known kind>", "params": {{"institution_id": "", "options": [],
             "stage": "", "key": "", "value": ""}}}}],
 "emits_events": [{{"etype": "...", "observers": ["public"]}}],
 "writes_terminal": false, "validation": ""}}"""


def compile_novel_action(*, chosen: str, intended: str, actor_id: str, day: str,
                         gateway, budget_ledger, cache, executor: TemplateExecutor
                         ) -> ConsequenceTemplate | None:
    """ONE consequence-compile call for a genuinely novel action (strong tier — consequence
    compilation is a STRONG_ONLY stage). Content-keyed cache; failures never cached."""
    from swm.engine.grounding import parse_json
    ok, _why = budget_ledger.can_afford(what=f"novel_consequence:{norm(chosen, 40)}",
                                        est_calls=1, novel_consequence=True)
    if not ok:
        return None
    deps = {"chosen": norm(chosen, 200), "intended": norm(intended, 200),
            "actor": actor_id, "backend": gateway.backend_fingerprint}
    executor.novel_requests += 1

    def _call():
        return gateway.call("consequence_compile", _NOVEL_PROMPT.format(
            actor_id=actor_id, day=day, chosen=norm(chosen, 200),
            intended=norm(intended, 300)))
    cached = cache.get("consequence_templates", deps)
    text = cached if cached is not None else _call()
    r = parse_json(text)
    if not isinstance(r, dict) or not r.get("effects"):
        return None                                        # failure — never cached
    if cached is None:
        cache.put("consequence_templates", deps, text)
        budget_ledger.record_novel_consequence()
    effects = [{"kind": str(e.get("kind")), "params": dict(e.get("params") or {})}
               for e in (r.get("effects") or []) if isinstance(e, dict)
               and str(e.get("kind")) in MECHANICAL_EFFECT_KINDS]
    tmpl = ConsequenceTemplate(
        action_id=str(r.get("action_id") or f"novel_{len(executor.templates)}"),
        description=norm(r.get("description"), 240), actor_ids=[actor_id],
        targets=list(r.get("targets") or []), effects=effects,
        emits_events=[{"etype": str(e.get("etype") or ""),
                       "observers": list(e.get("observers") or [])}
                      for e in (r.get("emits_events") or []) if isinstance(e, dict)],
        writes_terminal=bool(r.get("writes_terminal")), source="novel_compiled")
    executor.templates[tmpl.action_id] = tmpl
    return tmpl


def manifest(executor: TemplateExecutor) -> dict:
    return {"n_templates": len(executor.templates),
            "precompiled": sorted(a for a, t in executor.templates.items()
                                  if t.source == "blueprint_precompiled"),
            "novel_compiled": sorted(a for a, t in executor.templates.items()
                                     if t.source == "novel_compiled"),
            "template_hits": executor.hits,
            "novel_requests": executor.novel_requests,
            "rejections": executor.rejections[-20:],
            "boundary_contract": "delivery mechanics precompiled; interpretation/persuasion/"
                                 "reaction remain branch-local actor decisions",
            "templates": {a: json.loads(json.dumps(t.as_dict(), default=str))
                          for a, t in list(executor.templates.items())[:24]}}
