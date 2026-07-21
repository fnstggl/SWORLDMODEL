"""Three-valued answerability preflight — prove the world can answer BEFORE actors spend calls.

Returns exactly one of:

    answerable    — a valid executable terminal pathway is PROVEN (predicate exists, a writer
                    exists and is instantiated, its inputs are producible, a YES path and a NO
                    path both exist where both remain possible, the mechanism is accepted, and
                    unresolved branches can still contribute through forecast recovery);
    unanswerable  — a CONCRETE missing/invalid implementation mechanism makes terminal
                    resolution impossible (named, with the exact failing check);
    uncertain     — static analysis can neither prove nor disprove; ONE minimal bounded
                    pathway probe (deterministic symbolic micro-walk, zero LLM calls) runs
                    before deciding to repair or stop.

Failure to prove statically is NEVER treated as proof of impossibility, and no YES/NO path is
ever invented to satisfy a validator: a genuinely one-sided world is recorded as one-sided.
If a required mechanism is absent the caller attempts ONE targeted repair, rechecks, and if
still absent STOPS before expensive actor simulation, returning the best defensible labeled
forecast under the forecast-recovery contract with the exact blocking gap reported."""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import (ConsumerWorldBlueprint, parse_day,
                                                  vote_paths_possible)


@dataclass
class PreflightReport:
    verdict: str = "uncertain"                 # answerable | unanswerable | uncertain
    checks: list = field(default_factory=list)  # [{check, ok, note}]
    blocking: list = field(default_factory=list)
    probe: dict = field(default_factory=dict)
    one_sided: str = ""                        # non-empty => only this outcome is mechanically possible

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "checks": self.checks, "blocking": self.blocking,
                "probe": self.probe, "one_sided": self.one_sided}


def _check(rep: PreflightReport, name: str, ok, note: str = "", blocking_if_false=True):
    rep.checks.append({"check": name, "ok": bool(ok), "note": str(note)[:200]})
    if not ok and blocking_if_false:
        rep.blocking.append({"check": name, "note": str(note)[:200]})
    return bool(ok)


def run_preflight(bp: ConsumerWorldBlueprint, *, as_of: str, horizon: str,
                  consequence_templates: dict) -> PreflightReport:
    """Static answerability over the validated blueprint + instantiated templates."""
    rep = PreflightReport()
    term = bp.terminal
    tk = str(term.get("kind") or "")

    # 1. a terminal predicate exists
    has_pred = tk in ("institution_vote", "event_occurs", "state_predicate") \
        and (term.get("yes_when") or term.get("decision_rule"))
    _check(rep, "terminal_predicate_exists", has_pred,
           f"kind={tk or '(missing)'}")

    # 2-3. a terminal-writing operator exists AND is instantiated
    writers = list(term.get("written_by_action_ids") or [])
    writers += [t.get("action_id") for t in bp.action_templates
                if t.get("writes_terminal") or any(e.get("kind") == "record_vote"
                                                   for e in (t.get("effects") or []))]
    writers = list(dict.fromkeys(w for w in writers if w))
    _check(rep, "terminal_writer_exists", bool(writers),
           f"writers={writers[:4]}")
    instantiated = [w for w in writers if w in consequence_templates]
    _check(rep, "terminal_writer_instantiated", bool(instantiated),
           f"instantiated={instantiated[:4]} of {len(writers)}")

    # 4. required inputs producible: every writer has an actor who can take it, and the
    #    triggering moment exists before the horizon (a trigger, anchor or procedure stage)
    producible, why_prod = False, "no writer with an empowered actor and a pre-horizon trigger"
    d_hor = parse_day(horizon) or parse_day(term.get("evaluation_day"))
    trigger_days = [parse_day(d.get("when_day")) for d in bp.decision_triggers]
    anchor_days = [parse_day(t.get("day")) for t in bp.temporal_anchors]
    moments = [d for d in trigger_days + anchor_days if d is not None]
    for w in instantiated:
        t = next((x for x in bp.action_templates if x.get("action_id") == w), None)
        if t is None or not t.get("actor_ids"):
            continue
        if moments and (d_hor is None or min(moments) <= d_hor):
            producible, why_prod = True, f"writer {w} reachable at {min(moments)}"
            break
        if not moments and parse_day(term.get("evaluation_day")) is not None:
            producible = True
            why_prod = f"writer {w}; terminal evaluation day anchors the moment"
            break
    _check(rep, "writer_inputs_producible", producible, why_prod)

    # 5-6. YES path and NO path (where both remain possible)
    if tk == "institution_vote":
        yes_ok, no_ok, note = vote_paths_possible(bp)
    elif tk == "event_occurs":
        yes_ok = bool(writers) or any(m.get("writes_terminal") for m in bp.mechanisms)
        no_ok = True                     # non-occurrence by horizon is always a valid NO path
        note = "event_occurs: NO = censoring at horizon"
    else:
        yes_ok = no_ok = bool(term.get("yes_when")) and bool(term.get("no_when"))
        note = "state_predicate: both predicates stated"
    one_sided_confirmed = bp.validation.get("one_sided_confirmed")
    if one_sided_confirmed and (yes_ok ^ no_ok):
        rep.one_sided = ("YES" if yes_ok else "NO") + f" only — {one_sided_confirmed}"
        _check(rep, "yes_path_exists", yes_ok, note, blocking_if_false=False)
        _check(rep, "no_path_exists", no_ok, note, blocking_if_false=False)
    else:
        _check(rep, "yes_path_exists", yes_ok, note)
        _check(rep, "no_path_exists", no_ok, note)

    # 7. the outcome mechanism is accepted by the current causal rules (effect kinds known)
    unknown_effects = []
    for w in instantiated:
        tmpl = consequence_templates.get(w)
        unknown_effects += [e for e in (tmpl.unknown_effect_kinds if tmpl else [])]
    _check(rep, "outcome_mechanism_accepted", not unknown_effects,
           f"unknown effect kinds: {unknown_effects[:4]}" if unknown_effects else "all "
           "terminal-writing effects are known mechanical kinds")

    # 8. unresolved branches can still contribute (forecast-recovery pathway always present;
    #    a grounded rate or evidence-conditioned prior widens it — recorded, not required)
    _check(rep, "unresolved_recovery_pathway", True,
           f"forecast_recovery active; grounded_rates={len(bp.grounded_rates)}",
           blocking_if_false=False)

    # 9. not inevitably all-unresolved: at least one path from a trigger to a writer to the
    #    terminal evaluation exists within the horizon
    inevitable_dead = not (has_pred and writers and instantiated and producible)
    _check(rep, "not_inevitably_unresolved", not inevitable_dead,
           "a missing implementation mechanism would leave every particle unresolved"
           if inevitable_dead else "at least one resolving path exists")

    # ---- verdict ----------------------------------------------------------------------
    hard_missing = [b for b in rep.blocking
                    if b["check"] in ("terminal_predicate_exists", "terminal_writer_exists",
                                      "terminal_writer_instantiated",
                                      "not_inevitably_unresolved",
                                      "outcome_mechanism_accepted")]
    if not rep.blocking:
        rep.verdict = "answerable"
    elif hard_missing:
        rep.verdict = "unanswerable"
    else:
        rep.verdict = "uncertain"
        rep.probe = bounded_pathway_probe(bp, consequence_templates)
        if rep.probe.get("reached_terminal"):
            rep.verdict = "answerable"
            rep.checks.append({"check": "bounded_pathway_probe", "ok": True,
                               "note": "probe reached terminal evaluation"})
        elif rep.probe.get("proven_impossible"):
            rep.verdict = "unanswerable"
            rep.blocking.append({"check": "bounded_pathway_probe",
                                 "note": rep.probe.get("why", "")[:200]})
        # else: stays uncertain — the caller decides repair vs stop; static unprovability is
        # NOT proof of impossibility
    return rep


def bounded_pathway_probe(bp: ConsumerWorldBlueprint, consequence_templates: dict) -> dict:
    """One minimal bounded probe: symbolically walk the cheapest trigger→action→terminal path
    with an ARBITRARY valid vote/action assignment. Deterministic, zero LLM calls."""
    term = bp.terminal
    tk = str(term.get("kind") or "")
    if tk == "institution_vote":
        inst = bp.institution_by_id(term.get("institution_id"))
        if inst is None:
            return {"proven_impossible": True, "why": "terminal institution absent"}
        votes = {}
        for m in inst.get("members") or []:
            for t in bp.action_templates:
                if m not in (t.get("actor_ids") or []):
                    continue
                for e in t.get("effects") or []:
                    if e.get("kind") == "record_vote":
                        opts = [str(o) for o in ((e.get("params") or {}).get("options") or [])]
                        if opts:
                            votes[m] = opts[0]
        members = list(inst.get("members") or [])
        if members and all(m in votes for m in members):
            return {"reached_terminal": True,
                    "walk": f"assigned first option to all {len(members)} members; terminal "
                            f"evaluable under rule "
                            f"'{term.get('decision_rule') or inst.get('decision_rule')}'"}
        missing = [m for m in members if m not in votes]
        return {"proven_impossible": bool(missing), "why": f"members without any vote "
                f"action: {missing[:4]}"}
    if tk == "event_occurs":
        return {"reached_terminal": bool(term.get("evaluation_day")),
                "walk": "occurrence-or-censoring is evaluable at horizon"}
    return {"reached_terminal": bool(term.get("yes_when") and term.get("no_when")),
            "walk": "state predicates stated"}
