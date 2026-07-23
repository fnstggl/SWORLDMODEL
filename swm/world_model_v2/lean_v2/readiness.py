"""Simulation-readiness gate + terminal semantic mapping with synthetic round-trip proof.

Rollout may not begin until the compiled world is PROVEN ready to simulate to its measured
outcome. Two instruments:

  * `TerminalSemanticMapping` + `terminal_round_trip` — build a synthetic known-YES terminal
    state and a synthetic known-NO terminal state, push each through the SAME pure terminal
    evaluator and the SAME distribution/recovery path the run will use, and require P(yes)=1
    and P(yes)=0 exactly. Alias table covers label casings, option text and canonical
    outcome names, so a COMPLETED simulation can never again be discarded because YES was
    spelled differently (the visionOS failure class). A non-empty resolved distribution that
    cannot be mapped STOPS finalization for mapping repair — it never falls through to the
    prior.

  * `SimulationReadinessReport` — ready / repairable / not_ready across: every consequential
    actor has weighted states; shared conditions exist; every actor has a trigger and a
    feasible action set; every institutional process has a deadline/completion rule; every
    required intermediate variable and the terminal outcome have writers; a complete path
    from initial state to the measured outcome exists in the exact units the question needs.
    `repairable` triggers targeted repair (writer-key canonicalization, missing triggers,
    mechanism recovery); rollout never starts `not_ready`."""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm, norm_key, parse_day

READINESS_VERSION = "lean_v2.readiness.v1"

#: the canonical world-state key every event_occurs/state_predicate terminal reads
CANONICAL_TERMINAL_KEY = "__terminal_yes__"


# ------------------------------------------------------------------ pure terminal evaluator
def pure_terminal_outcome(bp, *, votes: dict = None, world_state: dict = None,
                          obligations: dict = None, mechanism: dict = None,
                          world_conditions: dict = None) -> dict:
    """The ONE terminal evaluation law, pure and testable (the engine calls this; the
    round-trip proves it). Returns {"resolved", "outcome" ('YES'|'NO'), "detail"} or
    {"resolved": False, "cause": ...}."""
    term = bp.terminal
    tk = str(term.get("kind") or "")
    votes = votes or {}
    world_state = world_state or {}
    if tk == "institution_vote":
        inst = bp.institution_by_id(term.get("institution_id")) or {}
        members = list(inst.get("members") or [])
        missing = [m for m in members if m not in votes]
        if missing:
            return {"resolved": False, "cause": f"votes_missing:{','.join(missing[:5])}"}
        rule = str(term.get("decision_rule") or inst.get("decision_rule") or "unanimity")
        substantive = {m: v for m, v in votes.items() if not str(v).startswith("__")}
        non_substantive = [m for m in members if str(votes.get(m, "")).startswith("__")]
        vals = [substantive[m] for m in members if m in substantive]
        nvals = [norm_key(v) for v in vals]         # case/whitespace-normalized comparison
        rp = term.get("rule_params") or {}
        if rule == "unanimity":
            yes = (not non_substantive and len(vals) == len(members)
                   and len(set(nvals)) == 1)
        elif rule in ("all_option", "single"):
            opt = norm_key(rp.get("option"))
            yes = (not non_substantive
                   and (all(v == opt for v in nvals) if opt else len(set(nvals)) == 1))
        elif rule in ("majority", "threshold"):
            # the target option: the rule's option (normalized), else the most-common vote
            opt = norm_key(rp.get("option")) or (max(set(nvals), key=nvals.count)
                                                 if nvals else "")
            count = sum(1 for v in nvals if v == opt)
            m = len(members)
            thr = rp.get("threshold")
            if thr is None or str(thr) == "":
                yes = count / max(1, m) > 0.5
            else:
                thr = float(thr)
                if thr < 1.0:
                    yes = count / max(1, m) > thr        # explicit fraction ("more than 50%")
                elif thr < m:
                    yes = count >= thr                   # absolute count achievable in-model
                else:
                    # an absolute threshold that meets-or-exceeds the MODELED member count is
                    # a real-body count (a majority of a larger parliament/board) collapsed
                    # onto representative actors/blocs — 26 of 50 modeled as 5, 5 of 9 modeled
                    # as 5. A 'majority' rule means a majority: translate to a majority of the
                    # modeled substantive votes rather than an unreachable absolute count.
                    yes = count / max(1, m) > 0.5
        else:
            return {"resolved": False, "cause": f"unknown_rule:{rule}"}
        return {"resolved": True, "outcome": "YES" if yes else "NO",
                "detail": {"votes": dict(votes), "rule": rule,
                           "non_substantive": non_substantive}}
    if tk in ("event_occurs", "state_predicate"):
        # canonical key first; mechanism (bounded numeric process) second; alias keys third
        if CANONICAL_TERMINAL_KEY in world_state:
            yes = bool(world_state[CANONICAL_TERMINAL_KEY])
            return {"resolved": True, "outcome": "YES" if yes else "NO",
                    "detail": {"predicate": CANONICAL_TERMINAL_KEY}}
        if mechanism:
            from swm.world_model_v2.lean_v2.mechanisms import evaluate_bounded_process
            return evaluate_bounded_process(mechanism, world_conditions=world_conditions
                                            or world_state)
        key = norm(term.get("yes_when"), 80) or "occurred"
        if key in world_state:
            yes = bool(world_state[key])
            return {"resolved": True, "outcome": "YES" if yes else "NO",
                    "detail": {"predicate": key}}
        if tk == "event_occurs":
            return {"resolved": True, "outcome": "NO",
                    "detail": {"predicate": key,
                               "note": "non-occurrence by evaluation day resolves NO"}}
        return {"resolved": False, "cause": "state_predicate_not_mechanically_bound"}
    return {"resolved": False, "cause": f"unknown_terminal_kind:{tk}"}


# ------------------------------------------------------------------ terminal mapping
@dataclass
class TerminalSemanticMapping:
    yes_label: str
    no_label: str
    aliases_yes: set = field(default_factory=set)
    aliases_no: set = field(default_factory=set)

    def canonical(self, label) -> str | None:
        l = norm_key(label)
        if l in self.aliases_yes:
            return "YES"
        if l in self.aliases_no:
            return "NO"
        return None

    def as_dict(self) -> dict:
        return {"yes_label": self.yes_label, "no_label": self.no_label,
                "aliases_yes": sorted(self.aliases_yes),
                "aliases_no": sorted(self.aliases_no)}


def build_terminal_mapping(bp) -> TerminalSemanticMapping:
    opts = [str(o) for o in (bp.resolution.get("options") or [])]
    yes_label = opts[0] if opts else "YES"
    no_label = opts[1] if len(opts) > 1 else "NO"
    m = TerminalSemanticMapping(yes_label=yes_label, no_label=no_label)
    m.aliases_yes = {norm_key(yes_label), "yes", "true", "1",
                     norm_key(bp.terminal.get("yes_when"))[:60]}
    m.aliases_no = {norm_key(no_label), "no", "false", "0",
                    norm_key(bp.terminal.get("no_when"))[:60]}
    m.aliases_yes.discard("")
    m.aliases_no.discard("")
    overlap = m.aliases_yes & m.aliases_no
    m.aliases_yes -= overlap
    m.aliases_no -= overlap
    return m


def synthetic_terminal_states(bp, obligations: dict = None) -> tuple:
    """(yes_case, no_case) synthetic inputs for the round-trip: votes for institution_vote,
    world_state for predicate terminals. Deterministic, from the blueprint's own mechanics."""
    term = bp.terminal
    if term.get("kind") == "institution_vote":
        inst = bp.institution_by_id(term.get("institution_id")) or {}
        members = list(inst.get("members") or [])
        from swm.world_model_v2.lean_v2.state_completeness import feasible_options_for
        opts = []
        for mmb in members:
            o = feasible_options_for(bp, mmb)
            opts.append(o[0] if o else "option_a")
        rule = str(term.get("decision_rule") or inst.get("decision_rule") or "unanimity")
        target = str((term.get("rule_params") or {}).get("option") or "")
        common = target or (opts[0] if opts else "option_a")
        # an alternative option distinct from `common`, drawn from the members' feasible sets
        alt = None
        for mmb in members:
            for o in feasible_options_for(bp, mmb):
                if norm_key(o) != norm_key(common):
                    alt = o
                    break
            if alt:
                break
        alt = alt or f"not_{common}"
        yes_votes = {m: common for m in members}
        if rule in ("majority", "threshold", "all_option", "single"):
            # YES = enough of the target option; NO = clearly below it → everyone votes
            # the alternative (a single flip may still clear an "at least N" threshold)
            no_votes = {m: alt for m in members}
        else:
            # unanimity: YES = all the same; NO = a genuine split (one member differs)
            no_votes = dict(yes_votes)
            if members:
                no_votes[members[-1]] = alt
        return {"votes": yes_votes}, {"votes": no_votes}
    return ({"world_state": {CANONICAL_TERMINAL_KEY: True}},
            {"world_state": {CANONICAL_TERMINAL_KEY: False}})


def terminal_round_trip(bp, *, obligations: dict = None, mechanism: dict = None) -> dict:
    """The proof: synthetic YES maps to P(yes)=1, synthetic NO to P(yes)=0, through the SAME
    evaluator + distribution + recovery yes-key path the live run uses."""
    from swm.world_model_v2.forecast_recovery import recover_forecast
    mapping = build_terminal_mapping(bp)
    yes_case, no_case = synthetic_terminal_states(bp, obligations)
    results = {"mapping": mapping.as_dict(), "checks": [], "ok": True}
    for name, case, want in (("known_yes", yes_case, "YES"), ("known_no", no_case, "NO")):
        out = pure_terminal_outcome(bp, votes=case.get("votes"),
                                    world_state=case.get("world_state"),
                                    obligations=obligations, mechanism=mechanism)
        ok_eval = bool(out.get("resolved")) and out.get("outcome") == want
        # distribution + recovery leg: a resolved outcome must map to p exactly 1/0
        dist = {mapping.yes_label: 1.0 if want == "YES" else 0.0,
                mapping.no_label: 0.0 if want == "YES" else 1.0}
        rec = recover_forecast(distribution=dist,
                               options=[mapping.yes_label, mapping.no_label],
                               unresolved_mass=0.0)
        p = rec.probability if rec is not None else None
        ok_map = (p == (1.0 if want == "YES" else 0.0)
                  and rec is not None and rec.probability_source == "completed_rollouts")
        results["checks"].append({"case": name, "evaluator_ok": ok_eval,
                                  "evaluator_out": out, "recovery_p": p,
                                  "mapping_ok": ok_map})
        results["ok"] = results["ok"] and ok_eval and ok_map
    return results


def canonicalize_terminal_writers(bp) -> dict:
    """Repair (the visionOS class, at the source): for predicate terminals, every
    terminal-writing template must SET the canonical terminal key — rewrite writers whose
    set_state key differs from what the terminal reads. Returns the repair record."""
    term = bp.terminal
    if term.get("kind") == "institution_vote":
        return {"needed": False, "why": "vote terminals read the tally directly"}
    rewritten = []
    for t in bp.action_templates:
        if not (t.get("writes_terminal") or any(e.get("kind") == "set_state"
                                                for e in t.get("effects") or [])):
            continue
        for e in t.get("effects") or []:
            if e.get("kind") != "set_state":
                continue
            p = e.setdefault("params", {})
            if p.get("key") != CANONICAL_TERMINAL_KEY:
                rewritten.append({"action_id": t.get("action_id"),
                                  "old_key": p.get("key"),
                                  "new_key": CANONICAL_TERMINAL_KEY})
                p["key"] = CANONICAL_TERMINAL_KEY
                p.setdefault("value", "true")
                t["writes_terminal"] = True
    return {"needed": bool(rewritten), "rewritten": rewritten,
            "canonical_key": CANONICAL_TERMINAL_KEY}


# ------------------------------------------------------------------ the readiness gate
@dataclass
class SimulationReadinessReport:
    verdict: str = "not_ready"                 # ready | repairable | not_ready
    checks: list = field(default_factory=list)
    repairs_needed: list = field(default_factory=list)
    repairs_applied: list = field(default_factory=list)
    round_trip: dict = field(default_factory=dict)
    version: str = READINESS_VERSION

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "checks": self.checks,
                "repairs_needed": self.repairs_needed,
                "repairs_applied": self.repairs_applied,
                "round_trip": self.round_trip, "version": self.version}


def simulation_readiness(*, bp, consequential_actors: list, completed_states: dict,
                         grounded_weights: dict, obligations: dict, executor,
                         mechanism: dict = None, shared_combos: list = None
                         ) -> SimulationReadinessReport:
    rep = SimulationReadinessReport()

    def check(name, ok, note="", repairable_as=""):
        rep.checks.append({"check": name, "ok": bool(ok), "note": str(note)[:180]})
        if not ok:
            if repairable_as:
                rep.repairs_needed.append({"check": name, "repair": repairable_as})
            else:
                rep.repairs_needed.append({"check": name, "repair": ""})
        return bool(ok)

    hard_fail = False
    for aid in consequential_actors:
        states = completed_states.get(aid) or []
        ok = check(f"actor_states:{aid}", bool(states),
                   f"{len(states)} state(s)", repairable_as="state_recovery_ladder")
        hard_fail = hard_fail or not ok
        gw = grounded_weights.get(aid) or {}
        wsum = sum((gw.get("mid") or {}).values())
        # tolerance accommodates rounded counted weights (they sum to ~0.9999); a genuinely
        # unnormalized set (far from 1) still fails and is renormalized
        check(f"actor_weights:{aid}", states and abs(wsum - 1.0) < 5e-3,
              f"weight sum {wsum:.4f}", repairable_as="renormalize_represented_weights")
    check("shared_conditions_exist", shared_combos is not None,
          f"{len(shared_combos or [])} combo(s)")
    from swm.world_model_v2.lean_v2.state_completeness import feasible_options_for
    for aid in consequential_actors:
        feas = feasible_options_for(bp, aid)
        triggered = any(d.get("actor_id") == aid for d in bp.decision_triggers) \
            or any(aid in (ob.required_participants or []) for ob in obligations.values())
        check(f"actor_trigger:{aid}", triggered, "",
              repairable_as="schedule_mandatory_trigger")
        check(f"actor_feasible_actions:{aid}", bool(feas), f"{len(feas)} option(s)",
              repairable_as="fallback_action_menu")
    for iid, ob in obligations.items():
        check(f"institution_deadline:{iid}",
              bool(ob.deadline_day and parse_day(ob.deadline_day)),
              ob.deadline_day, repairable_as="derive_deadline_from_terminal")
    # terminal pathway + units
    term = bp.terminal
    tk = str(term.get("kind") or "")
    if tk == "institution_vote":
        check("terminal_writer", any(t.writes_terminal
                                     for t in executor.templates.values()), "vote writers")
        check("terminal_units", True, "vote tally in members — exact units")
    else:
        has_writer = any(
            e.get("kind") == "set_state"
            and (e.get("params") or {}).get("key") == CANONICAL_TERMINAL_KEY
            for t in bp.action_templates for e in (t.get("effects") or []))
        check("terminal_writer", has_writer or mechanism is not None,
              "canonical-key writer or bounded mechanism",
              repairable_as="canonicalize_writers_or_mechanism_recovery")
        check("terminal_units", mechanism is not None or tk == "event_occurs",
              "numeric predicates need a bounded mechanism in question units",
              repairable_as="mechanism_recovery")
    rep.round_trip = terminal_round_trip(bp, obligations=obligations, mechanism=mechanism)
    check("terminal_round_trip", rep.round_trip.get("ok"),
          "synthetic YES→1 / NO→0 through the live path",
          repairable_as="terminal_mapping_repair")
    failed = [c for c in rep.checks if not c["ok"]]
    if not failed:
        rep.verdict = "ready"
    elif any(not r["repair"] for r in rep.repairs_needed) or hard_fail:
        rep.verdict = "not_ready"
    else:
        rep.verdict = "repairable"
    return rep
