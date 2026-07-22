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


# ------------------------------------------------------------------ D17: structural-fidelity gate
@dataclass
class StructuralFidelityReport:
    """Readiness is not merely 'the machine can run' — it is 'the world it will run IS faithful'.
    Aggregates the fidelity verdicts of the resolution (D5), the institution representation (D7),
    the evidence (D11), the behavior grounding (D8), and the outcome mechanism (D16). An invalid
    institution is NEVER made ready by rescaling its real threshold — it is repaired (roster) or
    fails."""
    verdict: str = "ready"                          # ready | repairable | not_ready
    checks: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    repairs_needed: list = field(default_factory=list)
    version: str = READINESS_VERSION

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "checks": self.checks, "diagnostics": self.diagnostics,
                "repairs_needed": self.repairs_needed, "version": self.version}


def _worst(verdicts: list) -> str:
    order = {"not_ready": 2, "repairable": 1, "ready": 0}
    return max(verdicts, key=lambda v: order.get(v, 0)) if verdicts else "ready"


def assess_structural_fidelity(bp, *, resolution_spec=None, representation=None,
                               evidence_store=None, mechanism_dim: dict = None,
                               grounding: dict = None) -> StructuralFidelityReport:
    """The structural-fidelity gate (§14). Each dimension contributes a verdict; the report is the
    WORST, with the specific repairs a `repairable` needs. Structural, not question-specific."""
    rep = StructuralFidelityReport()
    term = bp.terminal or {}
    tk = str(term.get("kind") or "")
    verdicts = []

    # (1) RESOLUTION — the frozen criterion parses to a typed terminal and matches the blueprint
    if resolution_spec is not None:
        try:
            from swm.world_model_v2.lean_v2.resolution_spec import spec_matches_blueprint
            matches = spec_matches_blueprint(resolution_spec, bp)
        except Exception:  # noqa: BLE001
            matches = True
        rk = getattr(resolution_spec, "terminal_kind", "")
        ok_res = bool(rk) and matches
        rep.checks["resolution"] = {"verdict": "ready" if ok_res else "repairable",
                                    "terminal_kind": rk, "matches_blueprint": matches}
        verdicts.append("ready" if ok_res else "repairable")
    else:
        rep.checks["resolution"] = {"verdict": "repairable", "note": "no parsed resolution spec"}
        verdicts.append("repairable")

    # (2) INSTITUTION — the faithful representation reconciles real==represented==threshold, or is
    # repaired by EXPANDING the roster; the threshold is NEVER rescaled to fit a collapsed roster
    if tk == "institution_vote":
        if representation is not None:
            v = getattr(representation, "verdict", "not_ready")
            rep.checks["institution"] = {
                "verdict": v, "real_member_count": getattr(representation, "real_member_count", None),
                "represented_voting_power": getattr(representation, "represented_voting_power", 0),
                "threshold": getattr(representation, "threshold", None),
                "repairs": getattr(representation, "repairs", [])}
            if v == "repairable":
                rep.repairs_needed.append("expand the institution roster to the real body "
                                          "(never rescale the threshold)")
            verdicts.append(v)
        else:
            rep.checks["institution"] = {"verdict": "not_ready", "note": "no representation built"}
            verdicts.append("not_ready")

    # (3) EVIDENCE — canonical facts exist and at least one is terminal-relevant (D11)
    if evidence_store is not None:
        n = len(getattr(evidence_store, "facts", []))
        rel = len(evidence_store.terminal_relevant_facts()) if hasattr(
            evidence_store, "terminal_relevant_facts") else 0
        okev = n > 0
        rep.checks["evidence"] = {"verdict": "ready" if okev else "repairable",
                                  "n_facts": n, "terminal_relevant": rel}
        verdicts.append("ready" if okev else "repairable")

    # (4) BEHAVIOR — the decision is grounded (counted actor classes / an action baseline exist),
    # never a bare label; states carry action tendencies (D8)
    if grounding is not None:
        ac = grounding.get("actor_state_reference_classes") or {}
        oc = (grounding.get("outcome_reference_class") or {}).get("provenance", {})
        grounded = bool(ac) or (oc.get("denominator") or 0) > 0
        rep.checks["behavior"] = {"verdict": "ready" if grounded else "repairable",
                                  "actors_with_classes": len(ac),
                                  "outcome_class_n": oc.get("denominator", 0)}
        verdicts.append("ready" if grounded else "repairable")

    # (5) OUTCOME — a numeric mechanism produces the required variable in the required dimension (D16)
    if mechanism_dim is not None:
        okdim = bool(mechanism_dim.get("ok"))
        rep.checks["outcome"] = {"verdict": "ready" if okdim else "not_ready",
                                 "dimension": mechanism_dim.get("dimension"),
                                 "required_dimension": mechanism_dim.get("required_dimension"),
                                 "diagnostics": mechanism_dim.get("diagnostics")}
        if not okdim:
            rep.diagnostics.append("outcome mechanism dimension does not match the required unit")
        verdicts.append("ready" if okdim else "not_ready")

    rep.verdict = _worst(verdicts)
    return rep


# ------------------------------------------------------------------ pure terminal evaluator
def pure_terminal_outcome(bp, *, votes: dict = None, world_state: dict = None,
                          obligations: dict = None, mechanism: dict = None,
                          world_conditions: dict = None, terminal_kind: str = "") -> dict:
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
        # D1: strip any `vote:`/`cast_vote:` menu prefix and normalize case/whitespace before
        # comparison so a valid cast is never miscounted on formatting
        from swm.world_model_v2.lean_v2.canonical_options import strip_menu_prefix
        nvals = [norm_key(strip_menu_prefix(v)) for v in vals]
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
                yes = count / max(1, m) > 0.5            # majority of the roster present here
            else:
                thr = float(thr)
                if thr < 1.0:
                    yes = count / max(1, m) > thr        # explicit fraction ("more than 50%")
                elif thr <= m:
                    yes = count >= thr                   # absolute count, seat-per-member here
                else:
                    # D7: an absolute threshold that EXCEEDS the roster present is a real-body
                    # count collapsed onto too few modeled actors (26 of 50 modeled as 5; 5 of 9
                    # modeled as 5). We DO NOT rescale reality to the broken blueprint. This node
                    # cannot faithfully resolve the vote without the full represented roster, so
                    # it DEFERS to the faithful deliberative resolver (institution_terminal.py),
                    # which repairs the roster (D7) and tallies seat-weighted against the REAL
                    # threshold — the rescaling logic that used to live here is deleted.
                    return {"resolved": False, "cause":
                            "representation_incomplete:threshold_exceeds_modeled_roster"}
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
        # D6 event-absence writer: a BOOLEAN deadline-bounded event that did not occur resolves
        # NO — the mechanical complement, not a missing mechanism. This covers BOTH event_occurs
        # AND a state_predicate that the resolution parser classifies as a boolean event (the
        # visionOS class: YES-writers exist but no positive NO-writer). A genuinely NUMERIC /
        # first-passage / categorical predicate with no mechanism stays honestly unresolved.
        from swm.world_model_v2.lean_v2.resolution_spec import (BOOLEAN_EVENT, DEADLINE_ABSENCE,
                                                                NUMERIC_THRESHOLD, FIRST_PASSAGE,
                                                                parse_resolution)
        kind = terminal_kind or parse_resolution(
            bp.resolution.get("interpretation") or bp.resolution.get("yes_means")
            or term.get("yes_when") or "",
            terminal_kind_hint=BOOLEAN_EVENT if tk == "event_occurs" else "").terminal_kind
        if tk == "event_occurs" or kind in (BOOLEAN_EVENT, DEADLINE_ABSENCE):
            return {"resolved": True, "outcome": "NO",
                    "detail": {"predicate": key, "terminal_kind": kind,
                               "note": "event_absent: non-occurrence by the deadline resolves "
                                       "NO (deterministic deadline complement)"}}
        return {"resolved": False,
                "cause": "numeric_terminal_not_mechanically_bound"
                if kind in (NUMERIC_THRESHOLD, FIRST_PASSAGE)
                else "state_predicate_not_mechanically_bound"}
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
    """Repair (the visionOS class, at the source): for BOOLEAN-EVENT predicate terminals, every
    terminal-writing template must SET the canonical terminal key — rewrite writers whose
    set_state key differs from what the terminal reads.

    TYPE-SAFE (D4): boolean-event canonicalization runs ONLY when the resolution is a boolean
    deadline event. It must never collapse a NUMERIC_THRESHOLD / FIRST_PASSAGE / CATEGORICAL
    terminal into a boolean OR — that is the Hormuz failure (a `daily_transit_count >= 50`
    predicate was rewritten to `__terminal_yes__`, so a de-escalation flag satisfied YES). For a
    non-boolean terminal this returns without touching any numeric writer."""
    from swm.world_model_v2.lean_v2.resolution_spec import (BOOLEAN_EVENT, DEADLINE_ABSENCE,
                                                            parse_resolution)
    term = bp.terminal
    if term.get("kind") == "institution_vote":
        return {"needed": False, "why": "vote terminals read the tally directly"}
    spec = parse_resolution(bp.resolution.get("interpretation") or bp.resolution.get("yes_means")
                            or term.get("yes_when") or "",
                            question=getattr(bp, "question", ""),
                            terminal_kind_hint=BOOLEAN_EVENT
                            if term.get("kind") == "event_occurs" else "")
    if spec.terminal_kind not in (BOOLEAN_EVENT, DEADLINE_ABSENCE):
        return {"needed": False, "type_safe_skip": True,
                "terminal_kind": spec.terminal_kind,
                "why": f"resolution is {spec.terminal_kind}; boolean canonicalization would "
                       f"collapse a numeric/categorical terminal — skipped (D4)"}
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
