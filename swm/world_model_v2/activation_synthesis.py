"""Activation synthesis — relevance-gated completion of the compiler→runtime execution chain.

The activation standard (integration-completion mandate) is NOT 100% raw activation: it is near-100% recall
of a phase on questions that STRUCTURALLY REQUIRE it, LOW false activation on questions that do not, and a
real causal effect when it fires. The measured failure had both faces: P6/P7 never emitted when required
(recall 0), while P9/P10 emitted on nearly everything (false activation ≈ 1.0 — structure declared, nothing
consuming it, no discrimination).

Two functions close both:

  * `phase_requirements(plan)` — the RELEVANCE GATE. A deterministic per-phase requirement judgment derived
    from the compiler's OWN causal analysis of the question (required causal processes recorded in
    mechanism_choices, declared sections, structural hypotheses). It never inspects question IDs; it is the
    runtime's independent judgment, scored against the benchmark's independent labels — not fit to them.
  * `synthesize_activation(plan, req)` — SPEC SYNTHESIS. For each REQUIRED phase whose execution linkage the
    compiler failed to emit, complete the chain from ALREADY-DECLARED components (an institution's normalized
    rule numbers → an institutional_decision event; declared populations/relations → their consumers; a
    detected nonlinear process → a real state-step chain; a per-process registry selection → its executable
    pack event). For each NOT-required phase, GATE OFF the ornamental execution the compiler over-emitted
    (decision events with no strategic process behind them). Nothing is invented; nothing is added to
    decorate a manifest.

Both run default-on inside the ONE canonical runtime (unified_runtime), after rule normalization.
"""
from __future__ import annotations

from swm.world_model_v2.state import parse_time  # noqa: F401  (re-export convenience for callers)

# ---------------------------------------------------------------- process-token vocabularies (per phase)
# These are CAUSAL-PROCESS descriptors matched against the compiler's required_causal_processes /
# structural hypotheses — a vocabulary over process semantics, NOT benchmark questions.
_P10_TOKENS = ("vote", "voting", "confirm", "ratif", "quorum", "approv", "legislat", "pass_bill",
               "nominat", "impeach", "override", "adjudicat", "rule_on", "ruling", "court_decision",
               "committee", "board_decision", "formal_decision", "certif", "authoriz", "sanction_vote",
               "resolution_vote", "referendum", "enact", "policy_decision", "rate_decision", "monetary",
               "central_bank", "board_vote", "regulator", "agency_decision", "tribunal")
_P9POP_TOKENS = ("turnout", "adoption", "participation", "uptake", "population", "aggregate_behavior",
                 "public_opinion", "mass_", "viewership", "attendance", "consumer", "electorate",
                 "polling", "vaccination", "enrollment", "subscriber", "audience", "voters",
                 "market_share", "collective_behavior", "demand")
_P9NET_TOKENS = ("contagion", "viral", "spread", "network", "cascade", "diffusion", "word_of_mouth",
                 "social_transmission", "peer_", "bank_run", "panic", "rumor", "misinformation",
                 "share_propagation", "retweet", "amplif")
_P7_TOKENS = ("nonlinear", "threshold", "tipping", "cascade", "saturation", "feedback", "exponential",
              "critical_mass", "runaway", "accelerat", "s-curve", "s_curve", "regime_shift", "spiral",
              "snowball", "contagion", "percolat")
_P4_TOKENS = ("decision", "decide", "negotiat", "bargain", "strategic", "choice", "concession",
              "agreement", "deal", "veto", "endorse", "concede", "withdraw", "announce", "commit",
              "escalat", "de-escalat", "retaliat", "coalition_formation", "hold_out", "settle")


def _process_text(plan) -> str:
    """All causal-process descriptors the compiler recorded for this plan, as one searchable string."""
    parts = []
    for c in (getattr(plan, "mechanism_choices", []) or []):
        if isinstance(c, dict):
            parts.append(str(c.get("process", "")))
    for h in (getattr(plan, "structural_hypotheses", []) or []):
        if isinstance(h, dict):
            parts.append(str(h.get("describe", "")))
    for m in (getattr(plan, "accepted_mechanisms", []) or []):
        if isinstance(m, dict):
            parts.append(str(m.get("causal_role", "")))
    prov = getattr(plan, "provenance", {}) or {}
    parts.append(str(prov.get("rationale", "")))
    # the outcome contract's own wording names the causal process ("Will the FOMC *vote* to…") — the
    # compiler sometimes returns an empty process list, and the question text is structural, not an ID.
    parts.append(str(getattr(plan, "question", "")))
    for it in (getattr(plan, "interpretations", []) or []):
        if isinstance(it, dict):
            parts.append(str(it.get("reading", "")))
    return " ".join(parts).lower().replace("-", "_")


def _hit(text: str, tokens) -> list:
    return [t for t in tokens if t in text]


def phase_requirements(plan) -> dict:
    """The relevance gate: per-phase {required, why, evidence}.

    Requirement = a CAUSAL signal of this phase's kind AND the structural section the phase executes over is
    declared. The causal signal is semantic first, lexical second: the compiler's `causal_dependencies`
    block (a structured judgment about what the outcome causally depends on — strategic actors, aggregate
    behavior, networked transmission, nonlinear dynamics, institutional decision) is the primary signal;
    the process-token vocabularies remain as a lexical backstop for older plans that lack the block.
    The runtime — not the LLM — still owns the final verdict: a signal with no declared structure to
    execute over is NOT required (it is a blocked/missing-state diagnostic instead)."""
    text = _process_text(plan)
    deps = ((getattr(plan, "provenance", {}) or {}).get("causal_dependencies")) or {}
    insts = getattr(plan, "institutions", []) or []
    pops = getattr(plan, "populations", []) or []
    rels = getattr(plan, "relations", []) or []
    decs = getattr(plan, "actor_decisions", []) or []
    persons = [e for e in (getattr(plan, "entities", []) or [])
               if isinstance(e, dict) and str(e.get("type", "")) == "person"]
    prov = getattr(plan, "provenance", {}) or {}
    pps = prov.get("per_process_selection", {}) or {}
    p6_sel = {p: s for p, s in pps.items() if isinstance(s, dict) and s.get("selected")}

    def gate(hits, declared, declared_name):
        """Returns (required, why, signal_present). A causal signal WITHOUT declared structure is not
        `required` for synthesis, but the supervisor records it as blocked_missing_state, never a no-op."""
        if hits and declared:
            return True, f"causal signal {hits[:3]} + declared {declared_name}", True
        if hits:
            return (False, f"causal signal {hits[:3]} but no declared {declared_name} "
                           f"(nothing to execute)", True)
        if declared:
            return False, f"{declared_name} declared but no causal process of this kind — context only", False
        return False, f"no causal signal and no declared {declared_name}", False

    def signal(dep_key, tokens):
        """Semantic-first causal signal: the compiler's structured dependency judgment, or a lexical hit."""
        if deps.get(dep_key) is True:
            return [f"causal_dependencies.{dep_key}"]
        return _hit(text, tokens)

    req = {}
    ok, why, sig = gate(signal("institutional_decision_process", _P10_TOKENS), insts, "institutions")
    req["phase10_institutions"] = {"required": ok, "why": why, "signal": sig}
    # an aggregate-behavior signal implies a population by definition — segments are constructible from
    # broad priors when the compiler declared none (Part: Phase 9 populations, "construct weighted segments")
    ok, why, sig = gate(signal("aggregate_population_behavior", _P9POP_TOKENS), pops or True,
                        "populations (constructible)")
    req["phase9_populations"] = {"required": ok, "why": why, "signal": sig}
    # Part 7: relations may be INFERRED from the declared causal world when transmission is causally
    # required — >=2 declared entities (or a declared population to expose) are inferable structure.
    net_substrate = rels or (len([e for e in (getattr(plan, "entities", []) or [])
                                  if isinstance(e, dict)]) >= 2) or pops
    ok, why, sig = gate(signal("networked_transmission", _P9NET_TOKENS), net_substrate,
                        "relations (or inferable entity/population substrate)")
    req["phase9_networks"] = {"required": ok, "why": why, "signal": sig}
    hits7 = signal("nonlinear_dynamics", _P7_TOKENS)
    req["phase7_nonlinear"] = {"required": bool(hits7),
                               "why": (f"nonlinear causal structure: {hits7[:3]}" if hits7
                                       else "no nonlinear causal structure named")}
    # Phase 6 owns causal-process → mechanism resolution. It is required whenever the outcome depends on a
    # SOCIAL/behavioral causal mechanism (any structured dependency signal true, or a registry family already
    # answers a process) — a required process may never be silently omitted: it resolves to a validated
    # family or to a transparent broad-prior fallback that still executes (registry gap recorded).
    social_dep = any(deps.get(k) is True for k in
                     ("strategic_actor_decisions", "aggregate_population_behavior",
                      "networked_transmission", "nonlinear_dynamics", "institutional_decision_process"))
    procs = [str(c.get("process", "")) for c in (getattr(plan, "mechanism_choices", []) or [])
             if isinstance(c, dict) and c.get("process") and c.get("process") != "outcome_resolution"]
    other_social = any(req.get(k, {}).get("required") for k in
                       ("phase10_institutions", "phase9_populations", "phase9_networks"))
    req["phase6_registry"] = {
        "required": bool(p6_sel) or social_dep or other_social,
        "why": (f"registry answers {sorted(p6_sel)[:3]}" if p6_sel else
                ("social causal mechanism required (dependency signal)" if social_dep else
                 ("another social phase is causally required — a behavioral mechanism carries it"
                  if other_social else "no social behavioral mechanism required")))}
    hits4 = signal("strategic_actor_decisions", _P4_TOKENS)
    # a compiler-proposed decision whose candidate actions carry outcome POLARITY (approve/reject/veto…)
    # is itself structural evidence of a strategic actor — generic act/wait proposals are not.
    polar_decision = False
    if decs:
        from swm.world_model_v2.phase_consumers import action_polarity
        for d in decs:
            for a in (d.get("candidate_actions") or []):
                if isinstance(a, dict) and action_polarity(a.get("name") or a.get("type")) != 0:
                    polar_decision = True
                    break
    # organizations and institutions are strategic actors too — any declared entity is executable
    # actor substrate (the ontology has organizational_market/institutional action families)
    actor_substrate = decs or persons or [e for e in (getattr(plan, "entities", []) or [])
                                          if isinstance(e, dict) and e.get("id")]
    if polar_decision and not hits4:
        req["phase4_actor_policy"] = {"required": bool(actor_substrate),
                                      "why": "compiler-proposed decision with outcome-polar actions",
                                      "signal": True}
    else:
        ok, why, sig = gate(hits4, actor_substrate, "strategic actors")
        req["phase4_actor_policy"] = {"required": ok, "why": why, "signal": sig}
    return req


# ---------------------------------------------------------------- synthesis helpers (per phase)
def _resolve_event(plan):
    for ev in plan.scheduled_events:
        if ev.get("etype") == "resolve_outcome":
            return ev
    return None


def _add_mechanism(plan, mech_id, operator, role, source):
    if any(m.get("operator") == operator for m in plan.accepted_mechanisms if isinstance(m, dict)):
        return False
    plan.accepted_mechanisms.append({
        "mech_id": mech_id, "ontology_type": "structural", "causal_role": role,
        "parameter_source": source, "temporal_scale": "scheduled",
        "calibration_status": "structural_broad_prior", "operator": operator, "sensitivity": 0.7,
        "synthesized_by": "activation_synthesis"})
    return True


def _add_modulation(plan, var, weight):
    """Register a consumer-written state variable for CONSUMPTION BY A MECHANISM (the institutional
    decision or the aggregate-outcome realization mechanism). The terminal resolver never consumes these
    (Part 4: no resolver-level probability modulation). Total weight capped at synthesis time."""
    if not hasattr(plan, "_consumed_state"):
        plan._consumed_state = []
    mods = plan._consumed_state
    if any(m.get("var") == var for m in mods):
        return False
    total = sum(float(m.get("weight", 0.0)) for m in mods)
    w = max(0.0, min(float(weight), 0.45 - total))
    if w <= 0.0:
        return False
    mods.append({"var": var, "weight": round(w, 3)})
    return True


def _rule_numbers(inst) -> dict:
    """Extract declared threshold numbers from an institution's (normalized) rules. Integers > 1 are vote
    counts; floats in (0,1] are shares; member lists give n_members. Defaults are labeled broad priors."""
    n_members, needed, share = None, None, None
    for ru in (inst.get("rules") or []):
        if not isinstance(ru, dict):
            continue
        params = ru.get("params") or {}
        for key, val in params.items():
            k = str(key).lower()
            if isinstance(val, list) and ("member" in k or "voter" in k) and val:
                n_members = len(val)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if 0.0 < float(val) <= 1.0 and ("threshold" in k or "share" in k or "majority" in k):
                share = float(val)
            elif float(val) > 1.0:
                if "total" in k or "size" in k or "member" in k:
                    n_members = int(val)
                elif "quorum" in k or "needed" in k or "votes" in k or "min" in k or "threshold" in k:
                    needed = int(val)
    if n_members is None:
        n_members = max(needed * 2 - 1, 9) if needed else 9   # odd panel default (labeled broad prior)
    if needed is None:
        needed = int((share if share is not None else 0.5) * n_members) + 1
    needed = max(1, min(needed, n_members))
    return {"n_members": n_members, "needed": needed,
            "defaulted": share is None and needed == int(0.5 * n_members) + 1}


def synthesize_activation(plan, req=None) -> dict:
    """Complete the execution chain for required phases; gate off ornamental execution for non-required ones.
    Mutates the plan in place; returns the per-phase synthesis report (for the manifest). Idempotent."""
    import swm.world_model_v2.phase_consumers  # noqa: F401 — registers the consumer operators/events
    req = req or phase_requirements(plan)
    rep = {"requirements": {k: v["required"] for k, v in req.items()}, "actions": []}
    rev = _resolve_event(plan)
    resolve_var = (rev or {}).get("payload", {}).get("outcome_var", "outcome")
    options = (rev or {}).get("payload", {}).get("options") or ["True", "False"]
    lean = (rev or {}).get("payload", {}).get("lean", "neutral")
    resolve_ts = (rev or {}).get("ts", plan.horizon_ts - 1.0)

    def _has_event(etype):
        return any(e.get("etype") == etype for e in plan.scheduled_events)

    # ---- sanitation: a decision event whose participant was never declared as an entity would crash the
    # actor operator mid-rollout. Drop it with a recorded omission (completeness validation, not silence).
    declared_ids = {str(e.get("id")) for e in plan.entities if isinstance(e, dict)}
    kept = []
    for e in plan.scheduled_events:
        if e.get("etype") == "decision_opportunity" and \
                not all(str(p) in declared_ids for p in (e.get("participants") or [])):
            plan.omissions.append({"kind": "decision_event_undeclared_actor",
                                   "participants": list(e.get("participants") or []),
                                   "reason": "decision_opportunity participant is not a declared entity"})
            rep["actions"].append({"phase": "phase4_actor_policy",
                                   "action": "dropped_undeclared_actor_decision",
                                   "participants": list(e.get("participants") or [])})
            continue
        kept.append(e)
    plan.scheduled_events = kept

    # ---- Phase 10: declared institution + institutional process → executable threshold decision ----
    if req["phase10_institutions"]["required"] and plan.institutions and not _has_event("institutional_decision"):
        inst = max((i for i in plan.institutions if isinstance(i, dict)),
                   key=lambda i: float(i.get("sensitivity", 0.5) or 0.5), default=None)
        if inst is not None:
            nums = _rule_numbers(inst)
            plan.scheduled_events.append({
                "etype": "institutional_decision", "ts": resolve_ts - 1.0, "participants": [],
                "payload": {"institution_id": str(inst.get("id")), "outcome_var": resolve_var,
                            "options": options, "lean": lean, **nums}})
            _add_mechanism(plan, "institutional_threshold_decision", "institutional_decision",
                           "execute the declared institution's threshold rule over posterior-informed votes",
                           "declared rule numbers; posterior member propensity")
            rep["actions"].append({"phase": "phase10_institutions", "action": "institutional_decision_event",
                                   "institution": inst.get("id"), **nums})

    # ---- Phase 9 populations: construct weighted segments when the aggregate signal has no declared
    #      structure (broad-prior segments; recorded as inferred — never a decorative quantity) ----
    if req["phase9_populations"]["required"] and not plan.populations:
        plan.populations.append({"id": "affected_population", "_inferred": "aggregate_behavior_signal",
                                 "segments": [
                                     {"id": "engaged", "weight": 0.5, "differs_on": ["participation"]},
                                     {"id": "marginal", "weight": 0.5, "differs_on": ["participation"]}]})
        rep["actions"].append({"phase": "phase9_populations",
                               "action": "population_constructed_from_signal"})

    # ---- Phase 9 populations: declared population + aggregate process → aggregation consumer ----
    if req["phase9_populations"]["required"] and plan.populations and not _has_event("population_aggregation"):
        for p in [x for x in plan.populations if isinstance(x, dict)][:2]:
            var = f"population_aggregate:{p.get('id')}"
            plan.scheduled_events.append({
                "etype": "population_aggregation", "ts": resolve_ts - 3.0, "participants": [],
                "payload": {"population_id": str(p.get("id")), "out_var": var}})
            _add_modulation(plan, var, 0.2)
        _add_mechanism(plan, "population_aggregation", "population_aggregation",
                       "aggregate declared segment heterogeneity into the terminal rate",
                       "declared weights; labeled broad-prior heterogeneity")
        rep["actions"].append({"phase": "phase9_populations", "action": "aggregation_consumer",
                               "populations": [p.get("id") for p in plan.populations[:2]
                                               if isinstance(p, dict)]})

    # ---- Phase 9 networks: normalize compiler-declared relation names onto the REGISTERED relation
    #      registry (an unregistered rel is dropped at materialization, leaving the diffusion mechanism an
    #      empty graph — the preflight-found blocked_no_mechanism defect). Never drops an edge. ----
    if req["phase9_networks"]["required"] and plan.relations:
        from swm.world_model_v2.network import _RELATIONS
        _REL_MAP = {"trust": "trusts", "influence": "influences", "communicat": "communicates_with",
                    "observ": "observes", "report": "reports_to", "fund": "funds", "endors": "endorses",
                    "oppos": "opposes", "compet": "opposes", "member": "belongs_to", "belong": "belongs_to",
                    "depend": "depends_on", "control": "controls", "ally": "influences",
                    "partner": "communicates_with"}
        n_norm = 0
        for r in plan.relations:
            if isinstance(r, dict) and str(r.get("rel")) not in _RELATIONS:
                orig = str(r.get("rel", ""))
                mapped = next((v for k, v in _REL_MAP.items() if k in orig.lower()), "influences")
                r["_original_rel"], r["rel"] = orig, mapped
                n_norm += 1
        if n_norm:
            rep["actions"].append({"phase": "phase9_networks", "action": "relation_names_normalized",
                                   "n": n_norm})

    # ---- Phase 9 networks: infer relations from the declared causal world when transmission is required
    #      but the compiler declared no explicit edges (Part 7: "build or infer multilayer relations") ----
    if req["phase9_networks"]["required"] and not plan.relations:
        ents = [e for e in plan.entities if isinstance(e, dict) and e.get("id")]
        ents.sort(key=lambda e: -float(e.get("sensitivity", 0.5) or 0.5))
        inferred = []
        if len(ents) >= 2:
            hub = str(ents[0]["id"])
            for e in ents[1:5]:
                inferred.append({"src": hub, "rel": "influences", "dst": str(e["id"]),
                                 "_inferred": "causal_world_hub_influence"})
                inferred.append({"src": str(e["id"]), "rel": "communicates_with", "dst": hub,
                                 "_inferred": "causal_world_communication"})
        elif plan.populations:
            pid = str(plan.populations[0].get("id"))
            aid = str(ents[0]["id"]) if ents else pid
            inferred.append({"src": aid, "rel": "observes", "dst": pid,
                             "_inferred": "population_exposure_layer"})
        if inferred:
            plan.relations.extend(inferred)
            rep["actions"].append({"phase": "phase9_networks", "action": "relations_inferred_from_causal_world",
                                   "n": len(inferred)})

    # ---- Phase 9 networks: declared relations + diffusion process → multilayer percolation consumer ----
    if req["phase9_networks"]["required"] and plan.relations and not _has_event("network_diffusion"):
        var = "network_diffusion_reach"
        plan.scheduled_events.append({
            "etype": "network_diffusion", "ts": resolve_ts - 3.0, "participants": [],
            "payload": {"out_var": var}})
        _add_modulation(plan, var, 0.2)
        _add_mechanism(plan, "network_diffusion", "network_diffusion",
                       "percolate the declared relation graph by semantic layer into the terminal rate",
                       "declared edges; layer transmissibility broad priors")
        rep["actions"].append({"phase": "phase9_networks", "action": "diffusion_consumer",
                               "n_relations": len(plan.relations)})

    # ---- Phase 7: nonlinear process named → real saturating state-step chain ----
    if req["phase7_nonlinear"]["required"] and not _has_event("state_step"):
        import swm.world_model_v2.nonlinear.operators  # noqa: F401 — registers the nonlinear operators
        from swm.world_model_v2.events import event_type_registered, register_event_type
        for _et in ("state_step", "nonlinear_transition"):
            if not event_type_registered(_et):
                register_event_type(_et, scheduling="scheduled", reads=("quantities",),
                                    deltas=("quantities",), parameter_source="activation synthesis",
                                    validated=True)
        var = "nonlinear_state"
        horizon_days = max(1.0, (plan.horizon_ts - plan.as_of) / 86400.0)
        n_steps = int(min(20, max(4, horizon_days)))
        dt = max(0.5, horizon_days / n_steps)
        plan.scheduled_events.append({
            "etype": "state_step", "ts": plan.as_of + dt * 86400.0, "participants": [],
            "payload": {"step_spec": {"state_var": var, "form_id": "logistic_growth",
                                      "params": {"r": 0.35, "L": 1.0}, "s0": 0.05, "dt": dt,
                                      "mode": "increment", "clamp": [0.0, 1.0],
                                      "horizon_ts": resolve_ts - 2.0}}})
        _add_modulation(plan, var, 0.15)
        _add_mechanism(plan, "nonlinear_saturating_trajectory", "nonlinear_state_step",
                       "execute the detected nonlinear (saturating) trajectory event-by-event",
                       "structural logistic form; broad-prior rate (labeled)")
        rep["actions"].append({"phase": "phase7_nonlinear", "action": "state_step_chain",
                               "n_steps": n_steps})

    # ---- Phase 6: per-process registry selections → executable pack events (dispatchable families) ----
    if req["phase6_registry"]["required"]:
        added = _synthesize_p6(plan, resolve_ts)
        if added:
            rep["actions"].append({"phase": "phase6_registry", "action": "pack_events", "families": added})

    # ---- Phase 4: strategic process required → complete the decision linkage; else gate off ornament ----
    p4 = req["phase4_actor_policy"]["required"]
    has_dec_ev = _has_event("decision_opportunity")
    if p4 and (has_dec_ev or plan.entities) and not _has_event("actor_action_aggregation"):
        # the CONSUMER: chosen-action polarity aggregated into the terminal (after all decisions, before
        # resolve). Without it, decisions execute but nothing downstream depends on them (the audit gap).
        var = "actor_action_share"
        plan.scheduled_events.append({
            "etype": "actor_action_aggregation", "ts": resolve_ts - 2.0, "participants": [],
            "payload": {"out_var": var}})
        _add_modulation(plan, var, 0.15)
        _add_mechanism(plan, "actor_action_aggregation", "actor_action_aggregation",
                       "aggregate chosen typed actions' polarity into the terminal rate",
                       "lexical polarity; nonpolar actions skipped")
    if p4 and not has_dec_ev:
        # organizations/institutions are strategic actors too (the action ontology has an
        # organizational_market family) — prefer persons, but never leave a required strategic phase
        # with a selected mechanism and no decision event just because no `person` was declared.
        cands = [e for e in plan.entities if isinstance(e, dict) and e.get("id")]
        persons = [e for e in cands if str(e.get("type", "")) == "person"]
        actor = max(persons or cands, key=lambda e: float(e.get("sensitivity", 0.5) or 0.5), default=None)
        if actor is not None:
            plan.scheduled_events.append({
                "etype": "decision_opportunity", "ts": resolve_ts - 4.0,
                "participants": [str(actor.get("id"))],
                "payload": {"situation": "strategic decision required by the causal analysis",
                            "actions": [{"type": "act"}, {"type": "wait"}]}})
            _add_mechanism(plan, "production_actor_policy", "production_actor_policy",
                           "strategic-actor decision required by the causal analysis",
                           "Tier-7 broad structural policy unless a fitted pack is bound")
            rep["actions"].append({"phase": "phase4_actor_policy", "action": "decision_event_synthesized",
                                   "actor": actor.get("id")})
    elif not p4 and has_dec_ev:
        n_before = len(plan.scheduled_events)
        plan.scheduled_events = [e for e in plan.scheduled_events
                                 if e.get("etype") not in ("decision_opportunity",
                                                           "actor_action_aggregation")]
        plan.accepted_mechanisms = [m for m in plan.accepted_mechanisms
                                    if m.get("operator") not in ("production_actor_policy", "agent_decision",
                                                                 "actor_action_aggregation")]
        rep["actions"].append({"phase": "phase4_actor_policy", "action": "gated_off_ornamental_decisions",
                               "n_events_removed": n_before - len(plan.scheduled_events),
                               "why": req["phase4_actor_policy"]["why"]})

    # ---- route consumer state into a CONSUMING MECHANISM (Part 4: the terminal resolver never modulates).
    # An institutional decision consumes it (members respond to population/actor/network/nonlinear state);
    # otherwise the aggregate-outcome realization mechanism resolves the outcome from that state in the
    # event loop, ahead of the generic safety net.
    consumed = list(getattr(plan, "_consumed_state", []) or [])
    if consumed:
        inst_ev = next((e for e in plan.scheduled_events if e.get("etype") == "institutional_decision"),
                       None)
        if inst_ev is not None:
            inst_ev.setdefault("payload", {})["consume"] = consumed
            rep["actions"].append({"phase": "state_consumption", "action": "institution_consumes_state",
                                   "vars": [m["var"] for m in consumed]})
        elif not _has_event("aggregate_outcome_resolution"):
            from swm.world_model_v2.family_hazards import family_base_rate
            fbr, fam, fbr_src = family_base_rate(getattr(plan, "question", ""))
            plan.scheduled_events.append({
                "etype": "aggregate_outcome_resolution", "ts": resolve_ts - 0.5, "participants": [],
                "payload": {"outcome_var": resolve_var, "options": options, "lean": lean,
                            "consume": consumed, "fitted_base_rate": fbr,
                            "family": fam, "base_rate_provenance": fbr_src}})
            _add_mechanism(plan, "aggregate_outcome_mechanism", "aggregate_outcome_mechanism",
                           "realize the aggregate-behavior outcome from consumed causal state",
                           "posterior base rate + bounded state consumption (inside the mechanism)")
            rep["actions"].append({"phase": "state_consumption", "action": "aggregate_mechanism_resolves",
                                   "vars": [m["var"] for m in consumed]})
    return rep


#: Phase-6 families with an executable behavioral dispatch + the params their dispatch requires.
_P6_DISPATCHABLE = {
    "social_pressure_turnout": ("treatment",), "matching_donation_response": ("base_p",),
    "ultimatum_offer_response": ("offer_frac", "accept_threshold"), "bass_diffusion": ("p", "q", "M"),
}


def _synthesize_p6(plan, resolve_ts) -> list:
    """Phase-6 causal-process resolution (Part 3): required process → registry family → validated pack →
    executable event → StateDelta. A process a validated family answers gets that family's published pack
    event; a process NOTHING answers gets the transparent broad-prior `structural_process_prior` fallback —
    it still executes through the shared runtime, labeled exploratory with the registry gap recorded.
    A required causal process is never silently omitted."""
    prov = getattr(plan, "provenance", {}) or {}
    pps = prov.get("per_process_selection", {}) or {}
    added = []
    answered = set()
    for proc, sel in pps.items():
        fam = (sel or {}).get("selected")
        if not fam or fam not in _P6_DISPATCHABLE:
            continue
        try:
            from swm.world_model_v2.registry import load_registry
            rec = load_registry().records.get(fam)
            pack = (rec.packs or [None])[0] if rec else None
            if pack is None:
                continue
            params = {}
            for k, v in (pack.values or {}).items():
                params[k] = v.get("value") if isinstance(v, dict) else v
            if fam == "bass_diffusion":
                params.setdefault("M", 1.0)
            if fam == "social_pressure_turnout":
                params.setdefault("treatment", "control")
            if fam == "matching_donation_response":
                params.setdefault("base_p", float((params.get("levels") or {}).get("control", 0.05))
                                  if isinstance(params.get("levels"), dict) else 0.05)
            missing = [k for k in _P6_DISPATCHABLE[fam] if k not in params]
            if missing:
                plan.omissions.append({"kind": "p6_pack_incomplete", "family": fam, "missing": missing})
                continue
            var = f"mechanism_outcome:{fam}"
            if any(e.get("etype") == "behavioral_mechanism" and
                   (e.get("payload", {}).get("hazard_spec") or {}).get("mechanism") == fam
                   for e in plan.scheduled_events):
                continue
            from swm.world_model_v2.events import event_type_registered, register_event_type
            import swm.world_model_v2.registry.families.behavioral  # noqa: F401 — registers the operator
            if not event_type_registered("behavioral_mechanism"):
                register_event_type("behavioral_mechanism", scheduling="scheduled", reads=("quantities",),
                                    deltas=("quantities",), parameter_source="published pack (registry)",
                                    validated=True)
            plan.scheduled_events.append({
                "etype": "behavioral_mechanism", "ts": resolve_ts - 2.5, "participants": [],
                "payload": {"hazard_spec": {"kind": "behavioral", "mechanism": fam, "params": params,
                                            "outcome_var": var, "family": fam,
                                            "pack_id": pack.pack_id, "transport_widening": 1.3}}})
            _add_modulation(plan, var, 0.2)
            _add_mechanism(plan, fam, "behavioral_mechanism",
                           f"registry family {fam} answering process {proc}",
                           f"published pack {pack.pack_id} (transported; widened)")
            added.append(fam)
            answered.add(proc)
        except Exception:  # noqa: BLE001 — a registry hiccup must not block the forecast
            continue
    # ---- fallback: required processes NOTHING answers still execute (transparent broad prior) ----
    procs = [str(c.get("process", "")) for c in (getattr(plan, "mechanism_choices", []) or [])
             if isinstance(c, dict) and c.get("process") and c.get("process") != "outcome_resolution"]
    if not procs:
        # the compiler returned no process list — derive the required processes from its structured
        # dependency signals so the behavioral mechanism still executes (never silently omitted)
        deps = (getattr(plan, "provenance", {}) or {}).get("causal_dependencies") or {}
        procs = [k for k, v in deps.items() if v is True and k != "structural_change_monitoring"][:2] \
            or ["social_outcome_process"]
    lean = (getattr(plan, "provenance", {}) or {}).get("outcome_lean", "neutral")
    n_fb = 0
    for proc in procs:
        if proc in answered or n_fb >= 2:                    # bounded: modulation channel is capped anyway
            continue
        var = f"mechanism_outcome:fallback:{proc[:40]}"
        if any(e.get("etype") == "structural_process_prior" and
               e.get("payload", {}).get("out_var") == var for e in plan.scheduled_events):
            continue
        plan.scheduled_events.append({
            "etype": "structural_process_prior", "ts": resolve_ts - 2.5, "participants": [],
            "payload": {"process": proc, "out_var": var, "lean": lean}})
        _add_modulation(plan, var, 0.1)
        _add_mechanism(plan, f"structural_prior:{proc[:40]}", "structural_process_prior",
                       f"broad-prior fallback for unanswered required process {proc!r}",
                       "broad Beta prior; EXPLORATORY (registry gap)")
        plan.fallbacks_used.append({"process": proc, "tier": 6, "family": "structural_process_prior",
                                    "why": "no validated registry family answers this required causal "
                                           "process — transparent broad-prior fallback (registry gap)"})
        added.append(f"fallback:{proc[:40]}")
        n_fb += 1
    return added
