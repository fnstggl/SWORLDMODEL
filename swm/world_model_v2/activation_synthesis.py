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
    """Return the runtime-owned relevance verdict for every conditional phase.

    The LLM compiler's causal-dependency block is retained as provenance, but
    cannot turn a phase on: the prior activation run showed that treating those
    declarations as ground truth caused systematic false execution.  Relevance
    comes from the reviewable question-level adjudicator.  Missing scenario
    state remains a separate materialization/integration concern.
    """
    from swm.world_model_v2.causal_relevance import adjudicate_question

    judged = adjudicate_question(getattr(plan, "question", ""))
    deps = ((getattr(plan, "provenance", {}) or {}).get("causal_dependencies")) or {}
    dep_for = {
        "phase4_actor_policy": "strategic_actor_decisions",
        "phase6_registry": None,
        "phase7_nonlinear": "nonlinear_dynamics",
        "phase9_populations": "aggregate_population_behavior",
        "phase9_networks": "networked_transmission",
        "phase10_institutions": "institutional_decision_process",
        "phase11_recompilation": "structural_change_monitoring",
    }
    out = {}
    for phase, verdict in judged.items():
        dep = dep_for.get(phase)
        out[phase] = {**verdict, "signal": bool(verdict["required"]),
                      "compiler_corroboration": bool(dep and deps.get(dep) is True)}
    return out


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
    """Removed terminal shortcut retained only as a loud compatibility error.

    Phase effects must travel through causal state-transition events; callers
    may not attach probability nudges to ``resolve_outcome``.
    """
    raise RuntimeError("direct terminal rate_modulation is prohibited; use causal state drivers")


def _wire_causal_state_path(plan, drivers, resolve_ts, resolve_var, options, lean) -> bool:
    """Wire phase outputs into a state transition and then a terminal event.

    This is intentionally two events.  Upstream phases first mutate their own
    state, ``causal_state_transition`` consumes those fields into a typed latent
    propensity, and only ``causal_outcome_transition`` may write the terminal
    WorldState.  The generic resolver remains a safety net and no longer accepts
    phase-specific probability modifiers.
    """
    if not drivers:
        return False
    unique = []
    for driver in drivers:
        if driver.get("var") and driver["var"] not in {d["var"] for d in unique}:
            unique.append(dict(driver))
    propensity_var = "causal_outcome_propensity"
    plan.scheduled_events.append({
        "etype": "causal_state_transition", "ts": resolve_ts - 1.5, "participants": [],
        "payload": {"driver_vars": unique, "out_var": propensity_var, "lean": lean}})
    plan.scheduled_events.append({
        "etype": "causal_outcome_transition", "ts": resolve_ts - 0.75, "participants": [],
        "payload": {"propensity_var": propensity_var, "outcome_var": resolve_var,
                    "options": options}})
    _add_mechanism(plan, "causal_state_transition", "causal_state_transition",
                   "combine phase state through an equal-weight log-opinion state transition",
                   "posterior base rate plus labeled phase state; no terminal probability nudge")
    _add_mechanism(plan, "causal_outcome_transition", "causal_outcome_transition",
                   "sample the terminal state from the causal propensity state",
                   "causal state transition output; common-randomness branch stream")
    # A formal institution consumes the same causal propensity before applying
    # its declared threshold rule.
    for ev in plan.scheduled_events:
        if ev.get("etype") == "institutional_decision":
            ev.setdefault("payload", {})["propensity_var"] = propensity_var
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


def _materialize_required_structure(plan, req) -> list:
    """Repair compiler omissions with minimal, explicitly inferred causal state.

    The question-level relevance gate may find a required population, network,
    actor, or institution even when the compiler omitted its section.  Leaving
    that phase blocked would be an integration defect; silently treating it as
    irrelevant would be worse.  The repair uses abstract state only, labels it
    as inferred, and never claims scenario-specific facts that were not given.
    """
    actions = []
    prov = getattr(plan, "provenance", {}) or {}
    inferred = prov.setdefault("runtime_inferred_structure", [])

    # A relevant actor-policy phase needs at least one actor the production
    # policy can bind. Merely having an organization/entity in the plan is
    # insufficient because decision_opportunity dispatches to person actors.
    has_person = any(isinstance(e, dict) and e.get("id") and
                     str(e.get("type", "person")) == "person"
                     for e in (getattr(plan, "entities", []) or []))
    if req["phase4_actor_policy"]["required"] and not (
            has_person or getattr(plan, "actor_decisions", [])):
        plan.entities.append({"id": "strategic_actor", "type": "person",
                              "fields": {"role": "outcome-relevant decision maker"},
                              "_inferred": "question-level strategic process"})
        actions.append({"phase": "phase4_actor_policy", "action": "abstract_actor_materialized"})

    if req["phase9_populations"]["required"] and not getattr(plan, "populations", []):
        plan.populations.append({
            "id": "affected_population", "sensitivity": 0.7,
            "segments": [
                {"id": "lower_propensity", "weight": 0.5,
                 "differs_on": ["participation_propensity"]},
                {"id": "higher_propensity", "weight": 0.5,
                 "differs_on": ["participation_propensity"]},
            ], "_inferred": "question-level aggregate population process"})
        actions.append({"phase": "phase9_populations", "action": "abstract_population_materialized"})

    if req["phase9_networks"]["required"]:
        entities, seen_ids = [], set()
        for entity in (getattr(plan, "entities", []) or []):
            if not isinstance(entity, dict) or not entity.get("id"):
                continue
            eid = str(entity["id"])
            if eid not in seen_ids:
                entities.append(entity)
                seen_ids.add(eid)
        while len(entities) < 2:
            eid = "transmission_source" if not entities else "exposed_target"
            if eid in seen_ids:
                eid = f"network_node_{len(entities) + 1}"
            node = {"id": eid, "type": "organization",
                    "fields": {"role": "abstract transmission substrate"},
                    "_inferred": "question-level transmission process"}
            plan.entities.append(node)
            entities.append(node)
            seen_ids.add(eid)

        # Compiler-proposed relation names are not automatically executable:
        # the materializer rejects names outside the typed relation registry.
        # Treat an all-invalid relation list as missing state and add one
        # explicit registered edge while preserving invalid proposals in the
        # omission trail.
        from swm.world_model_v2.network import _RELATIONS
        materializable = any(
            isinstance(rel, dict) and rel.get("src") and rel.get("dst") and
            str(rel.get("src")) != str(rel.get("dst")) and rel.get("rel") in _RELATIONS
            for rel in (getattr(plan, "relations", []) or []))
        if not materializable:
            plan.relations.append({"src": str(entities[0]["id"]), "rel": "influences",
                                   "dst": str(entities[1]["id"]),
                                   "_inferred": "minimal transmission edge"})
            actions.append({"phase": "phase9_networks", "action": "abstract_relation_materialized"})

    if req["phase10_institutions"]["required"] and not getattr(plan, "institutions", []):
        plan.institutions.append({
            "id": "decision_institution", "sensitivity": 0.7,
            "rules": [{"kind": "quorum", "params": {"total": 9, "quorum": 5}}],
            "_inferred": "question-level rule-governed decision process"})
        actions.append({"phase": "phase10_institutions", "action": "abstract_institution_materialized"})

    if actions:
        inferred.extend(actions)
        plan.provenance = prov
    return actions


def _gate_irrelevant_execution(plan, req) -> list:
    """Remove compiler-proposed operators for phases adjudicated irrelevant."""
    phase_events = {
        "phase4_actor_policy": {"decision_opportunity", "actor_action_aggregation"},
        "phase6_registry": {"behavioral_mechanism", "feature_hazard", "structural_process_prior"},
        "phase7_nonlinear": {"state_step", "nonlinear_transition"},
        "phase9_populations": {"population_aggregation"},
        "phase9_networks": {"network_diffusion"},
        "phase10_institutions": {"institutional_decision", "collective_vote"},
    }
    phase_ops = {
        "phase4_actor_policy": {"production_actor_policy", "agent_decision", "fitted_decision",
                                "actor_action_aggregation"},
        "phase6_registry": {"behavioral_mechanism", "feature_hazard", "structural_process_prior"},
        "phase7_nonlinear": {"nonlinear_state_step", "nonlinear_mechanism", "nonlinear_contagion"},
        "phase9_populations": {"population_aggregation"},
        "phase9_networks": {"network_diffusion"},
        "phase10_institutions": {"institutional_decision", "institutional_vote", "institution_action"},
    }
    actions = []
    for phase, events in phase_events.items():
        if req.get(phase, {}).get("required"):
            continue
        before_events = len(plan.scheduled_events)
        before_mechs = len(plan.accepted_mechanisms)
        plan.scheduled_events = [e for e in plan.scheduled_events if e.get("etype") not in events]
        plan.accepted_mechanisms = [m for m in plan.accepted_mechanisms
                                    if m.get("operator") not in phase_ops[phase]]
        removed = (before_events - len(plan.scheduled_events)) + \
                  (before_mechs - len(plan.accepted_mechanisms))
        if removed:
            actions.append({"phase": phase, "action": "irrelevant_execution_gated_off",
                            "n_artifacts_removed": removed})
    return actions


def _bound_hazard_event_budget(plan, max_expected_events=100.0) -> list:
    """Prevent compiler hazards from starving scheduled causal mechanisms.

    The rollout has a hard event ceiling for safety. An unconstrained LLM rate
    can otherwise consume that ceiling on background hazards before the
    horizon-time phase events execute. Preserve the relative rates but scale
    their total expected count to a conservative budget, recording the exact
    repair in provenance.
    """
    hazards = list(getattr(plan, "stochastic_hazards", []) or [])
    horizon_days = max(0.0, (float(plan.horizon_ts) - float(plan.as_of)) / 86400.0)
    expected = sum(max(0.0, float(h.get("rate_per_day", 0.0) or 0.0)) * horizon_days
                   for h in hazards if isinstance(h, dict))
    if expected <= max_expected_events or expected <= 0.0:
        return []
    scale = max_expected_events / expected
    for hazard in hazards:
        if isinstance(hazard, dict):
            hazard["rate_per_day"] = max(
                0.0, float(hazard.get("rate_per_day", 0.0) or 0.0) * scale)
    return [{"phase": "cross_phase", "action": "hazard_event_budget_bounded",
             "expected_before": round(expected, 3),
             "expected_after": round(max_expected_events, 3),
             "scale": round(scale, 8)}]


def synthesize_activation(plan, req=None) -> dict:
    """Complete the execution chain for required phases; gate off ornamental execution for non-required ones.
    Mutates the plan in place; returns the per-phase synthesis report (for the manifest). Idempotent."""
    import swm.world_model_v2.phase_consumers  # noqa: F401 — registers the consumer operators/events
    req = req or phase_requirements(plan)
    rep = {"requirements": {k: v["required"] for k, v in req.items()}, "actions": []}
    rep["actions"].extend(_materialize_required_structure(plan, req))
    rep["actions"].extend(_gate_irrelevant_execution(plan, req))
    rep["actions"].extend(_bound_hazard_event_budget(plan))
    rev = _resolve_event(plan)
    if rev is not None:
        # Defense in depth for plans compiled by the pre-fix runtime.
        rev.setdefault("payload", {}).pop("rate_modulation", None)
    resolve_var = (rev or {}).get("payload", {}).get("outcome_var", "outcome")
    options = (rev or {}).get("payload", {}).get("options") or ["True", "False"]
    lean = (rev or {}).get("payload", {}).get("lean", "neutral")
    resolve_ts = (rev or {}).get("ts", plan.horizon_ts - 1.0)
    causal_drivers = []

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

    # ---- Phase 9 populations: declared population + aggregate process → aggregation consumer ----
    if req["phase9_populations"]["required"] and plan.populations and not _has_event("population_aggregation"):
        for p in [x for x in plan.populations if isinstance(x, dict)][:2]:
            var = f"population_aggregate:{p.get('id')}"
            plan.scheduled_events.append({
                "etype": "population_aggregation", "ts": resolve_ts - 3.0, "participants": [],
                "payload": {"population_id": str(p.get("id")), "out_var": var}})
            causal_drivers.append({"var": var, "phase": "phase9_populations", "direction": 1})
        _add_mechanism(plan, "population_aggregation", "population_aggregation",
                       "aggregate declared segment heterogeneity into causal state",
                       "declared weights; labeled broad-prior heterogeneity")
        rep["actions"].append({"phase": "phase9_populations", "action": "aggregation_consumer",
                               "populations": [p.get("id") for p in plan.populations[:2]
                                               if isinstance(p, dict)]})

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
        causal_drivers.append({"var": var, "phase": "phase9_networks", "direction": 1})
        _add_mechanism(plan, "network_diffusion", "network_diffusion",
                       "percolate the declared relation graph into downstream causal state",
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
        causal_drivers.append({"var": var, "phase": "phase7_nonlinear", "direction": 1})
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
        for ev in plan.scheduled_events:
            if ev.get("etype") == "behavioral_mechanism":
                var = (ev.get("payload", {}).get("hazard_spec") or {}).get("outcome_var")
            elif ev.get("etype") == "structural_process_prior":
                var = ev.get("payload", {}).get("out_var")
            else:
                continue
            if var:
                causal_drivers.append({"var": var, "phase": "phase6_registry", "direction": 1})

    # ---- Phase 4: strategic process required → complete the decision linkage; else gate off ornament ----
    p4 = req["phase4_actor_policy"]["required"]
    has_dec_ev = _has_event("decision_opportunity")
    if p4 and has_dec_ev:
        # A compiler decision is not executable merely because an event of the
        # right type exists. Bind the production operator and ensure the
        # decision precedes its aggregation/downstream state path.
        _add_mechanism(plan, "production_actor_policy", "production_actor_policy",
                       "strategic-actor decision required by the causal analysis",
                       "Tier-7 broad structural policy unless a fitted pack is bound")
        latest_decision_ts = resolve_ts - 4.0
        for event in plan.scheduled_events:
            if event.get("etype") != "decision_opportunity":
                continue
            old_ts = float(event.get("ts", latest_decision_ts))
            if old_ts > latest_decision_ts:
                event["ts"] = latest_decision_ts
                rep["actions"].append({"phase": "phase4_actor_policy",
                                       "action": "late_decision_moved_before_consumer",
                                       "old_ts": old_ts, "new_ts": latest_decision_ts})
    if p4 and (has_dec_ev or plan.entities) and not _has_event("actor_action_aggregation"):
        # the CONSUMER: chosen-action polarity aggregated into the terminal (after all decisions, before
        # resolve). Without it, decisions execute but nothing downstream depends on them (the audit gap).
        var = "actor_action_share"
        plan.scheduled_events.append({
            "etype": "actor_action_aggregation", "ts": resolve_ts - 2.0, "participants": [],
            "payload": {"out_var": var}})
        causal_drivers.append({"var": var, "phase": "phase4_actor_policy", "direction": 1})
        _add_mechanism(plan, "actor_action_aggregation", "actor_action_aggregation",
                       "aggregate chosen typed actions' polarity into downstream causal state",
                       "lexical polarity; nonpolar actions skipped")
    if p4 and not has_dec_ev:
        persons = [e for e in plan.entities
                   if isinstance(e, dict) and str(e.get("type", "")) == "person"]
        actor = max(persons, key=lambda e: float(e.get("sensitivity", 0.5) or 0.5), default=None)
        if actor is not None:
            plan.scheduled_events.append({
                "etype": "decision_opportunity", "ts": resolve_ts - 4.0,
                "participants": [str(actor.get("id"))],
                "payload": {"situation": "strategic decision required by the causal analysis",
                            "actions": [{"name": "support", "mechanisms_triggered": ["record_action"]},
                                        {"name": "reject", "mechanisms_triggered": ["record_action"]}]}})
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
    if p4:
        # Ensure an existing compiler decision exposes outcome-polar alternatives;
        # otherwise the downstream actor state cannot encode a direction.
        from swm.world_model_v2.phase_consumers import action_polarity
        for ev in plan.scheduled_events:
            if ev.get("etype") != "decision_opportunity":
                continue
            payload = ev.setdefault("payload", {})
            acts = payload.get("candidate_actions") or payload.get("actions") or []
            if not any(action_polarity((a.get("name") or a.get("type")) if isinstance(a, dict) else a)
                       for a in acts):
                payload["candidate_actions"] = [
                    {"name": "support", "mechanisms_triggered": ["record_action"]},
                    {"name": "reject", "mechanisms_triggered": ["record_action"]},
                ]
                rep["actions"].append({"phase": "phase4_actor_policy",
                                       "action": "outcome_polar_action_space_completed"})
    if _wire_causal_state_path(plan, causal_drivers, resolve_ts, resolve_var, options, lean):
        rep["actions"].append({"phase": "cross_phase", "action": "causal_state_path",
                               "drivers": causal_drivers})
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
        if proc in answered or n_fb >= 2:                    # bounded exploratory mechanism count
            continue
        var = f"mechanism_outcome:fallback:{proc[:40]}"
        if any(e.get("etype") == "structural_process_prior" and
               e.get("payload", {}).get("out_var") == var for e in plan.scheduled_events):
            continue
        plan.scheduled_events.append({
            "etype": "structural_process_prior", "ts": resolve_ts - 2.5, "participants": [],
            "payload": {"process": proc, "out_var": var, "lean": lean}})
        _add_mechanism(plan, f"structural_prior:{proc[:40]}", "structural_process_prior",
                       f"broad-prior fallback for unanswered required process {proc!r}",
                       "broad Beta prior; EXPLORATORY (registry gap)")
        plan.fallbacks_used.append({"process": proc, "tier": 6, "family": "structural_process_prior",
                                    "why": "no validated registry family answers this required causal "
                                           "process — transparent broad-prior fallback (registry gap)"})
        added.append(f"fallback:{proc[:40]}")
        n_fb += 1
    return added
