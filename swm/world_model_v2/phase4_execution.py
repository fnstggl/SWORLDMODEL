"""Phase 4 actor-policy execution through the shared event world.

Selected actions become events and explicit ``StateDelta`` records.  Actual
feasibility is rechecked after actor-perceived feasibility; mistaken attempts are
represented as blocked-action deltas rather than silently disappearing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import threading
import time
from dataclasses import asdict

from swm.world_model_v2 import semantic_consequences as semcons
from swm.world_model_v2.events import Event, register_event_type
from swm.world_model_v2.phase4_policy import (
    ActionPosterior, ActionSpaceBuilder, ActorPolicyModel, ActorViewBuilder, DecisionTrace,
    FeasibilityEngine, TypedAction, build_trace,
)
from swm.world_model_v2.state import F, StateField
from swm.world_model_v2.transitions import StateDelta, ValidationResult, register_operator


for _name, _visibility in (
    ("actor_action", "participants"), ("action_blocked", "participants"),
    ("actor_reaction", "participants"), ("delayed_action_effect", "participants"),
    ("institution_submission", "institutional"), ("message_delivered", "participants"),
):
    register_event_type(_name, scheduling="endogenous", visibility=_visibility, validated=True,
                        parameter_source="phase4 typed action execution")


def _id(prefix: str, payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


class ActorPolicyRuntime:
    """Orchestrates one universal actor-policy decision across posterior worlds."""

    def __init__(self, model: ActorPolicyModel | None = None, *, view_builder=None,
                 action_builder=None, feasibility=None, consequence_mode: str | None = None,
                 consequence_llm=None):
        self.model = model or ActorPolicyModel()
        self.views = view_builder or ActorViewBuilder()
        self.actions = action_builder or ActionSpaceBuilder()
        self.feasibility = feasibility or FeasibilityEngine()
        self._lock = threading.RLock()
        # consequence routing: semantic structured world transitions are THE default; the
        # scalar `ACTION_PATHWAY_EFFECTS × pathway_step` writer runs only under the explicit
        # legacy benchmark mode. Every execution counts into consequence_report (fail-loud).
        requested = consequence_mode or semcons.resolve_consequence_mode()
        if requested not in semcons.CONSEQUENCE_MODES:
            raise ValueError(f"unknown consequence mode {requested!r} "
                             f"(known: {semcons.CONSEQUENCE_MODES})")
        self.consequence_mode = requested
        self.consequence_llm = consequence_llm
        self.consequence_compiler = None              # built lazily (needs the bound LLM)
        self.consequence_report = {"requested_mode": requested, "actual_mode": requested,
                                   **semcons.empty_report()}

    def decide(self, plan, posterior_worlds: list, actor_id: str, *, decision: dict,
               seed: int, question_id: str = "", observed_events=None,
               particle_weights: list[float] | None = None
               ) -> tuple[TypedAction, ActionPosterior, DecisionTrace]:
        """Compute and sample a policy without exposing a ``WorldState`` to numeric policy code."""
        started = time.monotonic()
        if not posterior_worlds:
            raise ValueError("posterior_worlds cannot be empty")
        views = [self.views.build(world, actor_id, observed_events=observed_events)
                 for world in posterior_worlds]
        # Build once from the first particle; feasibility and consequences are particle-specific.
        decision = {**decision, "plan": plan}
        actions = self.actions.build(plan, posterior_worlds[0], views[0], decision=decision)
        decisions = [[self.feasibility.classify(action, view, world) for action in actions]
                     for view, world in zip(views, posterior_worlds)]
        model_kwargs = {"seed": seed}
        if particle_weights is not None:
            model_kwargs["particle_weights"] = particle_weights
        posterior = self.model.decide(views, actions, decisions, **model_kwargs)
        selected_id = posterior.sample(random.Random(seed))
        selected = next(action for action in actions if action.action_id == selected_id)
        trace = build_trace(
            question_id=question_id or _id("question", getattr(plan, "question", "")),
            plan=plan, worlds=posterior_worlds, views=views, actions=actions,
            feasibility=decisions, posterior=posterior, selected_action_id=selected_id,
            seed=seed, started_at=started,
        )
        return selected, posterior, trace

    def execute(self, world, action: TypedAction, posterior: ActionPosterior, trace: DecisionTrace,
                *, seed: int = 0) -> tuple[StateDelta, list[Event]]:
        """Recheck actual feasibility, mutate shared state, and return follow-up events."""
        with self._lock:
            view = self.views.build(world, action.actor_id)
            fd = self.feasibility.classify(action, view, world)
            if not fd.actually_feasible:
                delta = StateDelta(
                    at=world.clock.now, event_type="action_blocked", operator="production_actor_policy",
                    reason_codes=["attempted_but_blocked", fd.actual_status] + fd.actual_reasons[:4],
                    uncertainty={"trace_id": trace.trace_id, "action": action.as_dict(),
                                 "p_dist": posterior.action_probabilities},
                )
                self._append_history(world, action, "blocked", delta)
                event = Event(ts=world.clock.now, etype="action_blocked",
                              participants=[action.actor_id],
                              payload={"action_id": action.action_id, "status": fd.actual_status,
                                       "reasons": fd.actual_reasons, "trace_id": trace.trace_id},
                              source="endogenous:production_actor_policy")
                trace.resulting_event_ids.append(_id("event", event.__dict__))
                trace.resulting_state_delta_ids.append(_id("delta", delta.as_dict()))
                trace.warnings.append("actor attempted an action it believed feasible; world blocked it")
                trace.seal()
                return delta, [event]

            delta = StateDelta(
                at=world.clock.now, event_type="actor_action", operator="production_actor_policy",
                reason_codes=["calibrated_actor_policy", f"support={posterior.support_grade}"],
                uncertainty={"trace_id": trace.trace_id, "p_dist": posterior.action_probabilities,
                             "policy_families": posterior.policy_family_posterior.weights,
                             "credible_intervals": posterior.credible_intervals},
                evidence_deps=list(trace.observed_evidence_ids),
            )
            actor = world.entity(action.actor_id)
            before = actor.value("current_action")
            actor.set("current_action", F(action.as_dict(), status="derived",
                                           method="production_actor_policy", updated_at=world.clock.now))
            delta.change(f"{action.actor_id}.current_action", before, action.as_dict())
            self._append_history(world, action, "executed", delta)
            self._consume_resources(world, action, delta)
            self._create_commitments(world, action, delta)
            semantic_events = self._apply_consequences(world, action, posterior, trace, delta)
            self._post_execute(world, action, posterior, trace, delta)
            events = self._follow_up_events(
                world, action, posterior, trace, seed,
                suppress={e.etype for e in semantic_events})
            events = list(semantic_events) + events
            delta.follow_up_events = [self._event_record(event) for event in events]
            event = Event(ts=world.clock.now, etype="actor_action",
                          participants=[x for x in (action.actor_id, action.target.target_id) if x],
                          payload={"action": action.as_dict(), "trace_id": trace.trace_id},
                          visibility=action.observability.get("default", "participants"),
                          source="endogenous:production_actor_policy")
            events.insert(0, event)
            trace.resulting_event_ids.extend(_id("event", e.__dict__) for e in events)
            trace.resulting_state_delta_ids.append(_id("delta", delta.as_dict()))
            trace.downstream_reactions.extend([self._event_record(e) for e in events[1:]])
            trace.seal()
            return delta, events

    def _post_execute(self, world, action: TypedAction, posterior: ActionPosterior,
                      trace: DecisionTrace, delta: StateDelta):
        """Subclass hook, called after the typed consequence writers and BEFORE the delta/trace
        seal, so actor-local post-action state (e.g. the LLM persona layer's cognition
        write-back) is recorded on the SAME StateDelta as the action that produced it."""
        return None

    # ---- consequence routing (semantic default / legacy benchmark / dual audit) --------
    def _apply_consequences(self, world, action: TypedAction, posterior: ActionPosterior,
                            trace: DecisionTrace, delta: StateDelta) -> list[Event]:
        """The action→world seam. Default (`semantic_world_consequences`): compile the chosen
        action into a validated CausalActionProgram and apply typed world changes — objects,
        facts, real-content communications, submissions entering real procedures, staged
        processes — then recompute `pathway_progress:*` as read-only projections of that typed
        state. `legacy_scalar_pathway_consequences` (benchmark-only) runs the historical
        `possible_consequences` + `ACTION_PATHWAY_EFFECTS × pathway_step` scalar writers.
        `dual_run_consequence_audit` applies semantic and RECORDS (never applies) what the
        legacy scalars would have written."""
        mode, report = self.consequence_mode, self.consequence_report
        if mode == "legacy_scalar_pathway_consequences":
            n0 = len(delta.changes)
            self._apply_immediate_consequences(world, action, delta, consequence_mode=mode)
            self._apply_pathway_effects(world, action, delta, consequence_mode=mode)
            report["legacy_scalar_writes"] += len(delta.changes) - n0
            delta.reason_codes.append("legacy_scalar_pathway_consequences")
            return []
        program = self._consequence_program(world, action, posterior, trace)
        events = semcons.execute_program(world, program, delta, report)
        semcons.project_decided_outcome_quantities(world, action, delta, report)
        semcons.derive_pathway_summaries(world, delta)
        delta.uncertainty["consequence_program"] = program.as_dict()
        if program.unmodeled:
            trace.warnings.append(
                f"semantic_consequence_unmodeled: {action.action_name} produced no validated "
                f"world transition (quarantined: {len(program.quarantined)})")
            report["fallbacks"] += 1
            report["fallback_reasons"].append(
                {"action": action.action_name, "reason": "unmodeled",
                 "quarantined": [q.get("reason", "") for q in program.quarantined[:4]]})
        elif program.partially_modeled:
            trace.warnings.append(
                f"semantic_consequence_partial: {len(program.quarantined)} op(s) of "
                f"{action.action_name} quarantined")
        if mode == "dual_run_consequence_audit":
            shadow_world = copy.deepcopy(world)
            shadow = StateDelta(at=world.clock.now, event_type="dual_run_legacy_shadow",
                                operator="legacy_scalar_pathway_consequences")
            self._apply_immediate_consequences(
                shadow_world, action, shadow, consequence_mode="legacy_scalar_pathway_consequences")
            self._apply_pathway_effects(
                shadow_world, action, shadow, consequence_mode="legacy_scalar_pathway_consequences")
            report.setdefault("dual_run_legacy_shadow", []).append(
                {"action_id": action.action_id, "unapplied_changes": shadow.changes})
            delta.uncertainty["legacy_shadow_unapplied"] = shadow.changes
        return events

    def _consequence_program(self, world, action: TypedAction, posterior: ActionPosterior,
                             trace: DecisionTrace):
        if self.consequence_compiler is None:
            self.consequence_compiler = semcons.SemanticConsequenceCompiler(self.consequence_llm)
        qualitative = self._qualitative_for_trace(trace, posterior)
        return self.consequence_compiler.compile(world, action, qualitative=qualitative)

    def _qualitative_for_trace(self, trace: DecisionTrace, posterior: ActionPosterior):
        """The qualitative decision content backing this action, when one exists — the
        consequence compiler's LLM path runs only for qualitatively-decided actions; numeric/
        Tier-3 actions compile deterministically. Subclasses with richer per-trace state
        (the qualitative runtime) override this."""
        q = (posterior.provenance or {}).get("qualitative")
        return q if isinstance(q, dict) and q.get("routed") \
            and q.get("decision_source") not in ("numeric_policy", "numeric_fallback") else None

    @staticmethod
    def _append_history(world, action: TypedAction, status: str, delta: StateDelta):
        actor = world.entity(action.actor_id)
        current = actor.get("past_actions")
        before = list(current.value) if isinstance(current, StateField) and isinstance(current.value, list) else []
        after = before + [{"at": world.clock.now, "action": action.action_name,
                           "action_id": action.action_id, "status": status,
                           "target": action.target.target_id, "public": True}]
        actor.set("past_actions", F(after, status="derived", method="production_actor_policy",
                                    updated_at=world.clock.now))
        delta.change(f"{action.actor_id}.past_actions", before, after)

    @staticmethod
    def _consume_resources(world, action: TypedAction, delta: StateDelta):
        actor = world.entity(action.actor_id)
        resources = actor.get("resources")
        for name, cost in action.resource_costs.items():
            sf = resources.get(name) if isinstance(resources, dict) else None
            before = float(sf.value) if isinstance(sf, StateField) and isinstance(sf.value, (int, float)) else 0.0
            after = before - abs(float(cost))
            actor.set("resources", F(after, status="derived", method="production_actor_policy",
                                     updated_at=world.clock.now), key=name)
            delta.change(f"{action.actor_id}.resources[{name}]", before, after)

    @staticmethod
    def _create_commitments(world, action: TypedAction, delta: StateDelta):
        if not action.commitments_created:
            return
        actor = world.entity(action.actor_id)
        sf = actor.get("commitments")
        before = list(sf.value) if isinstance(sf, StateField) and isinstance(sf.value, list) else []
        created = [{**c, "created_by_action_id": action.action_id, "created_at": world.clock.now}
                   for c in action.commitments_created]
        after = before + created
        actor.set("commitments", F(after, status="derived", method="production_actor_policy",
                                   updated_at=world.clock.now))
        delta.change(f"{action.actor_id}.commitments", before, after)

    @staticmethod
    def _apply_immediate_consequences(world, action: TypedAction, delta: StateDelta, *,
                                      consequence_mode: str = "legacy_scalar_pathway_consequences"):
        """Apply only typed, bounded consequences.  No action can assign a terminal probability.
        LEGACY-ONLY: compiler-proposed `quantity_delta`/`belief_delta` scalars are exactly the
        unmechanized numeric writes the semantic mode replaces — invoking this writer under
        `semantic_world_consequences` is a structural violation, not a fallback."""
        if consequence_mode == "semantic_world_consequences":
            raise RuntimeError(
                "scalar consequence writer invoked in semantic_world_consequences mode — "
                "actions change the world through semantic_consequences.execute_program")
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        for consequence in action.possible_consequences:
            if not isinstance(consequence, dict):
                continue
            kind = consequence.get("kind")
            if kind == "quantity_delta":
                name = str(consequence.get("name", ""))
                if not name or "probab" in name.lower():
                    continue
                before = float(world.quantities[name].value) if name in world.quantities else 0.0
                after = before + float(consequence.get("delta", 0.0) or 0.0)
                register_quantity_type(name, units=str(consequence.get("units", "unit")))
                world.quantities[name] = Quantity(name=name, qtype=name, value=after,
                                                  timestamp=world.clock.now)
                delta.change(f"quantities[{name}]", before, after)
            elif kind == "belief_delta" and action.target.target_id in world.entities:
                target = world.entity(action.target.target_id)
                key = str(consequence.get("belief", "action_effect"))
                beliefs = target.get("beliefs") or {}
                sf = beliefs.get(key) if isinstance(beliefs, dict) else None
                before = float(sf.value) if isinstance(sf, StateField) and isinstance(sf.value, (int, float)) else 0.5
                shift = max(-0.25, min(0.25, float(consequence.get("delta", 0.0) or 0.0)))
                after = min(1.0, max(0.0, before + shift))
                target.set("beliefs", F(after, status="derived", method="production_actor_policy",
                                        updated_at=world.clock.now), key=key)
                delta.change(f"{target.identity}.beliefs[{key}]", before, after)

    #: legacy point step, kept for direct callers/tests; production draws the per-branch SAMPLED
    #: coupling `pathway_step` (world_dynamics) so the structural magnitude is a distribution.
    PATHWAY_STEP = 0.04

    @classmethod
    def _apply_pathway_effects(cls, world, action: TypedAction, delta: StateDelta, *,
                               consequence_mode: str = "legacy_scalar_pathway_consequences"):
        """LEGACY-ONLY (`legacy_scalar_pathway_consequences` benchmark mode) — the historical
        action→world half of the endogenous causal chain: an EXECUTED action moves the
        process quantities its ontology entry names — and the hazard rounds CONSUME those
        quantities, so the timing/probability of resolution emerges from what the simulated actors
        actually do. Three refinements over a flat write:
          * the step size is the per-branch SAMPLED coupling constant (structural uncertainty
            propagates), scaled by the actor's live CAPACITY (an exhausted actor moves less);
          * on a SHARED process, PRINCIPALS (the declared approvers) move it at full step,
            bystanders at the sampled non-principal share — Trump's shuttle diplomacy moves talks
            less than a principal's own acceptance;
          * on a CONTESTED (non-shared) pathway the write goes to the ACTOR'S OWN pursued mode
            channels (+) while rival mode channels on the same pathway are suppressed (sampled
            contested_suppression) — two campaigns evolve separately, in tension; the pathway
            aggregate still moves as the spillover signal.
        Applies ONLY to quantities the plan declared — worlds without a mode graph are untouched."""
        if consequence_mode == "semantic_world_consequences":
            raise RuntimeError(
                "scalar pathway writer invoked in semantic_world_consequences mode — "
                "pathway bars are read-only projections (derive_pathway_summaries) there")
        from swm.world_model_v2.phase4_policy import action_pathway_effects
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        effects = action_pathway_effects(action.action_family, action.action_name)
        if not effects:
            # a compiled NOVEL action executes through its validated ontology anchor's effects
            # (NovelActionCompiler attaches parameters.ontology_anchor only when the anchor's
            # causal reading matched) — an unanchored novel action moves nothing, and its branch
            # carries the explicit novel_action_unmodeled mark instead of a silent no-op.
            anchor = (action.parameters or {}).get("ontology_anchor") or {}
            if isinstance(anchor, dict) and anchor.get("name"):
                effects = action_pathway_effects(str(anchor.get("family", "")),
                                                 str(anchor["name"]))
        if not effects:
            return
        from swm.world_model_v2.world_dynamics import live_capacity, sampled_coupling

        def _write(var, qtype, step_eff):
            q = world.quantities.get(var)
            if q is None:
                return                                       # not declared for this world — no-op
            before = float(q.value) if isinstance(q.value, (int, float)) else 0.5
            after = max(0.05, min(0.95, before + step_eff))
            if after == before:
                return
            register_quantity_type(qtype, units="process_state")
            world.quantities[var] = Quantity(name=var, qtype=qtype, value=round(after, 4),
                                             timestamp=world.clock.now)
            delta.change(f"quantities[{var}]", round(before, 4), round(after, 4))

        step = sampled_coupling(world, "pathway_step")
        cap = live_capacity(world).get(action.actor_id)
        if isinstance(cap, (int, float)):
            step *= 0.4 + 0.6 * max(0.0, min(1.0, float(cap)))
        actor_ent = world.entities.get(action.actor_id)
        my_stances = actor_ent.value("stances", default=None) if actor_ent is not None else None
        my_stances = my_stances if isinstance(my_stances, list) else []
        for pw, eff in effects.items():
            share = 1.0
            pq = world.quantities.get(f"pathway_principals:{pw}")
            principals = str(getattr(pq, "value", "") or "").split("|") if pq is not None else []
            if principals and action.actor_id not in principals:
                share = sampled_coupling(world, "nonprincipal_step_share")
            _write(f"pathway_progress:{pw}", "pathway_progress", step * share * float(eff))
            # contested pathways: route the push into the actor's OWN pursued mode channel(s) and
            # suppress rivals' channels on the same pathway
            own_modes = {str(s.get("target_mode")) for s in my_stances
                         if s.get("target_mode") and str(s.get("pathway")) == pw
                         and str(s.get("commitment_level", "")) in
                         ("inclined_toward", "actively_pursuing", "formally_committed")}
            if not own_modes:
                continue
            supp = sampled_coupling(world, "contested_suppression")
            prefix = f"mode_progress:{pw}:"
            for var in list(world.quantities):
                if not var.startswith(prefix):
                    continue
                mode_id = var[len(prefix):]
                if mode_id in own_modes:
                    _write(var, "mode_progress", step * float(eff))
                else:
                    _write(var, "mode_progress", -step * supp * float(eff))

    @staticmethod
    def _follow_up_events(world, action: TypedAction, posterior: ActionPosterior,
                          trace: DecisionTrace, seed: int, suppress=frozenset()) -> list[Event]:
        """`suppress` carries the etypes the semantic consequence program already produced with
        REAL content — the legacy mechanism emissions (empty-content message pings, inert
        institution_submission markers) are skipped for those so a single action never delivers
        the same communication twice."""
        events = []
        participants = [x for x in (action.actor_id, action.target.target_id) if x]
        for mechanism in action.mechanisms_triggered:
            if mechanism == "message_delivery" and "message_delivered" in suppress:
                continue
            if mechanism == "institution_processing" and "collective_vote" in suppress:
                continue
            if mechanism == "message_delivery" and action.target.target_id:
                events.append(Event(ts=world.clock.now + max(0.0, action.expected_duration_s),
                                    etype="message_delivered", participants=participants,
                                    payload={"action_id": action.action_id,
                                             "content": action.parameters.get("content", ""),
                                             "trace_id": trace.trace_id},
                                    visibility=action.observability.get("default", "participants"),
                                    source="endogenous:production_actor_policy"))
            elif mechanism == "institution_processing":
                events.append(Event(ts=world.clock.now, etype="institution_submission",
                                    participants=participants,
                                    payload={"action_id": action.action_id,
                                             "institution_id": action.target.target_id,
                                             "action": action.as_dict(), "trace_id": trace.trace_id},
                                    visibility="institutional",
                                    source="endogenous:production_actor_policy"))
            elif (mechanism == "reaction_scheduling" and action.target.target_id
                  and action.target.target_id in world.entities):
                delay = max(1.0, float(action.parameters.get("reaction_delay_s", 60.0) or 60.0))
                events.append(Event(ts=world.clock.now + delay, etype="actor_reaction",
                                    participants=[action.target.target_id, action.actor_id],
                                    payload={"trigger_action_id": action.action_id,
                                             "candidate_actions": action.parameters.get(
                                                 "reaction_actions", ["acknowledge", "ignore"]),
                                             "trace_id": trace.trace_id},
                                    visibility="participants",
                                    source="endogenous:production_actor_policy"))
        for delayed in action.possible_delayed_consequences:
            if not isinstance(delayed, dict):
                continue
            events.append(Event(ts=world.clock.now + max(1.0, float(delayed.get("delay_s", 1.0) or 1.0)),
                                etype="delayed_action_effect", participants=participants,
                                payload={"action_id": action.action_id, "effect": delayed,
                                         "trace_id": trace.trace_id},
                                visibility=action.observability.get("default", "participants"),
                                source="endogenous:production_actor_policy"))
        return events

    @staticmethod
    def _event_record(event: Event) -> dict:
        return {"etype": event.etype, "ts": event.ts, "participants": list(event.participants),
                "payload": copy.deepcopy(event.payload), "visibility": event.visibility}


class ProductionActorPolicyOperator:
    """Registry-compatible production decision operator.

    The generic constructor uses the deliberately broad Tier-7 parameter pack.  A
    deployment may bind a fitted ``ActorPolicyModel`` without changing the shared
    execution path.  Unlike the legacy ``FittedDecisionOperator``, numeric policy
    code receives ActorViews, never the omniscient WorldState.
    """

    name = "production_actor_policy"

    def __init__(self, model: ActorPolicyModel | None = None, *, runtime: ActorPolicyRuntime | None = None):
        # A deployment may bind a fitted model, or a whole runtime (e.g. the Phase-4L persona
        # runtime, which keeps this operator's execution semantics and adds actor cognition).
        self.runtime = runtime or ActorPolicyRuntime(model)
        self.traces = []

    def applicable(self, world, event):
        return event.etype in ("decision_opportunity", "actor_reaction") and bool(event.participants)

    def run(self, world, event, rng):
        actor_id = event.participants[0]
        seed = rng.randrange(0, 2**31 - 1)
        decision = dict(event.payload or {})
        if "candidate_actions" not in decision and "actions" in decision:
            decision["candidate_actions"] = decision["actions"]
        selected, posterior, trace = self.runtime.decide(
            None, [world], actor_id, decision=decision, seed=seed,
            question_id=str(decision.get("question_id", "")),
            observed_events=[event],
        )
        delta, _events = self.runtime.execute(world, selected, posterior, trace, seed=seed)
        self.traces.append(trace)
        return delta, ValidationResult(ok=True)


register_operator(
    "production_actor_policy", ProductionActorPolicyOperator,
    requires=("posterior_world_state", "actor_view", "typed_action_space"),
    modifies=("entity.current_action", "entity.past_actions", "entity.resources",
              "entity.commitments", "event_queue"), temporal_scale="event",
    parameter_source="hierarchical fitted pack or explicit Tier-7 broad structural mixture",
    invariants=("no omniscient actor view", "zero known-impossible action mass",
                "selected action emits StateDelta"), validated=True,
)


def decide_and_execute_particles(runtime: ActorPolicyRuntime, plan, worlds: list, actor_id: str, *,
                                 decision: dict, seed: int, question_id: str = ""):
    """Posterior-integrated decision followed by matched execution in every world particle.

    A single calibrated marginal posterior is computed across all particles.  The
    selected typed action is then attempted in each particle, where actual
    feasibility and consequences may differ.  This is the explicit Phase-3 to
    Phase-4 bridge used by validation and forensic traces.
    """
    selected, posterior, trace = runtime.decide(
        plan, worlds, actor_id, decision=decision, seed=seed, question_id=question_id)
    executions = []
    for index, world in enumerate(worlds):
        delta, events = runtime.execute(world, selected, posterior, trace, seed=seed * 7919 + index)
        executions.append({"world_id": world.world_id, "branch_id": world.branch_id,
                           "delta": delta, "events": events})
    return {"selected_action": selected, "posterior": posterior, "trace": trace,
            "executions": executions}


def apply_adaptation(world, *, actor_id: str, action_name: str, reward: float,
                     outcome: str, source_event_id: str, learning_rate: float = 0.2) -> StateDelta:
    """Phase 4 persistent-state contract: update actor-local policy state in WorldState.

    This is intentionally not a Phase 8 persistence service.  It stores a typed,
    provenance-bearing update in the current world so later decisions change.
    """
    actor = world.entity(actor_id)
    latent = actor.get("latent_state") or {}
    key = f"phase4_policy_value:{action_name}"
    sf = latent.get(key) if isinstance(latent, dict) else None
    before = float(sf.value) if isinstance(sf, StateField) and isinstance(sf.value, (int, float)) else 0.0
    lr = min(1.0, max(0.0, float(learning_rate)))
    after = before + lr * (float(reward) - before)
    actor.set("latent_state", F(after, status="derived", sources=[source_event_id],
                                method="phase4_temporal_difference_update", updated_at=world.clock.now), key=key)
    hist_key = "phase4_policy_updates"
    hs = latent.get(hist_key) if isinstance(latent, dict) else None
    hist = list(hs.value) if isinstance(hs, StateField) and isinstance(hs.value, list) else []
    hist.append({"at": world.clock.now, "action": action_name, "reward": reward, "outcome": outcome,
                 "source_event_id": source_event_id, "before": before, "after": after})
    actor.set("latent_state", F(hist, status="derived", sources=[source_event_id],
                                method="phase4_policy_update_log", updated_at=world.clock.now), key=hist_key)
    delta = StateDelta(at=world.clock.now, event_type="actor_reaction", operator="phase4_adaptation",
                       reason_codes=["reinforcement_update", outcome], evidence_deps=[source_event_id],
                       uncertainty={"learning_rate": lr})
    delta.change(f"{actor_id}.latent_state[{key}]", before, after)
    delta.change(f"{actor_id}.latent_state[{hist_key}]", hist[:-1], hist)
    return delta
