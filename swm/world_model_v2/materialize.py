"""Materializer — turns a validated WorldExecutionPlan into a runnable WorldModelV2Run.

Plan JSON → typed objects: entities (universal schema, causally-relevant fields only), populations
(segments + heterogeneity), relation graph, executable rule systems, typed quantities, information ledger,
latent records (InitialStateModel), scheduled events + hazards (queue builder), and the operator set from the
accepted mechanisms. This is configuration, not a new engine: every scenario class flows through here.

Production contract (Tier A1 of the gap audit):
  * PROVENANCE HONESTY — compiler-proposed values are `inferred` (LLM proposal), never `observed`;
    `observed` requires an evidence reference, which only the evidence layer may attach.
  * LOUD FAILURE — nothing is silently dropped: unknown fields/relations/rule-kinds/quantities and
    mechanisms that resolve no operator are recorded in `world.omissions` / returned in the run report,
    and high-sensitivity drops raise MaterializeAbstention.
  * READOUT BINDING — the outcome contract's readout must resolve against the materialized base world
    (or name a declared quantity); a dangling readout aborts before any rollout.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.events import Event, EventQueue, StochasticHazard
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.init_state import InitialStateModel
from swm.world_model_v2.institutions import EXECUTABLE_RULE_KINDS, Rule, RuleSystem
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.population import Population, PopulationSegment
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.rollout import WorldModelV2Run
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import get_operator


class MaterializeAbstention(Exception):
    """DEPRECATED (pre-no-abstention). A plan that cannot materialize is now a TECHNICAL failure
    (CompilerExecutionError → execution_failed), NOT an epistemic forecast refusal. Retained so old
    imports resolve; new code raises CompilerExecutionError with a failure taxonomy."""


def build_world(plan, *, world_id: str = "w0", evidence_hash: str = "", versions: dict = None) -> WorldState:
    clock = SimulationClock(now=plan.as_of, as_of=plan.as_of)
    w = WorldState(world_id=world_id, branch_id="root", clock=clock,
                   network=RelationGraph(), information=InformationLedger(),
                   evidence_hash=evidence_hash, versions=versions or {})
    omissions = []
    prompt_hash = (plan.provenance or {}).get("prompt_hash", "")
    for e in plan.entities:
        ent = Entity(identity=str(e.get("id")), entity_type=str(e.get("type", "person")))
        for fname, val in (e.get("fields") or {}).items():
            if val in ("?", None, ""):
                continue                                     # unknowns stay latent, not fabricated
            # PROVENANCE HONESTY: an LLM proposal is an inference, not an observation. The evidence
            # layer upgrades fields to `observed` only with an evidence reference attached.
            sf = F(val, status="inferred", method=f"compiler:proposal:{prompt_hash}",
                   confidence=0.45, updated_at=plan.as_of)
            from swm.world_model_v2.state import ENTITY_FIELDS, extension_fields
            if fname in (set(ENTITY_FIELDS) | extension_fields(ent.entity_type)):
                # Canonical keyed state must remain addressable by key.  Wrapping a
                # resource/belief dict in one StateField made feasibility and updates
                # unable to access individual entries.
                if fname in ("resources", "beliefs") and isinstance(val, dict):
                    for key, item in val.items():
                        ent.set(fname, F(item, status="inferred", method=f"compiler:proposal:{prompt_hash}",
                                         confidence=0.45, updated_at=plan.as_of), key=str(key))
                else:
                    ent.set(fname, sf)
            else:
                # scenario-specific proposed field → typed latent_state namespace (kept, not dropped),
                # and recorded as an omission-from-canonical-schema for the audit trail
                ent.set("latent_state", sf, key=fname)
                omissions.append({"kind": "entity_field_routed_to_latent_state", "entity": ent.identity,
                                  "field": fname, "reason": "not a canonical schema field — stored as "
                                  "a typed latent_state scalar with provenance"})
        w.entities[ent.identity] = ent
    for p in plan.populations:
        segs = []
        for s in (p.get("segments") or []):
            segs.append(PopulationSegment(
                segment_id=str(s.get("id")), weight=F(float(s.get("weight", 0.0) or 0.0),
                                                      status="inferred",
                                                      method=f"compiler:proposal:{prompt_hash}"),
                heterogeneity={str(d): F(None, dist={"mean": 0.5, "sd": 0.2, "lo": 0.0, "hi": 1.0},
                                         status="assumed",
                                         method="unparameterized-dimension broad prior (labeled)")
                               for d in (s.get("differs_on") or [])}))
        w.populations[str(p.get("id"))] = Population(population_id=str(p.get("id")), segments=segs)
    for r in plan.relations:
        try:
            w.network.add(str(r.get("src")), str(r.get("rel")), str(r.get("dst")))
        except KeyError:
            omissions.append({"kind": "relation", "src": r.get("src"), "rel": r.get("rel"),
                              "dst": r.get("dst"), "reason": "unregistered relation type"})
    for inst in plan.institutions:
        rules, inst_id = [], str(inst.get("id"))
        for i, ru in enumerate(inst.get("rules") or []):
            kind = str(ru.get("kind", "procedure"))
            if kind not in EXECUTABLE_RULE_KINDS:
                # CLOSED RULE-KIND REGISTRY: an inexecutable rule must not silently validate everything.
                omissions.append({"kind": "institutional_rule", "institution": inst_id, "rule_kind": kind,
                                  "reason": f"rule kind {kind!r} is not executable "
                                            f"(executable: {sorted(EXECUTABLE_RULE_KINDS)})"})
                continue
            rules.append(Rule(rule_id=f"{inst_id}:{i}", kind=kind, params=dict(ru.get("params") or {})))
        w.institutions[inst_id] = RuleSystem(institution_id=inst_id, rules=rules)
    for q in plan.quantities:
        name, qtype = str(q.get("name")), str(q.get("qtype", q.get("name")))
        register_quantity_type(qtype, units=str(q.get("units", "unit")))
        try:
            w.quantities[name] = Quantity(name=name, qtype=qtype,
                                          value=q.get("value"), sd=q.get("sd"), timestamp=plan.as_of)
        except KeyError:
            omissions.append({"kind": "quantity", "name": name, "reason": "quantity construction failed"})
    w.omissions = omissions
    return w


def check_readout_binding(plan, world) -> None:
    """READOUT BINDING: the contract's readout must reference something that exists in the materialized
    world (an entity field path or a registered quantity). A dangling readout would let empty no-op worlds
    produce confident {'None': 1.0} answers — abort instead."""
    var = getattr(plan.outcome_contract, "readout_var", "") or ""
    if not var:
        return                                                # hand-built contracts bind a closure directly
    if var in world.quantities:
        return
    eid, _, fpath = var.partition(".")
    if eid in world.entities and fpath:
        return                                                # field may be set during rollout; entity must exist
    # Post-repair this should be unreachable (the compiler synthesizes a canonical outcome quantity). If it
    # still fails, it is a compiler DEFECT / execution failure — NOT an epistemic abstention.
    from swm.world_model_v2.result import CompilerExecutionError
    raise CompilerExecutionError(
        f"terminal readout {var!r} does not bind after compile-time repair (entities: "
        f"{sorted(world.entities)[:8]}, quantities: {sorted(world.quantities)[:8]})",
        taxonomy="terminal_readout_unbindable")


def queue_builder_from_plan(plan):
    """Fresh queue per branch: scheduled events + hazards + first-passage crossings,
    horizon-capped. The scenario temporal model rides on the world (shared read-only object);
    per-branch temporal state (sampled latents, attention buffers, cumulative-hazard states)
    lives on the world itself."""
    def build(world) -> EventQueue:
        q = EventQueue(horizon_ts=plan.horizon_ts)
        rng = random.Random(int(world.branch_id.strip("b").split(":")[0] or 0)
                            if world.branch_id.startswith("b") else 0)
        tmodel = getattr(plan, "temporal_model", None)
        if tmodel is not None and getattr(world, "temporal_model", None) is None:
            world.temporal_model = tmodel
        for ev in plan.scheduled_events:
            q.schedule(Event(ts=ev["ts"], etype=ev["etype"], participants=list(ev["participants"]),
                             payload=dict(ev["payload"]), source="scheduled",
                             trigger=dict(ev.get("trigger") or {})))
        for hz in plan.stochastic_hazards:
            q.add_hazard(StochasticHazard(etype=hz["etype"], rate_per_day=hz["rate_per_day"],
                                          participants=list(hz["participants"])),
                         now=world.clock.now, rng=rng, world=world)
        # continuous-time first-passage processes (§15): per-branch persistent threshold +
        # cumulative intensity; the initial crossing is a REAL event at its projected time
        for spec in (getattr(plan, "first_passage_processes", None) or []):
            try:
                from swm.world_model_v2.event_time import ensure_first_passage_state
                from swm.world_model_v2.temporal_hazards import schedule_crossing
                st = ensure_first_passage_state(world, spec)
                schedule_crossing(q, world, st, etype="first_passage")
            except Exception as e:  # noqa: BLE001 — a broken process spec is recorded, not fatal
                world.omissions.append({"kind": "first_passage_process",
                                        "process": str(spec.get("process_id", "?")),
                                        "reason": f"{type(e).__name__}: {e}"[:120]})
        return q
    return build


def operators_from_plan(plan, *, llm=None, allow_experimental=False) -> tuple:
    """Instantiate operators for accepted mechanisms. Mechanisms that name NO operator are returned as
    rejections (they must have been rejected at compile; this is defense in depth) — never silently
    skipped. Returns (operators, rejections)."""
    ops, seen, rejections = [], set(), []
    for m in plan.accepted_mechanisms:
        opname = m.get("operator")
        if not opname:
            rejections.append({"mech_id": m.get("mech_id"),
                               "reason": "accepted mechanism names no executable operator"})
            continue
        if opname in seen:
            continue
        seen.add(opname)
        try:
            factory = get_operator(opname, allow_experimental=allow_experimental)
        except (KeyError, PermissionError) as e:
            rejections.append({"mech_id": m.get("mech_id"), "reason": str(e)[:200]})
            continue
        try:
            # A3: the LLM reaches agent_decision only when experimental execution is explicitly enabled;
            # even then probability-minting requires the operator's own opt-in flag.
            if opname == "agent_decision":
                ops.append(factory(llm=(llm if allow_experimental else None),
                                   allow_llm_probabilities=allow_experimental))
            elif opname == "production_actor_policy":
                # The actor-policy MODE router (docs/ARCHITECTURE_QUALITATIVE_ACTORS.md §2).
                # DEFAULT-ON: every run requests hybrid qualitative cognition for consequential
                # humans; numeric_policy serves only as an explicit benchmark/ablation, the
                # Tier-3 routine policy, or a LOUDLY-REPORTED degradation (no backend /
                # construction failure) — never a silent swap. The report rides on the operator
                # and is attached to every run result by attach_actor_decision_distributions.
                bound_runtime, policy_report = _actor_policy_runtime(plan, llm)
                op = factory(runtime=bound_runtime) if bound_runtime is not None else factory()
                op.actor_policy_report = policy_report
                if policy_report.get("degraded"):
                    rejections.append({"mech_id": "production_actor_policy",
                                       "reason": "actor_policy_degraded: "
                                                 + policy_report["construction_error"]})
                ops.append(op)
            elif hasattr(factory, "run") and hasattr(factory, "applicable") and not isinstance(factory, type):
                # Phase 7/10 registry entries are configured operator instances.  Treat
                # them as prototypes instead of assuming every entry is a zero-arg class.
                ops.append(copy.deepcopy(factory))
            else:
                ops.append(factory())
        except TypeError as e:
            rejections.append({"mech_id": m.get("mech_id"),
                               "reason": f"operator {opname!r} needs bound parameters "
                                         f"(e.g. a fitted policy pack): {e}"[:200]})
    # Consequence INFRASTRUCTURE (not mechanism hypotheses): the consumers of the consequence
    # architecture's control/follow-up events must exist in every run of that mode, or the
    # downstream half of the causal chain silently dies in the queue.
    from swm.world_model_v2 import semantic_consequences as _semcons
    mode = _semcons.resolve_consequence_mode()
    if mode != "legacy_scalar_pathway_consequences":
        if "communication_delivery" not in seen:
            seen.add("communication_delivery")
            ops.append(_semcons.CommunicationDeliveryOperator())
        if "generated_attention" not in seen:
            # availability → attention is mode-agnostic (§9): both consequence modes separate
            # delivered from noticed; the attention operator collects the noticed bundle
            from swm.world_model_v2 import generated_world as _genw_att
            seen.add("generated_attention")
            ops.append(_genw_att.GeneratedAttentionOperator(report=_semcons.empty_report()))
        if "institutional_vote" not in seen:
            from swm.world_model_v2.transitions import InstitutionalVoteOperator
            seen.add("institutional_vote")
            ops.append(InstitutionalVoteOperator())
    if mode == "generated_actor_mediated_world":
        # the generated actor-mediated control plane: semantic-event routing, observation
        # delivery, and persistent-actor invocation share the bound runtime + its report
        from swm.world_model_v2 import generated_world as _genw
        runtime = next((getattr(op, "runtime", None) for op in ops
                        if getattr(op, "name", "") == "production_actor_policy"
                        and getattr(op, "runtime", None) is not None), None)
        if runtime is not None:
            report = runtime.consequence_report
            ops.append(_genw.GeneratedSemanticEventOperator(
                report=report, frontier_llm=getattr(runtime, "consequence_llm", None) or llm))
            ops.append(_genw.GeneratedObservationDeliveryOperator(report=report))
            for op in ops:                                    # rebind the shared attention op
                if getattr(op, "name", "") == "generated_attention":
                    op.report = report
            ops.append(_genw.GeneratedActorInvocationOperator(runtime, report=report))
    return ops, rejections


ACTOR_POLICY_MODES = ("numeric_policy", "persona_blended_numeric_policy", "stateless_llm_policy",
                      "persistent_qualitative_llm_policy", "hybrid_relevant_actor_policy")


def resolve_actor_policy_mode(llm=None) -> str:
    """The REQUESTED actor-policy mode. DEFAULT-ON: hybrid qualitative cognition for
    consequential humans is the intended default for every run — numeric_policy exists only as
    an explicit benchmark/ablation request (SWM_ACTOR_POLICY=numeric_policy or the legacy
    SWM_LLM_ACTORS=off). Whether the request can actually be SERVED (a backend exists, the
    runtime constructs) is a separate question answered loudly by the actor_policy_report —
    resolution never silently depends on the backend."""
    import os
    if os.environ.get("SWM_LLM_ACTORS", "").strip().lower() == "off":
        return "numeric_policy"
    mode = os.environ.get("SWM_ACTOR_POLICY", "").strip().lower()
    if mode in ACTOR_POLICY_MODES:
        return mode
    return "hybrid_relevant_actor_policy"


def _actor_policy_runtime(plan, llm):
    """Bind the runtime for the requested mode. Returns (runtime_or_None, report). The report
    ALWAYS states requested vs actual mode and why they differ — a Tier-1 actor silently
    becoming the old numeric actor is prohibited. A construction error is recorded verbatim
    (and re-surfaced as a run degradation), never swallowed into a quiet numeric run."""
    requested = resolve_actor_policy_mode(llm)
    report = {"requested_actor_policy_mode": requested,
              "actual_actor_policy_mode": requested, "construction_error": "", "degraded": False}
    if requested == "numeric_policy":
        report["actual_actor_policy_mode"] = "numeric_policy"
        report["reason"] = "numeric_policy explicitly requested (benchmark/ablation)"
        return None, report
    if llm is None:
        # honest capability statement, loudly reported — there is no backend to run cognition
        report.update(actual_actor_policy_mode="numeric_policy", reason="no_llm_backend",
                      warning="requested LLM actor policy cannot run without a backend; "
                              "numeric policy served — supply llm= to simulate_world/run_from_plan")
        return None, report
    try:
        if requested == "persona_blended_numeric_policy":
            from swm.world_model_v2.llm_actor import build_persona_runtime
            runtime = build_persona_runtime(llm=llm)
        else:
            from swm.world_model_v2.qualitative_actor import build_qualitative_runtime
            runtime = build_qualitative_runtime(plan, llm=llm, mode=requested,
                                                fallback_llms=_fallback_backends(llm))
        if runtime is None:
            raise RuntimeError("runtime constructor returned None despite a backend")
        return runtime, report
    except Exception as e:  # noqa: BLE001 — LOUD degradation, never a silent numeric swap
        report.update(actual_actor_policy_mode="numeric_policy", degraded=True,
                      construction_error=f"{type(e).__name__}: {e}"[:300],
                      reason="qualitative_runtime_construction_failed",
                      warning="requested actor policy FAILED to construct; numeric policy "
                              "served as a degraded run — this is a defect, not a mode choice")
        return None, report


def _fallback_backends(primary):
    """Secondary model families for Tier-1 decisions when the primary call fails: any
    configured alternate providers (HF router today). Recorded per-decision when they serve."""
    out = []
    try:
        import os
        if os.environ.get("HF_TOKEN"):
            from swm.api.hf_backend import hf_chat_fn
            out.append(hf_chat_fn(max_tokens=2000, temperature=0.8))
    except Exception:  # noqa: BLE001 — an unavailable fallback family is simply absent
        pass
    return out


def attach_actor_decision_distributions(ops, container: dict) -> None:
    """After a rollout: attach the counted action distributions AND the mandatory
    actor-policy routing report (requested vs actual mode, who was routed where, every
    fallback and its reason). The report is attached for EVERY run — including numeric ones —
    so a bypassed qualitative layer is always visible on the result."""
    try:
        from swm.world_model_v2.qualitative_actor import aggregate_actor_decisions
        records, mode, runtime_report, consequence_report = [], "", None, None
        for op in ops:
            runtime = getattr(op, "runtime", None)
            if runtime is not None and hasattr(runtime, "decision_records"):
                records.extend(runtime.decision_records)
                mode = getattr(runtime, "mode", mode)
            if runtime is not None and getattr(runtime, "consequence_report", None):
                consequence_report = runtime.consequence_report
            if getattr(op, "actor_policy_report", None):
                runtime_report = op.actor_policy_report
        # the consequence report rides on EVERY result that executed actor actions — the
        # requested/actual consequence mode and every fallback are visible, never inferred
        if consequence_report is None:
            from swm.world_model_v2 import semantic_consequences as _semcons
            from swm.world_model_v2.generated_world import generated_report as _genrep
            _mode = _semcons.resolve_consequence_mode()
            consequence_report = {"requested_mode": _mode, "actual_mode": _mode,
                                  **_semcons.empty_report(), **_genrep(),
                                  "note": "no production_actor_policy runtime executed"}
        container["consequence_report"] = consequence_report
        report = dict(runtime_report or {"requested_actor_policy_mode": "numeric_policy",
                                         "actual_actor_policy_mode": "numeric_policy",
                                         "reason": "no production_actor_policy operator bound"})
        qual_actors, num_actors, fallback_reasons = set(), set(), {}
        for posterior, trace in records:
            q = (posterior.provenance or {}).get("qualitative") or {}
            source = q.get("decision_source", "numeric_policy")
            if source in ("persistent_qualitative_llm", "stateless_llm"):
                qual_actors.add(trace.actor_id)
            else:
                num_actors.add(trace.actor_id)
                key = f"{trace.actor_id}:{q.get('reason', source)}"
                fallback_reasons[key] = fallback_reasons.get(key, 0) + 1
        report.update(
            actors_routed_qualitatively=sorted(qual_actors),
            actors_routed_numerically=sorted(num_actors - qual_actors),
            fallbacks=int(sum(fallback_reasons.values())),
            fallback_reasons=[{"actor_and_reason": k, "n": n}
                              for k, n in sorted(fallback_reasons.items())])
        container["actor_policy_report"] = report
        if records:
            container["actor_decision_distributions"] = aggregate_actor_decisions(records)
            container["actor_policy_mode"] = mode
    except Exception as e:  # noqa: BLE001 — reporting must not kill the run, but must not vanish
        container.setdefault("actor_policy_report", {})["report_error"] = \
            f"{type(e).__name__}: {e}"[:200]


def _inject_posterior_rate(plan) -> bool:
    """Copy the plan's evidence-updated outcome-rate posterior particles onto every resolve_outcome scheduled
    event payload, so the terminal resolver (GenericOutcomeOperator) draws each particle's Bernoulli rate from
    the POSTERIOR. Returns True iff a posterior was present and injected. Idempotent (writes the same value).

    This is the WORLD-STATE→EXECUTION bridge: a posterior stored on the plan but never placed where a mechanism
    reads it is ornamental. Here the numbers cross into the execution plane where they are causally consumed."""
    post = getattr(plan, "posterior_rate_particles", None)
    if not post:
        return False
    parts = [[float(r), float(w)] for r, w in post]
    injected = False
    for ev in plan.scheduled_events:
        # institutional_decision (Phase 10) consumes the SAME evidence-updated base rate as the terminal
        # resolver — the institution's rule TRANSFORMS the posterior, it never invents its own rate.
        if ev.get("etype") in ("resolve_outcome", "institutional_decision",
                               "aggregate_outcome_resolution"):
            ev.setdefault("payload", {})["posterior_rate_particles"] = parts
            injected = True
        # event-time residual chains consume the same posterior as their per-particle TARGET MASS — the
        # calibrated information parameterizes the causal process; the answer stays a first-passage readout.
        elif ev.get("etype") == "hazard_round" and \
                isinstance((ev.get("payload") or {}).get("calibration"), dict):
            ev["payload"]["calibration"]["posterior_rate_particles"] = parts
            injected = True
    # continuous-time first-passage processes (§15) draw the SAME evidence-updated target mass
    for spec in (getattr(plan, "first_passage_processes", None) or []):
        if isinstance(spec, dict) and isinstance(spec.get("calibration"), dict):
            spec["calibration"]["posterior_rate_particles"] = parts
            injected = True
    return injected


def _bind_scenario_schema(plan, base, llm) -> None:
    """generated_actor_mediated_world: bind THIS plan's generated ScenarioSemanticModel onto
    the base world (particles deep-copy it → branch-local, extensible). A hand-built plan may
    supply `plan.scenario_schema` directly; otherwise the schema compiler generates one from
    the plan. No backend / compilation failure = LOUD stamped degradation (recorded on the
    plan and re-stamped per action by the runtime) — never a silent fixed-v1 swap."""
    from swm.world_model_v2 import semantic_consequences as _semcons
    if _semcons.resolve_consequence_mode() != "generated_actor_mediated_world":
        return
    from swm.world_model_v2.scenario_schema import (
        ScenarioSemanticModel, SchemaCompiler, validate_initial_records,
        validate_scenario_schema,
    )
    schema = getattr(plan, "scenario_schema", None)
    if schema is None and isinstance((plan.provenance or {}).get("scenario_schema"), dict):
        schema = ScenarioSemanticModel.from_dict(plan.provenance["scenario_schema"])
    if schema is None:
        if llm is None:
            plan.provenance = {**(plan.provenance or {}),
                               "scenario_schema_error": "no_llm_backend_for_schema_compilation"}
            return
        try:
            schema = SchemaCompiler(llm).compile(
                question=str(getattr(plan, "question", ""))[:400],
                as_of=float(getattr(plan, "as_of", 0.0) or 0.0),
                horizon=float(getattr(plan, "horizon_ts", 0.0) or 0.0),
                entities=[str(e.get("id")) for e in (plan.entities or [])],
                institutions=list(getattr(plan, "institutions", None) and
                                  [str(i.get("id")) for i in plan.institutions] or []),
                evidence=str((plan.provenance or {}).get("evidence_summary", ""))[:1500])
        except Exception as e:  # noqa: BLE001 — LOUD degradation, never silent
            plan.provenance = {**(plan.provenance or {}),
                               "scenario_schema_error": f"{type(e).__name__}: {e}"[:300]}
            return
    if isinstance(schema, ScenarioSemanticModel):
        if not schema.frozen:
            ok, issues = validate_scenario_schema(schema)
            if not ok:
                plan.provenance = {**(plan.provenance or {}),
                                   "scenario_schema_error": f"validation: {issues[:4]}"}
                return
            schema.freeze()
        ok0, smuggled = validate_initial_records(schema, list(base.objects.values()))
        if not ok0:
            plan.provenance = {**(plan.provenance or {}),
                               "scenario_schema_error":
                                   f"outcome smuggled into initial records: {smuggled}"}
            return
        base.scenario_schema = schema
        plan.scenario_schema = schema
        if schema.outcome_predicates and getattr(plan.outcome_contract, "readout", None):
            from swm.world_model_v2.generated_world import make_generated_predicate_readout
            plan.outcome_contract.readout = make_generated_predicate_readout(schema)
            # the contract's options must be the FROZEN predicates' own labels, or the
            # projection counts frequencies of strings the readout can never produce
            opts = []
            for p in schema.outcome_predicates:
                t = str(p.get("option_true", "True"))
                if t not in opts:
                    opts.append(t)
            f = str(schema.outcome_predicates[0].get("option_false", "False"))
            if f not in opts:
                opts.append(f)
            if len(opts) >= 2:
                plan.outcome_contract.options = opts


def run_from_plan(plan, *, llm=None, n_particles=None, seed=0):
    """The end-to-end: plan → world → InitialStateModel → rollout → native terminal distribution.
    The compiler's fallback hierarchy guarantees ≥1 executable mechanism + a binding readout, so a
    failure here is TECHNICAL (execution_failed), never an epistemic abstention. The result carries the
    world's omission log."""
    from swm.world_model_v2.result import CompilerExecutionError
    base = build_world(plan, evidence_hash=(plan.provenance or {}).get("evidence_bundle_hash", ""))
    check_readout_binding(plan, base)
    _bind_scenario_schema(plan, base, llm)
    # Phase 3: if the pipeline attached an evidence-updated outcome-rate posterior, hand its particles to the
    # canonical resolve_outcome event so the terminal resolver draws each particle's Bernoulli rate from the
    # POSTERIOR (not the broad lean-Beta prior). This is the single injection point shared by BOTH the
    # single-structure (run.run) and multi-hypothesis paths — both build queues from plan.scheduled_events.
    _inject_posterior_rate(plan)
    ops, rejections = operators_from_plan(plan, llm=llm)
    if not ops:
        # the fallback guarantees generic_outcome_prior is accepted; reaching here is a compiler defect
        raise CompilerExecutionError(
            "no accepted mechanism resolves to an executable operator despite the fallback hierarchy "
            f"(rejections: {[r['reason'][:60] for r in rejections]})",
            taxonomy="missing_required_operator")
    init = InitialStateModel(base_world=base, latents=list(plan.latents))
    npart = n_particles or plan.compute_plan.get("n_particles", 30)
    run = WorldModelV2Run(initial=init, queue_builder=queue_builder_from_plan(plan),
                          operators=ops, contract=plan.outcome_contract, n_particles=npart)
    # B5: materialize competing structural hypotheses as SEPARATE particles — stratify particles across
    # hypotheses, tag each world, and give each hypothesis its own outcome lean so structural disagreement
    # genuinely widens the terminal distribution. Phase-1 weights are the priors (no evidence assimilation
    # here — that is Phase 3's run_filtered, a documented dependency).
    hyps = list(getattr(plan, "structural_hypotheses", []) or [])
    if len(hyps) > 1:
        result, branches = _run_with_hypotheses(run, plan, hyps, seed)
    else:
        result, branches = run.run(seed=seed)
    result["omissions"] = list(getattr(base, "omissions", []))
    result["operator_rejections"] = rejections
    attach_actor_decision_distributions(ops, result)
    err = (plan.provenance or {}).get("scenario_schema_error")
    if err:
        rep = result.setdefault("consequence_report", {})
        rep["degraded"] = True
        rep["scenario_schema_error"] = err
    return result, branches


def _run_with_hypotheses(run, plan, hyps, seed):
    """Stratify particles across structural hypotheses; each hypothesis carries a lean the generic resolver
    reads, so competing structures produce genuinely different terminal outcomes. When a Phase-3 structural
    POSTERIOR is attached to the plan, strata are weighted by the evidence-updated posterior (not the prior);
    otherwise by the compiler prior."""
    from swm.world_model_v2.state import F
    total = run.n_particles
    # Phase 3: prefer the likelihood-updated structural posterior when present.
    struct_post = getattr(plan, "structural_posterior", None) or {}

    def _weight(h):
        hid = str(h.get("id", "H"))
        if struct_post and hid in struct_post:
            return max(0.0, float(struct_post[hid]))
        return max(0.0, float(h.get("prior", 1.0) or 1.0))
    z = sum(_weight(h) for h in hyps) or 1.0
    alloc, assigned = [], 0
    for i, h in enumerate(hyps):
        k = max(1, round(total * _weight(h) / z)) if i < len(hyps) - 1 else total - assigned
        alloc.append(max(0, k))
        assigned += alloc[-1]
    worlds = run.initial.sample_particles(total, seed=seed)
    from swm.world_model_v2.rollout import RolloutEngine
    engine = RolloutEngine(operators=run.operators)
    branches, wi = [], 0
    default_lean = (plan.provenance or {}).get("outcome_lean", "neutral")
    for h, k in zip(hyps, alloc):
        lean = str(h.get("lean") or h.get("outcome_lean") or default_lean)
        for _ in range(k):
            if wi >= len(worlds):
                break
            w = worlds[wi]; wi += 1
            w.uncertainty_meta.setdefault("model", {})["hypothesis"] = h.get("id", "H")
            w.uncertainty_meta["hypothesis_lean"] = lean       # picked up by the resolve_outcome event
            q = run.queue_builder(w)
            # override the resolve_outcome payload lean for this hypothesis
            for ev in q.events:
                if ev.etype == "resolve_outcome":
                    ev.payload["lean"] = lean
            branches.append(engine.run_branch(w, q, seed=seed * 7919 + wi))
    result = plan.outcome_contract.project(branches)
    result["n_deltas"] = sum(len(b.log) for b in branches)
    result["readout"] = "terminal_states"
    # realized structural mass = the fraction of particles allocated to each hypothesis (which equals the
    # posterior when a Phase-3 structural posterior drove the allocation, else the compiler prior).
    realized = {}
    for b in branches:
        hid = b.world.uncertainty_meta.get("model", {}).get("hypothesis", "H")
        realized[hid] = realized.get(hid, 0.0) + 1.0 / max(1, len(branches))
    result["structural_realized_mass"] = {k: round(v, 4) for k, v in realized.items()}
    if struct_post:
        result["structural_posterior"] = {k: round(float(v), 4) for k, v in struct_post.items()}
        result["structural_source"] = "phase3_evidence_posterior"
        result["structural_note"] = ("competing structures weighted by the LIKELIHOOD-UPDATED structural "
                                     "posterior (Phase 3); strata allocated by posterior mass")
    else:
        result["structural_posterior"] = result["structural_realized_mass"]
        result["structural_source"] = "compiler_prior"
        result["structural_note"] = ("priors materialized as competing particles; NOT evidence-reweighted "
                                     "(no Phase-3 posterior attached)")
    return result, branches
