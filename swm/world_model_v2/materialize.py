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
    """Fresh queue per branch: scheduled events + hazards, horizon-capped."""
    def build(world) -> EventQueue:
        q = EventQueue(horizon_ts=plan.horizon_ts)
        rng = random.Random(int(world.branch_id.strip("b").split(":")[0] or 0)
                            if world.branch_id.startswith("b") else 0)
        for ev in plan.scheduled_events:
            q.schedule(Event(ts=ev["ts"], etype=ev["etype"], participants=list(ev["participants"]),
                             payload=dict(ev["payload"]), source="scheduled"))
        for hz in plan.stochastic_hazards:
            q.add_hazard(StochasticHazard(etype=hz["etype"], rate_per_day=hz["rate_per_day"],
                                          participants=list(hz["participants"])),
                         now=world.clock.now, rng=rng, world=world)
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
                # Default-on: with an LLM backend the core V2 funnel runs
                # hybrid_relevant_actor_policy — persistent qualitative LLM cognition for
                # causally consequential actors (question-specific tiers from the compiled
                # plan), the numeric policy for routine actors; with no backend, the numeric
                # production runtime exactly as it was. SWM_ACTOR_POLICY selects any mode.
                bound_runtime = _actor_policy_runtime(plan, llm)
                ops.append(factory(runtime=bound_runtime) if bound_runtime is not None
                           else factory())
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
    return ops, rejections


ACTOR_POLICY_MODES = ("numeric_policy", "persona_blended_numeric_policy", "stateless_llm_policy",
                      "persistent_qualitative_llm_policy", "hybrid_relevant_actor_policy")


def resolve_actor_policy_mode(llm) -> str:
    """The run's actor-policy mode: SWM_ACTOR_POLICY when set (validated), else DEFAULT-ON
    hybrid qualitative cognition with a backend, numeric without. The legacy SWM_LLM_ACTORS=off
    switch still forces numeric."""
    import os
    if os.environ.get("SWM_LLM_ACTORS", "").strip().lower() == "off":
        return "numeric_policy"
    mode = os.environ.get("SWM_ACTOR_POLICY", "").strip().lower()
    if mode in ACTOR_POLICY_MODES:
        return mode
    return "hybrid_relevant_actor_policy" if llm is not None else "numeric_policy"


def _actor_policy_runtime(plan, llm):
    """Bind the runtime for the resolved mode, or None (plain numeric operator). A failure to
    construct an LLM runtime never blocks materialization — the numeric path serves — but it is
    RECORDED on the plan provenance (`actor_runtime_fallback`), never silent: the run
    classification and product epistemic contract surface it."""
    mode = resolve_actor_policy_mode(llm)
    if mode == "numeric_policy" or llm is None:
        return None
    try:
        if mode == "persona_blended_numeric_policy":
            from swm.world_model_v2.llm_actor import build_persona_runtime
            return build_persona_runtime(llm=llm)
        from swm.world_model_v2.qualitative_actor import build_qualitative_runtime
        return build_qualitative_runtime(plan, llm=llm, mode=mode)
    except Exception as e:  # noqa: BLE001
        try:
            (plan.provenance or {}).setdefault("actor_runtime_fallback", []).append({
                "requested_mode": mode, "fallback": "numeric_policy",
                "reason": f"llm_runtime_construction_failed: {type(e).__name__}: {e}"[:160]})
        except Exception:  # noqa: BLE001 — provenance recording must not mask the fallback
            pass
        return None


def attach_world_hypotheses_from_plan(init, plan, *, llm=None) -> int:
    """Generate the run's COHERENT JOINT WORLD HYPOTHESES (joint_world) from the compiled plan
    and bind them onto the InitialStateModel, so every sampled particle carries one shared
    hidden reality that all actor-private states condition on. Default-on; `SWM_JOINT_WORLD=off`
    disables (recorded on the plan provenance, never silent). Returns the number attached."""
    import os
    if os.environ.get("SWM_JOINT_WORLD", "").strip().lower() == "off":
        (plan.provenance or {}).setdefault("joint_world", {})["status"] = "disabled_by_env"
        return 0
    try:
        import time as _t
        from swm.world_model_v2.joint_world import JointWorldHypothesizer, attach_joint_hypotheses
        actors = [str(e.get("id")) for e in (plan.entities or [])
                  if isinstance(e, dict) and str(e.get("type", "person")) == "person"]
        insts = [str(i.get("id")) for i in (plan.institutions or []) if isinstance(i, dict)]
        ev_rows = []
        for s in (getattr(plan, "_intention_stances", None) or [])[:10]:
            if isinstance(s, dict):
                ev_rows.append(f"- {s.get('actor')}: [{s.get('commitment_level')}] on "
                               f"{s.get('pathway')}"
                               + (f" — \"{str(s.get('quote', ''))[:160]}\"" if s.get("quote") else ""))
        evidence = "\n".join(ev_rows)
        date = _t.strftime("%Y-%m-%d", _t.gmtime(float(plan.as_of))) if plan.as_of else "?"
        structural = {"hypotheses": [str(h.get("id", "H"))
                                     for h in (getattr(plan, "structural_hypotheses", None) or [])
                                     if isinstance(h, dict)]}
        hyp = JointWorldHypothesizer(llm, k=3)
        rows = hyp.generate(question=str(getattr(plan, "question", ""))[:300], actors=actors,
                            institutions=insts, evidence=evidence, date=date,
                            structural_model=structural)
        n = attach_joint_hypotheses(init, rows)
        (plan.provenance or {}).setdefault("joint_world", {}).update(
            {"status": "attached", "k": n, "llm_calls": hyp.llm_calls,
             "source": rows[0].provenance.get("source", "") if rows else "",
             "labels": [r.label for r in rows]})
        return n
    except Exception as e:  # noqa: BLE001 — never blocks the forecast, never silent
        (plan.provenance or {}).setdefault("joint_world", {}).update(
            {"status": "generation_failed", "error": f"{type(e).__name__}: {e}"[:160]})
        return 0


def attach_actor_decision_distributions(ops, container: dict) -> None:
    """After a rollout, read every qualitative runtime's (posterior, trace) records off the
    shared operators and attach the counted raw/calibrated action distributions — the
    statistical layer's output — to the run container. No-op for numeric/persona runs."""
    try:
        from swm.world_model_v2.qualitative_actor import aggregate_actor_decisions
        records = []
        mode = ""
        for op in ops:
            runtime = getattr(op, "runtime", None)
            if runtime is not None and hasattr(runtime, "decision_records"):
                records.extend(runtime.decision_records)
                mode = getattr(runtime, "mode", mode)
        if records:
            container["actor_decision_distributions"] = aggregate_actor_decisions(records)
            container["actor_policy_mode"] = mode
    except Exception:  # noqa: BLE001 — aggregation is reporting, never a run blocker
        pass


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
    return injected


def run_from_plan(plan, *, llm=None, n_particles=None, seed=0):
    """The end-to-end: plan → world → InitialStateModel → rollout → native terminal distribution.
    The compiler's fallback hierarchy guarantees ≥1 executable mechanism + a binding readout, so a
    failure here is TECHNICAL (execution_failed), never an epistemic abstention. The result carries the
    world's omission log."""
    from swm.world_model_v2.result import CompilerExecutionError
    base = build_world(plan, evidence_hash=(plan.provenance or {}).get("evidence_bundle_hash", ""))
    check_readout_binding(plan, base)
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
    attach_world_hypotheses_from_plan(init, plan, llm=llm)
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
            jw = w.uncertainty_meta.get("joint_world_hypothesis")
            if isinstance(jw, dict):                           # joint-world ↔ structural coherence
                jw.setdefault("structural_model", {})["assigned_hypothesis"] = h.get("id", "H")
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
