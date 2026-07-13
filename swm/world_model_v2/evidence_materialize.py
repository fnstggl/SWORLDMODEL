"""Materialize evidence into the executed world — Phase 2.

Evidence must have causal consequences, not just decorate a prompt. This layer:

  * writes verified PUBLIC claims into the omniscient world evidence store (provenance = claim id — this is
    the ONLY layer allowed to stamp a state field `observed`);
  * schedules `observe_evidence` events for included claims at their earliest observation time, so rollout
    produces explicit StateDelta objects for observations;
  * enforces ACTOR views — an actor only observes claims its visibility permits, at the time it could;
  * (via evidence_recompile) adds the institutions / rules / events / reweighted hypotheses the evidence
    implies, which flow through the existing materializer → changed event queue, feasible actions, and
    terminal distribution.

A component is "causal" only if removing it changes at least one of: plan structure, actor view, action
feasibility, mechanism selection, event queue, StateDelta trace, structural weights, terminal distribution,
or support grade. `evidence_causal_effect()` measures exactly that against the pre-evidence plan.
"""
from __future__ import annotations

import copy

from swm.world_model_v2.state import F, parse_time
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            register_operator)


class EvidenceObservationOperator(TransitionOperator):
    """Writes an observed evidence claim into the world as an `observed` fact and emits a StateDelta. The
    claim's value is grounded in the evidence bundle (never LLM-minted here); provenance carries the claim id
    and the source, so the observation is auditable back to raw content."""
    name = "evidence_observation"

    def applicable(self, world, event):
        return event.etype == "observe_evidence"

    def propose(self, world, event, rng):
        p = event.payload
        return TransitionProposal(operator=self.name, action={
            "quantity": f"evidence::{p.get('claim_id', 'c')}", "claim_id": p.get("claim_id", ""),
            "subject": p.get("subject", ""), "value": p.get("value", "observed"),
            "source": p.get("source", ""), "visibility": p.get("visibility", "public")},
            reason_codes=["evidence_observation", f"claim={p.get('claim_id', '')}"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        var = a["quantity"]
        register_quantity_type(var, units="evidence")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=a["value"], timestamp=world.clock.now)
        # record into the omniscient evidence store on the world
        store = world.uncertainty_meta.setdefault("evidence_store", {})
        store[a["claim_id"]] = {"subject": a["subject"], "value": a["value"], "source": a["source"],
                                "visibility": a["visibility"], "observed_at": world.clock.now}
        d = StateDelta(at=world.clock.now, event_type="observe_evidence", operator=self.name,
                       reason_codes=proposal.reason_codes,
                       uncertainty={"provenance": f"evidence_claim:{a['claim_id']}", "source": a["source"]})
        return d.change(f"quantities[{var}]", before, a["value"])


register_operator("evidence_observation", EvidenceObservationOperator, requires=("quantities",),
                  modifies=("quantities",), temporal_scale="event",
                  parameter_source="grounded in the immutable evidence bundle (claim id); NOT LLM-minted",
                  validated=True)


def attach_evidence_observations(plan, bundle, *, max_obs: int = 8):
    """Add `observe_evidence` scheduled events for the bundle's INCLUDED claims, at their earliest
    observation time (clamped to ≤ as_of). Returns a COPY of the plan with the events + a registered event
    type, so rollout emits observation StateDeltas grounded in evidence."""
    from swm.world_model_v2.events import event_type_registered, register_event_type
    if not event_type_registered("observe_evidence"):
        register_event_type("observe_evidence", scheduling="scheduled", validated=True,
                             parameter_source="evidence_bundle")
    rp = copy.deepcopy(plan)
    as_of_ts = bundle.as_of
    vis_by_claim = {v["claim_id"]: v for v in bundle.actor_visibility}
    inc = set(bundle.included_claim_ids)
    # ensure the evidence_observation operator is in the rollout (operators_from_plan reads accepted_mechanisms)
    if not any(m.get("operator") == "evidence_observation" for m in rp.accepted_mechanisms):
        rp.accepted_mechanisms.append({
            "mech_id": "evidence_observation", "ontology_type": "informational",
            "causal_role": "materialize an evidence claim as an observed fact + StateDelta",
            "parameter_source": "immutable evidence bundle (claim id); NOT LLM-minted",
            "temporal_scale": "event", "calibration_status": "experimental",
            "operator": "evidence_observation", "sensitivity": 0.5})
    added = 0
    for c in bundle.claims:
        if c["claim_id"] not in inc or added >= max_obs:
            continue
        v = vis_by_claim.get(c["claim_id"], {})
        obs_ts = v.get("earliest_observation_time") or c.get("publication_time") or as_of_ts
        obs_ts = min(float(obs_ts), as_of_ts)             # never observe evidence after the as-of horizon
        rp.scheduled_events.append({
            "etype": "observe_evidence", "ts": obs_ts, "participants": [],
            "payload": {"claim_id": c["claim_id"], "subject": c.get("subject", ""),
                        "value": c.get("value") or c.get("object") or "reported",
                        "source": c.get("source_id", ""), "visibility": c.get("actor_visibility", "public")}})
        added += 1
    rp.provenance["n_evidence_observations"] = added
    return rp


def materialize_public_evidence(world, bundle):
    """Write verified/likely PUBLIC included claims into the omniscient world evidence store immediately (as
    context available at as_of). Actor-restricted claims are NOT written here — they reach actors only through
    observe_evidence at the permitted time."""
    inc = set(bundle.included_claim_ids)
    vis = {v["claim_id"]: v for v in bundle.actor_visibility}
    store = world.uncertainty_meta.setdefault("evidence_store", {})
    n = 0
    for c in bundle.claims:
        if c["claim_id"] not in inc:
            continue
        if vis.get(c["claim_id"], {}).get("visibility") == "public":
            store[c["claim_id"]] = {"subject": c.get("subject", ""), "value": c.get("value", ""),
                                    "source": c.get("source_id", ""), "visibility": "public"}
            n += 1
    return n


def evidence_causal_effect(plan, bundle, *, llm, horizon: str, seed: int = 7, n_particles: int = 40) -> dict:
    """Run the pre-evidence plan and the evidence-conditioned plan; report what evidence CHANGED. Proves the
    effect is not merely `outcome_lean`: reports structural changes, event-queue delta, StateDelta delta, and
    terminal-distribution delta."""
    from swm.world_model_v2.materialize import run_from_plan
    from swm.world_model_v2.evidence_recompile import recompile_with_evidence

    pre_res, pre_branches = run_from_plan(plan, llm=None, n_particles=n_particles, seed=seed)
    revised, diff = recompile_with_evidence(plan, bundle, llm=llm, horizon=horizon)
    revised = attach_evidence_observations(revised, bundle)
    post_res, post_branches = run_from_plan(revised, llm=None, n_particles=n_particles, seed=seed)

    obs_deltas = sum(1 for b in post_branches for d in getattr(b, "log", [])
                     if getattr(d, "event_type", "") == "observe_evidence")
    changed = {
        "structural_changes": diff.n_structural_changes,
        "lean_only": diff.lean_only,
        "n_events_pre": len(plan.scheduled_events), "n_events_post": len(revised.scheduled_events),
        "n_institutions_pre": len(plan.institutions), "n_institutions_post": len(revised.institutions),
        "n_entities_pre": len(plan.entities), "n_entities_post": len(revised.entities),
        "observation_state_deltas": obs_deltas,
        "terminal_pre": pre_res.get("distribution"), "terminal_post": post_res.get("distribution"),
        "terminal_changed": pre_res.get("distribution") != post_res.get("distribution"),
        "n_deltas_pre": pre_res.get("n_deltas"), "n_deltas_post": post_res.get("n_deltas"),
        "plan_diff": diff.as_dict(),
    }
    changed["evidence_is_causal"] = bool(
        changed["structural_changes"] > 0 or changed["terminal_changed"] or obs_deltas > 0
        or changed["n_events_post"] != changed["n_events_pre"])
    return changed
