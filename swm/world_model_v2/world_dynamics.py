"""In-run WORLD DYNAMICS — the honest remainder after the §NAP quarantine.

What USED to live here — and why it is gone from production:

  * STANCE DYNAMICS (ripeness/winning/exhaustion/bandwagon): universal numeric rules that moved
    stance labels when a 0-1 progress bar crossed 0.70, capacity dropped below 0.30, etc. Every
    threshold was invented; the progress bars they read no longer exist. In production, an actor's
    stance changes the way a real actor's stance changes: THE ACTOR ITSELF decides — its situated
    LLM cognition observes concrete events (a collapsing position, a rival's offer, exhaustion
    made visible in typed state) at its real decision triggers and rewrites its own stance record.
    The numeric rule tables are buried in legacy_numeric_ablations.
  * CAPACITY: a normalized 0-1 "capacity" resource initialized from a 3-level label
    (0.85/0.6/0.35), drained 0.02 per effortful action and 0.0007/day under contest. No real
    unit, never fitted. Removed. Real resources (money, inventory, ammunition, staffing) may be
    modeled as typed quantities with real units and observed values.
  * SAMPLED COUPLING CONSTANTS (COUPLING_PRIORS): pathway_step 0.04, endogenous split 0.60,
    persistence survival 0.75/0.85 … — the entire unfitted behavior→state→hazard coupling layer.
    Sampling an invented prior does not make it real. Buried in legacy_numeric_ablations; a
    future fitted coupling pack must pass numeric_provenance.fitted_artifact_eligible AND the
    channels it parameterizes must be re-justified individually.

What REMAINS is qualitative or observational:

  1. LIVE STANCE ACCESS — read every actor's CURRENT qualitative stance records from entity
     fields (the same source actor views project), so cognition and reporting see one consistent,
     evolving stance state.
  2. PERSISTENCE SEMANTICS (`PersistenceCheckOperator`) — OBSERVATIONAL: a resolution criterion
     that requires the end-state to HOLD ("no active hostilities for >=30 consecutive days",
     an institutional_rule-provenance number) schedules a check at the window's real end. The
     provisional end-state CONFIRMS iff it still holds in the simulated world at that time —
     i.e. no modeled event broke it — and COLLAPSES iff a mechanism actually cleared it
     (`break_provisional_state`). There is no survival coin: whether a ceasefire holds is
     decided by the simulated actors and mechanisms, or it is not decided at all.
"""
from __future__ import annotations

import hashlib
import json

from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)


def coupling_pack_info() -> dict:
    """§NAP: there is no production coupling channel. The historical unfitted COUPLING_PRIORS are
    quarantined (legacy_numeric_ablations); a fitted pack would have to pass the provenance gate
    AND re-justify each coupling channel individually before anything numeric could serve."""
    return {"source": "quarantined_no_production_coupling_channel", "fitted_at": None,
            "n_trajectories": None}


# ---------------------------------------------------------------- live stance access
def live_stances(world) -> list:
    """Every actor's CURRENT qualitative stance records, read from entity fields (the same source
    the actor views project) — actor cognition may rewrite these, so policies and reports see one
    consistent, evolving stance state."""
    out = []
    for ent in (world.entities or {}).values():
        recs = ent.value("stances", default=None)
        if isinstance(recs, list):
            out.extend(r for r in recs if isinstance(r, dict))
    return out


def stance_state_hash(stances: list) -> str:
    from swm.world_model_v2.mode_graph import canon_level
    key = sorted((str(s.get("actor")), canon_level(s.get("commitment_level")),
                  str(s.get("pathway")), str(s.get("target_mode"))) for s in (stances or []))
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- persistence semantics
def break_provisional_state(world, *, reason: str, at_ts: float = None) -> bool:
    """A modeled mechanism (an actor's action consequence, an institutional reversal, a generated
    event) BREAKS the currently provisional end-state: clear the provisional marker and record the
    collapse cause. Returns True iff a provisional state existed. This is the ONLY way a
    provisional end-state collapses — never a survival coin (§NAP)."""
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    q = world.quantities.get("provisional_absorbing_mode")
    mode = getattr(q, "value", None)
    if not mode:
        return False
    now = at_ts if at_ts is not None else world.clock.now
    register_quantity_type("provisional_absorbing_mode", units="mode")
    world.quantities["provisional_absorbing_mode"] = Quantity(
        name="provisional_absorbing_mode", qtype="provisional_absorbing_mode", value=None,
        timestamp=now)
    breaks = getattr(world, "_provisional_breaks", None)
    if breaks is None:
        breaks = []
        try:
            world._provisional_breaks = breaks
        except Exception:  # noqa: BLE001
            pass
    breaks.append({"mode": str(mode), "reason": str(reason)[:200], "at_ts": now})
    return True


class PersistenceCheckOperator(TransitionOperator):
    """OBSERVATIONAL persistence semantics: a provisional end-state CONFIRMS iff it still holds
    when its criterion window completes in the simulated world (the absorbing predicate is then
    genuinely satisfied — the monitor stamps first passage at the check, which IS when the
    criterion completes). It COLLAPSES only when a modeled mechanism actually broke it
    (`break_provisional_state`) before the check — the named near-miss then really happened in
    the trajectory. No survival probability is drawn (§NAP): holding-vs-breaking is decided by
    the simulated actors and mechanisms, and where no breaking mechanism was modeled at all the
    confirmation is honestly labeled `no_breaking_mechanism_modeled` in the delta."""
    name = "persistence_check"

    def applicable(self, world, event):
        stamped = world.quantities.get("absorbed_at")
        return event.etype == "persistence_check" and getattr(stamped, "value", None) in (None, 0)

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload),
                                  reason_codes=["persistence_check"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        prov_mode = str(getattr(world.quantities.get("provisional_absorbing_mode"), "value",
                                None) or "")
        want = str(a.get("mode", "") or "")
        d = StateDelta(at=world.clock.now, event_type="persistence_check", operator=self.name,
                       reason_codes=[f"mode={want or prov_mode or '?'}"])
        if not prov_mode or (want and want != prov_mode):
            # the provisional state was already broken by a modeled mechanism before the window
            # completed — the near-miss realized; the mode's first-passage process (if any) resumes
            breaks = [b for b in (getattr(world, "_provisional_breaks", None) or [])
                      if not want or b.get("mode") == want]
            d.reason_codes.append("near_miss_realized_collapse")
            d.uncertainty = {"collapse_cause": (breaks[-1] if breaks else
                                                {"reason": "cleared_before_window_completed"}),
                             "semantics": "observational_no_survival_coin"}
            fp_pid = str(a.get("first_passage_process_id", ""))
            if fp_pid:
                from swm.world_model_v2.event_time import resume_first_passage_after_collapse
                st_fp, next_ts = resume_first_passage_after_collapse(world, fp_pid)
                if st_fp is not None and next_ts is not None and next_ts <= st_fp.horizon_ts:
                    d.follow_up_events.append({
                        "etype": "first_passage", "ts": max(float(next_ts), world.clock.now),
                        "participants": [],
                        "payload": {**{k: v for k, v in st_fp.payload.items() if k != "spec"},
                                    "spec": st_fp.payload.get("spec"),
                                    "mode": want or prov_mode,
                                    "hazard_process_id": fp_pid,
                                    "hazard_generation": st_fp.generation}})
                    d.reason_codes.append("first_passage_resumed_after_collapse")
            return d
        # the provisional state HELD its whole window in the simulated world → the criterion is
        # genuinely satisfied NOW
        register_quantity_type("absorbing_state_reached", units="bool")
        register_quantity_type("absorbing_mode", units="mode")
        world.quantities["absorbing_state_reached"] = Quantity(
            name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
            timestamp=world.clock.now)
        world.quantities["absorbing_mode"] = Quantity(
            name="absorbing_mode", qtype="absorbing_mode", value=prov_mode,
            timestamp=world.clock.now)
        d.reason_codes.append("persisted_criterion_satisfied")
        breaking_modeled = bool(getattr(world, "_provisional_breaks", None))
        d.uncertainty = {"semantics": "observational_no_survival_coin",
                         "held_full_window": True,
                         "breaking_mechanism_modeled": breaking_modeled or
                         "no_breaking_mechanism_modeled"}
        world.quantities["provisional_absorbing_mode"] = Quantity(
            name="provisional_absorbing_mode", qtype="provisional_absorbing_mode", value=None,
            timestamp=world.clock.now)
        return d.change("quantities[absorbing_state_reached]", None, True)


register_operator("persistence_check", PersistenceCheckOperator(),
                  requires=("quantities",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="observational: the provisional end-state confirms iff it "
                                   "still holds when its criterion window completes in the "
                                   "simulated world; collapse only via a modeled breaking "
                                   "mechanism (§NAP — no survival coin)", validated=True)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et in ("persistence_check",):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=("quantities",),
                            deltas=("quantities",),
                            parameter_source="world-dynamics layer", validated=True)
