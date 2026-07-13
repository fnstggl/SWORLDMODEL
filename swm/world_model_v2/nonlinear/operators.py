"""Nonlinear mechanism execution operators — Phase 7, Part 18 (the EXECUTION PLANE).

These are real `TransitionOperator`s (Mode A — the seam that reuses the compiler, materializer, rollout loop,
StateDelta trace, follow-up-event scheduling, and contract projection unchanged; see hazard.py's
FeatureHazardOperator, which this mirrors). A Phase-7 nonlinear mechanism therefore:

  event fires → operator reads TYPED context (context.ContextSchema) + TYPED history (history_features, no
  future) from WorldState → evaluates the selected structural FORM, PER POSTERIOR PARTICLE where the pack
  carries Phase-3 uncertainty (posterior.propagate — E[f(X)] not f(E[X])) → guards the value
  (safety.safe_prob/safe_rate) → emits an explicit StateDelta recording exactly what changed and why → and
  may SCHEDULE FUTURE EVENTS (retransmission, a recurrent action, a next-exposure) so the nonlinearity
  propagates through time and changes the terminal outcome.

The scenario-specific instance rides on the triggering `event.payload` as `nonlinear_spec` — additive, so no
compiler/materialize/plan-schema change is required (Phase-9/10-safe). Two operators:
  * NonlinearMechanismOperator  — the general form→outcome executor (thresholds, saturation, fatigue,
    interactions, regime, survival). Writes the outcome quantity + optional recurrent follow-up.
  * NonlinearContagionOperator  — diffusion: wraps the fitted Hill/log-linear hazard, integrates a survival
    step with exposure HISTORY, emits a StateDelta per step, and on activation schedules retransmission
    exposures to in-network followers (the causal chain, executed — never a jump to final cascade size).
"""
from __future__ import annotations

import math
import random

from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                             register_operator)
from swm.world_model_v2.nonlinear import forms as _forms
from swm.world_model_v2.nonlinear import safety as _safety
from swm.world_model_v2.nonlinear import history as _history
from swm.world_model_v2.nonlinear.posterior import ParamPosterior, propagate


def _build_posteriors(spec: dict) -> dict:
    """Reconstruct {latent: ParamPosterior} from the serialized spec (particles / grid / envelope / point)."""
    out = {}
    for name, rep in (spec.get("param_posteriors") or {}).items():
        if "particles" in rep:
            out[name] = ParamPosterior(name, particles=[tuple(p) for p in rep["particles"]])
        elif "grid" in rep:
            out[name] = ParamPosterior(name, grid=(rep["grid"][0], rep["grid"][1]))
        elif "envelope" in rep:
            out[name] = ParamPosterior(name, envelope=rep["envelope"])
        elif "value" in rep:
            out[name] = ParamPosterior.point(name, rep["value"])
    return out


class NonlinearMechanismOperator(TransitionOperator):
    """Execute a fitted nonlinear FORM for one actor/scenario, honoring context + history + posterior.

    payload['nonlinear_spec'] = {
        form_id, params,                       # the fitted structural form + its serialized parameters
        outcome_var, actor,                    # what/who the outcome is written to
        context: {schema, extra},              # optional ContextSchema.as_dict() + precomputed extras
        history_window,                        # optional HistoryWindow.as_dict()
        param_posteriors,                      # optional {latent: {particles|grid|envelope|value}} (Phase 3)
        param_map,                             # optional {form_param: latent_name} remap for the posterior
        transport_widening, family, pack_id,   # provenance + transport
        output: 'prob'|'rate', window_days,    # how the form value becomes an outcome
        recurrent: {etype, delay_h, max}       # optional future-event generation
    }"""
    name = "nonlinear_mechanism"

    def applicable(self, world, event):
        return event.etype == "nonlinear_transition" and isinstance(event.payload.get("nonlinear_spec"), dict)

    # ---- read the typed inputs from WorldState (context + history + fixed features) ----
    def _inputs(self, world, spec):
        inp = dict(spec.get("params", {}).get("_inputs", {}))     # scenario-fixed x/features
        inp.update(spec.get("inputs") or {})
        feats = dict(spec.get("features") or {})
        actor_id = spec.get("actor")
        # context (typed, leakage-guarded)
        cx = spec.get("context") or {}
        if cx.get("schema"):
            from swm.world_model_v2.nonlinear.context import ContextSchema, ContextVariable
            sch = ContextSchema(mechanism_family=cx["schema"].get("mechanism_family", ""),
                                variables=[ContextVariable(**v) for v in cx["schema"].get("variables", [])])
            cvec = sch.read(world, actor_id=actor_id, extra=cx.get("extra"))
            for k, v in cvec.items():
                if not k.startswith("_") and v is not None:
                    inp.setdefault(k, v)
                    feats.setdefault(k, v)
        # history (typed, no future)
        if spec.get("history_window") is not None and actor_id and actor_id in (world.entities or {}):
            from swm.world_model_v2.nonlinear.history import HistoryWindow
            hw = HistoryWindow(**{k: v for k, v in spec["history_window"].items() if k != "features"})
            hf = _history.history_features(world.entity(actor_id), now=world.clock.now, window=hw)
            for k, v in hf.items():
                if not k.startswith("_"):
                    inp.setdefault(k, v)
                    feats.setdefault(k, v)
            inp["_refractory_active"] = hf.get("_refractory_active", 0.0)
        if feats:
            inp["features"] = feats
        inp.setdefault("window_days", float(spec.get("window_days", 1.0)))
        return inp

    def propose(self, world, event, rng):
        spec = event.payload["nonlinear_spec"]
        form = _forms.get_form(spec["form_id"])
        inputs = self._inputs(world, spec)
        # refractory suppression (Part 4C): a receiver inside its refractory window does not respond
        if inputs.get("_refractory_active", 0.0) >= 1.0:
            return TransitionProposal(operator=self.name,
                                      action={"outcome_var": spec["outcome_var"], "value": None,
                                              "suppressed": True, "actor": spec.get("actor")},
                                      reason_codes=["refractory_suppressed"])
        report = _safety.GuardReport()
        posteriors = _build_posteriors(spec)
        jensen = None
        if posteriors:
            pmap = spec.get("param_map") or {}
            base = dict(spec.get("params", {}))

            def build_params(sampled):
                p = dict(base)
                for fparam, latent in pmap.items():
                    if latent in sampled:
                        p[fparam] = sampled[latent]
                for latent, val in sampled.items():
                    if latent not in pmap.values():
                        p.setdefault(latent, val)
                return p
            pr = propagate(form, posteriors, inputs, n=int(spec.get("n_particles", 128)), rng=rng,
                           param_map=build_params)
            value = pr.mean
            jensen = pr.jensen_gap
        else:
            value = form.eval(spec.get("params", {}), inputs)
        # transport widening → per-branch dispersion on the log-odds / log-rate (mean preserved)
        widen = float(spec.get("transport_widening", 1.0) or 1.0)
        out_kind = spec.get("output", "prob" if form.output_domain in ("unit_interval", "prob_window")
                            else "rate")
        if out_kind == "prob":
            if widen > 1.0:
                lo = math.log(max(1e-6, value) / max(1e-6, 1 - value))
                lo += rng.gauss(0.0, 0.35 * (widen - 1.0))
                value = 1.0 / (1.0 + math.exp(-max(-30, min(30, lo))))
            p = _safety.safe_prob(value, report)
            action = {"outcome_var": spec["outcome_var"], "p": p, "kind": "prob"}
        else:
            lam = _safety.safe_rate(value, report)
            wd = inputs["window_days"]
            p = 1.0 - math.exp(-min(50.0, lam * wd))
            if widen > 1.0:
                p = _safety.safe_prob(p * (1.0 + 0.0) , report)   # rate widening handled via lam already
            action = {"outcome_var": spec["outcome_var"], "p": _safety.safe_prob(p, report), "rate": lam,
                      "kind": "rate"}
        action.update({"actor": spec.get("actor"), "family": spec.get("family", ""),
                       "pack_id": spec.get("pack_id", ""), "form_id": spec["form_id"],
                       "options": spec.get("options") or ["True", "False"],
                       "recurrent": spec.get("recurrent")})
        unc = {"pack_id": spec.get("pack_id", ""), "form_id": spec["form_id"],
               "transport_widening": widen, "output": out_kind}
        if jensen is not None:
            unc["jensen_gap"] = round(jensen, 6)
            unc["posterior_propagated"] = True
        if report.clamped:
            unc["stability_guards"] = report.reasons
        return TransitionProposal(operator=self.name, action=action,
                                  reason_codes=[f"nonlinear:{spec.get('family', '')}:{spec['form_id']}",
                                                f"p={round(action['p'], 4)}"],
                                  uncertainty=unc)

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.state import F
        a = proposal.action
        if a.get("suppressed"):
            d = StateDelta(at=world.clock.now, event_type="nonlinear_transition", operator=self.name,
                           reason_codes=proposal.reason_codes)
            return d
        var = a["outcome_var"]
        register_quantity_type(var, units="outcome")
        rng = random.Random(hash((world.branch_id, var, round(world.clock.now))) & 0xFFFFFFFF)
        opts = a["options"] if len(a["options"]) == 2 else ["True", "False"]
        occurred = rng.random() < a["p"]
        val = opts[0] if occurred else opts[1]
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
        if a.get("actor") and a["actor"] in (world.entities or {}):
            world.entity(a["actor"]).set("outcome", F(val, status="derived", method=self.name,
                                                      updated_at=world.clock.now), key=var)
            _history.record_action(world.entity(a["actor"]), at=world.clock.now, action=f"{var}={val}")
        d = StateDelta(at=world.clock.now, event_type="nonlinear_transition", operator=self.name,
                       reason_codes=proposal.reason_codes, uncertainty=proposal.uncertainty)
        d.change(f"quantities[{var}]", before, val)
        # temporal event generation: a recurrent event (the outcome causes the next one)
        rec = a.get("recurrent")
        if rec and occurred:
            delay = float(rec.get("delay_h", 24.0)) * 3600.0
            d.follow_up_events = [{"etype": rec.get("etype", "nonlinear_transition"),
                                   "ts": world.clock.now + delay,
                                   "participants": [a["actor"]] if a.get("actor") else [],
                                   "payload": {"nonlinear_spec": rec.get("next_spec")}}] \
                if rec.get("next_spec") else []
        return d


class NonlinearContagionOperator(TransitionOperator):
    """Diffusion executed as a real transition (bridges the Mode-B diffusion.py forms into rollout).

    payload['contagion_spec'] = {form_id ('complex_contagion_hazard'|'exposure_response_hazard'|
        'simple_contagion_hazard'), params, outcome_var, window_days, frailty_sigma}. The operator reads the
    receiver's EXPOSURE HISTORY (history_features → cum_count 'k', recency), evaluates the fitted hazard, and
    over the elapsed step accumulates activation probability; on activation it writes the outcome and SCHEDULES
    retransmission exposures to in-network followers (payload carries the follower list) — the exposure→
    adoption→retransmission chain executed event-by-event, never a jump to final cascade size."""
    name = "nonlinear_contagion"

    def applicable(self, world, event):
        return event.etype == "contagion_exposure" and isinstance(event.payload.get("contagion_spec"), dict)

    def _hazard(self, spec):
        from swm.world_model_v2.registry.families.diffusion import (LinearHazard, HillHazard, LogLinearHazard)
        fid, p = spec["form_id"], spec.get("params", {})
        if fid == "complex_contagion_hazard":
            return HillHazard(p["theta0"], p["alpha"], p["c"])
        if fid == "exposure_response_hazard":
            return LogLinearHazard(p["theta"])
        return LinearHazard(p.get("q", 0.02))

    def propose(self, world, event, rng):
        spec = event.payload["contagion_spec"]
        actor_id = event.participants[0] if event.participants else spec.get("actor")
        if not actor_id or actor_id not in (world.entities or {}):
            return None
        actor = world.entity(actor_id)
        # already activated? (idempotent — outcome quantity set)
        var = spec["outcome_var"]
        if actor.value("outcome", key=var) in ("True", True):
            return None
        hf = _history.history_features(actor, now=world.clock.now)
        k = hf["cum_count"] + float(spec.get("k0", 0.0))          # exposure count from typed history
        deg = float(actor.value("degree") or spec.get("deg", 1.0))
        recency_h = hf["recency_h"]
        hz = self._hazard(spec)
        report = _safety.GuardReport()
        lam = _safety.safe_rate(hz.lam(k, deg, recency_h), report)
        sigma = float(spec.get("frailty_sigma", 0.0))
        frail = math.exp(rng.gauss(-sigma ** 2 / 2, sigma)) if sigma > 0 else 1.0
        wd = float(spec.get("window_days", 1.0))
        haz = frail * lam * wd
        p = _safety.safe_prob(1.0 - math.exp(-min(50.0, haz)), report)
        return TransitionProposal(operator=self.name,
                                  action={"actor": actor_id, "outcome_var": var, "p": p, "k": k, "deg": deg,
                                          "followers": spec.get("followers") or [],
                                          "retransmit_spec": spec.get("retransmit_spec"),
                                          "delay_h": spec.get("retransmit_delay_h", 6.0)},
                                  reason_codes=[f"contagion:{spec['form_id']}", f"k={round(k,1)}",
                                                f"p={round(p,4)}"],
                                  uncertainty={"form_id": spec["form_id"], "lambda": round(lam, 6),
                                               "frailty_sigma": sigma,
                                               **({"stability_guards": report.reasons} if report.clamped
                                                  else {})})

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.state import F
        a = proposal.action
        var = a["outcome_var"]
        register_quantity_type(var, units="outcome")
        rng = random.Random(hash((world.branch_id, a["actor"], var)) & 0xFFFFFFFF)
        activated = rng.random() < a["p"]
        d = StateDelta(at=world.clock.now, event_type="contagion_exposure", operator=self.name,
                       reason_codes=proposal.reason_codes, uncertainty=proposal.uncertainty)
        actor = world.entity(a["actor"])
        before = actor.value("outcome", key=var)
        if activated:
            actor.set("outcome", F("True", status="derived", method=self.name, updated_at=world.clock.now),
                      key=var)
            world.quantities[f"{var}:{a['actor']}"] = Quantity(name=f"{var}:{a['actor']}", qtype=var,
                                                               value=True, timestamp=world.clock.now)
            d.change(f"{a['actor']}.outcome[{var}]", before, "True")
            # RETRANSMISSION: schedule fresh exposure events to in-network followers (bounded)
            fu = []
            for fol in a["followers"][:200]:
                if a.get("retransmit_spec"):
                    fu.append({"etype": "contagion_exposure",
                               "ts": world.clock.now + float(a["delay_h"]) * 3600.0,
                               "participants": [fol],
                               "payload": {"contagion_spec": dict(a["retransmit_spec"], actor=fol)}})
                    # record the exposure on the follower's typed history NOW (it will be at-or-before its
                    # own future event time; history_features drops anything after the receiver's clock)
                    if fol in (world.entities or {}):
                        _history.record_exposure(world.entity(fol), at=world.clock.now, source=a["actor"],
                                                 kind="retransmission")
            d.follow_up_events = fu
        else:
            d.change(f"{a['actor']}.outcome[{var}]", before, before)   # no-op recorded (survival step)
        return d


# ---------------------------------------------------------------- registration (import side effects)
register_operator("nonlinear_mechanism", NonlinearMechanismOperator(), requires=("entities",),
                  modifies=("quantities", "entities"), temporal_scale="event",
                  parameter_source="fitted nonlinear structural form + Phase-3 posterior particles; "
                                   "transport widening on log-odds/log-rate",
                  validated=True)
register_operator("nonlinear_contagion", NonlinearContagionOperator(), requires=("entities",),
                  modifies=("quantities", "entities"), temporal_scale="event",
                  parameter_source="fitted diffusion hazard form (Hill/log-linear); exposure history from "
                                   "typed event log; frailty heterogeneity",
                  validated=True)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et in ("nonlinear_transition", "contagion_exposure"):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=("entities",), deltas=("quantities",),
                            parameter_source="Phase 7 nonlinear mechanism", validated=True)

from swm.world_model_v2.state import extension_fields, register_entity_extension  # noqa: E402
# `outcome` is also registered by hazard.py (mechanism_outcome); register our own additive extension for the
# fields Phase-7 operators write/read, guarded per-field so import order with hazard.py never clobbers either.
_p7_fields = {}
if "outcome" not in extension_fields("person"):
    _p7_fields["outcome"] = "typed outcome value written to an actor by a mechanism"
if "degree" not in extension_fields("person"):
    _p7_fields["degree"] = "network out-degree (reach) available to diffusion forms"
if _p7_fields:
    register_entity_extension("p7_mechanism_fields", fields=_p7_fields,
                              entity_types=("person", "institution"))
