"""Feature-conditioned discrete-outcome hazard family — Phase 6.

One reusable STRUCTURAL family: a Bernoulli outcome whose log-odds is linear in typed actor features,
  P(outcome | x) = σ(b + Σ_j w_j · z_j(x)),   z_j = standardize(x_j) for continuous features,
fitted by penalized MLE on a TRAIN split (registry.ingestion.fit_logistic). Distinct CAUSAL PROCESSES
reuse this form through distinct FAMILIES + PACKS (they are NOT the same mechanism):
  * attrition_dropout_hazard      — a relationship/subscription ends (telco churn pack)
  * response_occurrence_hazard    — a message/question receives a response (StackExchange pack)
  * argument_persuasion_success   — an argument earns a view change (CMV pack)
Each family declares its own applicability + transport limits; only the executable core is shared.

CAUSAL HONESTY. A fitted feature→outcome model is PREDICTIVE unless the design identifies a cause. Packs
carry `causally_identified` and `forbidden` fields (see registry data). This module NEVER upgrades a
prediction to a cause; it only executes the fitted hazard and records provenance + transport widening.

EXECUTION. FeatureHazardOperator reads an actor's typed features from WorldState, evaluates the hazard
with the scenario-instance parameters (a pack bound to feature paths + a transport-widening factor),
samples the outcome per branch, writes a typed outcome quantity, and emits an explicit StateDelta. The
prediction is read from resulting world state, never returned by a bypass.
"""
from __future__ import annotations

import math

from swm.world_model_v2.registry.ingestion import fit_logistic
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            register_operator)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


class FeatureHazard:
    """A fitted feature→probability model with its own standardizer (train-only stats). Deterministic."""

    def __init__(self, weights: dict, intercept: float, standardizer: dict | None = None):
        self.weights = dict(weights)
        self.intercept = float(intercept)
        self.standardizer = dict(standardizer or {})   # {feat: [mu, sd]} for continuous features

    def z(self, feats: dict) -> float:
        s = self.intercept
        for k, w in self.weights.items():
            v = float(feats.get(k, 0.0))
            if k in self.standardizer:
                mu, sd = self.standardizer[k]
                v = (v - mu) / (sd or 1.0)
            s += w * v
        return s

    def p(self, feats: dict, *, logodds_shift: float = 0.0) -> float:
        return _sigmoid(self.z(feats) + logodds_shift)

    def params(self) -> dict:
        return {"weights": {k: round(v, 5) for k, v in self.weights.items()},
                "intercept": round(self.intercept, 5), "standardizer": self.standardizer}


def fit_feature_hazard(rows, feat_keys, cont_keys, *, label="y", iters=600, lr=0.3, l2=1e-3):
    """Fit on TRAIN rows only. rows: [{features:{...}, y:0/1}]. Standardizer computed on the SAME rows
    (train). Returns a FeatureHazard. (The Phase 6 fitting harness in experiments/ is what actually runs
    this on the committed datasets and records the coefficients into packs — this is the shared core.)"""
    stats = {}
    for k in cont_keys:
        vals = [float(r["features"][k]) for r in rows]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 or 1.0
        stats[k] = [mu, sd]

    def vec(r):
        out = []
        for k in feat_keys:
            v = float(r["features"][k])
            if k in stats:
                mu, sd = stats[k]
                v = (v - mu) / sd
            out.append(v)
        return out

    X = [vec(r) for r in rows]
    Y = [int(r[label]) for r in rows]
    w, b = fit_logistic(X, Y, iters=iters, lr=lr, l2=l2)
    return FeatureHazard(dict(zip(feat_keys, w)), b, stats)


def hazard_from_pack(pack_values: dict) -> FeatureHazard:
    """Build the executable hazard from a registry ParameterPack's stored `values`. The pack stores the
    fitted coefficients under a single 'coefficients' entry (source=fitted); this reconstructs the model
    so execution uses the SAME numbers that were validated — never re-typed by hand."""
    c = pack_values["coefficients"]["value"]
    return FeatureHazard(c["weights"], c["intercept"], c.get("standardizer"))


# ------------------------------------------------------------------ the executable world transition
class FeatureHazardOperator(TransitionOperator):
    """Executes a feature-conditioned hazard for one actor. The scenario instance (built by the compiler
    from a pack) rides on the event payload as `hazard_spec`:
        {outcome_var, weights, intercept, standardizer, feature_source:{feat: state_path},
         transport_widening, family, pack_id}
    propose(): read the actor's typed features from WorldState, compute p (transport widening inflates the
    log-odds uncertainty, NOT the point estimate — a transported pack is less certain, not more extreme).
    apply(): draw the outcome for this branch, write the typed outcome quantity, emit StateDelta with the
    pack provenance + widening recorded in `uncertainty`. Deterministic per branch_id (replayable)."""
    name = "feature_hazard"

    def applicable(self, world, event):
        return event.etype == "outcome_hazard" and isinstance(event.payload.get("hazard_spec"), dict)

    def _read_features(self, world, spec):
        """Pull the actor's typed features from world state via feature_source paths; fall back to the
        literal features dict the instance was bound with (a scenario-fixed instance)."""
        feats = dict(spec.get("features") or {})
        actor_id = spec.get("actor")
        src = spec.get("feature_source") or {}
        if actor_id and src and actor_id in (world.entities or {}):
            actor = world.entity(actor_id)
            for feat, field in src.items():
                v = actor.value(field, default=feats.get(feat))
                if v is not None:
                    feats[feat] = v
        return feats

    def propose(self, world, event, rng):
        spec = event.payload["hazard_spec"]
        hz = FeatureHazard(spec["weights"], spec["intercept"], spec.get("standardizer"))
        feats = self._read_features(world, spec)
        # transport widening → a per-branch log-odds jitter (broader outcome dispersion, mean unchanged)
        widen = float(spec.get("transport_widening", 1.0) or 1.0)
        base_sd = 0.35                                   # a modest base parameter-uncertainty sd on log-odds
        shift = rng.gauss(0.0, base_sd * (widen - 1.0)) if widen > 1.0 else 0.0
        p = hz.p(feats, logodds_shift=shift)
        return TransitionProposal(
            operator=self.name,
            action={"outcome_var": spec["outcome_var"], "p": p, "actor": spec.get("actor"),
                    "family": spec.get("family", ""), "pack_id": spec.get("pack_id", ""),
                    "options": spec.get("options") or ["True", "False"]},
            reason_codes=[f"feature_hazard:{spec.get('family', '')}", f"p={round(p, 4)}"],
            uncertainty={"pack_id": spec.get("pack_id", ""), "transport_widening": widen,
                         "logodds_shift": round(shift, 4)})

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        import random
        a = proposal.action
        var = a["outcome_var"]
        register_quantity_type(var, units="outcome")
        rng = random.Random(hash((world.branch_id, var)) & 0xFFFFFFFF)
        opts = a["options"] if len(a["options"]) == 2 else ["True", "False"]
        val = opts[0] if rng.random() < a["p"] else opts[1]
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
        # also mark the actor if one is bound (so downstream actor-view mechanisms see the outcome)
        if a.get("actor") and a["actor"] in (world.entities or {}):
            from swm.world_model_v2.state import F
            world.entity(a["actor"]).set("outcome", F(val, status="derived", method=self.name,
                                                      updated_at=world.clock.now), key=var)
        d = StateDelta(at=world.clock.now, event_type="outcome_hazard", operator=self.name,
                       reason_codes=proposal.reason_codes, uncertainty=proposal.uncertainty)
        d.change(f"quantities[{var}]", before, val)
        return d


register_operator("feature_hazard", FeatureHazardOperator(), requires=("entities",),
                  modifies=("quantities", "entities"), temporal_scale="event",
                  parameter_source="fitted feature-hazard pack (logistic MLE, train split); "
                                   "transport widening on log-odds uncertainty",
                  validated=True)
