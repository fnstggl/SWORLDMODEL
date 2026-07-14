"""Causal CONSUMERS for Phases 9 and 10 — the operators that make declared structure move the terminal.

The integration-completion audit found that populations, relations and institutions were EMITTED by the
compiler and materialized into the world, but nothing CONSUMED them: no operator aggregated population state,
no operator traversed the relation graph, and no institutional decision was ever executed against the outcome.
Removal of the whole section changed nothing — ornamental structure.

Three consumers close that gap. Each is a registered TransitionOperator with explicit broad-prior parameter
provenance; each writes a typed quantity via a StateDelta; each is scheduled ONLY when the relevance gate
(activation_synthesis) says the phase is causally required — never to decorate a manifest.

  * CollectiveThresholdDecisionOperator (Phase 10): members' yes-propensity is drawn from the evidence-updated
    POSTERIOR rate particles (the same base rate the terminal resolver would use), votes are drawn per particle,
    and the institution's DECLARED threshold/quorum rule transforms them into the outcome — the majoritarian
    sharpening around the threshold is the institution's real causal effect. It writes the canonical outcome
    quantity BEFORE resolve_outcome, so the generic safety net no-ops (domain mechanism takes precedence).
  * PopulationAggregationOperator (Phase 9): samples each declared segment's heterogeneous fields from their
    labeled prior distributions, weights by segment share, and writes the aggregate — consumed by the terminal
    later causal state-transition event.
  * NetworkDiffusionOperator (Phase 9): classifies declared relations into semantic layers
    (communication/exposure/trust/influence/authority/alliance), runs a per-particle independent-cascade
    percolation with layer-specific transmissibility priors, and writes the reach fraction — same channel.

None of these operators invents structure: they execute only what the compiler DECLARED.
"""
from __future__ import annotations

import hashlib
import math
import random

from swm.world_model_v2.state import StateField
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)

#: relation-name keywords → semantic layer; layer → transmissibility broad prior (mean, sd). Labeled priors,
#: not fitted values — the layer DIFFERENCE (trust spreads slower than exposure) is the structural claim.
_LAYER_KEYWORDS = {
    "communication": ("message", "communicat", "talk", "report", "brief"),
    "exposure": ("follow", "expos", "view", "audience", "broadcast", "media", "subscrib"),
    "trust": ("trust", "friend", "ally", "family", "confid"),
    "influence": ("influenc", "advis", "endors", "lead", "mentor"),
    "authority": ("authority", "supervis", "command", "oversee", "regulat", "govern", "appoint"),
    "alliance": ("alliance", "coalition", "partner", "member", "caucus"),
}
_LAYER_TRANSMISSIBILITY = {
    "communication": (0.35, 0.15), "exposure": (0.45, 0.15), "trust": (0.25, 0.10),
    "influence": (0.30, 0.12), "authority": (0.40, 0.15), "alliance": (0.30, 0.12),
    "generic": (0.30, 0.15),
}


def classify_layer(rel: str) -> str:
    r = str(rel or "").lower()
    for layer, keys in _LAYER_KEYWORDS.items():
        if any(k in r for k in keys):
            return layer
    return "generic"


def _branch_rng(world, salt: str) -> random.Random:
    """Deterministic per-branch stream (replayable across processes; hash() is process-randomized)."""
    seed = int.from_bytes(hashlib.sha256(f"{world.branch_id}|{salt}".encode()).digest()[:8], "big")
    return random.Random(seed)


def _draw_rate(particles, rng, default=0.5):
    """Weighted draw from posterior rate particles [(rate, weight)]; broad default when absent."""
    if not particles:
        return default
    total = sum(w for _, w in particles) or 1.0
    r, acc = rng.random() * total, 0.0
    for rate, w in particles:
        acc += w
        if r <= acc:
            return max(0.0, min(1.0, float(rate)))
    return max(0.0, min(1.0, float(particles[-1][0])))


# ---------------------------------------------------------------- Phase 10: collective threshold decision
class CollectiveThresholdDecisionOperator(TransitionOperator):
    """Execute a declared institution's threshold/quorum rule against posterior-informed member votes.

    payload = {institution_id, n_members, needed (count) | threshold_share, outcome_var, options,
               posterior_rate_particles?, lean?}. Per particle: propensity p ~ posterior particles (the SAME
    evidence-updated base rate the terminal resolver consumes — the institution transforms it, it does not
    invent it); yes ~ Binomial(n_members, p); passed = yes >= needed. Writes outcome_var = options[0] iff
    passed (affirmative-first contract), so the generic resolver's already-resolved no-op yields precedence
    to the institutional mechanism. parameter_source: declared rule numbers; propensity from posterior."""
    name = "institutional_decision"

    def applicable(self, world, event):
        return event.etype == "institutional_decision" and bool(event.payload.get("outcome_var"))

    def validate(self, world, proposal):
        # This operator IS the institutional execution — the rule it runs comes from the declared
        # institution itself, so re-validating against every RuleSystem (whose member lists are not world
        # entities) would spuriously block it.
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        p = event.payload
        n = max(1, int(p.get("n_members", 9) or 9))
        needed = p.get("needed")
        if needed is None:
            share = float(p.get("threshold_share", 0.5) or 0.5)
            needed = int(share * n) + 1                       # strictly-more-than-share convention
        return TransitionProposal(operator=self.name, action={
            "institution_id": str(p.get("institution_id", "")), "n_members": n, "needed": int(needed),
            "outcome_var": str(p["outcome_var"]), "options": list(p.get("options") or ["True", "False"]),
            "posterior_rate_particles": p.get("posterior_rate_particles"),
            "propensity_var": str(p.get("propensity_var", "")),
            "lean": str(p.get("lean", "neutral"))},
            reason_codes=[f"institution={p.get('institution_id', '?')}", f"rule={needed}/{n}"])

    def apply(self, world, proposal):
        from swm.world_model_v2.fallback import LEAN_BETA, _beta_sample
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        rng = _branch_rng(world, f"inst:{a['institution_id']}")
        post = a.get("posterior_rate_particles")
        causal_q = world.quantities.get(a.get("propensity_var", ""))
        causal_prop = getattr(causal_q, "value", None)
        if isinstance(causal_prop, (int, float)):
            prop, src = max(0.0, min(1.0, float(causal_prop))), "causal_state_transition"
        elif post:
            prop, src = _draw_rate(post, rng), "posterior"
        else:
            av, bv = LEAN_BETA.get(a["lean"], (1.0, 1.0))
            prop, src = _beta_sample(rng, av, bv), "prior_beta"
        yes = sum(1 for _ in range(a["n_members"]) if rng.random() < prop)
        passed = yes >= a["needed"]
        opts = a["options"] if len(a["options"]) == 2 else ["True", "False"]
        val = opts[0] if passed else opts[1]
        var = a["outcome_var"]
        register_quantity_type(var, units="outcome")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="institutional_decision", operator=self.name,
                       reason_codes=proposal.reason_codes,
                       uncertainty={"member_propensity": round(prop, 4), "propensity_source": src,
                                    "yes": yes, "needed": a["needed"], "n_members": a["n_members"],
                                    "note": "declared threshold rule over posterior-informed member votes "
                                            "(structural; member correlation not modeled — broad prior)"})
        d.change(f"institution[{a['institution_id']}].decision", None, "passed" if passed else "failed")
        return d.change(f"quantities[{var}]", before, val)


# ---------------------------------------------------------------- Phase 9: population aggregation
class PopulationAggregationOperator(TransitionOperator):
    """Aggregate a DECLARED population's segment states into one typed quantity the terminal consumes.

    payload = {population_id, out_var}. Per particle, each segment's heterogeneous fields are sampled from
    their labeled prior distributions (mean/sd), averaged within the segment, then weighted by the segment's
    normalized share. The aggregate (in [0,1]) is written to out_var; the terminal resolver consumes it
    through a later causal state-transition event. Removing heterogeneity (or the population) removes the
    per-particle variation and shifts the terminal — the causal pathway the audit found missing."""
    name = "population_aggregation"

    def applicable(self, world, event):
        return event.etype == "population_aggregation" and \
            str(event.payload.get("population_id", "")) in (world.populations or {})

    def propose(self, world, event, rng):
        p = event.payload
        return TransitionProposal(operator=self.name,
                                  action={"population_id": str(p["population_id"]),
                                          "out_var": str(p.get("out_var") or
                                                         f"population_aggregate:{p['population_id']}")},
                                  reason_codes=["population_consumer"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        pop = world.populations[a["population_id"]]
        rng = _branch_rng(world, f"pop:{a['population_id']}")
        weights = pop.normalized_weights()
        agg, seg_values = 0.0, {}
        for seg in pop.segments:
            vals = []
            for sf in (seg.heterogeneity or {}).values():
                d = (sf.dist or {}) if isinstance(sf, StateField) else {}
                mean, sd = float(d.get("mean", 0.5)), float(d.get("sd", 0.2))
                vals.append(min(1.0, max(0.0, rng.gauss(mean, sd))))
            seg_val = sum(vals) / len(vals) if vals else 0.5
            seg_values[seg.segment_id] = round(seg_val, 4)
            agg += weights.get(seg.segment_id, 0.0) * seg_val
        agg = min(1.0, max(0.0, agg))
        var = a["out_var"]
        register_quantity_type(var, units="share")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=round(agg, 4), timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="population_aggregation", operator=self.name,
                       reason_codes=proposal.reason_codes,
                       uncertainty={"segments": seg_values,
                                    "note": "segment heterogeneity sampled from labeled broad priors"})
        return d.change(f"quantities[{var}]", before, round(agg, 4))


# ---------------------------------------------------------------- Phase 6: structural process fallback
class StructuralProcessPriorOperator(TransitionOperator):
    """The strongest TRANSPARENT broad-prior fallback for a required causal process no validated registry
    family answers (Phase 6, Part 3): the process still executes through the shared event runtime — it is
    never silently omitted. Per particle it draws the process's contribution from a broad Beta prior
    (uniform-ish; labeled exploratory), writes a typed quantity the terminal consumes via the bounded
    state-transition path, and records the registry gap on the delta. This is honest ignorance made
    executable — not a fabricated fitted mechanism.

    payload = {process, out_var, lean?}."""
    name = "structural_process_prior"

    def applicable(self, world, event):
        return event.etype == "structural_process_prior" and bool(event.payload.get("out_var"))

    def propose(self, world, event, rng):
        p = event.payload
        return TransitionProposal(operator=self.name,
                                  action={"process": str(p.get("process", ""))[:60],
                                          "out_var": str(p["out_var"]),
                                          "lean": str(p.get("lean", "neutral"))},
                                  reason_codes=["registry_gap_fallback", "exploratory"])

    def apply(self, world, proposal):
        from swm.world_model_v2.fallback import LEAN_BETA, _beta_sample
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        rng = _branch_rng(world, f"proc:{a['process']}")
        av, bv = LEAN_BETA.get(a["lean"], (1.5, 1.5))
        val = round(_beta_sample(rng, av, bv), 4)
        var = a["out_var"]
        register_quantity_type(var, units="share")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="structural_process_prior", operator=self.name,
                       reason_codes=proposal.reason_codes,
                       uncertainty={"process": a["process"], "prior": f"Beta({av},{bv})",
                                    "note": "no validated registry family answers this required causal "
                                            "process — transparent broad-prior fallback (registry gap)"})
        return d.change(f"quantities[{var}]", before, val)


# ---------------------------------------------------------------- Phase 4: actor-action polarity consumer
#: action-name tokens that read as AFFIRMATIVE toward the outcome (the negative side reuses the compiler's
#: curated negation lexicon). Actions with no polarity (act/wait/monitor) are skipped — never guessed.
_AFFIRMATIVE_TOKENS = frozenset((
    "approve", "accept", "sign", "support", "agree", "confirm", "endorse", "yes", "vote_yes", "ratify",
    "advance", "pass", "adopt", "settle", "concede", "cooperate", "join", "commit", "proceed", "deal"))


def action_polarity(action_name: str):
    """+1 affirmative, -1 negative, 0 no polarity. Lexical only — reuses the compiler's negation lexicon."""
    from swm.world_model_v2.compiler import _negativity
    s = str(action_name or "").lower().replace("-", "_")
    if set(s.split("_")) & {"oppose", "block", "prevent", "reject", "veto"}:
        return -1
    if _negativity(s) > 0:
        return -1
    toks = set(s.split("_"))
    if toks & _AFFIRMATIVE_TOKENS:
        return 1
    return 0


class ActorActionPolarityOperator(TransitionOperator):
    """Aggregate the polarity of actors' CHOSEN actions into a typed quantity the terminal consumes.

    Phase 4's audit gap was the same as Phase 9's: decisions executed (current_action StateDeltas) but
    nothing downstream consumed them, so removing the decision operator never changed the terminal. This
    consumer reads every entity's current_action at aggregation time, classifies polarity lexically (the
    compiler's own negation lexicon + a curated affirmative set), and writes the affirmative share. Actions
    without polarity are SKIPPED (never guessed); if no polar action exists, nothing is written and the
    state-transition path receives no driver — an honest no-op, not a fabricated signal."""
    name = "actor_action_aggregation"

    def applicable(self, world, event):
        return event.etype == "actor_action_aggregation"

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name,
                                  action={"out_var": str(event.payload.get("out_var") or
                                                         "actor_action_share")},
                                  reason_codes=["actor_action_consumer"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        pos, neg, per_actor = 0, 0, {}
        for eid, ent in (world.entities or {}).items():
            act = ent.value("current_action", default=None)
            if isinstance(act, dict):
                act = act.get("action_name") or act.get("type")
            pol = action_polarity(act) if act else 0
            if pol > 0:
                pos += 1; per_actor[eid] = f"+{act}"
            elif pol < 0:
                neg += 1; per_actor[eid] = f"-{act}"
        if pos + neg == 0:
            return None                                       # no polar action — honest no-op
        share = pos / (pos + neg)
        var = proposal.action["out_var"]
        register_quantity_type(var, units="share")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=round(share, 4),
                                         timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="actor_action_aggregation", operator=self.name,
                       reason_codes=proposal.reason_codes,
                       uncertainty={"actors": per_actor,
                                    "note": "lexical polarity of chosen typed actions; nonpolar skipped"})
        return d.change(f"quantities[{var}]", before, round(share, 4))


# ---------------------------------------------------------------- Phase 9: multilayer network diffusion
class NetworkDiffusionOperator(TransitionOperator):
    """Percolate one independent cascade over the DECLARED relation graph, with layer-specific
    transmissibility priors, and write the terminal reach fraction.

    payload = {out_var, seed_id?}. Edges are classified into semantic layers by relation name; each layer's
    transmissibility is a labeled broad prior. Per particle: start from seed_id (or the highest-degree node),
    each edge transmits independently with its layer's sampled transmissibility, reach = fraction of nodes
    activated. Rewiring or removing a layer changes reach — the layer-specific causal consumer."""
    name = "network_diffusion"

    def applicable(self, world, event):
        return event.etype == "network_diffusion" and world.network is not None \
            and bool(getattr(world.network, "edges", None))

    def propose(self, world, event, rng):
        p = event.payload
        return TransitionProposal(operator=self.name,
                                  action={"out_var": str(p.get("out_var") or "network_diffusion_reach"),
                                          "seed_id": p.get("seed_id")},
                                  reason_codes=["network_consumer"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        rng = _branch_rng(world, "net:diffusion")
        edges = list(world.network.edges)
        nodes, adj = set(), {}
        layer_used = {}
        for e in edges:
            layer = classify_layer(e.rel)
            mean, sd = _LAYER_TRANSMISSIBILITY[layer]
            t = min(0.95, max(0.02, rng.gauss(mean, sd)))
            layer_used[layer] = layer_used.get(layer, 0) + 1
            nodes.add(e.src); nodes.add(e.dst)
            adj.setdefault(e.src, []).append((e.dst, t))
            adj.setdefault(e.dst, []).append((e.src, t * 0.6))   # weaker reverse transmission
        if not nodes:
            return None
        seed = a.get("seed_id")
        if seed not in nodes:
            seed = max(nodes, key=lambda n: len(adj.get(n, [])))
        active, frontier = {seed}, [seed]
        while frontier:
            nxt = []
            for u in frontier:
                for v, t in adj.get(u, []):
                    if v not in active and rng.random() < t:
                        active.add(v); nxt.append(v)
            frontier = nxt
        reach = len(active) / len(nodes)
        var = a["out_var"]
        register_quantity_type(var, units="share")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=round(reach, 4),
                                         timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="network_diffusion", operator=self.name,
                       reason_codes=proposal.reason_codes,
                       uncertainty={"layers": layer_used, "n_nodes": len(nodes),
                                    "note": "layer transmissibility from labeled broad priors"})
        return d.change(f"quantities[{var}]", before, round(reach, 4))


# ---------------------------------------------------------------- Cross-phase state path
class CausalStateTransitionOperator(TransitionOperator):
    """Consume phase state into a typed causal propensity.

    This replaces the prohibited ``resolve_outcome.rate_modulation`` shortcut.
    The operator reads only quantities written by earlier causal events.  It
    combines the evidence-updated base propensity and available drivers with an
    equal-weight logarithmic opinion pool: no phase receives a hand-tuned nudge.
    The result is another WorldState quantity, not a forecast.
    """
    name = "causal_state_transition"

    def applicable(self, world, event):
        return event.etype == "causal_state_transition" and bool(event.payload.get("out_var"))

    def propose(self, world, event, rng):
        p = event.payload
        return TransitionProposal(operator=self.name, action={
            "driver_vars": list(p.get("driver_vars") or []),
            "out_var": str(p["out_var"]), "lean": str(p.get("lean", "neutral")),
            "posterior_rate_particles": p.get("posterior_rate_particles")},
            reason_codes=["phase_state_to_causal_state", "equal_weight_log_opinion_pool"])

    @staticmethod
    def _logit(value):
        value = min(1.0 - 1e-6, max(1e-6, float(value)))
        return math.log(value / (1.0 - value))

    def apply(self, world, proposal):
        from swm.world_model_v2.fallback import LEAN_BETA, _beta_sample
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        rng = _branch_rng(world, "causal:state")
        post = a.get("posterior_rate_particles")
        if post:
            base, base_source = _draw_rate(post, rng), "posterior"
        else:
            av, bv = LEAN_BETA.get(a["lean"], (1.0, 1.0))
            base, base_source = _beta_sample(rng, av, bv), "broad_prior"
        logits = [self._logit(base)]
        used = []
        for driver in a["driver_vars"]:
            var = str(driver.get("var", ""))
            q = world.quantities.get(var)
            value = getattr(q, "value", None)
            if not isinstance(value, (int, float)):
                continue
            value = max(0.0, min(1.0, float(value)))
            if float(driver.get("direction", 1) or 1) < 0:
                value = 1.0 - value
            # One simulated mechanism realization is weak evidence, not a
            # certainty.  Beta(1,1) prior-predictive smoothing gives it one
            # effective observation: (1 + value) / 3.  This prevents a binary
            # upstream state from swamping the posterior or an institution's
            # declared threshold.
            smoothed = (1.0 + value) / 3.0
            logits.append(self._logit(smoothed))
            used.append({"var": var, "phase": driver.get("phase"), "value": round(value, 4),
                         "prior_smoothed_value": round(smoothed, 4)})
        pooled_logit = sum(logits) / len(logits)
        propensity = 1.0 / (1.0 + math.exp(-pooled_logit))
        var = a["out_var"]
        register_quantity_type(var, units="share")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=round(propensity, 6),
                                         timestamp=world.clock.now)
        delta = StateDelta(at=world.clock.now, event_type="causal_state_transition", operator=self.name,
                           reason_codes=proposal.reason_codes,
                           uncertainty={"base_source": base_source, "base": round(base, 4),
                                        "drivers_consumed": used,
                                        "combiner": "Beta(1,1) one-observation smoothing then "
                                                    "equal-weight logarithmic opinion pool"})
        return delta.change(f"quantities[{var}]", before, round(propensity, 6))


class CausalOutcomeTransitionOperator(TransitionOperator):
    """Resolve terminal WorldState from the preceding causal propensity state."""
    name = "causal_outcome_transition"

    def applicable(self, world, event):
        return event.etype == "causal_outcome_transition" and bool(event.payload.get("outcome_var"))

    def propose(self, world, event, rng):
        p = event.payload
        return TransitionProposal(operator=self.name, action={
            "propensity_var": str(p.get("propensity_var", "causal_outcome_propensity")),
            "outcome_var": str(p["outcome_var"]),
            "options": list(p.get("options") or ["True", "False"])},
            reason_codes=["causal_state_to_terminal_world_state"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        existing = world.quantities.get(a["outcome_var"])
        before = getattr(existing, "value", None)
        if before is not None:
            return StateDelta(at=world.clock.now, event_type="causal_outcome_transition",
                              operator=self.name,
                              reason_codes=["already_resolved_by_domain_mechanism_noop"])
        q = world.quantities.get(a["propensity_var"])
        propensity = getattr(q, "value", None)
        if not isinstance(propensity, (int, float)):
            return None
        rng = _branch_rng(world, "causal:outcome")
        opts = a["options"] if len(a["options"]) == 2 else ["True", "False"]
        value = opts[0] if rng.random() < max(0.0, min(1.0, float(propensity))) else opts[1]
        var = a["outcome_var"]
        register_quantity_type(var, units="outcome")
        world.quantities[var] = Quantity(name=var, qtype=var, value=value, timestamp=world.clock.now)
        delta = StateDelta(at=world.clock.now, event_type="causal_outcome_transition", operator=self.name,
                           reason_codes=proposal.reason_codes,
                           uncertainty={"propensity_var": a["propensity_var"],
                                        "propensity": round(float(propensity), 6)})
        return delta.change(f"quantities[{var}]", before, value)


register_operator("institutional_decision", CollectiveThresholdDecisionOperator(),
                  requires=("institutions", "quantities"), modifies=("quantities",),
                  temporal_scale="scheduled",
                  parameter_source="declared threshold/quorum rule numbers; member propensity from the "
                                   "evidence-updated posterior (broad prior when absent)", validated=True)
register_operator("population_aggregation", PopulationAggregationOperator(),
                  requires=("populations",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="declared segment weights; heterogeneity from labeled broad priors",
                  validated=True)
register_operator("structural_process_prior", StructuralProcessPriorOperator(),
                  requires=("quantities",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="broad Beta prior; EXPLORATORY registry-gap fallback (labeled)",
                  validated=True)
register_operator("actor_action_aggregation", ActorActionPolarityOperator(),
                  requires=("entities",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="lexical polarity of chosen typed actions (nonpolar actions skipped)",
                  validated=True)
register_operator("network_diffusion", NetworkDiffusionOperator(),
                  requires=("network",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="declared relation graph; layer transmissibility broad priors",
                  validated=True)
register_operator("causal_state_transition", CausalStateTransitionOperator(),
                  requires=("quantities",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="equal-weight log opinion pool over posterior base and phase state",
                  validated=True)
register_operator("causal_outcome_transition", CausalOutcomeTransitionOperator(),
                  requires=("quantities",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="causal propensity WorldState; deterministic branch RNG",
                  validated=True)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et, _reads, _deltas in (("institutional_decision", ("institutions", "quantities"), ("quantities",)),
                             ("structural_process_prior", ("quantities",), ("quantities",)),
                             ("population_aggregation", ("populations",), ("quantities",)),
                             ("actor_action_aggregation", ("entities",), ("quantities",)),
                             ("network_diffusion", ("network",), ("quantities",)),
                             ("causal_state_transition", ("quantities",), ("quantities",)),
                             ("causal_outcome_transition", ("quantities",), ("quantities",))):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=_reads, deltas=_deltas,
                            parameter_source="activation synthesis from declared plan structure",
                            validated=True)
