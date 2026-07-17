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
    resolver through the bounded rate-modulation channel.
  * NetworkDiffusionOperator (Phase 9): classifies declared relations into semantic layers
    (communication/exposure/trust/influence/authority/alliance), runs a per-particle independent-cascade
    percolation with layer-specific transmissibility priors, and writes the reach fraction — same channel.

None of these operators invents structure: they execute only what the compiler DECLARED.
"""
from __future__ import annotations

import hashlib
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
            "consume": list(p.get("consume") or []),
            "absorbing": bool(p.get("absorbing")),
            "absorbing_mode": str(p.get("absorbing_mode", "")),
            "lean": str(p.get("lean", "neutral"))},
            reason_codes=[f"institution={p.get('institution_id', '?')}", f"rule={needed}/{n}"])

    def apply(self, world, proposal):
        from swm.world_model_v2.fallback import LEAN_BETA, _beta_sample
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        rng = _branch_rng(world, f"inst:{a['institution_id']}")
        post = a.get("posterior_rate_particles")
        if post:
            prop, src = _draw_rate(post, rng), "posterior"
        else:
            av, bv = LEAN_BETA.get(a["lean"], (1.0, 1.0))
            prop, src = _beta_sample(rng, av, bv), "prior_beta"
        # the institution CONSUMES upstream causal state: members respond to the population aggregate,
        # actor-action polarity, diffusion reach and nonlinear momentum written into this world (bounded,
        # inside the mechanism — never at the terminal resolver)
        prop, consumed = consume_state_rate(world, prop, a.get("consume") or [])
        if len(a["options"]) != 2:
            return None                                      # institutional YES/NO write only fits binary
        yes = sum(1 for _ in range(a["n_members"]) if rng.random() < prop)
        passed = yes >= a["needed"]
        unc = {"member_propensity": round(prop, 4), "propensity_source": src,
               "consumed_state": consumed,
               "yes": yes, "needed": a["needed"], "n_members": a["n_members"],
               "note": "declared threshold rule over posterior-informed member votes "
                       "(structural; member correlation not modeled — broad prior)"}
        if a.get("absorbing"):
            # EVENT-TIME semantics: the institution's vote is a real event in the trajectory. Passing
            # WRITES the absorbing state at this vote's date (the monitor observes first passage); failing
            # leaves the world unabsorbed (censored ⇒ the negative option at readout). The institution
            # never declares the answer variable — the answer is read out of the trajectory.
            stamped = world.quantities.get("absorbed_at")
            flag = world.quantities.get("absorbing_state_reached")
            if getattr(stamped, "value", None) not in (None, 0) or getattr(flag, "value", None):
                return None                                   # first passage already happened
            d = StateDelta(at=world.clock.now, event_type="institutional_decision", operator=self.name,
                           reason_codes=proposal.reason_codes + ["absorbing_writer"], uncertainty=unc)
            d.change(f"institution[{a['institution_id']}].decision", None,
                     "passed" if passed else "failed")
            if passed:
                mode = a.get("absorbing_mode") or f"institutional:{a['institution_id']}"
                register_quantity_type("absorbing_state_reached", units="bool")
                register_quantity_type("absorbing_mode", units="mode")
                world.quantities["absorbing_state_reached"] = Quantity(
                    name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
                    timestamp=world.clock.now)
                world.quantities["absorbing_mode"] = Quantity(
                    name="absorbing_mode", qtype="absorbing_mode", value=str(mode),
                    timestamp=world.clock.now)
                d.change("quantities[absorbing_state_reached]", None, True)
            return d
        val = a["options"][0] if passed else a["options"][1]
        var = a["outcome_var"]
        register_quantity_type(var, units="outcome")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="institutional_decision", operator=self.name,
                       reason_codes=proposal.reason_codes, uncertainty=unc)
        d.change(f"institution[{a['institution_id']}].decision", None, "passed" if passed else "failed")
        return d.change(f"quantities[{var}]", before, val)


# ---------------------------------------------------------------- Phase 9: population aggregation
class PopulationAggregationOperator(TransitionOperator):
    """Aggregate a DECLARED population's segment states into one typed quantity the terminal consumes.

    payload = {population_id, out_var}. Per particle, each segment's heterogeneous fields are sampled from
    their labeled prior distributions (mean/sd), averaged within the segment, then weighted by the segment's
    normalized share. The aggregate (in [0,1]) is written to out_var; the terminal resolver consumes it
    through the bounded rate-modulation channel. Removing heterogeneity (or the population) removes the
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


# ---------------------------------------------------------------- aggregate outcome mechanism
#: bounded weight each consumed state variable may carry inside a consuming MECHANISM (never at the
#: terminal resolver — Part 4 prohibits resolver-level probability modulation).
MAX_CONSUMED_STATE_WEIGHT = 0.45


def consume_state_rate(world, base_p: float, consume: list) -> tuple:
    """A MECHANISM's bounded consumption of upstream causal state: p' = (1-Σw)·p + Σ wᵢ·vᵢ over quantities
    that upstream operators (population aggregation, network diffusion, actor polarity, nonlinear state)
    actually wrote into THIS world's state. Unwritten variables contribute nothing. An entry may set
    `invert: true` (survival-polarity event-time chains: pro-YES state must SUPPRESS the state-breaking
    hazard, so v ↦ 1−v). This runs inside a consuming operator's apply() — the terminal resolver never
    calls it."""
    used, blended, total = [], 0.0, 0.0
    for m in (consume or []):
        var, w = str(m.get("var", "")), float(m.get("weight", 0.0) or 0.0)
        q = world.quantities.get(var)
        v = getattr(q, "value", None)
        if w <= 0.0 or not isinstance(v, (int, float)):
            continue
        v = max(0.0, min(1.0, float(v)))
        if m.get("invert"):
            v = 1.0 - v
        blended += w * v
        total += w
        used.append(var)
    if not used:
        return base_p, []
    if total > MAX_CONSUMED_STATE_WEIGHT:
        blended *= MAX_CONSUMED_STATE_WEIGHT / total
        total = MAX_CONSUMED_STATE_WEIGHT
    return (1.0 - total) * base_p + blended, used


class AggregateOutcomeOperator(TransitionOperator):
    """The aggregate-behavior REALIZATION mechanism: when the outcome the question asks about IS an
    aggregate-behavior event (adoption/turnout/diffusion/momentum reaching what was asked) and no
    institutional procedure decides it, this mechanism — not the terminal resolver — realizes the outcome
    from the causal state upstream consumers wrote (population aggregate, diffusion reach, actor-action
    polarity, nonlinear state), blended with the evidence-updated posterior base rate, inside the event
    loop, with a StateDelta naming exactly which state it consumed. The generic resolver then no-ops
    (already-resolved precedence). payload = {outcome_var, options, lean, consume:[{var,weight}],
    posterior_rate_particles?}."""
    name = "aggregate_outcome_mechanism"

    def applicable(self, world, event):
        return event.etype == "aggregate_outcome_resolution" and bool(event.payload.get("outcome_var"))

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        p = event.payload
        return TransitionProposal(operator=self.name, action={
            "outcome_var": str(p["outcome_var"]), "options": list(p.get("options") or ["True", "False"]),
            "lean": str(p.get("lean", "neutral")), "consume": list(p.get("consume") or []),
            "fitted_base_rate": p.get("fitted_base_rate"),
            "base_rate_provenance": p.get("base_rate_provenance"),
            "posterior_rate_particles": p.get("posterior_rate_particles")},
            reason_codes=["aggregate_realization"])

    def apply(self, world, proposal):
        from swm.world_model_v2.fallback import LEAN_BETA, _beta_sample
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        var = a["outcome_var"]
        if var in world.quantities and world.quantities[var].value is not None:
            return None                                      # a stronger domain mechanism already resolved it
        rng = _branch_rng(world, "aggregate_outcome")
        post = a.get("posterior_rate_particles")
        if post:
            base, src = _draw_rate(post, rng), "posterior"
        elif isinstance(a.get("fitted_base_rate"), (int, float)):
            # learned family hazard (fit on calibration-split outcomes only; partial pooling)
            base, src = float(a["fitted_base_rate"]), str(a.get("base_rate_provenance", "fitted_family_prior"))
        else:
            av, bv = LEAN_BETA.get(a["lean"], (1.0, 1.0))
            base, src = _beta_sample(rng, av, bv), "prior_beta"
        p, used = consume_state_rate(world, base, a["consume"])
        if not used:
            return None                                      # no upstream state written → nothing to realize
        if len(a["options"]) != 2:
            return None                                      # non-binary contract: never poison the option
                                                             # space with binary values (readout must bin)
        opts = a["options"]
        val = opts[0] if rng.random() < p else opts[1]
        register_quantity_type(var, units="outcome")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="aggregate_outcome_resolution", operator=self.name,
                       reason_codes=proposal.reason_codes + [f"consumed:{','.join(used)}"],
                       uncertainty={"base_rate_source": src, "realized_rate": round(p, 4),
                                    "consumed_state": used})
        return d.change(f"quantities[{var}]", before, val)


# ---------------------------------------------------------------- Phase 6: structural process fallback
class StructuralProcessPriorOperator(TransitionOperator):
    """The strongest TRANSPARENT broad-prior fallback for a required causal process no validated registry
    family answers (Phase 6, Part 3): the process still executes through the shared event runtime — it is
    never silently omitted. Per particle it draws the process's contribution from a broad Beta prior
    (uniform-ish; labeled exploratory), writes a typed quantity the terminal consumes via the bounded
    modulation channel, and records the registry gap on the delta. This is honest ignorance made
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
    modulation channel ignores the var — an honest no-op, not a fabricated signal."""
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


register_operator("institutional_decision", CollectiveThresholdDecisionOperator(),
                  requires=("institutions", "quantities"), modifies=("quantities",),
                  temporal_scale="scheduled",
                  parameter_source="declared threshold/quorum rule numbers; member propensity from the "
                                   "evidence-updated posterior (broad prior when absent)", validated=True)
register_operator("population_aggregation", PopulationAggregationOperator(),
                  requires=("populations",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="declared segment weights; heterogeneity from labeled broad priors",
                  validated=True)
register_operator("aggregate_outcome_mechanism", AggregateOutcomeOperator(),
                  requires=("quantities",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="posterior base rate + bounded consumption of upstream causal state "
                                   "(inside the mechanism; the terminal resolver never modulates)",
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

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et, _reads, _deltas in (("institutional_decision", ("institutions", "quantities"), ("quantities",)),
                             ("aggregate_outcome_resolution", ("quantities",), ("quantities",)),
                             ("structural_process_prior", ("quantities",), ("quantities",)),
                             ("population_aggregation", ("populations",), ("quantities",)),
                             ("actor_action_aggregation", ("entities",), ("quantities",)),
                             ("network_diffusion", ("network",), ("quantities",))):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=_reads, deltas=_deltas,
                            parameter_source="activation synthesis from declared plan structure",
                            validated=True)
