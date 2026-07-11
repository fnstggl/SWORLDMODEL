"""Transition operators — Phase 4. Every change to the world is a validated, machine-readable StateDelta.

The contract: applicable(world, event) → propose(world, event, rng) → validate(world, proposal) →
apply(world, proposal) → StateDelta. No operator may replace the world with prose. The LLM participates ONLY
inside AgentDecisionOperator's policy — and there it chooses among TYPED, institution-validated actions and
returns a distribution; it cannot mutate state, invent actions outside the valid set, or emit coefficients.
New operator families register through `register_operator` with required/modified state, temporal scale,
parameter source, invariants, and validation status — an unsupported transition is refused, never narrated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.state import F, StateField, rfc3339

_OPERATORS: dict = {}


def register_operator(name: str, operator, *, requires: tuple = (), modifies: tuple = (),
                      temporal_scale: str = "", parameter_source: str = "", invariants: tuple = (),
                      validated: bool = False, experimental: bool = False):
    """Controlled operator registry. `experimental=True` marks compiler-proposed mechanisms that have NOT
    earned validation — they may run only when the plan explicitly enables experimental mechanisms."""
    _OPERATORS[name] = {"operator": operator, "requires": requires, "modifies": modifies,
                        "temporal_scale": temporal_scale, "parameter_source": parameter_source,
                        "invariants": invariants, "validated": validated, "experimental": experimental}
    return name


def get_operator(name: str, *, allow_experimental=False):
    meta = _OPERATORS.get(name)
    if meta is None:
        raise KeyError(f"no operator {name!r} in the registry — an unsupported transition is refused, "
                       f"never narrated (known: {sorted(_OPERATORS)})")
    if meta["experimental"] and not allow_experimental:
        raise PermissionError(f"operator {name!r} is EXPERIMENTAL (unvalidated) — the plan must explicitly "
                              f"enable experimental mechanisms to run it")
    return meta["operator"]


@dataclass
class StateDelta:
    """The machine-readable record of one transition: exactly which fields changed, from what, to what, why."""
    at: float
    event_type: str
    operator: str
    changes: list = field(default_factory=list)   # [{path, before, after}] — typed values, not prose
    reason_codes: list = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)
    evidence_deps: list = field(default_factory=list)

    def change(self, path: str, before, after):
        self.changes.append({"path": path, "before": before, "after": after})
        return self

    def as_dict(self):
        return {"at": rfc3339(self.at), "event_type": self.event_type, "operator": self.operator,
                "changes": self.changes, "reason_codes": self.reason_codes,
                "uncertainty": self.uncertainty, "evidence_deps": self.evidence_deps}


@dataclass
class TransitionProposal:
    operator: str
    action: dict = field(default_factory=dict)     # the typed action / update to apply
    p_dist: dict = field(default_factory=dict)     # probability distribution over typed actions (decision ops)
    reason_codes: list = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)
    evidence_deps: list = field(default_factory=list)


@dataclass
class ValidationResult:
    ok: bool
    reasons: list = field(default_factory=list)


class TransitionOperator:
    """The common interface (Phase 4). Subclasses implement the four methods; `run` wires them with
    institutional validation so an invalid action can NEVER apply."""
    name = "abstract"

    def applicable(self, world, event) -> bool:
        raise NotImplementedError

    def propose(self, world, event, rng) -> TransitionProposal:
        raise NotImplementedError

    def validate(self, world, proposal) -> ValidationResult:
        # default: every institution's executable rules must admit the action
        action = proposal.action or {}
        if action:
            for inst in (world.institutions or {}).values():
                ok, reasons = inst.validate_action(world, action)
                if not ok:
                    return ValidationResult(ok=False, reasons=reasons)
        return ValidationResult(ok=True)

    def apply(self, world, proposal) -> StateDelta:
        raise NotImplementedError

    def run(self, world, event, rng):
        """propose → validate → apply. Returns (StateDelta|None, ValidationResult)."""
        proposal = self.propose(world, event, rng)
        if proposal is None:
            return None, ValidationResult(ok=True, reasons=["no-op"])
        vr = self.validate(world, proposal)
        if not vr.ok:
            return None, vr
        return self.apply(world, proposal), vr


# ---------------------------------------------------------------- A. agent decision
def observable_view(world, actor_id: str) -> dict:
    """THE LLM DECISION BOUNDARY's state selector: only what THIS actor can know. Excludes other actors'
    private fields, latent ground truth (items the actor was never exposed to), future queue events, anything
    post-as-of, and simulator metadata (particle weights, sampled-latent records). The policy prompt renders
    ONLY this view — the world is never serialized wholesale into an agent prompt."""
    actor = world.entity(actor_id)
    own = {}
    for fname, sf in actor.fields.items():
        if isinstance(sf, dict):
            own[fname] = {k: v.value for k, v in sf.items() if isinstance(v, StateField)}
        elif isinstance(sf, StateField):
            own[fname] = sf.value
    info = world.information.visible_to(actor_id, at=world.clock.now) if world.information else []
    rels = [{"rel": e.rel, "with": (e.dst if e.src == actor_id else e.src)}
            for e in ((world.network.out_edges(actor_id) + world.network.in_edges(actor_id))
                      if world.network else [])]
    # PUBLIC quantities only (visibility of quantities is public by default; latent measurement state is not
    # exposed — measurement operators publish observations as information items instead)
    return {"self": own,
            "observed_information": [{"content": it.content, "source": it.source,
                                      "credibility": it.credibility, "salience": round(e.salience, 2)}
                                     for it, e in info],
            "relationships": rels[:12],
            "now": world.clock.now_rfc3339()}


_DECIDE_PROMPT = """You are the decision POLICY for one actor in a structured world simulation. You choose among
TYPED actions only; you cannot invent actions or change the world directly.
ACTOR: {identity} ({etype})
GOALS: {goals}
RESOURCES: {resources}
COMMITMENTS: {commitments}
ATTENTION (0-1): {attention}
RELATIONSHIPS (typed edges): {relationships}
WHAT THIS ACTOR HAS ACTUALLY OBSERVED (their information set — others may know different things):
{information}
SITUATION/EVENT at {now}: {situation}
VALID ACTIONS (institution-checked): {actions}
Return ONLY JSON: {{"p": {{"<action>": <0..1>, ...}}, "reasons": ["<code>", ...]}}"""


class AgentDecisionOperator(TransitionOperator):
    """The LLM as policy over TYPED actions: reads the actor's typed state + their OWN information set,
    emits a distribution over the institution-valid action set; the sampled action becomes a typed delta.
    With llm=None a uniform policy runs (offline/testing) — the machinery is identical."""
    name = "agent_decision"

    def __init__(self, llm=None):
        self.llm = llm

    def applicable(self, world, event):
        return event.etype == "decision_opportunity" and bool(event.participants)

    def _policy(self, world, actor, actions, event):
        if self.llm is None:
            return {a["type"]: 1.0 / len(actions) for a in actions}, ["uniform_policy"]
        from swm.engine.grounding import parse_json
        view = observable_view(world, actor.identity)      # the boundary: ONLY what this actor can know
        raw = parse_json(self.llm(_DECIDE_PROMPT.format(
            identity=actor.identity, etype=actor.entity_type,
            goals=view["self"].get("goals", "-"),
            resources=view["self"].get("resources", "-"),
            commitments=view["self"].get("commitments", "-"),
            attention=view["self"].get("attention", 0.7),
            relationships=[f"{r['rel']}->{r['with']}" for r in view["relationships"]][:8],
            information="\n".join(f"- [{i['source']} cred={i['credibility']}] {i['content']}"
                                  for i in view["observed_information"][:10]) or "- (nothing observed)",
            now=view["now"], situation=event.payload.get("situation", event.etype),
            actions=[a["type"] for a in actions]))) or {}
        p = raw.get("p") or {}
        dist = {a["type"]: max(0.0, float(p.get(a["type"], 0.0) or 0.0)) for a in actions}
        z = sum(dist.values())
        if z <= 0:
            return {a["type"]: 1.0 / len(actions) for a in actions}, ["policy_parse_fallback"]
        return {k: v / z for k, v in dist.items()}, [str(r)[:40] for r in (raw.get("reasons") or [])][:4]

    def propose(self, world, event, rng):
        actor = world.entity(event.participants[0])
        candidates = event.payload.get("actions") or [{"type": "act"}, {"type": "wait"}]
        # institutional pre-filter: only actions every rule system admits are shown to the policy
        valid = []
        for a in candidates:
            a = {**a, "actor": actor.identity}
            if all(inst.validate_action(world, a)[0] for inst in (world.institutions or {}).values()):
                valid.append(a)
        if not valid:
            return None
        dist, reasons = self._policy(world, actor, valid, event)
        pick = rng.random()
        acc, chosen = 0.0, valid[-1]
        for a in valid:
            acc += dist[a["type"]]
            if pick <= acc:
                chosen = a
                break
        return TransitionProposal(operator=self.name, action=chosen, p_dist=dist, reason_codes=reasons)

    def apply(self, world, proposal):
        actor = world.entity(proposal.action["actor"])
        before = actor.value("current_action")
        actor.set("current_action", F(proposal.action["type"], status="derived",
                                      method=self.name, updated_at=world.clock.now))
        past = actor.get("past_actions")
        hist = list(past.value) if isinstance(past, StateField) and isinstance(past.value, list) else []
        hist.append({"at": world.clock.now, "action": proposal.action["type"]})
        actor.set("past_actions", F(hist, status="derived", method=self.name, updated_at=world.clock.now))
        d = StateDelta(at=world.clock.now, event_type="decision", operator=self.name,
                       reason_codes=proposal.reason_codes, uncertainty={"p_dist": proposal.p_dist})
        d.change(f"{actor.identity}.current_action", before, proposal.action["type"])
        return d


# ---------------------------------------------------------------- B. belief update
class BeliefUpdateOperator(TransitionOperator):
    """Bounded Bayesian-ish belief shift on exposure: prior (the actor's typed belief) moves toward the
    item's claim with step ∝ source credibility × edge trust × salience, damped by incompatibility. A rule
    core (auditable, ablatable); an LLM may LATER refine likelihoods, but never free-rewrite beliefs."""
    name = "belief_update"

    def applicable(self, world, event):
        return event.etype == "exposure" and bool(event.participants) and "item_id" in event.payload

    def propose(self, world, event, rng):
        item = world.information.items.get(event.payload["item_id"])
        if item is None:
            return None
        return TransitionProposal(operator=self.name,
                                  action={"actor": event.participants[0], "item_id": item.item_id,
                                          "claim": item.about or item.content, "credibility": item.credibility,
                                          "trust": float(event.payload.get("trust", 0.6)),
                                          "salience": float(event.payload.get("salience", 0.6))},
                                  evidence_deps=[item.item_id])

    def apply(self, world, proposal):
        a = proposal.action
        actor = world.entity(a["actor"])
        beliefs = actor.get("beliefs") or {}
        key = a["claim"][:80]
        cur = beliefs.get(key)
        prior = float(cur.value) if isinstance(cur, StateField) and isinstance(cur.value, (int, float)) else 0.5
        step = 0.9 * a["credibility"] * a["trust"] * a["salience"]
        compat = 1.0 - 0.5 * abs(prior - 0.5) * 2 * (1 if step else 0)   # entrenched priors move less
        post = min(0.98, max(0.02, prior + (0.95 - prior) * step * compat * 0.5))
        actor.set("beliefs", F(post, status="derived", method=self.name, confidence=a["credibility"],
                               updated_at=world.clock.now), key=key)
        d = StateDelta(at=world.clock.now, event_type="belief_update", operator=self.name,
                       evidence_deps=proposal.evidence_deps,
                       uncertainty={"step": round(step, 3)})
        d.change(f"{a['actor']}.beliefs[{key}]", round(prior, 3), round(post, 3))
        return d


# ---------------------------------------------------------------- C. relationship update
class RelationshipUpdateOperator(TransitionOperator):
    """Bounded typed shifts on an edge's strength/trust after an action (reciprocity, hostility, obligation).
    Payload: {src, rel, dst, dimension: strength|trust, shift: bounded float}."""
    name = "relationship_update"
    MAX_SHIFT = 0.25                       # no unconstrained narrative jumps

    def applicable(self, world, event):
        return event.etype in ("relationship_effect",) or "relationship_shift" in event.payload

    def propose(self, world, event, rng):
        s = event.payload.get("relationship_shift")
        if not s:
            return None
        return TransitionProposal(operator=self.name, action=dict(s))

    def apply(self, world, proposal):
        a = proposal.action
        e = world.network.edge(a["src"], a["rel"], a["dst"])
        if e is None:
            e = world.network.add(a["src"], a["rel"], a["dst"])
        shift = max(-self.MAX_SHIFT, min(self.MAX_SHIFT, float(a.get("shift", 0.0))))
        dim = a.get("dimension", "strength")
        if dim == "trust":
            before = e.trust if e.trust is not None else 0.5
            e.trust = min(1.0, max(0.0, before + shift))
            after = e.trust
        else:
            before = float(e.strength.value or 0.5)
            e.strength.value = min(1.0, max(0.0, before + shift))
            after = e.strength.value
        d = StateDelta(at=world.clock.now, event_type="relationship_update", operator=self.name)
        d.change(f"edge({a['src']},{a['rel']},{a['dst']}).{dim}", round(before, 3), round(after, 3))
        return d


# ---------------------------------------------------------------- D. resources & commitments
class ResourceUpdateOperator(TransitionOperator):
    """Conservation-checked resource consumption/creation. Payload: {actor, resource, delta, floor=0}."""
    name = "resource_update"

    def applicable(self, world, event):
        return "resource_delta" in event.payload

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload["resource_delta"]))

    def validate(self, world, proposal):
        a = proposal.action
        actor = world.entities.get(a["actor"])
        cur = actor.get("resources", key=a["resource"]) if actor else None
        have = float(cur.value) if isinstance(cur, StateField) and isinstance(cur.value, (int, float)) else 0.0
        if have + float(a["delta"]) < float(a.get("floor", 0.0)):
            return ValidationResult(ok=False, reasons=[f"insufficient {a['resource']}: {have} + {a['delta']} "
                                                       f"< floor {a.get('floor', 0.0)}"])
        return super().validate(world, proposal)

    def apply(self, world, proposal):
        a = proposal.action
        actor = world.entity(a["actor"])
        cur = actor.get("resources", key=a["resource"])
        before = float(cur.value) if isinstance(cur, StateField) else 0.0
        after = before + float(a["delta"])
        actor.set("resources", F(after, status="derived", method=self.name,
                                 updated_at=world.clock.now), key=a["resource"])
        d = StateDelta(at=world.clock.now, event_type="resource_update", operator=self.name)
        d.change(f"{a['actor']}.resources[{a['resource']}]", before, after)
        return d


# ---------------------------------------------------------------- F. institutional execution
class InstitutionalVoteOperator(TransitionOperator):
    """Deterministic collective-decision execution: gathers each participant's current_action as a vote,
    runs the institution's voting rule, writes the typed outcome quantity."""
    name = "institutional_vote"

    def applicable(self, world, event):
        return event.etype == "collective_vote"

    def propose(self, world, event, rng):
        votes = {}
        for pid in event.participants:
            act = world.entity(pid).value("current_action", default="abstain")
            votes[pid] = act if act in ("yes", "no", "abstain") else "abstain"
        return TransitionProposal(operator=self.name,
                                  action={"votes": votes, **{k: event.payload.get(k)
                                                             for k in ("threshold", "needed", "total",
                                                                       "institution", "outcome_var")}})

    def apply(self, world, proposal):
        a = proposal.action
        inst = world.institutions.get(a.get("institution")) or next(iter(world.institutions.values()), None)
        from swm.world_model_v2.institutions import RuleSystem
        rs = inst if inst is not None else RuleSystem(institution_id="adhoc")
        res = rs.run_vote(a["votes"], threshold=a.get("threshold"), needed=a.get("needed"),
                          total=a.get("total"))
        var = a.get("outcome_var", "vote_outcome")
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        register_quantity_type(var, units="bool")
        before = world.quantities.get(var).value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=res["passed"],
                                         timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="collective_vote", operator=self.name,
                       uncertainty={"tally": res})
        d.change(f"quantities[{var}]", before, res["passed"])
        return d


# ---------------------------------------------------------------- H. exogenous hazards (background)
class BackgroundDynamicsOperator(TransitionOperator):
    """Time passing: attention mean-reversion + memory decay over the ELAPSED interval. Applied on
    background_tick events; parameters are labeled broad priors."""
    name = "background_dynamics"

    def applicable(self, world, event):
        return event.etype == "background_tick"

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name,
                                  action={"elapsed_days": float(event.payload.get("elapsed_days", 1.0))})

    def apply(self, world, proposal):
        days = proposal.action["elapsed_days"]
        d = StateDelta(at=world.clock.now, event_type="background_tick", operator=self.name,
                       uncertainty={"note": "broad-prior background dynamics"})
        for eid, ent in world.entities.items():
            att = ent.get("attention")
            if isinstance(att, StateField) and isinstance(att.value, (int, float)):
                before = att.value
                after = before + (0.7 - before) * min(1.0, 0.2 * days)   # mean-revert toward 0.7
                ent.set("attention", F(round(after, 4), status="derived", method=self.name,
                                       updated_at=world.clock.now))
                if abs(after - before) > 1e-6:
                    d.change(f"{eid}.attention", round(before, 4), round(after, 4))
        if world.information is not None:
            world.information.decay(days)
        return d


# register the foundational families (validated = the contract itself is tested; parameters where they
# exist are labeled priors)
register_operator("agent_decision", AgentDecisionOperator, requires=("entity.*", "information"),
                  modifies=("entity.current_action", "entity.past_actions"), temporal_scale="event",
                  parameter_source="LLM policy over typed actions", validated=True)
register_operator("belief_update", BeliefUpdateOperator, requires=("information", "entity.beliefs"),
                  modifies=("entity.beliefs",), temporal_scale="event",
                  parameter_source="rule core (credibility×trust×salience), broad priors", validated=True)
register_operator("relationship_update", RelationshipUpdateOperator, requires=("network",),
                  modifies=("network.edges",), temporal_scale="event",
                  parameter_source="bounded shifts (|Δ|<=0.25)", validated=True)
register_operator("resource_update", ResourceUpdateOperator, requires=("entity.resources",),
                  modifies=("entity.resources",), temporal_scale="event",
                  parameter_source="conservation-checked deltas", validated=True)
register_operator("institutional_vote", InstitutionalVoteOperator, requires=("institutions",),
                  modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="deterministic rule execution", validated=True)
register_operator("background_dynamics", BackgroundDynamicsOperator, requires=("entities",),
                  modifies=("entity.attention", "information.salience"), temporal_scale="interval",
                  parameter_source="broad priors (labeled)", validated=True)
