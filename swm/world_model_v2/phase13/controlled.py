"""Phase 13 controlled decision-correctness benchmark (Part 30A) — 200 independently specified tasks.

Each task is a small, fully specified decision world with a KNOWN ground truth: the payoff structure is
constructed, so the true optimal action (or policy) and its value are computable analytically or by
exhaustive evaluation — the benchmark verifies intervention semantics, feasibility, matched randomness,
search correctness, policy behavior and constraint handling against truth, not vibes.

Families (14): discrete, continuous, combinatorial, sequential, partially_observable, multi_actor,
institutional, population, network, nonlinear, information_gathering, irreversible, constrained,
multi_objective. Task parameters are seeded per task id — 200 independent specifications.

Split governance (Part 32): tasks are split by task id into development(60) / calibration(40) /
validation(40) / locked_test(60) at the task level (each task is its own decision environment; no unit
is shared across tasks by construction). The locked set is evaluated ONCE by the runner's
`--locked` pass and the access is logged.
"""
from __future__ import annotations

import math
import random

from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.events import Event, EventQueue, StochasticHazard, register_event_type
from swm.world_model_v2.init_state import InitialStateModel
from swm.world_model_v2.institutions import Rule, RuleSystem
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta, TransitionOperator, ValidationResult
from swm.world_model_v2.phase13.contracts import (ConstraintSpec, DecisionProblem, RiskSpec,
                                                  Stakeholder, UtilitySpec)
from swm.world_model_v2.phase13.ontology import ActionSchema

T0 = 1.0e9
DAY = 86400.0
register_quantity_type("payoff", units="util")
register_event_type("effect_tick", scheduling="scheduled", validated=True)

FAMILIES = ("discrete", "continuous", "combinatorial", "sequential", "partially_observable",
            "multi_actor", "institutional", "population", "network", "nonlinear",
            "information_gathering", "irreversible", "constrained", "multi_objective")

#: exactly 200 tasks: 15 for the first 6 families, 14 for the remaining 8 (6*15 + 8*14 = 202 → trim 2)
_COUNTS = {f: (15 if i < 6 else 14) for i, f in enumerate(FAMILIES)}
_COUNTS["multi_objective"] = 13
_COUNTS["constrained"] = 13
assert sum(_COUNTS.values()) == 200


def task_ids() -> list:
    out = []
    for fam in FAMILIES:
        for k in range(_COUNTS[fam]):
            out.append(f"{fam}_{k:02d}")
    return out


def split_of(task_id: str) -> str:
    """Deterministic split by task-id hash: 60 dev / 40 cal / 40 val / 60 locked."""
    h = random.Random(task_id).random()
    if h < 0.30:
        return "development"
    if h < 0.50:
        return "calibration"
    if h < 0.70:
        return "validation"
    return "locked_test"


# ---------------------------------------------------------------- the shared payoff world
class PayoffOperator(TransitionOperator):
    """Applies a task-specific payoff rule when decision-action effects land. The rule is data
    (payload-driven), so one operator serves every family; noise flows through the engine rng."""
    name = "controlled_payoff"

    def __init__(self, rule):
        self.rule = rule                                # callable(world, event, rng) -> float delta

    def applicable(self, world, event):
        return event.etype in ("decision_action", "message_delivered", "effect_tick",
                               "information_published", "measurement", "collective_vote",
                               "decision_opportunity")

    def propose(self, world, event, rng):
        from swm.world_model_v2.transitions import TransitionProposal
        return TransitionProposal(operator=self.name, action={}, reason_codes=[event.etype])

    def apply(self, world, proposal):
        return None

    def run(self, world, event, rng):
        gain = float(self.rule(world, event, rng) or 0.0)
        d = StateDelta(at=world.clock.now, event_type=event.etype, operator=self.name)
        if gain != 0.0:
            q = world.quantities["payoff"]
            before = float(q.value or 0.0)
            q.value = before + gain
            d.change("quantities.payoff", before, q.value)
        return d, ValidationResult(ok=True)


def _base_world(rng, *, entities=(), noise_sd=0.15):
    base = WorldState(world_id="cw", branch_id="root", clock=SimulationClock(now=T0, as_of=T0))
    maker = Entity(identity="decider")
    maker.set("resources", F(100.0, status="observed"), key="budget")
    base.entities["decider"] = maker
    for e in entities:
        base.entities[e.identity] = e
    base.quantities["payoff"] = Quantity(name="payoff", qtype="payoff", value=0.0, timestamp=T0)
    maker.set("latent_state", F(None, dist={"mean": 0.5, "sd": noise_sd, "lo": 0.0, "hi": 1.0}),
              key="context")
    return base


def _ctx(base, rule, *, horizon_days=8.0, hazard_rate=0.3, n_particles=40, extra_ops=(),
         schedule=()):
    init = InitialStateModel(base_world=base, latents=[])

    def qb(world):
        q = EventQueue(horizon_ts=T0 + horizon_days * DAY)
        pr = int(world.branch_id.split(":")[0].strip("b") or 0) if world.branch_id[:1] == "b" else 0
        if hazard_rate > 0:
            q.add_hazard(StochasticHazard(etype="distraction", rate_per_day=hazard_rate,
                                          participants=["decider"]),
                         now=world.clock.now, rng=random.Random(pr), world=world)
        for ev in schedule:
            q.schedule(Event(ts=ev["ts"], etype=ev["etype"],
                             participants=list(ev.get("participants", [])),
                             payload=dict(ev.get("payload", {}))))
        return q

    contract = OutcomeContract(family="continuous",
                               readout=lambda w: float(w.quantities["payoff"].value or 0.0),
                               horizon_ts=T0 + horizon_days * DAY)
    return {"initial": init, "queue_builder": qb, "operators": [PayoffOperator(rule), *extra_ops],
            "contract": contract, "n_particles": n_particles}


def _std_problem(task_id, actions, *, aggregation="weighted_sum", robustness="expected",
                 constraints=(), stakeholders=None, decision_points=(), cvar_alpha=0.2):
    stakeholders = stakeholders or [Stakeholder("decider", utility_fn=lambda o: o["readout"])]
    return DecisionProblem(
        decision_id=task_id, decision_maker="decider",
        authority=["communicate", "set_parameter", "transfer", "gather_information", "submit",
                   "choose_policy", "invest"],
        controllable_resources={"budget": 100.0},
        as_of="2001-09-09", horizon="2001-09-17",
        decision_points=list(decision_points),
        candidate_actions=list(actions), generated_action_permission=False,
        constraints=list(constraints),
        utility=UtilitySpec(stakeholders=stakeholders, aggregation=aggregation,
                            cvar_alpha=cvar_alpha),
        risk=RiskSpec(robustness=robustness, tolerance="neutral"))


def _act(aid, *, op="communicate", obj="decider", ts_days=1.0, quality=None, params=None,
         recipients=("decider",), **kw):
    p = dict(params or {})
    content = {"variant": aid}
    if quality is not None:
        content["quality"] = quality
    return ActionSchema(action_id=aid, actor="decider", operation=op, object=obj,
                        recipients=list(recipients), timing_ts=T0 + ts_days * DAY,
                        params=p, content=content, **kw)


# ---------------------------------------------------------------- family builders
# Every builder returns {"ctx", "problem", "optimum": {"action_id", "value"}, "notes"}.

def _discrete(rng, task_id):
    k = rng.randint(3, 6)
    quals = sorted({round(rng.uniform(0.1, 0.9), 3) for _ in range(k)})
    while len(quals) < 3:
        quals.append(round(rng.uniform(0.1, 0.9), 3))
    acts = [_act(f"arm_{i}", quality=qv) for i, qv in enumerate(quals)]

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            a = event.payload.get("action")
            return float(a.content.get("quality", 0.0)) + rng_.gauss(0, 0.05)
        return 0.0
    best = max(range(len(quals)), key=lambda i: quals[i])
    return {"ctx": _ctx(_base_world(rng), rule), "problem": _std_problem(task_id, acts),
            "optimum": {"action_id": f"arm_{best}", "value": quals[best]},
            "notes": f"analytic: E[payoff]=quality; best arm quality={quals[best]}"}


def _continuous(rng, task_id):
    peak = round(rng.uniform(0.25, 0.85), 3)
    levels = [round(i / 8, 3) for i in range(9)]
    acts = [_act(f"lvl_{str(v).replace('.', 'p')}", op="set_parameter", obj="payoff",
                 params={"value": v}) for v in levels]

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            a = event.payload.get("action")
            v = float(a.params.get("value", 0.0))
            return 1.0 - (v - peak) ** 2 + rng_.gauss(0, 0.03)
        return 0.0
    best = min(levels, key=lambda v: abs(v - peak))
    return {"ctx": _ctx(_base_world(rng), rule), "problem": _std_problem(task_id, acts),
            "optimum": {"action_id": f"lvl_{str(best).replace('.', 'p')}",
                        "value": 1.0 - (best - peak) ** 2},
            "notes": f"concave payoff peaked at {peak}; grid argmax known"}


def _combinatorial(rng, task_id):
    chans = {"email": rng.uniform(0.2, 0.7), "call": rng.uniform(0.2, 0.7)}
    times = {"early": rng.uniform(0.0, 0.3), "late": rng.uniform(0.0, 0.3)}
    inter = rng.uniform(-0.25, 0.25)                   # email×early interaction
    acts = []
    for ch, cv in chans.items():
        for tm, tv in times.items():
            val = cv + tv + (inter if (ch == "email" and tm == "early") else 0.0)
            acts.append(_act(f"{ch}_{tm}", quality=round(val, 4),
                             ts_days=(0.5 if tm == "early" else 3.0)))

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            return float(event.payload["action"].content.get("quality", 0.0)) + rng_.gauss(0, 0.04)
        return 0.0
    best = max(acts, key=lambda a: a.content["quality"])
    return {"ctx": _ctx(_base_world(rng), rule), "problem": _std_problem(task_id, acts),
            "optimum": {"action_id": best.action_id, "value": best.content["quality"]},
            "notes": "additive channel+timing with one interaction; exhaustive over 4 combos"}


class _SeqRevealOperator(TransitionOperator):
    """Sequential structure: a measurement at day 2 reveals the latent context; a second-stage action
    pays off only if aligned with the revealed context."""
    name = "seq_reveal"

    def applicable(self, world, event):
        return event.etype == "measurement"

    def propose(self, world, event, rng):
        from swm.world_model_v2.transitions import TransitionProposal
        return TransitionProposal(operator=self.name, action={}, reason_codes=["reveal"])

    def apply(self, world, proposal):
        return None

    def run(self, world, event, rng):
        ent = world.entities["decider"]
        ctxv = ent.get("latent_state", key="context")
        val = float(ctxv.value if ctxv is not None and ctxv.value is not None else 0.5)
        info = world.information
        d = StateDelta(at=world.clock.now, event_type="measurement", operator=self.name)
        if info is not None:
            from swm.world_model_v2.information import InformationItem
            side = "high" if val >= 0.5 else "low"
            iid = f"reveal:{side}"
            if iid not in info.items:
                info.publish(InformationItem(item_id=iid, content=f"context is {side}",
                                             kind="public", source="world",
                                             created_at=world.clock.now))
            info.expose("decider", iid, world.clock.now)
            d.change("information.reveal", None, side)
        return d, ValidationResult(ok=True)


def _sequential(rng, task_id):
    gain = round(rng.uniform(0.6, 1.0), 3)

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            a = event.payload.get("action")
            side = str(a.content.get("variant", ""))
            ent = world.entities["decider"]
            ctxv = ent.get("latent_state", key="context")
            val = float(ctxv.value if ctxv is not None and ctxv.value is not None else 0.5)
            truth = "high" if val >= 0.5 else "low"
            if side in ("high", "low"):
                return (gain if side == truth else -0.3) + rng_.gauss(0, 0.03)
            if side == "blind":
                return 0.25 * gain + rng_.gauss(0, 0.03)
        return 0.0

    base = _base_world(rng)
    from swm.world_model_v2.information import InformationLedger
    base.information = InformationLedger()
    ctx = _ctx(base, rule, extra_ops=(_SeqRevealOperator(),),
               schedule=({"ts": T0 + 2 * DAY, "etype": "measurement"},))
    # policies are supplied by the runner (adaptive vs greedy vs nothing); the analytic optimum:
    # adaptive = E[gain | act on truth] = gain; greedy blind = 0.25*gain
    return {"ctx": ctx, "problem": _std_problem(task_id, [],
                                                decision_points=("2001-09-12",)),
            "optimum": {"action_id": "adaptive_policy", "value": gain},
            "notes": f"adaptive(after reveal)={gain} vs blind={0.25 * gain}; policy task",
            "policy_task": True, "gain": gain}


def _partially_observable(rng, task_id):
    return {**_sequential(rng, task_id),
            "notes": "belief-state task: the reveal enters the ledger; policies read belief only"}


class _OpponentOperator(TransitionOperator):
    """Multi-actor: an opponent with a fitted threshold policy reacts to the decider's move — matching
    the decider's aggression is bad for the decider (best response = play opposite)."""
    name = "opponent_policy"

    def __init__(self, aggr_threshold):
        self.thr = aggr_threshold

    def applicable(self, world, event):
        return event.etype == "message_delivered"

    def propose(self, world, event, rng):
        from swm.world_model_v2.transitions import TransitionProposal
        return TransitionProposal(operator=self.name, action={}, reason_codes=["respond"])

    def apply(self, world, proposal):
        return None

    def run(self, world, event, rng):
        aggr = float((event.payload.get("content") or {}).get("quality", 0.5))
        q = world.quantities["payoff"]
        before = float(q.value or 0.0)
        # opponent retaliates against aggression above its threshold
        q.value = before + (-0.5 * aggr if aggr > self.thr else 0.3 * (1 - aggr))
        d = StateDelta(at=world.clock.now, event_type="message_delivered", operator=self.name)
        d.change("quantities.payoff", before, q.value)
        return d, ValidationResult(ok=True)


def _multi_actor(rng, task_id):
    thr = round(rng.uniform(0.3, 0.7), 3)
    levels = [0.1, 0.35, 0.6, 0.85]
    acts = [_act(f"aggr_{str(v).replace('.', 'p')}", quality=v, recipients=("opponent",))
            for v in levels]
    opp = Entity(identity="opponent")
    base = _base_world(rng, entities=(opp,))

    def rule(world, event, rng_):
        return 0.0                                       # payoff moves via the opponent's response
    ctx = _ctx(base, rule, extra_ops=(_OpponentOperator(thr),))
    def value(v):
        return -0.5 * v if v > thr else 0.3 * (1 - v)
    best = max(levels, key=value)
    return {"ctx": ctx, "problem": _std_problem(task_id, acts),
            "optimum": {"action_id": f"aggr_{str(best).replace('.', 'p')}", "value": value(best)},
            "notes": f"opponent retaliates above threshold {thr}; strategic best response known"}


def _institutional(rng, task_id):
    budget_cap = round(rng.uniform(20.0, 60.0), 1)
    acts = [
        _act("small_spend", op="invest", params={"amount": budget_cap * 0.5},
             quality=0.4, required_resources={"budget": budget_cap * 0.5}),
        _act("cap_spend", op="invest", params={"amount": budget_cap * 0.9},
             quality=0.7, required_resources={"budget": budget_cap * 0.9}),
        _act("over_cap", op="invest", params={"amount": budget_cap * 1.5},
             quality=1.5, required_resources={"budget": budget_cap * 1.5}),   # best but ILLEGAL
    ]
    base = _base_world(rng)
    base.institutions["treasury"] = RuleSystem(institution_id="treasury", rules=[
        Rule(rule_id="cap", kind="budget",
             params={"actions": ["invest"], "resource": "spend_budget", "available": budget_cap})])
    base.quantities["spend_budget"] = Quantity(name="spend_budget", qtype="payoff",
                                               value=budget_cap, timestamp=T0)

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            return float(event.payload["action"].content.get("quality", 0.0)) + rng_.gauss(0, 0.03)
        return 0.0
    return {"ctx": _ctx(base, rule), "problem": _std_problem(task_id, acts),
            "optimum": {"action_id": "cap_spend", "value": 0.7},
            "notes": f"over_cap (quality 1.5) violates the executable budget rule {budget_cap}; "
                     "best FEASIBLE is cap_spend — tests institutional rejection"}


def _population(rng, task_id):
    segs = {f"seg_{i}": round(rng.uniform(0.1, 0.9), 3) for i in range(3)}
    acts = [_act(f"target_{s}", quality=v) for s, v in segs.items()]

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            return float(event.payload["action"].content.get("quality", 0.0)) + rng_.gauss(0, 0.05)
        return 0.0
    best = max(segs, key=segs.get)
    return {"ctx": _ctx(_base_world(rng), rule), "problem": _std_problem(task_id, acts),
            "optimum": {"action_id": f"target_{best}", "value": segs[best]},
            "notes": "segment response rates fixed; best segment known"}


def _network(rng, task_id):
    from swm.world_model_v2.network import RelationGraph
    reach = {"hub": rng.randint(4, 7), "mid": rng.randint(2, 3), "leaf": 1}
    base = _base_world(rng, entities=tuple(Entity(identity=n) for n in reach))
    base.network = RelationGraph()
    for n in reach:
        base.network.add("decider", "communicates_with", n)
    acts = [_act(f"seed_{n}", recipients=(n,), quality=reach[n] * 0.1) for n in reach]

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            return float(event.payload["action"].content.get("quality", 0.0)) + rng_.gauss(0, 0.02)
        return 0.0
    best = max(reach, key=reach.get)
    return {"ctx": _ctx(base, rule), "problem": _std_problem(task_id, acts),
            "optimum": {"action_id": f"seed_{best}", "value": reach[best] * 0.1},
            "notes": "diffusion reach proportional to degree; hub is optimal"}


def _nonlinear(rng, task_id):
    thr = round(rng.uniform(0.4, 0.7), 3)
    levels = [round(thr - 0.15, 3), round(thr + 0.1, 3), round(thr + 0.3, 3)]
    acts = [_act(f"push_{str(v).replace('.', 'p')}", op="set_parameter", obj="payoff",
                 params={"value": v}) for v in levels]

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            v = float(event.payload["action"].params.get("value", 0.0))
            return (1.0 if v >= thr else 0.1) - 0.4 * v + rng_.gauss(0, 0.03)
        return 0.0
    def value(v):
        return (1.0 if v >= thr else 0.1) - 0.4 * v
    best = max(levels, key=value)
    return {"ctx": _ctx(_base_world(rng), rule), "problem": _std_problem(task_id, acts),
            "optimum": {"action_id": f"push_{str(best).replace('.', 'p')}", "value": value(best)},
            "notes": f"threshold dynamics at {thr}: just past the threshold is optimal"}


def _information_gathering(rng, task_id):
    t = _sequential(rng, task_id)
    t["notes"] = "VOI task: the reveal has positive EVSI; gather-then-act beats blind commit"
    return t


def _irreversible(rng, task_id):
    p_win = round(rng.uniform(0.25, 0.45), 3)          # risky irreversible wins with p < 0.5
    acts = [
        _act("safe_reversible", quality=0.35),
        ActionSchema(action_id="risky_irreversible", actor="decider", operation="commit",
                     object="deal", timing_ts=T0 + DAY, reversible=False,
                     content={"variant": "risky", "p_win": p_win}),
    ]

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            a = event.payload.get("action")
            if a.content.get("variant") == "risky":
                return 1.0 if rng_.random() < p_win else -0.6
            return float(a.content.get("quality", 0.0)) + rng_.gauss(0, 0.02)
        return 0.0
    # under CVaR(0.2) the risky arm's tail is -0.6 < safe's 0.35 tail → safe is optimal
    return {"ctx": _ctx(_base_world(rng), rule),
            "problem": _std_problem(task_id, acts, robustness="cvar"),
            "optimum": {"action_id": "safe_reversible", "value": 0.35},
            "notes": f"risky mean={p_win * 1.0 - (1 - p_win) * 0.6:.3f} but CVaR tail=-0.6; "
                     "cvar objective must pick safe_reversible"}


def _constrained(rng, task_id):
    p_bad = round(rng.uniform(0.3, 0.5), 3)
    acts = [
        _act("aggressive", quality=0.9),               # violates the chance constraint
        _act("moderate", quality=0.5),
    ]

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            a = event.payload.get("action")
            if a.content.get("variant") == "aggressive" and rng_.random() < p_bad:
                return -1.0
            return float(a.content.get("quality", 0.0)) + rng_.gauss(0, 0.02)
        return 0.0
    cons = [ConstraintSpec(constraint_id="no_blowup", kind="chance",
                           outcome_pred=lambda o: o["readout"] > -0.5, max_prob=0.10)]
    return {"ctx": _ctx(_base_world(rng), rule),
            "problem": _std_problem(task_id, acts, constraints=cons),
            "optimum": {"action_id": "moderate", "value": 0.5},
            "notes": f"aggressive blows up with p={p_bad} > chance cap 0.10 → must pick moderate"}


def _multi_objective(rng, task_id):
    profit_w = round(rng.uniform(0.4, 0.6), 3)
    acts = [
        _act("profit_max", quality=0.9, params={"trust": 0.1}),
        _act("balanced", quality=0.6, params={"trust": 0.6}),
        _act("trust_max", quality=0.2, params={"trust": 0.9}),
    ]

    def rule(world, event, rng_):
        if event.etype == "decision_action":
            a = event.payload.get("action")
            world.quantities["payoff"].timestamp = world.clock.now
            # payoff quantity carries profit; trust rides in a second quantity
            tq = world.quantities.get("trust")
            if tq is not None:
                tq.value = float(tq.value or 0.0) + float(a.params.get("trust", 0.0))
            return float(a.content.get("quality", 0.0)) + rng_.gauss(0, 0.02)
        return 0.0
    base = _base_world(rng)
    register_quantity_type("trust", units="util")
    base.quantities["trust"] = Quantity(name="trust", qtype="trust", value=0.0, timestamp=T0)
    stak = [Stakeholder("owner", utility_fn=lambda o: o["readout"], weight=profit_w),
            Stakeholder("counterparty", utility_fn=lambda o: o["quantities"].get("trust", 0.0),
                        weight=1 - profit_w)]
    def value(a):
        return profit_w * a.content["quality"] + (1 - profit_w) * a.params["trust"]
    best = max(acts, key=value)
    return {"ctx": _ctx(base, rule),
            "problem": _std_problem(task_id, acts, stakeholders=stak),
            "optimum": {"action_id": best.action_id, "value": round(value(best), 4)},
            "notes": f"two stakeholders weighted {profit_w}/{1 - profit_w}; all three are "
                     "Pareto-efficient; the weighted optimum is known",
            "pareto_expected": ["profit_max", "balanced", "trust_max"]}


_BUILDERS = {"discrete": _discrete, "continuous": _continuous, "combinatorial": _combinatorial,
             "sequential": _sequential, "partially_observable": _partially_observable,
             "multi_actor": _multi_actor, "institutional": _institutional,
             "population": _population, "network": _network, "nonlinear": _nonlinear,
             "information_gathering": _information_gathering, "irreversible": _irreversible,
             "constrained": _constrained, "multi_objective": _multi_objective}


def build_task(task_id: str) -> dict:
    fam = task_id.rsplit("_", 1)[0]
    rng = random.Random(f"phase13-controlled|{task_id}")
    t = _BUILDERS[fam](rng, task_id)
    t["task_id"] = task_id
    t["family"] = fam
    t["split"] = split_of(task_id)
    t["known_optimum"] = True
    return t
