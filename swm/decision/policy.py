"""Sequential policies — the action layer for PLANS, not just single actions (Component 6).

A single action is a special case. The general decision is `argmax over POLICIES π  E[U(outcome | do(π))]`,
where a policy is a short ordered plan whose steps land on the world (and the agent) the EARLIER steps left
behind — a multi-touch outreach, a negotiation, a pricing schedule, a launch sequence. This is the
reflexivity the spec calls out: the follow-up lands on the person the opener already moved, so the same
closing ask converts differently after a pushy vs a kind opener (validated in EXP-060 C). A static
single-action layer cannot express that; this can.

The design reuses the whole single-action machinery: a policy is just an "arm" whose inner Monte-Carlo rolls
the ENTIRE sequence forward with state carried across steps. So `best_policy` races candidate policies with
the same best-arm identification, returns the same navigable object + confidence + honest-tie, and gets the
same scoreboard — it is `best_action` over sequences, not a parallel system.

Two rollout adapters (a policy's steps mean different things per world, but `best_policy` is generic over a
`rollout_fn(policy, rng) -> (outcome, factors)`):
  - `individual_rollout` — the person as a dynamical system (`IndividualAgent`): steps are messages; state
    (mood/busyness/reciprocity) carries forward via the validated Level-1 dynamics. The message-to-Andreessen
    / multi-touch case.
  - `structural_schedule_rollout` — a generic_scm world: steps are timed do-operators (temporal `inject_event`
    / parameter changes) applied over one diffusion. The pricing-schedule / launch-timing case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.decision.best_action import best_action
from swm.simulation.individual_agent import IndividualAgent
from swm.variables.variable_map import VariableMap


@dataclass
class Policy:
    """An ordered plan. `steps` semantics depend on the rollout adapter (messages for individual_rollout,
    timed Actions for structural_schedule_rollout). `label` names it for the race/report."""
    steps: list
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = "→".join(getattr(s, "label", None) or (s.get("label") if isinstance(s, dict) else None)
                                  or f"s{i}" for i, s in enumerate(self.steps))


def best_policy(rollout_fn, policies, utility, *, provenance=None, **kw):
    """argmax over candidate `policies` of `utility(final outcome | do(policy))`, via the single-action racing
    machinery. `rollout_fn(policy, rng) -> (outcome, factors)` rolls the whole sequence forward with state
    carried across steps. Returns a DecisionResult (navigable + confidence + honest tie), same as best_action."""
    pols = list(policies)
    labels = [p.label for p in pols]
    if len(set(labels)) != len(labels):                          # racing keys on label; make them unique
        for i, p in enumerate(pols):
            p.label = f"{p.label}#{i}"
    return best_action(lambda pol, rng: rollout_fn(pol, rng), pols, utility, provenance=provenance, **kw)


# ---- individual: the person as a dynamical system (state carries across the thread) ----
def _person_vm(person_vars, rng, est_sd):
    vm = VariableMap(entity_id="p")
    factors = {}
    for k, v in person_vars.items():
        sd = (est_sd or {}).get(k, 0.0)
        z = rng.gauss(0, 1)
        vm.set(k, v + z * sd, provenance="user", confidence=0.9)
        if sd > 0:
            factors[f"{k}~est"] = z
    return vm, factors


def individual_rollout(person_vars, response_fn, *, gap_steps=1, readout="last", est_sd=None, respond="sample"):
    """Build `rollout_fn(policy, rng)` for a message SEQUENCE on one person. Each step's response is read
    THROUGH the current state, then the message + (sampled) response UPDATE the state before the next step —
    the reflexivity a static vector can't express. `readout`: 'last' (P respond to the final ask), 'any'
    (P respond at all), 'count' (expected # of responses). `respond`: 'sample' (Bernoulli, so the thread
    genuinely branches) or 'threshold' (deterministic, reproduces EXP-060 C)."""
    def rollout(policy, rng):
        vm, factors = _person_vm(person_vars, rng, est_sd)
        agent = IndividualAgent(agent_id="p", variables=vm)
        ps = []
        for k, msg in enumerate(policy.steps):
            if k > 0:
                agent.relax(steps=gap_steps)                     # time passes; state decays toward baseline
            p = agent.response_p(msg, response_fn)["p"]
            responded = (rng.random() < p) if respond == "sample" else (p >= 0.5)
            ps.append(p)
            agent.apply(msg, responded, p)                       # the contact acts on the person (dynamics)
        if not ps:
            return 0.0, factors
        if readout == "last":
            outcome = ps[-1]
        elif readout == "any":
            prod = 1.0
            for pi in ps:
                prod *= (1.0 - pi)
            outcome = 1.0 - prod
        else:                                                    # 'count' — expected number of responses
            outcome = sum(ps)
        return outcome, factors
    return rollout


# ---- structural: a schedule of timed do-operators over one generic_scm diffusion ----
def structural_schedule_rollout(spec):
    """Build `rollout_fn(policy, rng)` where a policy's steps are timed Actions (temporal `inject_event` /
    parameter changes) applied to the compiled spec; the diffusion then carries state across the schedule.
    The pricing-schedule / launch-timing case."""
    from swm.api.compiler import build_sampler

    def rollout(policy, rng):
        s = spec
        for action in policy.steps:                              # each step is a spec->spec do-operator
            s = action.apply(s)
        return build_sampler(s).traced(rng)
    return rollout


# ---- policy-space builders ----
def message_sequences(openers, closer, *, opener_labels=None) -> list:
    """The classic two-touch decision: which OPENER best sets up a fixed CLOSER ask (the state-carryover
    test). Returns one Policy [opener, closer] per opener."""
    return [Policy([o, closer], label=(opener_labels[i] if opener_labels else f"open{i}") + "→ask")
            for i, o in enumerate(openers)]


def enumerate_policies(step_lists, *, labels=None) -> list:
    """Arbitrary candidate plans: each item is an ordered list of steps."""
    return [Policy(list(steps), label=(labels[i] if labels else f"plan{i}"))
            for i, steps in enumerate(step_lists)]
