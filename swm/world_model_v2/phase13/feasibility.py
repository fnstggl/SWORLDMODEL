"""Phase 13 feasibility engine (Part 6) — typed verdicts, never silent drops, never simulated impossibles.

Checks an `ActionSchema` against the CURRENT WorldState: authority, resource availability, timing,
institutional rules (through the canonical `RuleSystem.validate_action` — the same executable rules the
rollout enforces), prohibitions, prerequisites, network access, reversibility requirements, and mutual
exclusivity. Every rejection carries a typed reason code. Feasibility is STATE-DEPENDENT: the same
check runs again inside the rollout when the action's event fires (interventions.py attaches the
precondition closure), so an action that becomes infeasible mid-policy fails loudly there too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REASON_CODES = ("unauthorized", "insufficient_resources", "timing_violation", "institutional_rule",
                "prohibited", "precondition_failed", "no_network_access", "irreversible_disallowed",
                "mutually_exclusive", "unregistered_operation", "actor_missing", "target_missing",
                "duplicate", "info_gathering_disallowed")


@dataclass
class FeasibilityVerdict:
    action_id: str
    feasible: bool
    reasons: list = field(default_factory=list)     # [{"code":..., "detail":...}]

    def as_dict(self):
        return {"action_id": self.action_id, "feasible": self.feasible, "reasons": self.reasons}


def _reason(code: str, detail: str) -> dict:
    assert code in REASON_CODES, f"untyped feasibility reason {code!r}"
    return {"code": code, "detail": detail[:200]}


class FeasibilityEngine:
    """`check(world, action, problem)` → FeasibilityVerdict. Pure reads; never mutates the world."""

    def __init__(self, *, exclusivity_groups: dict = None):
        # group name -> [action_id]; at most one action per group may be selected together (Part 6)
        self.exclusivity_groups = dict(exclusivity_groups or {})

    def check(self, world, action, problem=None) -> FeasibilityVerdict:
        from swm.world_model_v2.phase13.ontology import operation_registered, operation_spec
        v = FeasibilityVerdict(action_id=action.action_id, feasible=True)

        def fail(code, detail):
            v.feasible = False
            v.reasons.append(_reason(code, detail))

        # 0. operation must exist in the ontology
        if not operation_registered(action.operation):
            fail("unregistered_operation", f"operation {action.operation!r} not in the ontology registry")
            return v
        spec = operation_spec(action.operation)

        # 1. acting actor must exist in the world (do_nothing/defer are exempt structural baselines)
        actor = (world.entities or {}).get(action.actor) if world is not None else None
        if world is not None and actor is None and action.operation not in ("do_nothing", "defer"):
            fail("actor_missing", f"actor {action.actor!r} not in world entities")

        # 2. authority: operation-required capability AND problem-declared authority
        req = spec["required_authority"] or action.authority_basis
        if problem is not None and action.operation not in ("do_nothing", "defer"):
            auth = set(problem.authority or [])
            basis = action.authority_basis or action.operation
            if auth and basis not in auth and action.operation not in auth and \
               f"{action.operation}:{action.object}" not in auth:
                fail("unauthorized",
                     f"decision-maker authority {sorted(auth)[:6]} does not cover "
                     f"{basis!r}/{action.operation!r}")
        if actor is not None and req:
            held = actor.get("authority")
            held_vals = set()
            if isinstance(held, dict):
                held_vals = set(held)
            elif held is not None and getattr(held, "value", None) is not None:
                hv = held.value
                held_vals = set(hv) if isinstance(hv, (list, tuple, set)) else {hv}
            if held_vals and req not in held_vals:
                fail("unauthorized", f"actor holds {sorted(held_vals)[:6]}, operation needs {req!r}")

        # 3. resources: required amounts must be available (actor resources, then problem-controlled)
        for rname, amount in (action.required_resources or {}).items():
            avail = None
            if actor is not None:
                rf = actor.get("resources", key=rname)
                if rf is not None and getattr(rf, "value", None) is not None:
                    avail = float(rf.value)
            if avail is None and problem is not None:
                avail = problem.controllable_resources.get(rname)
            if avail is None:
                fail("insufficient_resources", f"resource {rname!r} unknown to actor and contract")
            elif float(amount) > float(avail) + 1e-12:
                fail("insufficient_resources", f"needs {amount} {rname}, available {avail}")

        # 4. timing: action must fire inside [now, horizon] and before its deadline
        if world is not None and action.timing_ts:
            if action.timing_ts < world.clock.now - 1e-6:
                fail("timing_violation", f"fires at {action.timing_ts} before world now {world.clock.now}")
        if problem is not None and problem.horizon and action.timing_ts:
            from swm.world_model_v2.state import parse_time
            try:
                hz = parse_time(problem.horizon)
                if action.timing_ts > hz:
                    fail("timing_violation", "fires after the decision horizon")
            except ValueError:
                pass

        # 5. institutional rules — the CANONICAL executable rules, same dict contract the rollout uses
        if world is not None:
            targets = ([action.institutional_permission] if action.institutional_permission
                       else list((world.institutions or {}).keys()))
            adict = {"actor": action.actor, "type": action.operation, "target": action.object,
                     "amount": float(action.params.get("amount", action.required_resources.get(
                         next(iter(action.required_resources), ""), 0.0) or 0.0))}
            for iid in targets:
                inst = (world.institutions or {}).get(iid)
                if inst is None:
                    continue
                ok, reasons = inst.validate_action(world, adict)
                if not ok:
                    for r in reasons:
                        fail("institutional_rule", f"{iid}: {r}")

        # 6. prohibitions from the decision contract
        if problem is not None:
            for p in problem.prohibited or []:
                if callable(p):
                    try:
                        if p(action):
                            fail("prohibited", "matched a prohibition predicate")
                    except Exception:  # noqa: BLE001 — a broken prohibition fails CLOSED
                        fail("prohibited", "prohibition predicate errored — failing closed")
                elif str(p) == action.operation or str(p) == action.action_id:
                    fail("prohibited", f"operation/action {p!r} is prohibited by contract")
            if action.operation in ("gather_information", "investigate", "observe",
                                    "run_experiment") and not problem.information_gathering_allowed:
                fail("info_gathering_disallowed", "contract forbids information gathering")
            if problem.reversibility_required and not action.is_reversible():
                fail("irreversible_disallowed", "contract requires reversible actions only")

        # 7. state-dependent preconditions (closures over the world)
        for i, pre in enumerate(action.preconditions or []):
            try:
                if not pre(world):
                    fail("precondition_failed", f"precondition[{i}] returned False")
            except Exception as e:  # noqa: BLE001 — failing closed, with the error surfaced
                fail("precondition_failed", f"precondition[{i}] errored: {type(e).__name__}")

        # 8. network access: contacting/communicating with a recipient needs SOME edge or public channel
        if world is not None and getattr(world, "network", None) is not None and \
                action.spec()["family"] in ("relationships", "information") and action.recipients:
            edges = getattr(world.network, "edges", None) or []
            known = {(e.src, e.dst) for e in edges} | {(e.dst, e.src) for e in edges
                                                      if not _directed(e)}
            for r in action.recipients:
                if r in (world.entities or {}) and (action.actor, r) not in known \
                        and action.params.get("channel", "") == "" and not edges:
                    # no network declared at all → treat channels as open (recorded, not blocking)
                    break
                if edges and r in (world.entities or {}) and (action.actor, r) not in known and \
                        not action.params.get("channel"):
                    fail("no_network_access", f"no edge {action.actor}→{r} and no channel given")
        return v

    def check_bundle(self, world, actions: list, problem=None) -> list:
        """Feasibility for a set, including mutual-exclusivity groups. Never silently drops: returns a
        verdict per action, in order."""
        out = [self.check(world, a, problem) for a in actions]
        for group, members in self.exclusivity_groups.items():
            chosen = [a.action_id for a in actions if a.action_id in members]
            if len(chosen) > 1:
                for vd in out:
                    if vd.action_id in chosen[1:]:
                        vd.feasible = False
                        vd.reasons.append(_reason("mutually_exclusive",
                                                  f"group {group!r} already has {chosen[0]}"))
        return out


def _directed(edge) -> bool:
    try:
        from swm.world_model_v2.network import _RELATIONS
        return bool(_RELATIONS.get(edge.rel, {}).get("directed", True))
    except Exception:  # noqa: BLE001
        return True
