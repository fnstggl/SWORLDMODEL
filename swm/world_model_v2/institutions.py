"""Executable institutional state — Phase 1.5. Rules that CONSTRAIN actions, not descriptions of rules.

An institution here is any rule-governed structure: formal orgs, platform policies, market/game rules, legal
systems, contracts, household procedures, protocols. The compiler discovers which rule systems govern a
scenario. The hard property: `validate_action` runs BEFORE any transition applies — an LLM cannot produce an
action that violates an executable rule (acceptance test 5). Voting/thresholds/deadlines/budgets execute
deterministically (or stochastically where specified), never by narration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.state import Provenance, parse_time

#: The CLOSED set of rule kinds with executable semantics. The docstring used to advertise kinds that
#: silently validated everything; now: materialize refuses non-executable kinds (recorded omission), and
#: a Rule constructed with an unknown kind FAILS CLOSED at check time (defense in depth) — a rule the
#: engine cannot execute must not silently admit every action (Tier A1 of the gap audit).
EXECUTABLE_RULE_KINDS = ("decision_right", "deadline", "budget", "eligibility", "procedure",
                         "capacity", "quorum")


@dataclass
class Rule:
    rule_id: str
    kind: str                             # one of EXECUTABLE_RULE_KINDS
    params: dict = field(default_factory=dict)
    prov: Provenance = field(default_factory=Provenance)

    def check(self, world, action: dict):
        """Return (ok, reason). `action` is a typed dict: {actor, type, target?, amount?, …}."""
        k, p = self.kind, self.params
        if k not in EXECUTABLE_RULE_KINDS:
            return False, (f"rule kind {k!r} has no executable semantics — failing closed "
                           f"(executable: {EXECUTABLE_RULE_KINDS})")
        if k == "decision_right":
            if action.get("type") in p.get("actions", []) and action.get("actor") not in p.get("holders", []):
                return False, f"{action.get('actor')} lacks the decision right for {action.get('type')}"
        elif k == "deadline":
            if action.get("type") in p.get("actions", []):
                try:
                    by = parse_time(p["by_ts"])
                except (KeyError, ValueError, TypeError):
                    return False, f"deadline rule {self.rule_id} has no parseable by_ts — failing closed"
                if world.clock.now > by:
                    return False, f"deadline passed for {action.get('type')}"
        elif k == "budget":
            if action.get("type") in p.get("actions", []):
                spent = float(action.get("amount", 0.0))
                holder = p.get("resource_holder"), p.get("resource")
                q = world.quantities.get(p.get("resource")) if p.get("resource") else None
                avail = q.value if q is not None else p.get("available", 0.0)
                if spent > avail:
                    return False, f"budget exceeded: {spent} > {avail} {p.get('resource', '')}"
        elif k == "eligibility":
            if action.get("type") in p.get("actions", []):
                need = p.get("require", {})
                actor = world.entities.get(action.get("actor"))
                for fname, val in need.items():
                    if actor is None or actor.value(fname) != val:
                        return False, f"{action.get('actor')} not eligible: needs {fname}={val}"
        elif k == "procedure":
            stage = p.get("stage_var", "procedural_stage")
            allowed = p.get("allowed_in_stage", {})
            cur = world.quantities.get(stage)
            cur_v = cur.value if cur is not None else p.get("default_stage")
            if action.get("type") in allowed and cur_v not in allowed[action["type"]]:
                return False, f"{action.get('type')} not allowed in stage {cur_v!r}"
        elif k == "capacity":
            # bounded concurrent/total uses of an action, tracked in a counter quantity
            if action.get("type") in p.get("actions", []):
                var = p.get("counter_var", f"capacity_used:{self.rule_id}")
                used = world.quantities.get(var)
                used_v = float(used.value) if used is not None and used.value is not None else 0.0
                if used_v + 1 > float(p.get("max", 0)):
                    return False, f"capacity exhausted for {action.get('type')} ({used_v}/{p.get('max')})"
        elif k == "quorum":
            # a collective action requires >= min_present eligible participants marked present/eligible
            if action.get("type") in p.get("actions", []):
                present = 0
                for eid in p.get("members", []):
                    ent = world.entities.get(eid)
                    if ent is not None and ent.value("attention", default=1.0) not in (None, 0.0):
                        present += 1
                if present < int(p.get("min_present", 0)):
                    return False, (f"quorum not met for {action.get('type')}: {present} present "
                                   f"< {p.get('min_present')}")
        return True, ""


@dataclass
class RuleSystem:
    institution_id: str
    rules: list = field(default_factory=list)     # [Rule]
    prov: Provenance = field(default_factory=Provenance)

    def validate_action(self, world, action: dict):
        """Every rule must pass. Returns (ok, [reasons])."""
        reasons = []
        for r in self.rules:
            ok, why = r.check(world, action)
            if not ok:
                reasons.append(f"[{r.rule_id}] {why}")
        return (not reasons), reasons

    # ---------------- executable collective decisions ----------------
    def run_vote(self, votes: dict, *, threshold: float = None, needed: int = None, total: int = None,
                 weights: dict = None) -> dict:
        """Deterministic vote execution: votes={voter: 'yes'|'no'|'abstain'}, optional per-voter weights,
        threshold as share of TOTAL (not of cast) unless `needed` given. Returns the typed outcome."""
        w = weights or {}
        yes = sum(w.get(v, 1.0) for v, c in votes.items() if c == "yes")
        no = sum(w.get(v, 1.0) for v, c in votes.items() if c == "no")
        tot = total if total is not None else sum(w.get(v, 1.0) for v in votes)
        if needed is None:
            needed = (threshold if threshold is not None else 0.5) * tot
            passed = yes > needed if (threshold in (None, 0.5)) else yes >= needed
        else:
            passed = yes >= needed
        return {"passed": bool(passed), "yes": yes, "no": no, "total": tot,
                "needed": round(float(needed), 3)}
