"""Executable institutional state — Phase 1.5. Rules that CONSTRAIN actions, not descriptions of rules.

An institution here is any rule-governed structure: formal orgs, platform policies, market/game rules, legal
systems, contracts, household procedures, protocols. The compiler discovers which rule systems govern a
scenario. The hard property: `validate_action` runs BEFORE any transition applies — an LLM cannot produce an
action that violates an executable rule (acceptance test 5). Voting/thresholds/deadlines/budgets execute
deterministically (or stochastically where specified), never by narration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.state import Provenance


@dataclass
class Rule:
    rule_id: str
    kind: str                             # decision_right | voting | threshold | deadline | budget |
    #                                       eligibility | legal | procedure | capacity
    params: dict = field(default_factory=dict)
    prov: Provenance = field(default_factory=Provenance)

    def check(self, world, action: dict):
        """Return (ok, reason). `action` is a typed dict: {actor, type, target?, amount?, …}."""
        k, p = self.kind, self.params
        if k == "decision_right":
            if action.get("type") in p.get("actions", []) and action.get("actor") not in p.get("holders", []):
                return False, f"{action.get('actor')} lacks the decision right for {action.get('type')}"
        elif k == "deadline":
            if action.get("type") in p.get("actions", []) and world.clock.now > p["by_ts"]:
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
