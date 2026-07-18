"""Deterministic feasibility — checked against EVERY structural world hypothesis, then again
at execution time.

Typed verdicts, never scores. A candidate infeasible in some worlds and feasible in others
is reported with per-hypothesis frequencies and reasons — a recommendation may not owe its
apparent success to worlds where the action could not actually occur (the goal evaluation
reads these frequencies alongside success counts). Nothing here converts an infeasible
action into a no-op: infeasible-everywhere candidates never reach simulation, and runtime
infeasibility is a loud recorded step failure (execution.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.phase13.scenario_actions.candidates import ConcreteAction
from swm.world_model_v2.scenario_schema import ScenarioSemanticModel


@dataclass
class FeasibilityVerdict:
    candidate_id: str
    feasible: bool = True
    reasons: list = field(default_factory=list)          # typed {code, detail}
    conditional: list = field(default_factory=list)      # holds only under these conditions
    unresolved: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"candidate_id": self.candidate_id, "feasible": self.feasible,
                "reasons": self.reasons, "conditional": self.conditional,
                "unresolved": self.unresolved}


def _fail(v: FeasibilityVerdict, code: str, detail: str):
    v.feasible = False
    v.reasons.append({"code": code, "detail": detail[:200]})


def check_candidate(world, language, problem, candidate: ConcreteAction, *,
                    goal=None) -> FeasibilityVerdict:
    """Deterministic checks against ONE world (one hypothesis' initial state)."""
    v = FeasibilityVerdict(candidate_id=candidate.candidate_id)
    schema = getattr(world, "scenario_schema", None)
    if isinstance(schema, dict):
        schema = ScenarioSemanticModel.from_dict(schema)
    entities = set(getattr(world, "entities", {}) or {})
    if candidate.actor_id not in entities:
        _fail(v, "actor_missing", f"decision maker {candidate.actor_id!r} not in this world")
    if candidate.actor_id != problem.decision_maker:
        _fail(v, "wrong_actor", f"candidate acts as {candidate.actor_id!r}, contract says "
                                f"{problem.decision_maker!r}")

    # prohibited predicates from the decision contract (callables over the candidate)
    for p in (problem.prohibited or []):
        if callable(p):
            try:
                if p(candidate):
                    _fail(v, "prohibited", f"contract prohibition {getattr(p, '__name__', 'rule')}")
            except Exception:  # noqa: BLE001 — a broken prohibition fails CLOSED
                _fail(v, "prohibited_check_failed",
                      "a prohibition predicate could not evaluate; failing closed")
        elif isinstance(p, str) and p and p.lower() in (candidate.title + " " + " ".join(
                s.intent for s in candidate.steps)).lower():
            _fail(v, "prohibited", f"contract prohibits {p!r}")
    for c in (problem.constraints or []):
        if getattr(c, "kind", "") == "hard" and getattr(c, "action_pred", None) is not None:
            try:
                if not c.action_pred(candidate):
                    _fail(v, "hard_constraint", f"{c.constraint_id}: {c.description}")
            except Exception:  # noqa: BLE001
                _fail(v, "hard_constraint_check_failed",
                      f"{c.constraint_id} could not evaluate; failing closed")

    horizon = getattr(world, "horizon", None) or (schema.horizon if schema else 0.0)
    now = float(getattr(getattr(world, "clock", None), "now", 0.0) or 0.0)
    institutions = set(getattr(world, "institutions", {}) or {}) | \
        set((schema.institutional_definitions or {}) if schema else {})
    # a later step may legitimately target a record an EARLIER step's compiled program
    # creates — collect plan-internal record ids in step order
    created_by_plan: set = set()
    resources_needed: dict = {}
    for step in candidate.steps:
        for t in step.target_ids:
            t = str(t)
            if t not in entities and t not in (getattr(world, "objects", {}) or {}) \
                    and t not in institutions and t not in created_by_plan:
                _fail(v, "target_missing", f"step {step.step_id}: target {t!r} does not exist")
        for op in step.compiled_ops or []:
            rid = str(op.get("record_id", "") or "")
            if rid and str(op.get("op")) == "create_or_update_record":
                created_by_plan.add(rid)
        if step.timing_ts is not None:
            if horizon and float(step.timing_ts) > float(horizon):
                _fail(v, "timing_after_horizon",
                      f"step {step.step_id} fires after the decision horizon")
            if float(step.timing_ts) < now:
                _fail(v, "timing_in_past", f"step {step.step_id} is scheduled before now")
        for name, amt in (step.resource_commitments or {}).items():
            resources_needed[str(name)] = resources_needed.get(str(name), 0.0) + float(amt)
        if step.conditions:
            v.conditional.append({"step": step.step_id,
                                  "conditions": [getattr(c, "description", "") or
                                                 getattr(c, "record_type", "")
                                                 for c in step.conditions][:4]})
        if problem.reversibility_required and step.reversible is False:
            _fail(v, "irreversible", f"step {step.step_id} is irreversible and the contract "
                                     f"requires reversibility")
    ent = (getattr(world, "entities", {}) or {}).get(candidate.actor_id)
    holdings = {}
    if ent is not None:
        res = ent.get("resources")
        if isinstance(res, dict):
            holdings = {str(k): float(sf.value) for k, sf in res.items()
                        if isinstance(getattr(sf, "value", None), (int, float))}
    for name, amt in resources_needed.items():
        have = holdings.get(name, float((problem.controllable_resources or {}).get(name, 0.0)))
        if have < amt:
            _fail(v, "insufficient_resources",
                  f"needs {amt} {name}, holds {have} in this world")
    # institutional entries must reference procedures that exist here
    if schema is not None:
        insts = set(schema.institutional_definitions or {}) | set(
            getattr(world, "institutions", {}) or {})
        for step in candidate.steps:
            for t in step.target_ids:
                if str(t) in (schema.institutional_definitions or {}) and str(t) not in insts:
                    _fail(v, "institution_missing", f"institution {t!r} not in this world")
    v.unresolved = [u for u in candidate.all_unresolved()][:8]
    return v


def check_across_particles(particles: list, hypothesis_assignment: list, language, problem,
                           candidate: ConcreteAction, *, goal=None) -> dict:
    """Run the deterministic check against EVERY particle's initial world; report frequencies
    and per-hypothesis reasons. `feasible_everywhere` gates simulation entry;
    `feasible_somewhere` candidates run but their evaluation carries the infeasibility mask
    so success can never be credited to worlds where the action could not occur."""
    verdicts = [check_candidate(w, language, problem, candidate, goal=goal)
                for w in particles]
    n = max(1, len(verdicts))
    by_hyp: dict = {}
    for hid, vd in zip(hypothesis_assignment or ["H0"] * n, verdicts):
        h = by_hyp.setdefault(str(hid), {"n": 0, "feasible": 0, "reasons": {}})
        h["n"] += 1
        h["feasible"] += 1 if vd.feasible else 0
        for r in vd.reasons:
            h["reasons"][r["code"]] = h["reasons"].get(r["code"], 0) + 1
    n_feasible = sum(1 for vd in verdicts if vd.feasible)
    return {"candidate_id": candidate.candidate_id,
            "n_particles": len(verdicts), "n_feasible": n_feasible,
            "feasible_everywhere": n_feasible == len(verdicts),
            "feasible_somewhere": n_feasible > 0,
            "feasibility_mask": [vd.feasible for vd in verdicts],
            "by_hypothesis": by_hyp,
            "rejection_reasons": _merge_reasons(verdicts),
            "conditional": verdicts[0].conditional if verdicts else [],
            "unresolved": verdicts[0].unresolved if verdicts else []}


def _merge_reasons(verdicts) -> list:
    counts: dict = {}
    for vd in verdicts:
        for r in vd.reasons:
            key = (r["code"], r["detail"])
            counts[key] = counts.get(key, 0) + 1
    return [{"code": c, "detail": d, "in_n_worlds": n}
            for (c, d), n in sorted(counts.items(), key=lambda kv: -kv[1])][:12]
