"""Action — a typed intervention on a compiled world (a `do`-operator), never a string an LLM judges.

The robustness decision at the heart of the action layer: an Action is a transform `ModelSpec -> ModelSpec'`
that modifies the *world* before it is rolled forward, so the effect of the action is *simulated*, not
asserted. Three kinds mirror the three ways you can intervene on a causal model:

  - PARAMETER  `do(X := x)` — set/shift a variable's value (price = 49, tone = warm), or choose the message.
  - STRUCTURAL — add/remove/modify an entity or edge (a competitor's strength, a cell's turnout).
  - TEMPORAL   — inject an exogenous event over the horizon (a shock at time t).

The LLM's role is to *propose* actions (swm/decision/space.py); scoring is always simulation. Every
transform clones the spec first, so an intervention can never mutate the base model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class Action:
    """`apply(spec) -> spec'`. A None transform is the identity — the status-quo / do-nothing action, the
    honest baseline every recommendation is contrasted against."""
    label: str
    transform: object = None
    kind: str = "parameter"
    meta: dict = field(default_factory=dict)

    def apply(self, spec):
        return self.transform(spec) if self.transform else copy.deepcopy(spec)


def noop(label="status_quo") -> Action:
    """Do nothing — the baseline. Returns a fresh clone so downstream never mutates the base spec."""
    return Action(label, None, "none")


# ---- parameter interventions (generic_scm variables) ----
def set_var(name, value, *, est_sd=None, label=None) -> Action:
    """`do(name := value)` on a generic_scm variable (e.g. price, discount). Optionally reset its est_sd."""
    def t(spec):
        s = copy.deepcopy(spec)
        v = s.var(name)
        if v is None:
            raise KeyError(f"no variable {name!r} in spec")
        v.value = float(value)
        if est_sd is not None:
            v.est_sd = float(est_sd)
        return s
    return Action(label or f"{name}={round(float(value), 4)}", t, "parameter", {"name": name, "value": value})


def shift_var(name, delta, *, label=None) -> Action:
    """Shift a variable by `delta` (a nudge relative to its current value)."""
    def t(spec):
        s = copy.deepcopy(spec)
        v = s.var(name)
        if v is None:
            raise KeyError(f"no variable {name!r} in spec")
        v.value = float(v.value + delta)
        return s
    sign = "+" if delta >= 0 else ""
    return Action(label or f"{name}{sign}{delta}", t, "parameter", {"name": name, "delta": delta})


# ---- the message do-operator (single_agent) — the best_message intervention ----
def set_message(message: dict, *, label=None) -> Action:
    """Choose the message sent to the person (single_agent). This is the `do(x)` behind best-message."""
    def t(spec):
        s = copy.deepcopy(spec)
        s.extra = dict(s.extra)
        s.extra["message"] = dict(message)
        return s
    return Action(label or "message", t, "parameter", {"message": message})


# ---- structural interventions ----
def set_competitor(name, *, strength=None, est_sd=None, label=None) -> Action:
    """Change a competitor's strength/uncertainty in a bracket (a trade, an injury, a new entrant's rating)."""
    def t(spec):
        s = copy.deepcopy(spec)
        for c in s.extra.get("competitors", []):
            if c.get("name") == name:
                if strength is not None:
                    c["strength"] = strength
                if est_sd is not None:
                    c["est_sd"] = est_sd
        return s
    return Action(label or f"{name}~strength", t, "structural", {"name": name, "strength": strength})


def set_cell(cell_id, *, stance=None, turnout=None, label=None) -> Action:
    """Intervene on a demographic cell in an electorate (a mobilization campaign raises turnout/stance)."""
    def t(spec):
        s = copy.deepcopy(spec)
        for i, c in enumerate(s.extra.get("cells", [])):
            if c.get("id", str(i)) == cell_id:
                if stance is not None:
                    c["stance"] = stance
                if turnout is not None:
                    c["turnout"] = turnout
        return s
    return Action(label or f"cell:{cell_id}", t, "structural", {"cell_id": cell_id})


# ---- escape hatch: any spec -> spec transform (temporal shocks, novel structures) ----
def custom(label, transform, *, kind="custom", **meta) -> Action:
    return Action(label, transform, kind, meta)
