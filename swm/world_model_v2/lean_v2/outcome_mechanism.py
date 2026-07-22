"""D16 — dimensional outcome mechanisms. The terminal must be produced as the EXACT variable the
question asks for, in the EXACT units, by a chain whose every link exists — never collapsed to a
qualitative boolean or computed in the wrong dimension.

The EXP-113 failures this eliminates:
  * a `count >= 50 tankers/day` terminal was resolved as a boolean OR of unrelated events (Hormuz);
  * an interest-rate `level` question was scored against a `votes` mechanism; a `rate` (per-time)
    threshold was compared to a bare `count`.

`OutcomeMechanismSpec` types the whole pathway:

    observations / actor behavioral inputs (with units)
      → deterministic transitions (each dimensionally checked)
      → the output variable (with a unit)
      → aggregation over a window
      → comparator vs threshold (in the SAME dimension as the output)

A hard dimensional validator reconciles the mechanism's output dimension with the resolution's
required dimension (D5), rejects unit mismatches and dimensionless-boolean collapses, and checks
the backward dependency chain terminal ← variable ← transitions ← inputs is unbroken. HYBRID by
construction: actors decide the BEHAVIORAL inputs; deterministic code computes the counts / tallies
/ rates. Bounded ranges propagate; the output is never a qualitative label.

Universal: dimensions are inferred from the unit strings, never hardcoded per question."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm, norm_key

OUTCOME_MECHANISM_VERSION = "lean_v2.outcome_mechanism.v1"

# base dimensions the terminal can require
DIM_COUNT = "count"                 # a pure count of things (tankers, votes, seats, filings)
DIM_RATE = "rate"                   # count per unit time (tankers/day, cases/week)
DIM_RATIO = "ratio"                 # dimensionless fraction / percentage / probability
DIM_CURRENCY = "currency"           # money (USD, EUR, ...)
DIM_LEVEL = "level"                 # a level in some scale (rate %, price, index, temperature)
DIM_EVENT = "event"                 # a boolean occurrence by a deadline
DIM_DURATION = "duration"           # a span of time
DIM_UNKNOWN = "unknown"

#: unit-string → dimension inference (checked in order; first match wins)
_UNIT_RULES = [
    (re.compile(r"\bper\s*(day|week|month|hour|year|annum)\b|/\s*(day|week|month|hr|hour|yr|year)"
                r"|\bdaily\b|\bweekly\b|\bmonthly\b", re.I), DIM_RATE),
    (re.compile(r"\b(usd|eur|gbp|jpy|dollars?|euros?|yen|pounds?|\$|€|£|¥)\b", re.I), DIM_CURRENCY),
    (re.compile(r"%|percent|percentage|share of|fraction|probability|ratio|bps?|basis points?",
                re.I), DIM_RATIO),
    (re.compile(r"\b(vote|votes|seat|seats|tanker|tankers|ship|ships|filing|filings|case|cases|"
                r"unit|units|barrels?|people|members?|days count|number of)\b", re.I), DIM_COUNT),
    (re.compile(r"\b(basis points?|bp|price|index|level|rate of|degrees?|°|temperature|"
                r"yield)\b", re.I), DIM_LEVEL),
    (re.compile(r"\b(occurs?|occurred|happens?|announced?|passes?|by (the )?deadline|before|"
                r"event|yes/no|boolean)\b", re.I), DIM_EVENT),
    (re.compile(r"\b(days|weeks|months|hours|years|duration)\b", re.I), DIM_DURATION),
]


def infer_dimension(unit: str) -> str:
    """The base dimension of a unit string. Deterministic; 'tankers/day' → rate, 'votes' → count,
    '%' → ratio, 'by the deadline' → event."""
    u = str(unit or "").strip()
    if not u:
        return DIM_UNKNOWN
    for patt, dim in _UNIT_RULES:
        if patt.search(u):
            return dim
    return DIM_UNKNOWN


def dimensions_compatible(unit_a: str, unit_b: str) -> bool:
    da, db = infer_dimension(unit_a), infer_dimension(unit_b)
    if DIM_UNKNOWN in (da, db):
        # fall back to a normalized string comparison when a dimension can't be inferred
        return norm_key(unit_a) == norm_key(unit_b) or not unit_a or not unit_b
    return da == db


@dataclass
class MechanismInput:
    name: str
    unit: str = ""
    source: str = "observation"        # observation | actor_decision (hybrid boundary)

    def as_dict(self) -> dict:
        return {"name": self.name, "unit": self.unit, "source": self.source}


@dataclass
class Transition:
    op: str                            # sum | count_over_window | rate | average | level | max | last
    inputs: list = field(default_factory=list)     # input names consumed
    output: str = ""                   # the variable produced
    unit: str = ""                     # the unit of the produced variable

    def as_dict(self) -> dict:
        return {"op": self.op, "inputs": list(self.inputs), "output": self.output, "unit": self.unit}


@dataclass
class OutcomeMechanismSpec:
    output_variable: str
    output_unit: str
    comparator: str = ">="             # >= | > | <= | < | ==
    threshold: float = None
    threshold_unit: str = ""
    aggregation: str = "level"         # sum | average | count | level | max | last | any_day
    window: str = ""
    inputs: list = field(default_factory=list)         # MechanismInput
    transitions: list = field(default_factory=list)    # Transition
    evidence: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)    # {lo, hi} bounded range on the output
    version: str = OUTCOME_MECHANISM_VERSION

    def output_dimension(self) -> str:
        return infer_dimension(self.output_unit)

    def as_dict(self) -> dict:
        return {"output_variable": self.output_variable, "output_unit": self.output_unit,
                "output_dimension": self.output_dimension(), "comparator": self.comparator,
                "threshold": self.threshold, "threshold_unit": self.threshold_unit,
                "aggregation": self.aggregation, "window": self.window,
                "inputs": [i.as_dict() for i in self.inputs],
                "transitions": [t.as_dict() for t in self.transitions],
                "evidence": list(self.evidence), "assumptions": list(self.assumptions),
                "uncertainty": dict(self.uncertainty), "version": self.version}


# operations and the input→output dimension they are allowed to produce
_OP_DIMENSION = {
    "sum": {DIM_COUNT: DIM_COUNT, DIM_CURRENCY: DIM_CURRENCY},
    "count_over_window": {DIM_EVENT: DIM_COUNT, DIM_COUNT: DIM_COUNT},   # events → a count
    "rate": {DIM_COUNT: DIM_RATE},                                       # count over time → rate
    "average": {DIM_COUNT: DIM_COUNT, DIM_LEVEL: DIM_LEVEL, DIM_RATE: DIM_RATE,
                DIM_CURRENCY: DIM_CURRENCY, DIM_RATIO: DIM_RATIO},
    "level": {DIM_LEVEL: DIM_LEVEL, DIM_RATIO: DIM_RATIO},
    "max": {DIM_COUNT: DIM_COUNT, DIM_LEVEL: DIM_LEVEL, DIM_RATE: DIM_RATE},
    "last": {DIM_LEVEL: DIM_LEVEL, DIM_RATIO: DIM_RATIO, DIM_COUNT: DIM_COUNT},
}


def validate_outcome_mechanism(spec: OutcomeMechanismSpec, resolution_spec=None) -> tuple:
    """(ok, diagnostics). Hard dimensional + structural checks:
      1. the output dimension MATCHES the resolution's required dimension (D5) — a votes/count/rate/
         level/event terminal is scored by a mechanism of that dimension, never another;
      2. the comparator threshold is in the OUTPUT dimension (no count vs rate);
      3. no dimensionless-boolean collapse of a genuinely numeric terminal;
      4. every transition is a legal operation for its input dimension and yields its stated unit;
      5. the backward chain is unbroken: the output variable is produced by some transition whose
         inputs trace back to declared inputs (no orphan variable)."""
    d = []
    out_dim = spec.output_dimension()

    # (1) output dimension vs the resolution's required unit
    req_unit = getattr(resolution_spec, "unit", "") if resolution_spec is not None else ""
    req_dim = infer_dimension(req_unit) if req_unit else DIM_UNKNOWN
    if req_dim != DIM_UNKNOWN and out_dim != DIM_UNKNOWN and out_dim != req_dim:
        d.append(f"output dimension {out_dim} ({spec.output_unit!r}) != required {req_dim} "
                 f"({req_unit!r}) — mechanism computes the wrong quantity")

    # (3) numeric terminal must not be a boolean collapse
    if req_dim in (DIM_COUNT, DIM_RATE, DIM_LEVEL, DIM_CURRENCY, DIM_RATIO) and out_dim == DIM_EVENT:
        d.append(f"required {req_dim} terminal collapsed to a boolean event — dimensionless")

    # (2) threshold in the output dimension
    if spec.threshold is not None and spec.threshold_unit:
        if not dimensions_compatible(spec.threshold_unit, spec.output_unit):
            d.append(f"threshold unit {spec.threshold_unit!r} ({infer_dimension(spec.threshold_unit)})"
                     f" != output unit {spec.output_unit!r} ({out_dim})")

    # (4) each transition legal for its input dimension and consistent with its declared unit
    produced = {}
    input_units = {i.name: i.unit for i in spec.inputs}
    for t in spec.transitions:
        in_dims = {infer_dimension(input_units.get(n) or produced.get(n, "")) for n in t.inputs}
        in_dims.discard(DIM_UNKNOWN)
        allowed = _OP_DIMENSION.get(t.op)
        if allowed is not None and in_dims:
            for idim in in_dims:
                exp = allowed.get(idim)
                if exp is not None and infer_dimension(t.unit) not in (exp, DIM_UNKNOWN):
                    d.append(f"transition '{t.op}' on {idim} should yield {exp}, but its unit "
                             f"{t.unit!r} is {infer_dimension(t.unit)}")
        produced[t.output] = t.unit

    # (5) backward chain: the output variable must be produced (or be a declared input)
    all_outputs = {t.output for t in spec.transitions} | {i.name for i in spec.inputs}
    if spec.transitions and spec.output_variable not in all_outputs:
        d.append(f"output variable {spec.output_variable!r} is produced by no transition and is "
                 f"not a declared input — the dependency chain is broken")
    # every transition input must exist upstream (declared input or earlier transition output)
    seen = set(input_units)
    for t in spec.transitions:
        for n in t.inputs:
            if n not in seen:
                d.append(f"transition '{t.op}' consumes undefined variable {n!r}")
        seen.add(t.output)

    return (not d, d)


# ------------------------------------------------------------------ adapter for recovered processes
_AGG_TO_OP = {"any_day": "max", "average": "average", "total": "sum", "level": "last",
              "sum": "sum", "count": "count_over_window", "max": "max", "last": "last"}


def outcome_mechanism_from_bounded_process(mechanism: dict, resolution_spec=None
                                           ) -> OutcomeMechanismSpec:
    """Adapt a recovered `bounded_numeric_process` (mechanisms.py) into a typed OutcomeMechanismSpec
    so its dimensions can be validated against the resolution. The observations' unit is the output
    unit; the aggregation maps to a transition op."""
    obs = mechanism.get("observations") or []
    unit = next((o.get("unit") for o in obs if o.get("unit")), "") \
        or (getattr(resolution_spec, "unit", "") if resolution_spec is not None else "")
    var = mechanism.get("variable") or "terminal_variable"
    agg = mechanism.get("aggregation") or "level"
    names = [f"obs_{o.get('obs_id', i)}" for i, o in enumerate(obs[:8])]
    return OutcomeMechanismSpec(
        output_variable=var, output_unit=unit, comparator=mechanism.get("comparator") or ">=",
        threshold=mechanism.get("threshold"), threshold_unit=unit, aggregation=agg,
        inputs=[MechanismInput(n, unit, "observation") for n in names],
        transitions=[Transition(_AGG_TO_OP.get(agg, "last"), names, var, unit)],
        uncertainty={"lo": mechanism.get("min_rate"), "hi": mechanism.get("max_rate")})


def dimensional_check_mechanism(mechanism: dict, resolution_spec=None) -> dict:
    """Validate a recovered mechanism's dimensions against the resolution. Returns
    {ok, dimension, required_dimension, diagnostics, spec}."""
    spec = outcome_mechanism_from_bounded_process(mechanism, resolution_spec)
    ok, diag = validate_outcome_mechanism(spec, resolution_spec)
    return {"ok": ok, "dimension": spec.output_dimension(),
            "required_dimension": infer_dimension(getattr(resolution_spec, "unit", "")
                                                  if resolution_spec is not None else ""),
            "diagnostics": diag, "spec": spec.as_dict()}
