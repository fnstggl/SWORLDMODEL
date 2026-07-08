"""Structural-model spec — the compiler's intermediate representation + a SAFE equation evaluator.

The world-model compiler's contract: the LLM reads a question + retrieved context and emits a SPEC — the
mechanism, the variables (each with a value, an estimate-uncertainty, and a real per-unit-time volatility),
the structural equations that couple them, the outcome to read, and the horizon. This module is the typed
IR for that spec and, critically, a **whitelisted expression evaluator** so the LLM-authored structural
equations can be evaluated WITHOUT `eval()` of arbitrary code — only arithmetic, the declared variables,
numeric constants, and a small set of math functions are permitted. Anything else raises.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field

_FUNCS = {"min": min, "max": max, "abs": abs, "exp": math.exp, "log": lambda x: math.log(max(1e-12, x)),
          "sqrt": lambda x: math.sqrt(max(0.0, x)), "tanh": math.tanh, "clip": lambda x, a, b: max(a, min(b, x))}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b if b else 0.0, ast.Pow: lambda a, b: a ** b,
           ast.Mod: lambda a, b: a % b if b else 0.0}
_CMP = {ast.Gt: lambda a, b: a > b, ast.Lt: lambda a, b: a < b, ast.GtE: lambda a, b: a >= b,
        ast.LtE: lambda a, b: a <= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


def _ev(node, ns):
    if isinstance(node, ast.Expression):
        return _ev(node.body, ns)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"only numeric constants allowed, got {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in ns:
            return float(ns[node.id])
        raise ValueError(f"unknown variable {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_ev(node.left, ns), _ev(node.right, ns))
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, ns)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_ev(a, ns) for a in node.args])
    if isinstance(node, ast.IfExp):
        return _ev(node.body, ns) if _ev(node.test, ns) else _ev(node.orelse, ns)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
        return _CMP[type(node.ops[0])](_ev(node.left, ns), _ev(node.comparators[0], ns))
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v, ns) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def safe_eval(expr, ns: dict) -> float:
    """Evaluate a structural-equation string against a variable namespace, safely (whitelisted AST only)."""
    try:
        return float(_ev(ast.parse(str(expr), mode="eval"), ns))
    except ValueError:
        raise
    except Exception as e:                                # malformed expr -> 0 drift, never crash a rollout
        raise ValueError(f"bad expression {expr!r}: {e}")


@dataclass
class SpecVar:
    name: str
    value: float
    est_sd: float = 0.0            # epistemic uncertainty in the current estimate
    volatility: float = 0.0        # aleatoric per-unit-time volatility (calibrated to timescale)
    lo: float = 0.0
    hi: float = 1.0
    # calibrated-readout fields: the variable's ELASTICITY toward the outcome, with uncertainty + provenance
    weight: float = None          # signed per-unit push on the outcome logit (None => not a readout variable)
    weight_sd: float = None       # uncertainty in that elasticity (the CI); wide => the model is unsure of it
    center: float = 0.5           # the neutral value the push is measured from
    weight_source: str = "llm"    # provenance: llm | literature | registry | fit


@dataclass
class ModelSpec:
    """The compiled structural model, mechanism-tagged. `extra` carries mechanism-specific payloads
    (competitors for a bracket, agents for a committee, cells for an electorate)."""
    mechanism: str
    variables: list = field(default_factory=list)     # list[SpecVar]  (for generic_scm)
    equations: dict = field(default_factory=dict)     # var_name -> drift expression string
    outcome: dict = field(default_factory=dict)       # {"variable","event":{"op","value"}} or {"target"}
    horizon: float = 6.0
    dt: float = 1.0
    extra: dict = field(default_factory=dict)
    rationale: str = ""

    def var(self, name):
        return next((v for v in self.variables if v.name == name), None)


def parse_spec(raw) -> ModelSpec:
    """Tolerant parse of an LLM spec payload (dict or raw JSON string) into a ModelSpec."""
    obj = raw if isinstance(raw, dict) else json.loads(str(raw)[str(raw).find("{"):str(raw).rfind("}") + 1])
    def _optf(v, k):
        return float(v[k]) if v.get(k) is not None else None
    variables = [SpecVar(name=v["name"], value=float(v.get("value", 0.5)),
                         est_sd=float(v.get("est_sd", 0.0)), volatility=float(v.get("volatility", 0.0)),
                         lo=float(v.get("lo", 0.0)), hi=float(v.get("hi", 1.0)),
                         weight=_optf(v, "weight"), weight_sd=_optf(v, "weight_sd"),
                         center=float(v.get("center", 0.5)), weight_source=str(v.get("weight_source", "llm")))
                 for v in obj.get("variables", [])]
    return ModelSpec(mechanism=str(obj.get("mechanism", "generic_scm")), variables=variables,
                     equations={k: str(v) for k, v in (obj.get("equations") or {}).items()},
                     outcome=obj.get("outcome", {}), horizon=float(obj.get("horizon", 6.0)),
                     dt=float(obj.get("dt", 1.0)), extra=obj.get("extra", {}),
                     rationale=str(obj.get("rationale", ""))[:500])
