"""Spec validator + repair loop — a linter and smoke-test for LLM-compiled structural models.

EXP-066 exposed the one real failure of autonomous compilation: the LLM gets the STRUCTURE right but makes
numeric bugs in the EQUATIONS (its inflation model's mean-reversion term had an equilibrium at ~35% with a
hi bound of 10, so the variable pinned to the bound and the forecast degenerated to P=1.0). That is exactly
the class of error a linter + test-run catches before you trust generated code. This module is that:

  - `validate(spec)` runs STATIC checks (undeclared variables, out-of-range values, an event threshold
    outside the variable's support) and DYNAMIC checks (it actually simulates the spec and inspects the
    result): equilibrium-out-of-bounds, variable-saturates-a-bound, a degenerate/collapsed outcome, and
    volatility too large for the horizon. Each is an `Issue` with a severity.
  - `ValidatingCompiler` wraps any compiler: compile -> validate -> if there are errors, hand the spec and
    the concrete issues back to the LLM to REPAIR, re-validate, up to a few rounds — then return a spec
    that either passes or is flagged as unrepaired. Pluggable `repair_fn` (LLM backend or cached), same
    pattern as the rest of the system.

The equilibrium check is the load-bearing one: for a `generic_scm` variable it finds where the drift is
zero (the level the variable is pulled toward) and flags it if that level is outside the declared bounds —
catching the inflation bug automatically, from first principles, with no knowledge of the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.api.compiler import CompiledModel
from swm.api.model_spec import ModelSpec, parse_spec, safe_eval


@dataclass
class Issue:
    code: str
    severity: str          # "error" (will produce garbage) | "warn" (suspicious)
    message: str
    variable: str = ""

    def as_dict(self):
        return {"code": self.code, "severity": self.severity, "message": self.message,
                "variable": self.variable}


def _state(spec: ModelSpec) -> dict:
    return {v.name: v.value for v in spec.variables}


def _drift_at(spec: ModelSpec, var: str, x: float) -> float:
    st = _state(spec)
    st[var] = x
    return safe_eval(spec.equations[var], st)


def _equilibrium(spec: ModelSpec, var: str):
    """Find where drift(var)=0 (the level the variable is pulled toward), holding others at their values.
    Scans a wide band around the bounds and interpolates the first sign change; None if drift is monotone
    (variable is pushed to a bound with no interior rest point)."""
    v = spec.var(var)
    lo, hi = v.lo, v.hi
    span = (hi - lo) or 1.0
    n = 240
    prev = None
    for i in range(n + 1):
        x = (lo - span) + i * (3 * span) / n          # scan [lo-span, hi+2*span]
        try:
            g = _drift_at(spec, var, x)
        except Exception:
            return "unparseable"
        if prev is not None:
            x0, g0 = prev
            if (g0 <= 0 <= g or g0 >= 0 >= g) and g != g0:
                return x0 + (0 - g0) * (x - x0) / (g - g0)
        prev = (x, g)
    return None


def validate(spec: ModelSpec, n: int = 400) -> list:
    """Return the list of Issues for a compiled spec (empty = clean). Static checks + a simulate-and-inspect
    dynamic pass. Focused on generic_scm (where equations live); light sanity for other mechanisms."""
    issues = []
    if spec.mechanism == "generic_scm":
        names = {v.name for v in spec.variables}
        if not spec.variables:
            issues.append(Issue("no_variables", "error", "generic_scm has no variables"))
        target = spec.outcome.get("variable") or (spec.variables[0].name if spec.variables else None)
        # --- static ---
        for v in spec.variables:
            if not (v.lo <= v.value <= v.hi):
                issues.append(Issue("value_out_of_bounds", "error",
                                    f"{v.name} value {v.value} outside [{v.lo},{v.hi}]", v.name))
            if v.est_sd < 0 or v.volatility < 0:
                issues.append(Issue("negative_uncertainty", "error",
                                    f"{v.name} has negative est_sd/volatility", v.name))
            if (v.hi - v.lo) > 0 and v.volatility * (spec.horizon ** 0.5) > 2 * (v.hi - v.lo):
                issues.append(Issue("volatility_too_large", "warn",
                                    f"{v.name} volatility*sqrt(horizon) exceeds 2x its range -> "
                                    f"uninformatively wide", v.name))
        for var, expr in spec.equations.items():
            try:
                safe_eval(expr, {**_state(spec), var: spec.var(var).value if spec.var(var) else 0.0})
            except Exception as e:
                issues.append(Issue("bad_equation", "error", f"equation for {var} fails: {str(e)[:80]}", var))
                continue
            # --- equilibrium: the level the variable is pulled toward must be inside its bounds ---
            v = spec.var(var)
            if v is None:
                issues.append(Issue("equation_unknown_var", "error",
                                    f"equation targets undeclared variable {var}", var))
                continue
            eq = _equilibrium(spec, var)
            if eq == "unparseable":
                issues.append(Issue("bad_equation", "error", f"equation for {var} unparseable", var))
            elif eq is None:
                d_hi = _drift_at(spec, var, v.hi)
                issues.append(Issue("saturates_bound", "error",
                                    f"{var} drift never reaches zero in-range -> it saturates the "
                                    f"{'upper' if d_hi > 0 else 'lower'} bound (no interior equilibrium)", var))
            elif not (v.lo <= eq <= v.hi):
                issues.append(Issue("equilibrium_out_of_bounds", "error",
                                    f"{var} is pulled toward {eq:.2f}, outside its bounds [{v.lo},{v.hi}] "
                                    f"-> it will pin to a bound", var))
        # event threshold must lie within the outcome variable's support to be non-trivial
        ev = spec.outcome.get("event")
        tv = spec.var(target) if target else None
        if ev and tv is not None and not (tv.lo <= float(ev.get("value", 0.5)) <= tv.hi):
            issues.append(Issue("event_threshold_outside_support", "error",
                                f"event threshold {ev.get('value')} outside {target}'s [{tv.lo},{tv.hi}] "
                                f"-> P is trivially 0 or 1", target))

    # --- dynamic: actually simulate and inspect (all mechanisms) ---
    try:
        out = CompiledModel(spec).run(n=n)
        iv = out.get("interval_80")
        if iv and (iv[1] - iv[0]) < 1e-4:
            issues.append(Issue("degenerate_outcome", "error",
                                "outcome interval collapsed to a point -> no uncertainty (a bound pin or "
                                "a constant)"))
        pe = out.get("p_event")
        if pe is not None and (pe < 1e-3 or pe > 1 - 1e-3) and iv and (iv[1] - iv[0]) < 1e-3:
            issues.append(Issue("trivial_event", "error", f"P(event)={pe} with no spread -> the event is "
                                "trivially certain; check bounds/threshold/equation"))
    except Exception as e:
        issues.append(Issue("simulation_error", "error", f"spec fails to simulate: {str(e)[:80]}"))
    return issues


def build_repair_prompt(spec_json: dict, issues: list) -> str:
    import json
    bullets = "\n".join(f"  - [{i.severity}] {i.code}: {i.message}" for i in issues)
    return ("Your structural-model spec has bugs found by a validator that SIMULATED it. Fix ONLY what the "
            "issues call out; keep the mechanism, the sensible values, and the intent. Common fix: an "
            "equilibrium/mean-reversion term must pull the variable toward a level INSIDE its [lo,hi] "
            "bounds (e.g. for mean-reversion to level L write '-k*(x - L)', not 'k*(C - x)' with C outside "
            "the range).\n\nSPEC:\n" + json.dumps(spec_json, indent=1) + "\n\nISSUES:\n" + bullets +
            "\n\nReturn ONLY the corrected JSON spec.")


@dataclass
class ValidatingCompiler:
    """Wrap a compiler with validate -> repair -> re-validate. `repair_fn(prompt) -> spec JSON|dict`."""
    compiler: object
    repair_fn: object = None
    max_repairs: int = 2
    last_report: dict = None

    def compile(self, question: str, context: str = "", *, key: str = None) -> CompiledModel:
        compiled = self.compiler.compile(question, context, key=key)
        issues = validate(compiled.spec)
        rounds = 0
        while any(i.severity == "error" for i in issues) and self.repair_fn and rounds < self.max_repairs:
            spec_json = _spec_to_json(compiled.spec)
            errs = [i for i in issues if i.severity == "error"]
            fixed = self.repair_fn(build_repair_prompt(spec_json, errs))
            compiled = CompiledModel(parse_spec(fixed))
            issues = validate(compiled.spec)
            rounds += 1
        self.last_report = {"clean": not any(i.severity == "error" for i in issues),
                            "repairs": rounds, "issues": [i.as_dict() for i in issues]}
        return compiled


def _spec_to_json(spec: ModelSpec) -> dict:
    return {"mechanism": spec.mechanism,
            "variables": [{"name": v.name, "value": v.value, "est_sd": v.est_sd, "volatility": v.volatility,
                           "lo": v.lo, "hi": v.hi} for v in spec.variables],
            "equations": spec.equations, "outcome": spec.outcome, "horizon": spec.horizon, "dt": spec.dt,
            "extra": spec.extra, "rationale": spec.rationale}
