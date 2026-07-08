"""WorldModel — the front door. One `simulate(question)` call, the whole system underneath.

This realizes the vision in a single call: retrieve the relevant slice of the world, COMPILE it into the
right structural model (mechanism + variables + equations + calibrated timescales), roll it forward as a
Monte-Carlo ensemble, and return a calibrated outcome DISTRIBUTION with an honest reducible/irreducible
split and a forecastability verdict — never a false-confident point.

    wm = WorldModel(retriever, compiler)
    wm.simulate("Will the FOMC raise rates in July?")   # -> {mechanism, forecast, uncertainty, horizon, ...}

Both dependencies are pluggable: `retriever` from swm/api/retrieval.py (web search in prod, as-of in eval),
`compiler` a StructuralCompiler with an LLM or cached backend. The Levels 1-3, the bracket, and the
generic SCM are the compiler's mechanism library — this is the layer that turns them into "ask anything."

Self-correction is ON by default: the compiler is wrapped in a `ValidatingCompiler`, so every simulate()
call VALIDATES the compiled spec (simulate-and-inspect) and, if a `repair_fn` is supplied, REPAIRS
degeneracies (the EXP-067 loop) before running — the generated model is tested before it is trusted. The
`validation` report rides along in the output. Pass `validate=False` for the raw path.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.api.spec_validator import ValidatingCompiler


@dataclass
class WorldModel:
    compiler: object                      # swm.api.compiler.StructuralCompiler
    retriever: object = None              # swm.api.retrieval.Retriever (optional; context can be passed in)
    n: int = 8000
    validate: bool = True                 # self-correct by default: validate (+ repair if repair_fn) each spec
    repair_fn: object = None              # LLM backend that fixes a flagged spec (None = validate-only)

    def _compiler(self):
        if not self.validate:
            return self.compiler
        if isinstance(self.compiler, ValidatingCompiler):
            return self.compiler
        return ValidatingCompiler(self.compiler, repair_fn=self.repair_fn)

    def simulate(self, question: str, *, context: str = "", as_of: str = "", key: str = None,
                 n: int = None) -> dict:
        if self.retriever is not None and not context:
            ctx = self.retriever.retrieve(question, as_of=as_of)
            context = ctx.to_prompt()
            n_evidence = len(ctx)
        else:
            n_evidence = 0
        comp = self._compiler()
        compiled = comp.compile(question, context, key=key)
        validation = getattr(comp, "last_report", None)
        try:                                              # a spec validation could not repair may not run
            forecast = compiled.run(n=n or self.n)
        except Exception as e:
            forecast = {"mechanism": compiled.spec.mechanism, "error": str(e)[:120]}
        return {"question": question, "n_evidence": n_evidence,
                "mechanism": compiled.spec.mechanism, "forecast": forecast,
                "forecastable": _forecastable(forecast),
                "validation": validation,
                "headline": _headline(question, forecast),
                "spec": {"mechanism": compiled.spec.mechanism,
                         "variables": [(v.name, v.value, v.volatility) for v in compiled.spec.variables],
                         "equations": compiled.spec.equations, "outcome": compiled.spec.outcome,
                         "horizon": compiled.spec.horizon, "rationale": compiled.spec.rationale}}


def _forecastable(forecast: dict):
    u = forecast.get("uncertainty")
    if isinstance(u, dict) and "forecastable" in u:
        return u["forecastable"]
    return None                           # not defined for categorical/agent mechanisms


def _headline(question: str, forecast: dict) -> str:
    m = forecast.get("mechanism")
    if m == "bracket":
        return (f"{forecast.get('favorite')} most likely ({forecast.get('p_target') or '-'} for the named "
                f"target); most of the spread is irreducible tournament variance")
    if forecast.get("p_event") is not None:
        return f"P(event) = {forecast['p_event']}  (80% interval {forecast.get('interval_80')})"
    if "mean" in forecast:
        return f"outcome ~ {forecast['mean']}  (80% interval {forecast.get('interval_80')})"
    return "see forecast"
