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
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldModel:
    compiler: object                      # swm.api.compiler.StructuralCompiler
    retriever: object = None              # swm.api.retrieval.Retriever (optional; context can be passed in)
    n: int = 8000

    def simulate(self, question: str, *, context: str = "", as_of: str = "", key: str = None,
                 n: int = None) -> dict:
        if self.retriever is not None and not context:
            ctx = self.retriever.retrieve(question, as_of=as_of)
            context = ctx.to_prompt()
            n_evidence = len(ctx)
        else:
            n_evidence = 0
        compiled = self.compiler.compile(question, context, key=key)
        forecast = compiled.run(n=n or self.n)
        return {"question": question, "n_evidence": n_evidence,
                "mechanism": compiled.spec.mechanism, "forecast": forecast,
                "forecastable": _forecastable(forecast),
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
