"""State grounding — make the model know what the world CURRENTLY is, not what the LLM guesses it is.

The compiler emits each variable's WEIGHT (elasticity) now calibrated from the corpus; but its VALUE (the
current state of that variable) is still largely an LLM guess. A calibrated weight times a guessed value is
still a guess. This layer replaces the guessed values of the HIGH-LEVERAGE variables (chosen by variance
triage — we don't ground 100 variables, we ground the ~10 the outcome turns on) with values MEASURED from
real evidence, each carrying provenance and a confidence interval — exactly like the weights do now. A
variable we cannot ground stays at its prior with a wide CI, so the forecast widens honestly instead of
faking precision.

Grounders are pluggable per variable:
  - `DataGrounder` — a structured data source (FRED, polls, market prices, product metrics): as-of value + sd.
  - `RetrievalGrounder` — as-of retrieval + an LLM value-extractor for text-derivable variables.
Feed the grounded spec to the calibrated_readout / compiler runtime; the model then simulates THIS world.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class GroundedValue:
    value: float
    sd: float                          # the CI on the measured value (0 = certain; wide = unsure)
    source: str


class Grounder:
    def ground(self, variable, question=None, as_of=None):
        raise NotImplementedError


@dataclass
class DataGrounder(Grounder):
    """Ground from a structured data source. `fetch(variable, as_of) -> (value, sd) | None`."""
    fetch: object
    name: str = "data"

    def ground(self, variable, question=None, as_of=None):
        r = self.fetch(variable, as_of)
        return GroundedValue(float(r[0]), float(r[1]), self.name) if r is not None else None


@dataclass
class RetrievalGrounder(Grounder):
    """Ground a text-derivable variable via as-of retrieval + a pluggable LLM value-extractor.
    `retriever.retrieve(query, as_of)` and `extract_fn(variable, question, evidence) -> {value, sd} | None`."""
    retriever: object
    extract_fn: object
    name: str = "retrieval"

    def ground(self, variable, question=None, as_of=None):
        ev = self.retriever.retrieve(f"{variable} {question or ''}", as_of=as_of)
        r = self.extract_fn(variable, question, ev)
        if r is None:
            return None
        return GroundedValue(float(r["value"]), float(r.get("sd", 0.2)), self.name)


@dataclass
class StateGrounder:
    """Triage a spec's variables, ground the high-leverage ones from evidence, return the grounded spec + a
    provenance report. `grounders` maps a variable name to a Grounder (or a list tried in order); a `default`
    grounder is tried for any variable without a specific one."""
    grounders: dict = field(default_factory=dict)
    default: object = None
    keep_frac: float = 0.9
    ground_all: bool = False           # if True, skip triage and try to ground every variable

    def _for(self, name):
        g = self.grounders.get(name, [])
        gs = g if isinstance(g, list) else [g]
        return gs + ([self.default] if self.default is not None else [])

    def _ground_var(self, name, question, as_of):
        for g in self._for(name):
            gv = g.ground(name, question, as_of)
            if gv is not None:
                return gv
        return None

    def _high_leverage(self, spec):
        readout_vars = [v for v in spec.variables if v.weight is not None]
        if self.ground_all or not readout_vars:
            return {v.name for v in spec.variables}
        from swm.api.adaptive_fidelity import triage
        return set(triage(spec, keep_frac=self.keep_frac)["invest_in"])

    def ground_spec(self, spec, question=None, *, as_of=None):
        invest = self._high_leverage(spec)
        s = copy.deepcopy(spec)
        report = []
        for v in s.variables:
            gv = self._ground_var(v.name, question, as_of) if v.name in invest else None
            if gv is not None:
                v.value, v.est_sd = gv.value, gv.sd
                report.append({"var": v.name, "grounded": True, "value": round(gv.value, 4),
                               "sd": round(gv.sd, 4), "source": gv.source, "high_leverage": True})
            else:
                report.append({"var": v.name, "grounded": False, "source": "llm_prior",
                               "high_leverage": v.name in invest})
        return s, report


def ground_features(names, grounder: StateGrounder, question=None, *, as_of=None, guess=None):
    """Convenience for a feature-vector model: return the grounded value for each name (or the `guess`
    fallback if ungroundable). `guess` is a dict name->value (the ungrounded prior)."""
    out = []
    for nm in names:
        gv = grounder._ground_var(nm, question, as_of)
        out.append(gv.value if gv is not None else (guess or {}).get(nm, 0.0))
    return out
