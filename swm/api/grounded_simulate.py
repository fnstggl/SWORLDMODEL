"""GroundedSimulator — the assembled end-to-end pipeline on a real question.

The whole thesis in one callable: take a real social question and a population, map each person's grounded
variables onto their latent value profile, estimate their answer with the unified GroundedReadout
(structure + world-knowledge prior + reliability weighting), and aggregate bottom-up to a calibrated
outcome — with an auditable value-factor decomposition of *why*.

    question + population
       │  map grounded variables (VariableMap provenance -> reliability)
       ▼
    GroundedReadout: decorrelate -> latent value factors; regularize toward the LLM world-knowledge prior
       │
       ▼  simulate each individual -> P(answer)   [+ their value profile]
    aggregate bottom-up -> P(population outcome)   [+ which value axes drove it]

This is the simulate-the-event pipeline for opinion/behavior outcomes, finally assembled with grounded
variables and a unified estimator — not a price model. `simulate_person` gives an individual's answer +
value profile; `simulate_population` gives the collective outcome + confidence + the driver axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.variables.grounded_readout import GroundedReadout


@dataclass
class GroundedForecast:
    p_outcome: float                 # calibrated P(outcome / support share)
    n: int                           # population size simulated
    confidence: float                # [0,1] from population size + estimator support
    value_drivers: list             # [(factor_index, mean_contribution)] — which value axes moved it
    per_person: list = field(default_factory=list)   # optional individual P(answer)s

    def as_dict(self):
        return {"p_outcome": round(self.p_outcome, 4), "n": self.n,
                "confidence": round(self.confidence, 3),
                "value_drivers": [[i, round(c, 4)] for i, c in self.value_drivers]}


@dataclass
class GroundedSimulator:
    """A fitted, end-to-end grounded question simulator."""
    attrs: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    readout: GroundedReadout = None                  # type: ignore

    def fit(self, rows, k=3, **kw) -> "GroundedSimulator":
        """rows: {qid, answer_idx, demo}. Fits the unified GroundedReadout on the training population."""
        self.readout = GroundedReadout(attrs=self.attrs, provenance=self.provenance, k=k, **kw).fit(rows)
        return self

    # ---- individual ----
    def simulate_person(self, question, demo) -> dict:
        p = self.readout.predict(question, demo)
        profile = {f"value_factor_{i+1}": round(s, 3) for i, s in enumerate(self.readout._features(demo))} \
            if self.readout.use_factors else {}
        return {"p_answer": round(p, 4), "value_profile": profile}

    # ---- population (bottom-up aggregation) ----
    def simulate_population(self, question, population, keep_individuals=False) -> GroundedForecast:
        ps = [self.readout.predict(question, d) for d in population]
        n = len(ps)
        p_outcome = sum(ps) / n if n else 0.5
        # which value axes drove it: mean per-factor contribution to the logit across the population
        drivers = []
        entry = self.readout._models.get(question)
        if entry and self.readout.use_factors:
            w = entry[0]
            scores = [self.readout._features(d) for d in population]
            for c in range(len(w)):
                mc = sum(w[c] * s[c] for s in scores) / n if n else 0.0
                drivers.append((c + 1, mc))
            drivers.sort(key=lambda t: -abs(t[1]))
        # confidence: grows with n and with the estimator actually having a fitted model for this question
        supported = 1.0 if (entry and entry[3] >= 20) else 0.4
        conf = supported * min(1.0, n / 200.0) ** 0.5
        return GroundedForecast(p_outcome=p_outcome, n=n, confidence=conf, value_drivers=drivers[:3],
                                per_person=(ps if keep_individuals else []))
