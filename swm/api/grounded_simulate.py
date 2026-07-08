"""IndependentPopulationReadout — a DEMOTED library leaf, NOT a simulation.

⚠️ This is deliberately no longer called "simulate". The simulation audit (SIMULATION_AUDIT.md) established
that `GroundedSimulator.simulate_population` was `sum(ps)/n` — a mean of INDEPENDENT per-person regressions,
`∂pᵢ/∂pⱼ = 0`: no agent reads any other, no state, no interaction, no dynamics. It is a well-calibrated
COMPOSITOR of a fitted logistic readout, not a model of a world. Believable ≠ a simulation.

Its correct place in the architecture: ONE leaf in the compiler's mechanism library — the "independent /
non-interacting population" mechanism — invoked ONLY when the compiler determines a question is a
marginal-recovery problem (e.g. a well-calibrated opinion share where coupling adds nothing, EXP-053/061).
The word **simulate** is reserved for the compiler selecting and running the RIGHT mechanism, rolling it
forward at calibrated time, and returning the navigable outcome — see `swm/api/world_model.py` (the front
door) and `swm/api/action_simulate.py` (the interventional layer). Use those for "simulate the event".

What this leaf still does well, honestly labeled: map each person's grounded variables onto their latent
value profile, estimate their answer with the unified `GroundedReadout` (structure + world-knowledge prior +
reliability weighting), and aggregate bottom-up to a calibrated share — with an auditable value-factor
decomposition of *why*. `predict_person` gives an individual's answer + value profile; `predict_share` gives
the collective share + confidence + the driver axes. The historical names `GroundedSimulator` /
`simulate_population` / `simulate_person` remain as thin back-compat aliases (see bottom of file); they are
deprecated and carry no claim to being a simulation.
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
class IndependentPopulationReadout:
    """A fitted, bottom-up INDEPENDENT-population readout (the compiler's non-interacting mechanism leaf).
    Aggregates independent per-person predictions — no coupling. Not a simulation; see the module docstring."""
    attrs: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    readout: GroundedReadout = None                  # type: ignore

    def fit(self, rows, k=3, **kw) -> "IndependentPopulationReadout":
        """rows: {qid, answer_idx, demo}. Fits the unified GroundedReadout on the training population."""
        self.readout = GroundedReadout(attrs=self.attrs, provenance=self.provenance, k=k, **kw).fit(rows)
        return self

    # ---- individual ----
    def predict_person(self, question, demo) -> dict:
        p = self.readout.predict(question, demo)
        profile = {f"value_factor_{i+1}": round(s, 3) for i, s in enumerate(self.readout._features(demo))} \
            if self.readout.use_factors else {}
        return {"p_answer": round(p, 4), "value_profile": profile}

    # ---- population (bottom-up aggregation of INDEPENDENT predictions; ∂pᵢ/∂pⱼ = 0) ----
    def predict_share(self, question, population, keep_individuals=False) -> GroundedForecast:
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

    # ---- deprecated aliases (do NOT imply a simulation; kept so existing call sites keep working) ----
    def simulate_person(self, question, demo) -> dict:
        return self.predict_person(question, demo)

    def simulate_population(self, question, population, keep_individuals=False) -> GroundedForecast:
        return self.predict_share(question, population, keep_individuals)


# Deprecated name. `GroundedSimulator` was never a simulation (audit: mean of independent regressions). Kept
# as an alias for back-compat; new code should use IndependentPopulationReadout + predict_share, or the real
# simulator front doors (WorldModel / ActionWorldModel).
GroundedSimulator = IndependentPopulationReadout
