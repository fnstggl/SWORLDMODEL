"""GeneralSimulator — the front door: any question flows all the way through to an outcome.

The assembled parts finally behind one call. A question arrives; the simulator ROUTES it to the machinery
that actually carries its signal, and fuses whatever evidence is available into one calibrated forecast
with an auditable breakdown:

  - POPULATION question with a modelled population  → `GroundedSimulator` (EXP-050): map each person's
    grounded variables, estimate with the unified readout, aggregate bottom-up to the collective outcome.
  - NOVEL question with as-of NEWS                  → `SemanticStanceJudge` (EXP-047): read the news for
    THIS outcome's specific resolution, turn the stance into P.
  - NOVEL question with world-knowledge only        → `QuestionEngine` (EXP-037): infer the drivers and
    aggregate them in log-odds (Tetlock's incremental update).

When several fire, they are fused in log-odds weighted by each source's confidence — so a question with a
population AND news AND drivers uses all three. This is the concrete "put in any scenario → simulate →
most-likely outcome," routing to real simulation where a population exists and to grounded reading where
it does not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.api.grounded_simulate import GroundedSimulator
from swm.api.question_engine import QuestionEngine


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(z):
    if z < -35:
        return 1e-15
    if z > 35:
        return 1 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class GeneralForecast:
    p_outcome: float
    method: str                       # which source(s) fired
    confidence: float
    sources: dict = field(default_factory=dict)     # per-source p + weight (auditable)
    value_drivers: list = field(default_factory=list)

    def as_dict(self):
        return {"p_outcome": round(self.p_outcome, 4), "method": self.method,
                "confidence": round(self.confidence, 3),
                "sources": {k: {"p": round(v["p"], 4), "weight": round(v["weight"], 3)}
                            for k, v in self.sources.items()},
                "value_drivers": [[i, round(c, 4)] for i, c in self.value_drivers]}


@dataclass
class GeneralSimulator:
    """One `answer()` for any social question — routes to simulation / reading / drivers and fuses them."""
    grounded: GroundedSimulator = None                # type: ignore; a fitted population simulator (optional)
    question_engine: QuestionEngine = field(default_factory=QuestionEngine)
    stance_judge: object = None                       # a SemanticStanceJudge (optional)
    stance_scale: float = 1.6                          # how hard a unit of confident stance moves the logit

    def answer(self, question: str, *, population=None, known_item: str = None, news=None,
               base_rate: float = 0.5, resolution_hint: str = "", driver_infer_fn=None,
               n_views: int = 1) -> GeneralForecast:
        sources = {}
        drivers = []

        # 1. population simulation (real bottom-up simulation where a population exists)
        if self.grounded is not None and known_item is not None and population:
            fc = self.grounded.simulate_population(known_item, population)
            sources["population_simulation"] = {"p": fc.p_outcome, "weight": max(0.2, fc.confidence)}
            drivers = fc.value_drivers

        # 2. semantic news reading (grounded content for THIS resolution)
        if self.stance_judge is not None and news:
            s = self.stance_judge.stance(question, news, resolution_hint)
            p_news = _sigmoid(_logit(base_rate) + self.stance_scale * s["stance"] * s["confidence"])
            sources["news_stance"] = {"p": p_news, "weight": max(0.1, s["confidence"])}

        # 3. driver inference (world-knowledge decomposition)
        if driver_infer_fn is not None:
            qf = self.question_engine.forecast(question, driver_infer_fn, n_views=n_views)
            sources["driver_engine"] = {"p": qf.p_outcome, "weight": max(0.1, qf.confidence)}

        if not sources:
            return GeneralForecast(base_rate, "prior_only", 0.0, {}, [])

        # fuse in log-odds, weighted by source confidence
        wsum = sum(v["weight"] for v in sources.values())
        z = sum(v["weight"] * _logit(v["p"]) for v in sources.values()) / wsum
        p = _sigmoid(z)
        conf = min(0.98, (abs(p - 0.5) * 2) * min(1.0, wsum))
        return GeneralForecast(p_outcome=p, method="+".join(sources), confidence=conf,
                               sources=sources, value_drivers=drivers)
