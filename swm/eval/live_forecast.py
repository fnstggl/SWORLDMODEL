"""Live forecast harness — retrieve → simulate → forecast → log, for a MEASURED general predictor.

This is the production entry point and the measurement mechanism in one. For any question it:
  1. RETRIEVES real context from the outside world (`swm/api/retrieval.py`);
  2. runs the generative loop (`GenerativeSimulator`) — identify agents, map variables from the retrieved
     context, deliberate — to produce a forecast;
  3. LOGS the forecast to a `PostMortemLog` with its resolution metadata, so it is scored the moment the
     question resolves — a leakage-free skill number by construction (forecast made before, scored after).

On the CUTOFF, precisely: the model's training cutoff does NOT limit what this can DO — retrieval supplies
current evidence, so it forecasts genuinely-future events. The cutoff matters ONLY for honest
*measurement*: a backtest against a KNOWN outcome can be gamed if the model recalls it (training) or
retrieves it (search surfaces the resolved result). So the two clean measurement paths are:
  - FORWARD (this harness): forecast an as-yet-unresolved question, log, score on resolution — no answer
    exists to leak;
  - AS-OF BACKTEST: a question resolving AFTER the cutoff, with retrieval restricted to AS-OF evidence
    (`asof_retriever`) — no memorization, no retrieved outcome.
In production, serving a real user question, there is no leakage to worry about — you are forecasting a
future that has not happened.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiveForecaster:
    retriever: object                 # swm.api.retrieval.Retriever
    simulator: object                 # swm.api.generative_simulator.GenerativeSimulator (identify+position fns set)
    log: object = None                # swm.eval.postmortem.PostMortemLog (optional; enables scoring on resolution)

    def forecast(self, question: str, *, fid: str = None, made_at=None, resolves_at=None,
                 as_of: str = "") -> dict:
        ctx = self.retriever.retrieve(question, as_of=as_of)
        fc = self.simulator.simulate(question, context=ctx.to_prompt())
        p = fc.p_outcome
        if self.log is not None and made_at is not None and resolves_at is not None:
            self.log.log(fid or question, p, made_at, resolves_at,
                         meta={"n_agents": fc.n_agents, "n_evidence": len(ctx), "as_of": as_of,
                               "independent_p": fc.independent_p})
        return {"question": question, "p_outcome": round(p, 4), "n_agents": fc.n_agents,
                "n_evidence": len(ctx), "independent_p": round(fc.independent_p, 4),
                "emergent_shift": round(p - fc.independent_p, 4), "audit": fc.agents,
                "trajectory": [round(x, 3) for x in fc.trajectory]}

    def resolve(self, fid: str, outcome: int) -> None:
        if self.log is not None:
            self.log.resolve(fid, outcome)

    def skill(self, **kw) -> dict:
        return self.log.skill(**kw) if self.log is not None else {"note": "no log attached"}
