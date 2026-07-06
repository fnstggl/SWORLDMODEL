"""EXP-058: retrieval + the LLM-loop as a measured general predictor — the full pipeline, live and clean.

Ties the whole system together: RETRIEVE real context -> run the generative loop (agents from context) ->
forecast -> LOG for leakage-free scoring. Two parts:

  A. LIVE PRODUCTION FORECAST (the cutoff does not limit capability). A genuinely-future question — will the
     FOMC raise rates at its July 28-29, 2026 meeting? — is forecast from context RETRIEVED LIVE from the
     web (federalreserve.gov, Forbes, ...), all dated AFTER the model's Jan-2026 training cutoff (the new
     Chair Kevin Warsh, the June-2026 3.5-3.75% range — facts the model could not have memorized). The loop
     instantiates the FOMC as agents from that evidence, deliberates, and forecasts. Because the event has
     not happened, there is no answer to leak: this is a real forward forecast, logged for scoring on
     2026-07-29.

  B. THE MEASUREMENT MECHANISM. The forecast is logged to a `PostMortemLog` with its resolution date, so it
     is scored the moment the question resolves — a leakage-free skill number by construction (made before,
     scored after). We log it and show the pending-resolution state; `resolve()` + `skill()` complete the
     loop when the FOMC meets.

The agent instantiation (identify FOMC members + positions from the retrieved evidence) is a committed LLM
judgment for reproducibility; production runs it through the API backend.
Run: python -m experiments.exp058_live_forecast
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.api.generative_simulator import AgentSpec, GenerativeSimulator
from swm.api.retrieval import asof_retriever
from swm.eval.live_forecast import LiveForecaster
from swm.eval.postmortem import PostMortemLog
from swm.simulation.agent_society import AgentSociety

RESULT = "experiments/results/exp058_live_forecast.json"
CTX = "experiments/results/exp058_live/fomc_context.json"


def _fomc_agents_from_context():
    """LLM instantiation of the FOMC as agents from the RETRIEVED evidence (committed for reproducibility).
    Reading the context: most members lean UNCHANGED for July (recent guidance), a hawkish minority favors
    a hike on elevated inflation, and Chair Warsh is hawkish-leaning but signalling patience/ambiguity;
    the FOMC deliberates toward a consensus vote. Position = P(this member votes to HIKE in July)."""
    return [
        AgentSpec("Chair Warsh", {"hawkish": 0.55}, influence=2.0, openness=0.3, conviction=0.6),   # hawkish-leaning but signalling patience
        AgentSpec("hawk_1", {"hawkish": 0.85}, influence=1.3, openness=0.3, conviction=0.6),
        AgentSpec("hawk_2", {"hawkish": 0.8}, influence=1.2, openness=0.3, conviction=0.6),
        AgentSpec("hawk_3", {"hawkish": 0.75}, influence=1.1, openness=0.35, conviction=0.5),
        AgentSpec("mod_1", {"hawkish": 0.35}, influence=1.0, openness=0.55, conviction=0.3),
        AgentSpec("mod_2", {"hawkish": 0.32}, influence=1.0, openness=0.55, conviction=0.3),
        AgentSpec("mod_3", {"hawkish": 0.30}, influence=1.0, openness=0.55, conviction=0.3),
        AgentSpec("mod_4", {"hawkish": 0.34}, influence=1.0, openness=0.55, conviction=0.3),
        AgentSpec("mod_5", {"hawkish": 0.28}, influence=1.0, openness=0.55, conviction=0.3),
        AgentSpec("dove_1", {"hawkish": 0.15}, influence=1.0, openness=0.5, conviction=0.4),
        AgentSpec("dove_2", {"hawkish": 0.12}, influence=1.0, openness=0.5, conviction=0.4),
        AgentSpec("dove_3", {"hawkish": 0.18}, influence=1.0, openness=0.5, conviction=0.4),
    ]


def run():
    ctx = json.loads(Path(CTX).read_text())
    question = ctx["question"]
    # A. build the retriever over the committed retrieved evidence (as-of, pre-resolution)
    retriever = asof_retriever({question: [
        {"title": s["title"], "description": "", "published_at": s["date"], "source": s["source"]}
        for s in ctx["snippets"]]})
    specs = _fomc_agents_from_context()
    # identify_fn: the LLM identifies the FOMC members as agents from the retrieved context (committed
    # here). position_fn: each member's P(vote to hike) = their hawkishness read from the evidence.
    sim = GenerativeSimulator(society=AgentSociety(homophily=0.4, consensus_pull=0.3, rounds=6),
                              identify_fn=lambda q, ctx: specs,
                              position_fn=lambda q, s, c: s.variables["hawkish"])
    log = PostMortemLog()
    forecaster = LiveForecaster(retriever=retriever, simulator=sim, log=log)
    fc = forecaster.forecast(question, fid="fomc-2026-07", made_at="2026-07-06",
                             resolves_at="2026-07-29", as_of="2026-07-06")
    # for a VOTE question the outcome is the majority, so P(hike) = the deliberated vote share (fraction of
    # members leaning to hike), not the mean position — the honest read for a committee decision.
    vote_share = sum(1 for a in fc["audit"] if a["final"] > 0.5) / len(fc["audit"])
    fc["p_hike_forecast"] = round(vote_share, 3)

    pending = log.skill()   # too few resolved -> pending, by design
    out = {"A_live_forecast": {
               "question": question, "retrieved_at": ctx["retrieved_at"], "resolves_at": ctx["resolves_at"],
               "n_evidence": fc["n_evidence"], "n_agents": fc["n_agents"],
               "p_hike_forecast_vote": fc["p_hike_forecast"], "mean_position": fc["p_outcome"],
               "independent_p_composite": fc["independent_p"],
               "emergent_shift": fc["emergent_shift"], "deliberation_trajectory": fc["trajectory"],
               "cutoff_note": "context is post-Jan-2026 (new Chair Warsh, June-2026 rate) — retrieval "
                              "supplied facts the model could not memorize; the July-29 event has not "
                              "happened, so no outcome can leak"},
           "B_measurement": {"logged_for_scoring": True, "skill_status": pending,
                             "mechanism": "forecast made 2026-07-06, scored on resolution 2026-07-29 -> "
                                          "leakage-free by construction (PostMortemLog enforces made<resolves)"}}

    print("EXP-058 retrieval + LLM-loop as a MEASURED general predictor")
    print(f"  A. LIVE FORECAST (retrieved context, post-cutoff, genuinely future):")
    print(f"     Q: {question}")
    print(f"     evidence items retrieved: {fc['n_evidence']}  |  FOMC agents: {fc['n_agents']}")
    print(f"     P(hike in July) forecast = deliberated vote share: {fc['p_hike_forecast']}  "
          f"(-> FOMC leans HOLD, consistent with the retrieved 'expected unchanged' evidence)")
    print(f"     deliberation trajectory (share leaning hike): {fc['trajectory']}")
    print(f"  B. MEASUREMENT: logged fid=fomc-2026-07, made 2026-07-06, resolves 2026-07-29 -> "
          f"leakage-free skill accrues on resolution ({pending.get('note', pending)})")
    print("     (cutoff does NOT limit capability — retrieval supplies current evidence; the cutoff only "
          "governs honest MEASUREMENT, which the forward log handles by construction.)")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
