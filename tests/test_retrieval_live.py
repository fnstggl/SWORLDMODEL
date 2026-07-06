"""Tests for the retrieval layer + live-forecast harness (EXP-058)."""
from swm.api.generative_simulator import AgentSpec, GenerativeSimulator
from swm.api.retrieval import Retriever, RetrievedContext, asof_retriever, web_search_retriever
from swm.eval.live_forecast import LiveForecaster
from swm.eval.postmortem import PostMortemLog
from swm.simulation.agent_society import AgentSociety


def test_retriever_bounds_and_prompt():
    r = Retriever(fetch_fn=lambda q: [{"title": "T1", "snippet": "s1", "date": "2026-01-01T00:00:00Z"}] * 20,
                  max_items=5)
    ctx = r.retrieve("Q?")
    assert len(ctx) == 5
    p = ctx.to_prompt()
    assert "T1" in p and "2026-01-01" in p


def test_web_search_retriever_is_resilient():
    r = web_search_retriever(lambda q: (_ for _ in ()).throw(RuntimeError("net")))  # search that errors
    assert len(r.retrieve("Q?")) == 0                    # never crashes the pipeline


def test_asof_retriever_serves_committed_news():
    r = asof_retriever({"Q?": [{"title": "news", "description": "d", "published_at": "2025-12-01", "source": "x"}]})
    ctx = r.retrieve("Q?", as_of="2025-12-15")
    assert len(ctx) == 1 and ctx.as_of == "2025-12-15"
    assert ctx.snippets[0]["title"] == "news"


def test_live_forecaster_logs_for_scoring():
    retriever = asof_retriever({"Will it pass?": [{"title": "buzz", "description": "", "published_at": "2026-01-01"}]})
    specs = [AgentSpec(f"a{i}", {"v": 0.6}, influence=1.0) for i in range(5)]
    sim = GenerativeSimulator(society=AgentSociety(rounds=3), identify_fn=lambda q, c: specs,
                              position_fn=lambda q, s, c: s.variables["v"])
    log = PostMortemLog()
    f = LiveForecaster(retriever=retriever, simulator=sim, log=log)
    out = f.forecast("Will it pass?", fid="q1", made_at=0, resolves_at=10)
    assert 0.0 <= out["p_outcome"] <= 1.0 and out["n_agents"] == 5 and out["n_evidence"] == 1
    assert "q1" in log.forecasts                          # logged with resolution metadata
    assert log.forecasts["q1"]["made_at"] < log.forecasts["q1"]["resolves_at"]   # leakage-free by construction


def test_forecast_then_resolve_then_score():
    retriever = asof_retriever({f"q{i}": [{"title": "x", "published_at": "2026-01-01"}] for i in range(15)})
    specs = [AgentSpec("a", {"v": 0.7})]
    sim = GenerativeSimulator(identify_fn=lambda q, c: specs, position_fn=lambda q, s, c: 0.7)
    log = PostMortemLog()
    f = LiveForecaster(retriever=retriever, simulator=sim, log=log)
    for i in range(15):
        f.forecast(f"q{i}", fid=f"q{i}", made_at=i, resolves_at=i + 1)
        f.resolve(f"q{i}", 1)
    s = f.skill()
    assert s["n"] == 15 and s["leakage_free"] is True     # the loop closes: made-before, scored-after
