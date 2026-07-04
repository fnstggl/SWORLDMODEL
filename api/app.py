"""FastAPI service (audit J) + the v1 dashboard.

Run:  uvicorn api.app:app --reload         (DB path via SWM_DB, default data/events.db)

Endpoints (audit J contract: every payload carries report_type = prediction | insight):
  POST /v1/import           import normalized thread JSON  {threads:[...], owner_id}
  POST /v1/fit              fit the world + run the temporal backtest (grade)
  POST /v1/predict          {contact_id, text, channel?}            -> PREDICTION
  POST /v1/compare-actions  {contact_id, texts[], channel?}         -> PREDICTION
  POST /v1/suggest          {contact_id, goal, channel?, k?}        -> INSIGHT drafts, PREDICTION ranking
  GET  /v1/persona/{id}     persona posterior (correct-a-guess card)-> INSIGHT
  POST /v1/elicit           {contact_id, draft_text}   -> the one VOI question (or none)
  POST /v1/correct          {contact_id, factor, answer}
  POST /v1/backtest         run the exp001 ladder                    -> PREDICTION metrics
  GET  /v1/contacts         known recipients
  GET  /                    dashboard (static single page)
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from swm import llm
from swm.eval.baselines import noise_floor_brier
from swm.ingestion.importers import import_threads
from swm.ingestion.store import EventStore
from swm.worlds.world import World

DB_PATH = os.environ.get("SWM_DB", "data/events.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="sworldmodel", version="0.0.1")
world = World(store=EventStore(DB_PATH))


class ImportReq(BaseModel):
    threads: list[dict]
    owner_id: str


class PredictReq(BaseModel):
    contact_id: str
    text: str
    channel: str = "email"


class CompareReq(BaseModel):
    contact_id: str
    texts: list[str]
    channel: str = "email"


class SuggestReq(BaseModel):
    contact_id: str
    goal: str
    channel: str = "email"
    k: int = 3
    context: str = ""


class ElicitReq(BaseModel):
    contact_id: str
    draft_text: str
    channel: str = "email"


class CorrectReq(BaseModel):
    contact_id: str
    factor: str
    answer: str


@app.post("/v1/import")
def import_data(req: ImportReq):
    counts = import_threads(world.store, req.threads, req.owner_id)
    return {"imported": counts, "labeled_sends": len(world.store.labeled_sends())}


@app.post("/v1/fit")
def fit():
    return world.fit()


@app.post("/v1/predict")
def predict(req: PredictReq):
    return world.predict(req.contact_id, req.text, channel=req.channel)


@app.post("/v1/compare-actions")
def compare(req: CompareReq):
    return world.compare(req.contact_id, req.texts, channel=req.channel)


@app.post("/v1/suggest")
def suggest(req: SuggestReq):
    """Drafts are INSIGHT (generated); the ranking over them is PREDICTION (scored by readout)."""
    persona = world.persona(req.contact_id)
    drafts = llm.generate_drafts(persona, req.goal, channel=req.channel, k=req.k,
                                 context=req.context)
    ranking = world.compare(req.contact_id, [d.text for d in drafts], channel=req.channel)
    return {
        "drafts": [{"report_type": "insight", "text": d.text, "rationale": d.rationale,
                    "source": d.source} for d in drafts],
        "ranking": ranking,
    }


@app.get("/v1/persona/{contact_id}")
def persona(contact_id: str):
    return {"report_type": "insight",
            "caveat": "inferred posterior, not observed truth — correct anything wrong",
            "persona": world.persona(contact_id).summary()}


@app.post("/v1/elicit")
def elicit(req: ElicitReq):
    q = world.voi(req.contact_id, req.draft_text, channel=req.channel)
    return q or {"report_type": "insight", "question": None,
                 "note": "no single question would move this prediction by >= 2 points"}


@app.post("/v1/correct")
def correct(req: CorrectReq):
    return world.correct(req.contact_id, req.factor, req.answer)


@app.post("/v1/backtest")
def backtest():
    result = dict(world.backtest) if world.backtest else {}
    if not result:
        from swm.eval.harness import run_ladder
        result = run_ladder(world.store)
    result["noise_floor"] = noise_floor_brier(world.store.labeled_sends())
    result["report_type"] = "prediction"
    return result


@app.get("/v1/contacts")
def contacts():
    return {"contacts": world.store.recipients()}


class RolloutReq(BaseModel):
    world_id: str = "generic"
    action_plan: list[dict]          # [{actor_id, title, domain?, hour?}, ...]
    horizon: int | None = None
    n_samples: int = 200
    as_of: float | None = None


# Domains/horizons that have a validated state-transition backtest (exp005). Everything else is
# labeled 'unvalidated' by the honesty gate. HN one-step is the only backtested cell today.
_VALIDATED = {"hn": {"grade": "C", "horizon": 1}}


@app.post("/v1/rollout")
def rollout_ep(req: RolloutReq):
    """Multi-step state evolution -> a DISTRIBUTION of futures (audit C.8). Distinct from /predict
    (one-step calibrated). Honesty gate: 'validated' only where a backtest exists, else labeled
    unvalidated with a warning."""
    import time as _t

    from swm.state.factors import build_hn_registry, tag_topic
    from swm.state.state import Action, WorldState
    from swm.state.trajectory import rollout as _rollout
    from swm.state.transition import PriorHead, TransitionModel

    reg = build_hn_registry()
    model = TransitionModel(reg, PriorHead())          # prior head => unvalidated by construction
    t0 = req.as_of or _t.time()
    initial = WorldState(timestamp=t0)
    plan = []
    for i, a in enumerate(req.action_plan):
        title = a.get("title", "")
        plan.append(Action(
            action_id=f"plan-{i}", actor_id=a.get("actor_id", "anon"),
            content_features={"title_len": min(1.0, len(title) / 80),
                              "is_show": 1.0 if title.lower().startswith("show hn") else 0.0,
                              "is_ask": 1.0 if title.lower().startswith("ask hn") else 0.0,
                              "is_text": 1.0 if not a.get("domain") else 0.0,
                              "topic": tag_topic(title)},
            timing={"hour": a.get("hour", 12), "weekday": a.get("weekday", 2), "ts": t0 + i * 86400},
            meta={"domain": a.get("domain", ""), "title": title}))
    v = _VALIDATED.get(req.world_id)
    ro = _rollout(model, initial, plan, n_samples=req.n_samples,
                  validated_grade=v["grade"] if v else None,
                  validated_horizon=v["horizon"] if v else 0, domain=req.world_id)
    return {
        "report_type": ro.report_type,              # always "simulation" (never "prediction")
        "calibration_grade": ro.calibration_grade,  # "unvalidated" unless backtested
        "warning": ro.warning,
        "world_id": req.world_id, "horizon": ro.steps, "n_samples": ro.n_samples,
        "as_of": t0, "trajectory_distribution": ro.per_step,
        "note": "SIMULATION: a distribution of plausible futures, not a prediction. "
                "Use /predict for one-step calibrated prediction.",
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text()
