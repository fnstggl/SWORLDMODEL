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


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text()
