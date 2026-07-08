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
# A public-figure resolver is wired in by default so /predict never hard-blocks on an unknown contact:
# when there's no private history we infer the recipient online. A live web/LLM backend can be injected
# via env-configured search_fn/infer_fn; absent one, it degrades to the transparent offline fallback.
from swm.entities.public_figure import default_resolver  # noqa: E402

world = World(store=EventStore(DB_PATH), resolver=default_resolver())


class ImportReq(BaseModel):
    threads: list[dict]
    owner_id: str


class PredictReq(BaseModel):
    contact_id: str
    text: str
    channel: str = "email"
    name: str | None = None          # public-figure name to resolve when we have no private history
    domain: str = ""                 # topic/domain of the ask (sharpens the online lookup)
    ask: str = ""


class CompareReq(BaseModel):
    contact_id: str
    texts: list[str]
    channel: str = "email"
    name: str | None = None


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
    return world.predict(req.contact_id, req.text, channel=req.channel, name=req.name,
                         domain=req.domain, ask=req.ask)


@app.post("/v1/compare-actions")
def compare(req: CompareReq):
    return world.compare(req.contact_id, req.texts, channel=req.channel, name=req.name)


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


# A persisted, backtested aggregate world (models/<domain>_aggregate.json) makes a domain's
# one-step cell 'validated' AND makes the rollout use the real state-sensitive head. Without it we
# fall back to a state-IGNORING PriorHead, which the honesty gate labels 'unvalidated' — and we say
# so, rather than stamping a grade the model didn't earn (the EXP-008 audit fix).
_MODELS_DIR = Path(os.environ.get("SWM_MODELS", "models"))


def _load_aggregate_world(world_id: str):
    from swm.worlds.aggregate_world import AggregateWorld
    p = _MODELS_DIR / f"{world_id}_aggregate.json"
    if p.exists():
        try:
            return AggregateWorld.load(p)
        except Exception:
            return None
    return None


def _make_action(a: dict, t0: float, i: int):
    from swm.state.factors import tag_topic
    from swm.state.state import Action
    title = a.get("title", "")
    return Action(
        action_id=f"plan-{i}", actor_id=a.get("actor_id", "anon"),
        content_features={"title_len": min(1.0, len(title) / 80),
                          "is_show": 1.0 if title.lower().startswith("show hn") else 0.0,
                          "is_ask": 1.0 if title.lower().startswith("ask hn") else 0.0,
                          "is_text": 1.0 if not a.get("domain") else 0.0,
                          "topic": tag_topic(title)},
        timing={"hour": a.get("hour", 12), "weekday": a.get("weekday", 2), "ts": t0 + i * 86400},
        meta={"domain": a.get("domain", ""), "title": title})


@app.post("/v1/rollout")
def rollout_ep(req: RolloutReq):
    """Multi-step state evolution -> a DISTRIBUTION of futures (audit C.8). Distinct from /predict
    (one-step calibrated). Honesty gate: a fitted, backtested world makes horizon 1 'validated' and
    the rollout state-sensitive; otherwise a state-ignoring PriorHead is used and everything is
    labeled 'unvalidated' with a warning."""
    import time as _t

    t0 = req.as_of or _t.time()
    plan = [_make_action(a, t0, i) for i, a in enumerate(req.action_plan)]
    world = _load_aggregate_world(req.world_id)

    if world is not None:
        # REAL fitted, state-sensitive transition; horizon 1 carries the backtest grade.
        from swm.simulation.rollout import simulate
        grade_letter = world.grade.get("grade")
        ro = simulate(world.transition, world.pop, plan, n_samples=req.n_samples,
                      validated_grade=grade_letter if grade_letter not in (None, "ungraded") else None,
                      validated_horizon=1, domain=req.world_id)
        head = "fitted state-transition (state genuinely conditions the prediction)"
    else:
        from swm.state.factors import build_hn_registry
        from swm.state.state import WorldState
        from swm.state.trajectory import rollout as _rollout
        from swm.state.transition import PriorHead, TransitionModel
        model = TransitionModel(build_hn_registry(), PriorHead())  # state-IGNORING => unvalidated
        ro = _rollout(model, WorldState(timestamp=t0), plan, n_samples=req.n_samples)
        head = "PriorHead (state-ignoring fallback; no fitted model for this world_id)"

    return {
        "report_type": ro.report_type,              # always "simulation" (never "prediction")
        "calibration_grade": ro.calibration_grade,  # "unvalidated" unless a fitted+graded world exists
        "warning": ro.warning,
        "engine": head,
        "world_id": req.world_id, "horizon": ro.steps, "n_samples": ro.n_samples,
        "as_of": t0, "trajectory_distribution": ro.per_step,
        "note": "SIMULATION: a distribution of plausible futures, not a prediction. "
                "Use /predict for one-step calibrated prediction.",
    }


class SimulateReq(BaseModel):
    title: str
    domain: str = ""
    author_rep: float = 0.0          # as-of author reputation (centered log mean past score)
    features: dict = {}              # optional LLM-extracted action features
    n_samples: int = 200


def _load_sim_engine():
    from swm.simulation.engine import HNSimulationEngine
    p = _MODELS_DIR / "hn_simulation.json"
    if p.exists():
        try:
            return HNSimulationEngine.load(p)
        except Exception:
            return None
    return None


@app.post("/v1/simulate")
def simulate_ep(req: SimulateReq):
    """REAL multi-step, multi-actor simulation (Phase 6): 8 HN community segments react over steps,
    world state (score/exposure/social-proof/front-page/novelty) updates each step, and P(hit) is
    the FRACTION OF SIMULATED TRAJECTORIES that cross — not a classifier over initial features.
    Returns stepwise state, segment reactions, uncertainty by horizon, and the outcome distribution.
    Uses the fitted+graded engine (models/hn_simulation.json) when present, else a default engine
    labeled 'unvalidated'."""
    import json as _json
    from pathlib import Path as _P

    from swm.simulation.engine import HNSimulationEngine
    from swm.simulation.world_rollout import WorldRollout
    from swm.state.factors import tag_topic

    eng = _load_sim_engine()
    gp = _MODELS_DIR / "hn_simulation_grade.json"
    if eng is not None and gp.exists():
        grade = _json.loads(gp.read_text())
        validated = True
    else:
        eng = eng or HNSimulationEngine()
        grade = {"grade": "unvalidated", "note": "no fitted engine; qualitative simulation"}
        validated = False

    topic = tag_topic(req.title)
    feats = {**req.features, f"topic_{topic}": 1.0,
             "title_len": min(1.0, len(req.title) / 80),
             f"cat_{req.features.get('cat', 'Other')}": 1.0}
    ctx = {"exposure_mult": 1.0, "state_boost": req.author_rep}
    # one recorded trajectory (stepwise events + segment reactions) + the distribution
    import random as _r
    _, st = eng._trajectory(feats, req.author_rep, ctx, _r.Random(0), record=True)
    ro = WorldRollout(eng).rollout(feats, author_rep=req.author_rep, ctx=ctx, n_samples=req.n_samples)
    reactions = {}
    for rx in st.reactions:
        reactions[rx.actor_id] = reactions.get(rx.actor_id, 0.0) + rx.intensity
    return {
        "report_type": "simulation",
        "calibration_grade": grade.get("grade"),
        "validated": validated,
        "title": req.title, "topic": topic,
        "p_hit": ro["p_hit"], "p_hit_raw": ro["p_hit_raw"],
        "outcome_distribution_bands": ro["band_probs"],   # [<10,10-39,40-99,100-299,300+]
        "uncertainty_by_horizon": ro["per_step"],         # per-timestep score interval
        "final_interval80": ro["final_interval80"],
        "example_trajectory": {"steps": [snap for snap in [st.snapshot()]],
                               "segment_upvotes": {k: round(v, 1) for k, v in reactions.items()}},
        "note": "P(hit) is the fraction of simulated multi-step trajectories that cross the "
                "front-page cascade — derived from the trajectory distribution, not a classifier.",
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text()
