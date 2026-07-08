"""Tests for the peak-architecture builds: corpus harvest, regime router, adaptive fidelity (triage),
embedding-keyed registry, event model, and the full-covariance weight posterior."""
import random

from swm.api.adaptive_fidelity import triage, variable_leverage
from swm.api.model_spec import parse_spec
from swm.eval.harvest import harvest_source, onehot
from swm.eval.regime_router import RegimeRouter, examples_from_portfolio
from swm.simulation.event_model import EventModel, interval_coverage
from swm.variables.bayes_logistic import BayesianLogistic
from swm.variables.embedding_registry import EmbeddingPriorRegistry, lexical_embed
from swm.variables.prior_registry import PriorRegistry


# ---- #1 corpus harvest ----
def test_harvest_source_fits_and_registers():
    rng = random.Random(0)
    rows = [{"x": rng.random()} for _ in range(300)]
    X = [[r["x"]] for r in rows]
    y = [1 if r["x"] > 0.5 else 0 for r in rows]
    reg = PriorRegistry()
    cw = harvest_source(reg, X, y, ["driver"], "some_outcome", source="test")
    assert cw is not None and cw.model.w[0] > 0        # positive elasticity learned
    assert reg.get("driver", "some_outcome") is not None


# ---- #2 regime router ----
def test_router_prior_routes_by_regime():
    r = RegimeRouter()                                  # unfit -> world-knowledge prior
    pop = r.route("population", baseline_strength=0.0, irreducible_frac=0.3)
    macro = r.route("macro", baseline_strength=0.9, irreducible_frac=0.5)
    assert pop["decision"] == "rich_sim" and macro["decision"] == "baseline"


def test_router_learns_from_portfolio_examples():
    portfolio = {"domains": {
        "opinion": {"kind": "population", "by_fidelity": {"full": {"skill_vs": {"persistence": 0.15},
                    "baseline_loss": {"base_rate": 0.2, "persistence": 0.19}, "beats_all_baselines": True}}},
        "fx": {"kind": "macro", "by_fidelity": {"full": {"skill_vs": {"persistence": 0.0},
               "baseline_loss": {"base_rate": 0.2, "persistence": 0.05}, "beats_all_baselines": False}}}}}
    ex = examples_from_portfolio(portfolio)
    assert len(ex) == 2 and {e["kind"] for e in ex} == {"population", "macro"}
    r = RegimeRouter().fit(ex)
    assert r.route("population", baseline_strength=0.05)["p_fidelity_wins"] > \
           r.route("macro", baseline_strength=0.7)["p_fidelity_wins"]


# ---- #3 adaptive fidelity (variance triage) ----
def test_triage_finds_the_high_leverage_variable():
    spec = parse_spec({"mechanism": "calibrated_readout", "extra": {"intercept": 0.0}, "variables": [
        {"name": "big", "value": 0.9, "est_sd": 0.3, "weight": 2.0, "weight_sd": 0.1},   # varies + strong
        {"name": "flat", "value": 0.5, "est_sd": 0.0, "weight": 2.0, "weight_sd": 0.1}],  # constant -> no leverage
        "outcome": {"event": {"op": ">", "value": 0.5}}})
    lev = variable_leverage(spec)
    assert lev[0][0] == "big"
    t = triage(spec, keep_frac=0.9)
    assert "big" in t["invest_in"]


# ---- #4 embedding-keyed registry ----
def test_embedding_registry_transfers_across_phrasing():
    base = PriorRegistry()
    base.update("inflation rate", "rate hike", mean=1.5, sd=0.2, n=500)
    emb = EmbeddingPriorRegistry(base, threshold=0.5).build_index()
    assert emb.get("inflation rate", "rate hike").source.startswith("registry")   # exact hit
    got = emb.get("inflation", "rate hike")                                        # phrasing variant -> transfer
    assert got is not None and got.mean == 1.5 and "transfer" in got.source
    assert got.sd > 0.2                                                            # widened by transfer distance


# ---- #5 event model ----
def test_event_model_calibrates_and_covers():
    rng = random.Random(0)
    moves = [(rng.gauss(0, 0.3) if rng.random() < 0.5 else 0.0) for _ in range(400)]
    em = EventModel.calibrate(moves, threshold=0.05, calendar=list(range(1, 7)))
    assert 0.3 < em.p_active < 0.7 and em.impact_sd > 0
    r = em.rollout(1.0, 6, n=2000)
    assert r["p05"] < r["mean"] < r["p95"]              # a real interval (variance placed at the events)


def test_interval_coverage():
    truths = [0.5] * 100
    cov = interval_coverage(truths, [0.0] * 100, [1.0] * 100, nominal=0.9)
    assert cov["empirical_coverage"] == 1.0


# ---- #6 full-covariance posterior ----
def test_full_covariance_posterior_samples():
    rng = random.Random(0)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(300)]
    y = [1 if x[0] + x[1] > 0 else 0 for x in X]
    m = BayesianLogistic(l2=1.0).fit(X, y)
    d_full = m.predict_dist([1.0, 1.0], full_cov=True)
    d_diag = m.predict_dist([1.0, 1.0], full_cov=False)
    assert 0.0 <= d_full["p"] <= 1.0 and d_full["sd"] >= 0.0 and d_diag["sd"] >= 0.0
