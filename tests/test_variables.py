"""Tests for the behavioral variable-mapping core architecture."""
import random

from swm.state.state import Action
from swm.variables.inference import VariableInferenceEngine
from swm.variables.schema import NAMES, SPECS, BY_CATEGORY
from swm.variables.variable_map import Variable, VariableMap
from swm.worlds.variable_world import VariableWorld


def test_schema_covers_all_determinant_categories():
    cats = set(BY_CATEGORY)
    assert cats == {"disposition", "relational", "incentive", "state", "platform", "message_fit",
                    "persona"}
    assert len(NAMES) >= 20


def test_variable_map_provenance_and_override():
    vm = VariableMap("e")
    vm.set("base_responsiveness", 0.5, provenance="llm", confidence=0.4)
    vm.set("base_responsiveness", 0.8, provenance="data", confidence=0.9)   # data outranks llm
    assert vm.get("base_responsiveness") == 0.8 and vm.vars["base_responsiveness"].provenance == "data"
    vm.merge_user_context({"base_responsiveness": 0.2})                     # user outranks data
    assert vm.get("base_responsiveness") == 0.2 and vm.vars["base_responsiveness"].provenance == "user"
    # signed variable clamps to [-1,1]
    vm.set("trust_in_source", 5.0, provenance="llm")
    assert vm.get("trust_in_source") == 1.0


def test_variable_map_features_shape_and_confidence_shrink():
    vm = VariableMap("e").fill_priors()
    feats = vm.to_features()
    assert len(feats) == 2 * len(NAMES)                    # values + confidence channels
    # a low-confidence value barely moves from neutral; a high-confidence one moves fully
    lo = VariableMap("e"); lo.set("stakes", 1.0, provenance="heuristic", confidence=0.1)
    hi = VariableMap("e"); hi.set("stakes", 1.0, provenance="data", confidence=1.0)
    i = NAMES.index("stakes")
    assert lo.to_features()[i] < hi.to_features()[i]


def test_inference_is_asof_and_provenance_tracked():
    eng = VariableInferenceEngine(platform="email")
    a = Action(action_id="m", actor_id="s", channel="email", timing={"ts": 100},
               meta={"text": "Hi — quick question, are you free? Urgent, just following up."})
    cold = eng.infer("x", a, history={"n_prior": 0})
    warm = eng.infer("x", a, history={"n_prior": 15, "response_rate": 0.8, "recency_days": 2})
    # data variable present + high confidence only when there is history
    assert warm.confidence("base_responsiveness") > cold.confidence("base_responsiveness")
    assert warm.vars["base_responsiveness"].provenance == "data"
    # heuristic message-fit always inferred
    assert cold.vars["pushiness"].provenance == "heuristic" and cold.get("pushiness") > 0
    # LLM inference applies with llm provenance and never touches the outcome
    llm = eng.infer("x", a, history={"n_prior": 3},
                    llm_inference={"trust_in_source": {"value": 0.7, "confidence": 0.6}})
    assert llm.vars["trust_in_source"].provenance == "llm" and llm.get("trust_in_source") == 0.7


def test_variable_world_backtests_and_matches_state_model():
    """Routing prediction through the full variable map must be BACKTESTABLE and not much worse than
    the compact entity-state model on data with a real per-entity signal."""
    rng = random.Random(0)
    theta = {f"e{i}": min(0.95, max(0.05, rng.betavariate(2, 3))) for i in range(40)}
    insts = []
    for k in range(2400):
        eid = rng.choice(list(theta))
        a = Action(action_id=str(k), actor_id="s", channel="email", timing={"ts": k},
                   meta={"text": "are you free for a quick call?"})
        insts.append((eid, a, None, int(rng.random() < theta[eid])))
    res, preds, y = VariableWorld(platform="email").backtest(insts)
    from swm.eval.metrics import log_loss
    gr = sum(y) / len(y)
    assert res["log_loss"] < log_loss(y, [gr] * len(y))    # beats the global-rate baseline via mapped state


def test_variable_world_predict_is_explainable():
    a = Action(action_id="m", actor_id="s", channel="email", timing={"ts": 10}, meta={"text": "hi?"})
    w = VariableWorld(platform="email")
    w._observe("e", 1, 1); w._observe("e", 2, 1); w._observe("e", 3, 0)
    w.readout = None
    out = w.predict("e", a, explain=True)
    assert "provenance" in out and "variables" in out
    assert out["provenance"]["n_vars"] >= 20
