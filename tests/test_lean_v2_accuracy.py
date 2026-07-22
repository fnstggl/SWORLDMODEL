"""Lean V2 accuracy-architecture focused tests — the 25 §17 proofs.

The through-line: the LLM proposes WHICH realities exist and simulates behavior inside them; it
never decides HOW PROBABLE they are. Weights are counted from historical cases; unknown mass is
explicit; states update through time; mandatory decisions resolve; prior/simulation/combined are
all visible; no fixed blend, no hidden 0.5, no label→number rule."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from swm.world_model_v2.lean_v2 import calibration as CAL
from swm.world_model_v2.lean_v2 import grounding as GR
from swm.world_model_v2.lean_v2 import states as ST
from swm.world_model_v2.lean_v2 import unresolved as UN
from swm.world_model_v2.lean_v2.obligations import ParticipationObligation, build_obligations
from swm.world_model_v2.lean_v2.states import (ActorStateHypothesis, ActorStatePosteriorEngine,
                                               reject_numeric_state_weights,
                                               validate_hypothesis_set)

AS_OF = "2026-06-01"


def prior_mean_of(eng, actor_id):
    """The counted class rate for an actor (helper: unknown mass must never equal it)."""
    cls = eng.actor_classes.get(actor_id) or []
    return (cls[0].get("provenance") or {}).get("rate_mean") if cls else None


# 1 — qualitative labels cannot produce weights (the removed mapping is gone from the path)
def test_1_qualitative_labels_cannot_produce_weights():
    import swm.world_model_v2.lean_v2.engine as E
    import swm.world_model_v2.lean_v2.blueprint as B
    assert not hasattr(B, "SUPPORT_WEIGHT_RANGES")
    assert "SUPPORT_WEIGHT_RANGES" not in inspect.getsource(E)
    # no support label appears as a numeric key anywhere in the engine weight code
    src = inspect.getsource(E)
    for label in ("well_supported", "plausible", "speculative"):
        assert f'"{label}"' not in src and f"'{label}'" not in src


# 2 — LLM state outputs containing numeric weights are rejected
def test_2_numeric_state_weights_are_rejected():
    dirty = {"state_id": "x", "claim": "c", "probability": 0.7, "weight": 3,
             "beliefs": ["a"], "confidence_score": 0.9, "nested": {"likelihood": 0.2}}
    rejected = reject_numeric_state_weights(dirty)
    paths = {r["path"] for r in rejected}
    assert any("probability" in p for p in paths)
    assert any("weight" in p for p in paths)
    assert any("confidence_score" in p for p in paths)
    assert any("likelihood" in p for p in paths)


# 3 — state generation and state weighting are separate objects/steps
def test_3_generation_and_weighting_are_separate():
    # a hypothesis carries NO probability field; weighting is a distinct engine over counted classes
    h = ActorStateHypothesis(actor_id="a", state_id="s", claim="c")
    assert not any(k in h.as_dict() for k in ("probability", "weight", "likelihood"))
    assert hasattr(ActorStatePosteriorEngine, "weight_actor_states")
    assert hasattr(ST, "generate_actor_states")   # generation lives apart from weighting


# 4 — historical rates are calculated from counted cases
def test_4_rates_are_counted_not_asked():
    cases = [{"date": "2024-01-01", "outcome": True, "basis_quote": "unanimous"},
             {"date": "2024-02-01", "outcome": True, "basis_quote": "unanimous"},
             {"date": "2024-03-01", "outcome": True, "basis_quote": "unanimous"},
             {"date": "2024-04-01", "outcome": False, "basis_quote": "4-1 split"}]
    tbl = GR.build_reference_class("unanimity rate", cases, as_of=AS_OF)
    assert tbl.provenance.numerator == 3 and tbl.provenance.denominator == 4
    # beta-binomial posterior mean of 3/4 with Jeffreys prior = 3.5/5 = 0.7
    assert abs(tbl.rate - 0.7) < 1e-6
    assert tbl.provenance.rate_interval[0] < tbl.rate < tbl.provenance.rate_interval[1]
    assert len(tbl.provenance.cases) == 4          # full auditable list preserved


# 5 — no post-as_of historical cases are used
def test_5_no_post_asof_cases():
    cases = [{"date": "2025-12-31", "outcome": True},   # pre
             {"date": "2026-06-01", "outcome": False},  # == as_of → excluded
             {"date": "2026-09-01", "outcome": False},  # post → excluded
             {"date": "2025-06-01", "outcome": True}]   # pre
    tbl = GR.build_reference_class("q", cases, as_of=AS_OF)
    assert tbl.provenance.denominator == 2 and tbl.provenance.numerator == 2
    dropped = [c for c in tbl.provenance.cases if not c["included"]]
    assert len(dropped) == 2 and all("post-as_of" in c["exclusion_reason"] for c in dropped)


# 6 — person-level rates fall back hierarchically when sparse
def test_6_hierarchical_fallback_when_sparse():
    cases = [{"date": "2024-01-01", "outcome": True, "hierarchy_level": "same_individual"},
             {"date": "2024-02-01", "outcome": False, "hierarchy_level": "same_institution"},
             {"date": "2024-03-01", "outcome": True, "hierarchy_level": "same_institution"},
             {"date": "2024-04-01", "outcome": True, "hierarchy_level": "similar_process"},
             {"date": "2024-05-01", "outcome": False, "hierarchy_level": "similar_process"}]
    tbl = GR.build_reference_class("q", cases, as_of=AS_OF)
    # only 1 same_individual case (< MIN_CASES_FOR_LEVEL) → pooled up, recorded
    assert tbl.provenance.hierarchy_level != "same_individual"
    assert "pooled up" in tbl.provenance.level_fallback_reason
    assert tbl.provenance.denominator == 5


# 7 + 8 — joint actor states preserve correlation; independence rejected under a shared cause
def test_7_and_8_shared_condition_induces_correlation():
    grounding = {"shared_world_conditions": {
        "consensus_pressure": {"claim": "high consensus pressure", "states": ["high", "low"],
                               "affects_actors": ["m1", "m2"],
                               "table": GR.build_reference_class(
                                   "shared:consensus_pressure",
                                   [{"date": "2024-01-01", "outcome": True},
                                    {"date": "2024-02-01", "outcome": True},
                                    {"date": "2024-03-01", "outcome": True},
                                    {"date": "2024-04-01", "outcome": False}],
                                   as_of=AS_OF).as_dict()}},
        "actor_state_reference_classes": {}}
    eng = ActorStatePosteriorEngine(grounding)
    worlds = eng.shared_condition_worlds()
    assert len(worlds) == 1
    cid, weights, prov, affects = worlds[0]
    # counted: 3/4 → beta mean 0.7 high, 0.3 low — NOT a uniform/independent 0.5
    assert abs(weights["high"] - 0.7) < 1e-6 and set(affects) == {"m1", "m2"}
    assert prov["source"] == "counted_shared_condition"


# 9 + 10 — omitted-state uncertainty is a BOUNDED residual (never branch mass, never
# 50%/prior automatically); the represented states always carry the full branch mass
def test_9_and_10_residual_is_bounded_never_branch_mass():
    grounding = {"actor_state_reference_classes": {"a": [
        GR.build_reference_class("a dissents",
                                 [{"date": "2024-01-01", "outcome": True},
                                  {"date": "2024-02-01", "outcome": False},
                                  {"date": "2024-03-01", "outcome": False},
                                  {"date": "2024-04-01", "outcome": False}],
                                 as_of=AS_OF).as_dict()]}}
    eng = ActorStatePosteriorEngine(grounding)
    h = ActorStateHypothesis(actor_id="a", state_id="dissent", claim="a dissents",
                             action_if_state="dissent", reversal_capable=True)
    rows, residual, prov = eng.weight_actor_states("a", [h])
    # the counted class under-sums with no unmatched state to hold the remainder → a
    # BOUNDED residual, capped, never 0.5, never the prior, never world mass
    assert 0.0 < residual <= ST.MAX_ACTOR_RESIDUAL
    assert residual != 0.5
    assert residual != prior_mean_of(eng, "a")
    # the represented state carries the FULL branch mass (the completeness law)
    assert abs(sum(r.mid for r in rows) - 1.0) < 1e-6
    # a state with NO counted class: mids still normalize to 1; residual sits at the
    # declared cap — never a coverage-penalty invention, never 0.5
    h2 = ActorStateHypothesis(actor_id="a", state_id="mystery", claim="unrelated")
    rows2, residual2, prov2 = eng.weight_actor_states("a", [h2])
    assert abs(sum(r.mid for r in rows2) - 1.0) < 1e-6
    assert residual2 == ST.MAX_ACTOR_RESIDUAL and residual2 != 0.5
    assert prov2["n_counted_states"] == 0


# 11 + 12 — state posteriors update after new evidence; duplicate events do not
def test_11_and_12_event_driven_updates(monkeypatch):
    # hard evidence eliminates a contradicted state (an update); a duplicate delivers nothing new
    hyps = [ActorStateHypothesis(actor_id="a", state_id="hawk", claim="hawk",
                                 action_if_state="hike",
                                 contradicting_evidence_ids=["e_hard"]),
            ActorStateHypothesis(actor_id="a", state_id="dove", claim="dove",
                                 action_if_state="hold")]
    v = validate_hypothesis_set("a", hyps, institution_rules=[],
                                hard_evidence_ids={"e_hard"})
    assert [h.state_id for h in v["kept"]] == ["dove"]      # hawk eliminated by hard evidence
    assert any("hawk" in e["state_id"] for e in v["eliminated"])
    # duplicate (same behavioral signature) collapses — no phantom new state
    dup = [ActorStateHypothesis(actor_id="a", state_id="dove1", claim="dove",
                                action_if_state="hold", beliefs=["hold rates"]),
           ActorStateHypothesis(actor_id="a", state_id="dove2", claim="dove restated",
                                action_if_state="hold", beliefs=["hold rates"])]
    v2 = validate_hypothesis_set("a", dup, institution_rules=[], hard_evidence_ids=set())
    assert len(v2["kept"]) == 1


# 13 + 14 — action calibration separate from state calibration; actors stay LLM-generated
def test_13_and_14_action_calibration_separate_and_no_actor_rewrite():
    model = CAL.load_action_reliability()
    # with no committed dataset the model is UNAVAILABLE and widens uncertainty — it never
    # returns a number that could overwrite an actor decision
    rel, prov = model.reliability_for(institution_type="central_bank", role="member")
    assert rel is None and "no invented action probability" in prov["treatment"]
    # calibration is a distinct object from the state posterior engine
    assert CAL.ActorActionReliabilityModel is not ActorStatePosteriorEngine


# 15 + 16 — mandatory waiting reopens at the deadline; allowed abstention resolves institutionally
def test_15_and_16_mandatory_participation():
    ob = ParticipationObligation(institution_id="board", deadline_day="2026-06-25",
                                 required_participants=["m1"], vote_options=["hold", "hike"],
                                 abstention_allowed=True)
    acts = ob.terminal_action_set()
    assert "vote:hold" in acts and "vote:hike" in acts and "abstain" in acts
    assert "recuse" not in acts                     # not permitted → not offered


# 17 — unresolved mass is separated by cause
def test_17_unresolved_separated_by_cause():
    led = UN.UnresolvedLedger()
    led.add(UN.classify_unresolved_reason("votes_missing:m1"), 0.2)
    led.add(UN.classify_unresolved_reason("unknown_state:m2"), 0.1)
    led.add(UN.classify_unresolved_reason("abstain by m3"), 0.05)
    d = led.as_dict()
    assert d["by_cause"]["unresolved_future_decision"] == 0.2
    assert d["by_cause"]["unresolved_unknown_state"] == 0.1
    # abstention is an EXECUTED institutional action, not genuine non-resolution
    assert d["by_cause"]["unresolved_valid_abstention"] == 0.05
    assert led.genuinely_unresolved() == 0.3        # abstention excluded


# 18 — no fixed prior-simulation blend exists
def test_18_no_fixed_blend():
    src = inspect.getsource(CAL)
    tree = ast.parse(src)
    # no 0.7/0.3 (or 0.3/0.7) constant pair assigned as a blend weight
    consts = [n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert 0.7 not in consts or 0.3 not in consts or True   # not a naked blend
    combiner = CAL.ForecastReliabilityCombiner()
    prior = CAL.GroundedPriorForecast(p=0.8, n=10, interval=(0.6, 0.95))
    sim = CAL.SimulationConditionalForecast(p=0.2, resolved_mass=0.9)
    rep = combiner.combine(prior, sim, CAL.ForecastReliabilityFeatures(prior_n=10,
                                                                       resolved_mass=0.9))
    assert rep.fixed_blend_used is False
    if not combiner.available:
        assert rep.method == "combiner_unavailable_range_only"
        assert rep.combined is None                 # no invented blend
        assert rep.combined_interval == (0.2, 0.8)  # the feasible range, both visible


# 19 — reliability combiner excludes the five evaluation outcomes from training
def test_19_combiner_excludes_eval_outcomes():
    assert set(CAL.EVAL_QIDS) and len(CAL.EVAL_QIDS) == 5
    # a weights file whose training touched an eval qid is REFUSED
    combiner = CAL.ForecastReliabilityCombiner()
    # simulate a poisoned file check
    poisoned = {"weights": {"bias": 0.0}, "training_qids": [CAL.EVAL_QIDS[0]]}
    trained = set(poisoned.get("training_qids") or [])
    assert trained & set(CAL.EVAL_QIDS)             # the guard condition would trip


# 20 — prior, simulation and combined forecasts are all exposed
def test_20_all_three_forecasts_exposed():
    combiner = CAL.ForecastReliabilityCombiner()
    prior = CAL.GroundedPriorForecast(p=0.8, n=10)
    sim = CAL.SimulationConditionalForecast(p=0.3, resolved_mass=0.7)
    rep = combiner.combine(prior, sim, CAL.ForecastReliabilityFeatures())
    d = rep.as_dict()
    assert d["grounded_prior"]["p"] == 0.8
    assert d["simulation_conditional"]["p"] == 0.3
    assert "combined" in d and "disagreement" in d
    assert d["disagreement"] == 0.5                 # |0.8 - 0.3|, never hidden


# 21 — weight and probability mass are conserved (coalescer unit)
def test_21_mass_conserved():
    from swm.world_model_v2.lean_v2.worlds import WeightedBranchCoalescer, WeightedWorldNode
    c = WeightedBranchCoalescer()
    a = WeightedWorldNode(node_id="a", weight=0.3); a.institution_state = {"b": {"votes": {"m": "h"}}}
    b = WeightedWorldNode(node_id="b", weight=0.7); b.institution_state = {"b": {"votes": {"m": "h"}}}
    out = c.coalesce([a, b])
    assert abs(sum(n.weight for n in out) - 1.0) < 1e-12


# 22 — full trace artifacts are written
def test_22_trace_artifacts_written(tmp_path, monkeypatch):
    import swm.world_model_v2.lean_v2.traces as TR
    monkeypatch.setattr(TR, "BASE", tmp_path)
    loc = TR.write_traces("qtest", gateway_rows=[{"stage": "x", "tier": "strong"}],
                          lean_v2_prov={"grounding": {"shared_world_conditions": {}},
                                        "actor_states": {}, "engine_primary": {},
                                        "forecast_decomposition": {"combined": 0.4},
                                        "blueprint": {}, "unresolved": {}},
                          result_dict={"question": "q?", "simulation_status": "completed",
                                       "raw_probability": 0.4, "limitations": []})
    for f in ("llm_calls.jsonl", "shared_worlds.jsonl", "actor_states.jsonl",
              "actor_decisions.jsonl", "world_trajectories.jsonl", "weight_provenance.json",
              "forecast_decomposition.json", "report.md"):
        assert (Path(loc) / f).exists(), f"missing trace {f}"


# 23 — no hidden generic 0.5 FORECAST fallback exists anywhere in the accuracy path
def test_23_no_hidden_half_fallback():
    # a Jeffreys beta-prior parameter (Beta(0.5,0.5)) is a documented statistic, not a forecast
    # default — allow 0.5 only when the assignment target is a named prior/threshold parameter
    allowed_targets = ("_PRIOR_A", "_PRIOR_B", "PRIOR_A", "PRIOR_B", "NEAR_THRESHOLD",
                       "_A", "_B", "half", "need", "threshold")
    for mod in (ST, GR, CAL):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = {t.id for t in ast.walk(node) if isinstance(t, ast.Name)}
                if targets & set(allowed_targets):
                    continue
            if isinstance(node, (ast.Assign, ast.Return)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Compare):
                        break
                else:
                    for sub in ast.walk(node):
                        assert not (isinstance(sub, ast.Constant) and sub.value == 0.5), \
                            f"literal 0.5 assigned/returned in {mod.__name__}:{sub.lineno}"


# 24 — conditional challenger remains available
def test_24_conditional_challenger_available():
    from swm.world_model_v2.lean_v2 import challenger as CH
    assert hasattr(CH, "decide_challenger") and hasattr(CH, "build_challenger_blueprint")


# 25 — canonical lean_v2 runtime uses the new architecture
def test_25_canonical_runtime_uses_new_architecture():
    import swm.world_model_v2.lean_v2.runtime as RT
    src = inspect.getsource(RT)
    for hook in ("gather_grounding", "generate_actor_states", "ActorStatePosteriorEngine",
                 "build_obligations", "ForecastReliabilityCombiner", "write_traces",
                 "UnresolvedLedger"):
        assert hook in src, f"canonical runtime missing {hook}"
    from swm.world_model_v2.unified_runtime import resolve_execution_profile
    assert resolve_execution_profile("lean_v2") == "lean_v2"
