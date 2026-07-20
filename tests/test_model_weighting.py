"""Pins the grounded-weighted simulation forecast (WSim): the rich forecast is a plausibility-weighted
average of the SIMULATED worlds (Bayesian model averaging), not an equal-weight mean, not an arbitrary LLM
confidence, and never a silent substitution by the outside-view fallback when valid simulations ran.

Tests: malformed-world rejection, duplicate-world merging, unequal grounded weights, uncited-confidence
capping, weight normalization, correct weighted aggregation, and the no-silent-substitution contract."""
import json

from swm.world_model_v2.model_weighting import (
    WorldInfo, curate_worlds, merge_duplicates, grounded_world_weights, weighted_distribution, p_yes,
    objective_quality, final_forecast_selection, combined_forecast, simulation_confidence)


def _stub(payload):
    return lambda _prompt: json.dumps(payload)


def _world(mid, dist, **kw):
    return WorldInfo(model_id=mid, dist=dist, **kw)


# ---- curation: malformed / unsupported / fallback-derived worlds are dropped with a reason ----

def test_curate_rejects_malformed_and_unsupported():
    worlds = [
        _world("empty", {}),                                            # no distribution
        _world("failed", {"True": 0.5, "False": 0.5}, status="execution_failed"),
        _world("unresolved", {"True": 0.4, "False": 0.6}, unresolved_share=1.0),
        _world("fellback", {"True": 0.3, "False": 0.7}, fallback_derived=True),
        _world("good", {"True": 0.6, "False": 0.4}, support_grade="transfer_supported"),
    ]
    valid, rejected = curate_worlds(worlds)
    assert [w.model_id for w in valid] == ["good"]
    reasons = dict(rejected)
    assert "no_simulated_distribution" in reasons["empty"]
    assert "execution_failed" in reasons["failed"]
    assert "unresolved" in reasons["unresolved"]
    assert "fallback_derived" in reasons["fellback"]


# ---- duplicate-world merging: same causal story counted once, weight combined ----

def test_merge_duplicates_collapses_same_causal_story():
    a = _world("A", {"True": 0.6, "False": 0.4}, support_grade="exploratory",
               actors=["BoJ board"], institutions=["Bank of Japan"], mechanisms=["policy vote"])
    b = _world("B", {"True": 0.62, "False": 0.38}, support_grade="transfer_supported",
               actors=["BoJ board"], institutions=["Bank of Japan"], mechanisms=["policy vote"])  # dup of A
    c = _world("C", {"True": 0.2, "False": 0.8}, actors=["markets"], institutions=["FX"],
               mechanisms=["carry unwind"])
    merged, records = merge_duplicates([a, b, c])
    ids = {w.model_id for w in merged}
    assert len(merged) == 2                                            # A/B collapsed to one
    assert "C" in ids
    kept = [w for w in merged if w.model_id in ("A", "B")][0]
    assert kept.support_grade == "transfer_supported"                 # best-support representative kept
    assert kept.merged_from                                           # records the collapsed id
    assert records and records[0]["kept"] in ("A", "B")


# ---- grounded weights: UNEQUAL, normalized, evidence-anchored ----

def test_weights_are_unequal_and_normalized():
    strong = _world("strong", {"True": 0.7, "False": 0.3}, support_grade="transfer_supported",
                    posterior_consumed=True)
    weak = _world("weak", {"True": 0.2, "False": 0.8}, support_grade="highly_speculative")
    llm = _stub({"worlds": [
        {"model_id": "strong", "plausibility": 0.8, "rationale": "matches the hawkish CPI print",
         "citations": ["April core CPI 3.1% > target"], "unsupported_assumptions": []},
        {"model_id": "weak", "plausibility": 0.3, "rationale": "needs a shock with no evidence",
         "citations": [], "unsupported_assumptions": ["an unannounced emergency meeting"]}]})
    weights = grounded_world_weights("Will BoJ hike?", [strong, weak], "April core CPI 3.1%", llm=llm)
    w = {ww.model_id: ww.weight for ww in weights}
    assert abs(sum(w.values()) - 1.0) < 1e-6                          # normalized
    assert w["strong"] > w["weak"]                                    # unequal, evidence favors strong
    assert w["strong"] > 0.6                                          # decisively, not 50/50


def test_uncited_high_confidence_is_capped():
    # two equal-quality worlds; one claims 0.95 plausibility with NO citation -> capped, loses weight
    a = _world("cited", {"True": 0.5, "False": 0.5}, support_grade="exploratory")
    b = _world("uncited", {"True": 0.9, "False": 0.1}, support_grade="exploratory")
    llm = _stub({"worlds": [
        {"model_id": "cited", "plausibility": 0.6, "rationale": "grounded", "citations": ["a real fact"],
         "unsupported_assumptions": []},
        {"model_id": "uncited", "plausibility": 0.95, "rationale": "just confident", "citations": [],
         "unsupported_assumptions": []}]})
    weights = {ww.model_id: ww for ww in grounded_world_weights("q", [a, b], "ev", llm=llm)}
    assert weights["uncited"].plausibility <= 0.35 + 1e-9             # capped
    assert weights["cited"].weight > weights["uncited"].weight        # citation beats bare confidence


def test_missing_llm_falls_back_to_objective_anchor_not_equal():
    # no LLM: weights come from the objective anchor (support/evidence/unresolved), NOT equal weights
    a = _world("hi", {"True": 0.5, "False": 0.5}, support_grade="transfer_supported", posterior_consumed=True)
    b = _world("lo", {"True": 0.5, "False": 0.5}, support_grade="highly_speculative")
    weights = {ww.model_id: ww.weight for ww in grounded_world_weights("q", [a, b], "", llm=None)}
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["hi"] > weights["lo"]                             # anchor still discriminates


# ---- correct weighted aggregation: plausibility × outcome, NOT the mean ----

def test_weighted_aggregation_is_plausibility_weighted_not_mean():
    from swm.world_model_v2.model_weighting import WorldWeight
    ws = [WorldWeight("A", 0.6, 0.6, 0.6, {"True": 0.2, "False": 0.8}),
          WorldWeight("B", 0.3, 0.3, 0.3, {"True": 0.7, "False": 0.3}),
          WorldWeight("C", 0.1, 0.1, 0.1, {"True": 0.9, "False": 0.1})]
    dist = weighted_distribution(ws)
    assert abs(p_yes(dist) - 0.42) < 1e-6                            # 0.6*.2+0.3*.7+0.1*.9, not mean 0.6
    assert abs(sum(dist.values()) - 1.0) < 1e-6                      # normalized distribution


# ---- the no-silent-substitution contract ----

def test_no_silent_substitution_when_valid_sims_ran():
    # valid simulated forecast exists -> source MUST be weighted_simulation, never the fallback
    sel = final_forecast_selection(n_worlds_valid=3, weighted_p=0.42, outside_p=0.05, combined=0.30)
    assert sel["source"] == "weighted_simulation" and sel["forecast"] == 0.42
    # no valid world -> only THEN the grounded fallback
    sel2 = final_forecast_selection(n_worlds_valid=0, weighted_p=None, outside_p=0.05)
    assert sel2["source"] == "grounded_fallback" and sel2["forecast"] == 0.05


def test_combined_shrinks_toward_baseline_only_when_sim_weak():
    assert combined_forecast(0.42, 0.05, alpha=1.0) == 0.42          # strong sim -> pure simulation
    assert combined_forecast(0.42, 0.05, alpha=0.0) == 0.05          # zero confidence -> baseline
    mid = combined_forecast(0.42, 0.05, alpha=0.5)
    assert 0.05 < mid < 0.42                                          # partial shrinkage


def test_simulation_confidence_ignores_baseline_agreement():
    from swm.world_model_v2.model_weighting import WorldWeight
    ws = [WorldWeight("A", 0.7, 0.7, 0.7, {"True": 0.6, "False": 0.4}),
          WorldWeight("B", 0.3, 0.4, 0.4, {"True": 0.5, "False": 0.5})]
    worlds = [_world("A", {"True": 0.6}, posterior_consumed=True, n_particles=30),
              _world("B", {"True": 0.5}, n_particles=30)]
    conf = simulation_confidence(ws, worlds)
    assert 0.0 <= conf["alpha"] <= 1.0
    assert "does NOT reward agreement" in conf["components"]["note"]  # anti-circular by construction
