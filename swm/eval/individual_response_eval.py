"""Individual-response eval (spec Phase 4): the model vs its honest alternatives.

Compares, on ONE time-ordered labeled dataset of (entity, message-features, outcome), the regimes
the spec names:

  segment            : sources = {segment}           (know the population, not the person)
  + person history   : sources = {segment, person}   (hierarchical partial pooling)
  + message features : sources = {segment, message}  (content, no person)
  full individual    : sources = {segment, person, message}
  raw LLM            : an external predictor callable (blind to history)         [pluggable]
  raw LLM + context  : an external predictor callable (given as-of history)      [pluggable]

Each regime is graded on the SAME temporal holdout with proper scoring + decision lift, so the
question "does modeling the individual beat the segment / beat raw LLM + context?" gets a number.
Raw-LLM arms are optional (they need an LLM/agent predictor); when absent the report marks them
blocked and still answers the segment-vs-individual question, which needs no LLM.
"""
from __future__ import annotations

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.worlds.individual_world import IndividualWorld


def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4),
            "uplift@20": round(uplift_at_k(y, p, 0.2), 4)}


def evaluate(samples: list[tuple[str, dict, int]], message_feature_names: list[str], *,
             split: float = 0.7, prior_strength: float = 4.0,
             raw_llm_pred: dict[int, float] | None = None,
             raw_llm_context_pred: dict[int, float] | None = None) -> dict:
    """samples time-ordered. raw_llm_pred / raw_llm_context_pred: optional {test_index -> p} from an
    external predictor over the SAME test rows (test index = position within the held-out slice)."""
    n = len(samples)
    cut = int(split * n)
    test = samples[cut:]
    y = [o for _, _, o in test]
    seg_rate = (sum(o for _, _, o in samples[:cut]) + 1) / (cut + 2)

    regimes = {
        "segment": frozenset({"segment"}),
        "+person": frozenset({"segment", "person"}),
        "+message": frozenset({"segment", "message"}),
        "full_individual": frozenset({"segment", "person", "message"}),
    }
    results = {}
    preds_full = None
    for name, sources in regimes.items():
        w = IndividualWorld(message_feature_names=message_feature_names, sources=sources,
                            prior_strength=prior_strength)
        # replicate the world's as-of test prediction to capture per-row preds
        from swm.transition.individual_transition import IndividualTransition
        m = IndividualTransition(message_feature_names=message_feature_names, segment_rate=seg_rate,
                                 prior_strength=prior_strength, sources=sources)
        m.fit_stream(samples[:cut], segment_rate=seg_rate)
        preds = []
        for eid, mf, o in test:
            preds.append(m.predict(eid, mf)["p_mean"])
            m.transition(eid, o)
        results[name] = {**_score(y, preds), "sources": sorted(sources)}
        if name == "full_individual":
            preds_full = preds

    # optional raw-LLM arms (external predictions keyed by test index)
    for arm, table in (("raw_llm", raw_llm_pred), ("raw_llm+context", raw_llm_context_pred)):
        if table:
            preds = [table.get(i, seg_rate) for i in range(len(test))]
            results[arm] = _score(y, preds)
        else:
            results[arm] = {"status": "BLOCKED: needs an LLM/agent predictor over the test rows"}

    # headline comparisons
    comp = {}
    if preds_full is not None:
        comp["individual_beats_segment_logloss"] = round(
            results["segment"]["log_loss"] - results["full_individual"]["log_loss"], 4)
    return {"n": n, "n_test": len(test), "test_base_rate": round(sum(y) / len(y), 4),
            "segment_rate_train": round(seg_rate, 4), "regimes": results, "comparison": comp}
