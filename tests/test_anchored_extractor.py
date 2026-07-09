"""Tests for AnchoredExtractor — the three pillars as the default value grounding (EXP-086 wired in)."""
from swm.api.retrieval_grounding import CalibratedExtractor
from swm.api.anchored_extractor import AnchoredExtractor


def _llm(evidence_val, evidence_ci, ref_val, ref_ci):
    """A mock LLM that answers the reference-class prompt with the base rate and the evidence prompt with the
    specific value (distinguished by the 'reference-class' phrasing)."""
    def fn(prompt):
        if "reference-class" in prompt or "base-rate current value" in prompt:
            return {"value": ref_val, "ci95": ref_ci, "confidence": 0.6}
        return {"value": evidence_val, "ci95": evidence_ci, "confidence": 0.8}
    return fn


def test_strong_evidence_barely_shrinks():
    # tight evidence CI (small sd) -> the value stays near the evidence, not the far-away base rate
    ext = AnchoredExtractor(CalibratedExtractor(_llm(8.0, 0.05, 2.0, 1.0)))
    r = ext.extract("x", "q", ["ev"])
    assert r["value"] > 7.5                      # evidence 8.0 dominates; base rate 2.0 barely pulls it


def test_weak_evidence_pulls_to_base_rate():
    # wide evidence CI (large sd) -> the value is pulled hard toward the reference-class base rate
    ext = AnchoredExtractor(CalibratedExtractor(_llm(8.0, 20.0, 2.0, 0.5)))
    r = ext.extract("x", "q", ["ev"])
    assert r["value"] < 4.0                      # uncertain evidence -> shrink toward base rate 2.0


def test_no_evidence_falls_back_to_reference():
    def fn(prompt):
        if "reference-class" in prompt or "base-rate current value" in prompt:
            return {"value": 3.0, "ci95": 0.5, "confidence": 0.6}
        return {"value": None}                   # evidence undeterminable
    r = AnchoredExtractor(CalibratedExtractor(fn)).extract("x", "q", ["ev"])
    assert r is not None and abs(r["value"] - 3.0) < 1e-6   # uses the outside view instead of nulling


def test_anchor_off_returns_raw_evidence():
    ext = AnchoredExtractor(CalibratedExtractor(_llm(8.0, 20.0, 2.0, 0.5)), anchor=False)
    r = ext.extract("x", "q", ["ev"])
    assert abs(r["value"] - 8.0) < 1e-6          # no shrink -> the raw evidence value
