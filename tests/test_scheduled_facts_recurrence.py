"""Pins the recurrence-aware scheduled-facts fix: a strong recurring pattern RAISES the outcome and
composes with disrupting evidence, instead of only outcome-entailing negatives moving the forecast
(the visionOS-WWDC failure). Parser stays back-compatible with the old entailing schema."""
import math

from swm.world_model_v2.scheduled_facts import entailment_nudge, extract_scheduled_facts


def _net(facts):
    acc = 0.0
    for f in facts:
        acc = max(-4.0, min(4.0, acc + entailment_nudge(f["outcome_influence"], f["influence_strength"],
                                                         f["confidence"])))
    return 1.0 / (1.0 + math.exp(-acc))


def test_nudge_sign_and_cap():
    assert entailment_nudge("raises", 1.0, 1.0) > 0 and entailment_nudge("lowers", 1.0, 1.0) < 0
    assert entailment_nudge("neutral", 1.0, 1.0) == 0.0
    assert entailment_nudge("raises", 1.0, 1.0) <= 1.8            # per-fact cap: no lone certainty


def test_strong_recurrence_beats_a_single_speculative_disruptor():
    # visionOS shape: two strong recurrences (raise) vs one speculative abandonment (lower) => net YES-leaning
    facts = [{"outcome_influence": "raises", "influence_strength": 0.8, "confidence": 0.9},
             {"outcome_influence": "raises", "influence_strength": 0.7, "confidence": 0.9},
             {"outcome_influence": "lowers", "influence_strength": 0.6, "confidence": 0.7}]
    assert _net(facts) > 0.6


def test_parser_reads_new_influence_schema(monkeypatch):
    def fake_llm(_prompt):
        return ('{"facts":[{"fact":"Annual conference in June","date":"2026-06-08","entity":"Apple",'
                '"kind":"recurring_event","source":"model_knowledge","confidence":0.9,'
                '"pattern_strength":"strong_recurrence","outcome_influence":"raises",'
                '"influence_strength":0.8,"reason":"annual OS at WWDC"}]}')
    facts = extract_scheduled_facts("Will Apple announce X at WWDC 2026?", as_of="2026-05-07",
                                    horizon="2026-06-12", evidence_text="", llm=fake_llm)
    assert len(facts) == 1
    f = facts[0]
    assert f["outcome_influence"] == "raises" and f["influence_strength"] == 0.8
    assert f["pattern_strength"] == "strong_recurrence"
    assert f["outcome_entailing"] is True and f["entailed_direction"] == "yes"   # back-compat derived


def test_parser_back_compat_old_entailing_schema(monkeypatch):
    def fake_llm(_prompt):
        return ('{"facts":[{"fact":"Term ends May 15","date":"2026-05-15","entity":"X","kind":"term_expiry",'
                '"source":"evidence","confidence":0.9,"outcome_entailing":true,"entailed_direction":"no"}]}')
    facts = extract_scheduled_facts("q", as_of="2026-05-01", horizon="2026-06-01", evidence_text="", llm=fake_llm)
    f = facts[0]
    # old schema => synthesized influence so the accumulator still works
    assert f["outcome_influence"] == "lowers" and f["influence_strength"] > 0
    assert f["outcome_entailing"] is True and f["entailed_direction"] == "no"
