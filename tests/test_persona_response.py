"""Persona-response engine invariants: counting-not-asking, hypothesis competition, fragility
reporting, fail-closed parsing, universal dossier construction. Offline (deterministic fake LLM)."""
from __future__ import annotations

import json

from swm.decision.persona_response import (DEFAULT_OUTCOME_UTILITIES, GENERIC_INBOX_HYPOTHESES,
                                           OUTCOMES, PersonaDossier, ensemble_evaluate,
                                           fragility_report, simulate_response,
                                           specialize_hypotheses)


def _fake_chat(script):
    """A chat_fn whose reply depends on the hypothesis text present in the prompt: script maps
    hypothesis-id substring -> outcome."""
    def fn(prompt, **kw):
        for key, outcome in script.items():
            if key in prompt:
                return json.dumps({"outcome": outcome, "reply_text":
                                   ("Send it." if outcome == "requests_material" else ""),
                                   "reasoning": "test"})
        return json.dumps({"outcome": "no_response", "reply_text": "", "reasoning": "default"})
    return fn


def _dossier():
    return PersonaDossier(name="Peter Thiel", role="investor",
                          evidence=[("fellowship", "pays young people to skip college"),
                                    ("profile", "screens heavily, rarely answers cold outreach")])


def test_dossier_renders_qualitative_text_no_invented_numbers():
    d = _dossier()
    r = d.render()
    assert "You are Peter Thiel" in r and "skip college" in r
    assert "0.85" not in r and "status_orientation" not in r, \
        "the dossier must carry evidence text, never invented numeric traits"


def test_dossier_universal_from_user_context():
    d = PersonaDossier.from_user_context("Dana", "my manager; hates long emails; replies at night")
    assert "Dana" in d.render() and "hates long emails" in d.render()


def test_simulate_response_parses_categorical_outcome():
    fn = _fake_chat({"You personally skim": "curious_reply"})
    r = simulate_response(fn, _dossier(), GENERIC_INBOX_HYPOTHESES[0], "hello")
    assert r.outcome == "curious_reply"
    assert r.hypothesis_id == "reads_own_bursts"


def test_unparseable_draw_fails_closed_to_no_response():
    r = simulate_response(lambda p, **k: "I would probably reply enthusiastically!",
                          _dossier(), GENERIC_INBOX_HYPOTHESES[0], "hello")
    assert r.outcome == "no_response" and not r.raw_available


def test_ensemble_counts_choices_never_asks_for_probability():
    # under intros_only the persona ignores; under reads_own_bursts it asks for the memo
    fn = _fake_chat({"You personally skim": "requests_material",
                     "trusted": "no_response", "assistant screens": "no_response",
                     "read artifacts": "requests_material", "do not respond": "no_response"})
    res = ensemble_evaluate(fn, _dossier(), GENERIC_INBOX_HYPOTHESES, "msg", draws_per_hypothesis=4)
    assert res.n_draws == 20
    assert res.counts["reads_own_bursts"]["requests_material"] == 4
    assert res.counts["intros_only"]["no_response"] == 4
    dist = res.outcome_dist()
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    # prior-weighted: requests_material mass = priors of reads_own_bursts + evidence_first = 0.30
    assert abs(dist["requests_material"] - 0.30) < 1e-9


def test_valenced_utilities_penalize_dismissive():
    assert DEFAULT_OUTCOME_UTILITIES["dismissive_reply"] < 0 < DEFAULT_OUTCOME_UTILITIES["curious_reply"]


def test_fragility_flags_single_hypothesis_winner():
    # arm A wins ONLY under reads_own_bursts; arm B wins under the other four -> A fragile if ahead
    fnA = _fake_chat({"You personally skim": "meeting_offer"})
    fnB = _fake_chat({"assistant screens": "requests_material", "trusted": "requests_material",
                      "read artifacts": "requests_material", "do not respond": "no_response"})
    d = _dossier()
    rA = ensemble_evaluate(fnA, d, GENERIC_INBOX_HYPOTHESES, "A", draws_per_hypothesis=3)
    rB = ensemble_evaluate(fnB, d, GENERIC_INBOX_HYPOTHESES, "B", draws_per_hypothesis=3)
    rep = fragility_report({"A": rA, "B": rB})
    assert rep["per_hypothesis_winner"]["reads_own_bursts"] == "A"
    assert rep["per_hypothesis_winner"]["assistant_screens"] == "B"
    if rep["winner"] == "A":
        assert rep["fragile"] is True                    # A leads only under one hypothesis
    else:
        assert set(rep["winner_wins_under_hypotheses"]) >= {"assistant_screens", "evidence_first"}
        assert rep["fragile"] is False


def test_specialize_hypotheses_falls_back_to_generic_and_normalizes():
    hyps = specialize_hypotheses(None, _dossier())
    assert [h["id"] for h in hyps] == [h["id"] for h in GENERIC_INBOX_HYPOTHESES]
    good = json.dumps([{"id": "a", "prior": 2, "reality": "You screen."},
                       {"id": "b", "prior": 1, "reality": "You read."},
                       {"id": "c", "prior": 1, "reality": "You ignore."}])
    hyps2 = specialize_hypotheses(lambda p, **k: good, _dossier(), k=3)
    assert abs(sum(h["prior"] for h in hyps2) - 1.0) < 1e-9
    assert hyps2[0]["prior"] == 0.5


def test_arrival_context_carries_action_semantics():
    """Different ACTIONS are different arrival contexts through the same engine — the seam the
    action-level search uses (warm intro vs cold email is not a wording change)."""
    seen = {}
    def fn(prompt, **kw):
        seen["intro"] = "forwards this to you with a note" in prompt
        return json.dumps({"outcome": "requests_material", "reply_text": "", "reasoning": ""})
    simulate_response(fn, _dossier(), GENERIC_INBOX_HYPOTHESES[2], "msg",
                      arrival_context="A portfolio founder you trust forwards this to you with a "
                                      "note: 'worth two minutes.'")
    assert seen["intro"]
