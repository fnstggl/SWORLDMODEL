"""Offline tests for the reply-first planner and the separated human-language judge.

The invariants under test are the architectural promises:
  * three SEPARATED judges — the outcome judge ranks ONLY candidates that already passed the
    truth gate and the language gate (it can never resurrect a fabrication or bot-register text);
  * the outcome judge is BLIND (shuffled labels) and its counts stay machine-readable only;
  * exactly ONE recommended message, and no simulated reply percentages anywhere a human reads;
  * the truth gate fails CLOSED (unverifiable => not passed);
  * the language judge exposes the preference-learning hook and folds stored human A/B choices
    into its prompt;
  * the hard writing rules (one number, no jargon compounds, human-typed request) ride along on
    every instantiation prompt.
"""
import json
import os
import random
import re

import pytest

from swm.decision.language_judge import (LanguageVerdict, llm_language_judge, load_preferences,
                                         record_preference)
from swm.decision.llm_moves import SenderBrief
from swm.decision.persona_response import PersonaDossier
from swm.decision.reply_first import (_NO_PERCENT_LABEL, BEAT_ROLE, BEATS, STRUCTURES,
                                      ReplyFirstPlanner)

BRIEF = SenderBrief(
    sender="Beckett",
    thesis="AI infrastructure has a planning problem disguised as a power problem",
    ask="get the recipient to ask for the one-page memo",
    facts=["17 years old, starting Princeton in the fall",
           "building Aurelius (runaurelius.com)",
           "+724% SLA-safe goodput per dollar in simulated replay of ~1.5M requests",
           "results are simulated replay, not a production deployment"])

DOSSIER = PersonaDossier(name="Peter Thiel", role="investor",
                         evidence=[("contrarian", "skeptical of consensus and credentials")])

HYPS = [{"id": "assistant_screens", "prior": 0.5, "description": "an assistant screens"},
        {"id": "reads_own", "prior": 0.5, "description": "reads his own inbox in bursts"}]


def planner(chat=None, **kw):
    kw.setdefault("sender_brief", BRIEF)
    kw.setdefault("dossier", DOSSIER)
    return ReplyFirstPlanner(chat, **kw)


# --------------------------------------------------------------------------- language judge
def test_language_judge_offline_lexical_fallback():
    judge = llm_language_judge(None)
    v = judge("Peter, I'm Beckett. We found the bottleneck is planning, not power. Want the "
              "one-pager? Beckett")
    assert isinstance(v, LanguageVerdict)
    assert v.source == "lexical"
    assert 0.0 <= v.score <= 1.0


def test_language_judge_parses_llm_json_and_gates_on_flags():
    ok_judge = llm_language_judge(lambda p, **k: '{"score": 0.9, "flags": []}')
    assert ok_judge("hi").ok and ok_judge("hi").score == pytest.approx(0.9)
    flagged = llm_language_judge(
        lambda p, **k: '{"score": 0.9, "flags": [{"sentence": "May I send you the memo?", '
                       '"problem": "ceremonious permission construction"}]}')("hi")
    assert not flagged.ok and flagged.flags[0]["problem"].startswith("ceremonious")


def test_language_judge_falls_back_lexical_when_llm_dies():
    def boom(p, **k):
        raise RuntimeError("api down")
    v = llm_language_judge(boom)("A plain sentence. Another plain sentence.")
    assert v.source == "lexical_fallback(llm_judge_failed)"
    assert 0.0 <= v.score <= 1.0


def test_language_judge_rubric_names_the_bot_register_problems():
    seen = {}
    def capture(p, **k):
        seen["prompt"] = p
        return '{"score": 1.0, "flags": []}'
    llm_language_judge(capture)("some email text")
    for needle in ("ceremonious permission constructions", "jargon compounds",
                   "ONE big number", "assistant-speak"):
        assert needle in seen["prompt"]


def test_preference_hook_roundtrip_and_prompt_folding(tmp_path):
    prefs = str(tmp_path / "prefs.jsonl")
    record_preference("Want the one-pager?", "May I send you the one-page technical memo?",
                      path=prefs, note="human chose the plain ask")
    rows = load_preferences(prefs)
    assert rows[-1]["chosen"] == "Want the one-pager?"
    seen = {}
    def capture(p, **k):
        seen["prompt"] = p
        return '{"score": 1.0, "flags": []}'
    llm_language_judge(capture, prefs_path=prefs)("text")
    assert "Want the one-pager?" in seen["prompt"]
    assert "May I send you the one-page technical memo?" in seen["prompt"]
    assert "CHOSEN over REJECTED" in seen["prompt"]


def test_load_preferences_keeps_only_last_k(tmp_path):
    prefs = str(tmp_path / "p.jsonl")
    for i in range(9):
        record_preference(f"c{i}", f"r{i}", path=prefs)
    rows = load_preferences(prefs, k=4)
    assert [r["chosen"] for r in rows] == ["c5", "c6", "c7", "c8"]


# --------------------------------------------------------------------------- truth gate
def test_truth_gate_blocks_numbers_not_in_facts():
    p = planner()
    verdict = p.truth("Peter, I'm Beckett, building Aurelius. We cut costs 9999% last week. "
                      "Want the memo? Beckett")
    assert not verdict["ok"]
    assert any("number not in facts" in str(v) for v in verdict["violations"])


def test_truth_gate_fails_closed_when_judge_unavailable():
    def broken(p, **k):
        raise RuntimeError("api down")
    from swm.decision.outreach_contract import plain_baseline_draft
    p = planner(chat=broken)
    verdict = p.truth(plain_baseline_draft(BRIEF, "Peter Thiel"))
    assert not verdict["ok"]                     # unverifiable text does NOT pass
    assert any("failing closed" in str(v) for v in verdict["violations"])


def test_truth_gate_offline_accepts_plain_baseline():
    from swm.decision.outreach_contract import plain_baseline_draft
    verdict = planner().truth(plain_baseline_draft(BRIEF, "Peter Thiel"))
    assert verdict["ok"]


# --------------------------------------------------------------------------- separation invariant
def test_outcome_judge_never_sees_gate_failures(monkeypatch):
    """The core separation guarantee: a candidate failing truth or language never reaches the
    outcome judge, so persona appeal can never resurrect a fabrication or bot-register text."""
    evaluated = []

    class _Ens:
        counts = {"assistant_screens": {"no_response": 3}}
        n_draws = 3
        failures = 0
        def expected_utility(self, *a, **k):
            return 0.0

    import swm.decision.persona_response as pr
    monkeypatch.setattr(pr, "ensemble_evaluate",
                        lambda chat, dossier, hyps, text, **k: evaluated.append(text) or _Ens())

    p = planner(chat=lambda pr_, **k: "", hypotheses=HYPS)
    texts = {
        "S:iden>surp>evid>requ": "CLEAN: plain, true, human.",
        "S:surp>iden>evid>requ": "FABRICATED: we grew 9999% and raised $5M.",
        "S:reco>surp>evid>iden>requ": "BOT REGISTER: May I most humbly request your attention?",
    }
    monkeypatch.setattr(p, "instantiate",
                        lambda st, reqs: texts.get("S:" + ">".join(b[:4] for b in st), ""))
    monkeypatch.setattr(p, "truth",
                        lambda t: {"ok": "9999" not in t, "violations": []})
    p.language = lambda t: LanguageVerdict(ok="humbly" not in t,
                                           score=0.2 if "humbly" in t else 0.9)
    monkeypatch.setattr(p, "wording_pass", lambda t: t)
    monkeypatch.setattr(p, "beat_variants",
                        lambda st, t: [{"label": "keep", "text": t, "origin": "structure"}])

    result = p.run()
    joined = "\n".join(evaluated)
    assert "9999" not in joined, "truth-gate failure reached the outcome judge"
    assert "humbly" not in joined, "language-gate failure reached the outcome judge"
    assert "CLEAN" in joined
    assert "9999" not in result.winner_text and "humbly" not in result.winner_text


def test_outcome_judge_is_blind_shuffled(monkeypatch):
    seen_order = []

    class _Ens:
        counts = {"h": {"no_response": 3}}
        n_draws = 3
        failures = 0
        def expected_utility(self, *a, **k):
            return 0.0

    import swm.decision.persona_response as pr
    monkeypatch.setattr(pr, "ensemble_evaluate",
                        lambda chat, dossier, hyps, text, **k: seen_order.append(text) or _Ens())
    # pick a seed whose 4-element shuffle is a real permutation
    seed = next(s for s in range(50)
                if (lambda l: (random.Random(s).shuffle(l), l)[1])(list("abcd")) != list("abcd"))
    p = planner(chat=lambda q, **k: "", hypotheses=HYPS, seed=seed)
    finalists = [{"label": f"c{i}", "text": f"text_{i}"} for i in range(4)]
    p.outcome_rank(finalists)
    assert sorted(seen_order) == [f"text_{i}" for i in range(4)]
    assert seen_order != [f"text_{i}" for i in range(4)], "evaluation order was not shuffled"


def test_outcome_counts_go_to_trace_not_to_result(monkeypatch, tmp_path):
    class _Ens:
        counts = {"assistant_screens": {"requests_material": 2, "no_response": 1}}
        n_draws = 3
        failures = 0
        def expected_utility(self, *a, **k):
            return 0.5

    import swm.decision.persona_response as pr
    monkeypatch.setattr(pr, "ensemble_evaluate", lambda *a, **k: _Ens())
    trace = str(tmp_path / "plan.jsonl")
    p = planner(chat=lambda q, **k: "", hypotheses=HYPS, trace_path=trace)
    out = p.outcome_rank([{"label": "a", "text": "ta"}, {"label": "b", "text": "tb"}])
    assert set(out) == {"order", "separable"}          # ordinal info only, no utilities upward
    rows = [json.loads(l) for l in open(trace)]
    internal = [r for r in rows if r.get("stage") == "step6_outcome_internal"]
    assert internal and "counts" in internal[0]


# --------------------------------------------------------------------------- single-output promise
def test_offline_run_yields_one_output_and_no_percentages():
    result = planner().run()
    assert isinstance(result.winner_text, str) and result.winner_text
    s = result.summary()
    assert s["report_type"] == "reply_first_single_output"
    assert isinstance(s["recommended_message"], str)
    # the message BODY may cite the sender's own factual stats (e.g. "+724%"); the invariant is
    # that no simulated OUTCOME percentage appears anywhere else in the human-facing report
    meta = {k: v for k, v in s.items() if k != "recommended_message"}
    blob = json.dumps(meta)
    assert "%" not in blob, "simulated percentage leaked into the human-facing summary"
    assert "expected_utility" not in blob
    assert not re.search(r"\b0\.\d+\s*(probability|chance)", blob)
    assert "best-supported candidate" in result.label


def test_no_percent_label_wording():
    assert "%" not in _NO_PERCENT_LABEL
    assert "no reliable distinction" in _NO_PERCENT_LABEL
    assert "real outreach outcomes" in _NO_PERCENT_LABEL


def test_tie_break_prefers_language_score_then_brevity(monkeypatch):
    p = planner()          # offline: outcome judge cannot separate -> tie-break path
    monkeypatch.setattr(p, "instantiate", lambda st, reqs: "")
    monkeypatch.setattr(p, "truth", lambda t: {"ok": True, "violations": []})
    scores = {"long text with many extra words here": 0.9, "short text": 0.9}
    p.language = lambda t: LanguageVerdict(ok=True, score=scores.get(t, 0.9))
    monkeypatch.setattr(p, "beat_variants",
                        lambda st, t: [{"label": "keep", "text": t, "origin": "s"}])
    monkeypatch.setattr(p, "wording_pass",
                        lambda t: "long text with many extra words here")
    import swm.decision.outreach_contract as oc
    monkeypatch.setattr(oc, "plain_baseline_draft", lambda *a, **k: "short text")
    result = p.run()
    assert result.winner_text == "short text"          # equal language score -> fewer words wins
    assert result.label == _NO_PERCENT_LABEL


# --------------------------------------------------------------------------- prompts carry the rules
def test_writing_rules_ride_on_every_instantiation_prompt():
    prompts = []
    def capture(q, **k):
        prompts.append(q)
        return "Peter, plain email text. Beckett"
    p = planner(chat=capture)
    p.instantiate(STRUCTURES[0], {"worthwhile": "w"})
    joined = prompts[-1]
    for needle in ("AT MOST ONE number", "no ceremonious permission constructions",
                   "NO jargon compounds", "45-85 words"):
        assert needle in joined


def test_backward_requirements_prompt_works_backward_from_replies():
    prompts = []
    def capture(q, **k):
        prompts.append(q)
        return ""
    p = planner(chat=capture)
    reqs = p.backward_requirements([{"reply": "Interesting. Send it.",
                                     "outcome": "requests_material"}])
    assert "Work BACKWARD" in prompts[-1]
    assert '"Interesting. Send it."' in prompts[-1]
    assert set(reqs) == {"worthwhile", "surprising", "believable", "noticed", "effortless"}


def test_beat_vocabulary_is_complete():
    assert set(BEATS) == set(BEAT_ROLE)
    for st in STRUCTURES:
        assert st[-1] == "request"
        assert set(st) <= set(BEATS)


# --------------------------------------------------------------------------- run-1 forensic fixes
def test_request_beat_role_is_sender_directed():
    """Run-1 forensic: 'the reply being asked for' made the LLM paste the recipient's desired
    reply ('Send me the one-pager.') as the sender's closing line. The role text must direct the
    ask from the sender."""
    role = BEAT_ROLE["request"]
    assert "sender's own words" in role.lower() or "SENDER" in role
    assert "NEVER paste" in role
    prompts = []
    def capture(q, **k):
        prompts.append(q)
        return "email"
    p = planner(chat=capture)
    p.instantiate(STRUCTURES[0], {})
    assert "never the recipient's hoped-for reply" in prompts[-1]


def test_request_swaps_precede_drops_in_variant_pool():
    """Run-1 forensic: a [:4] ranking slice silently excluded both request swaps because they
    were appended after the necessity drops."""
    def chat(q, **k):
        return '{"a": "Want the one-pager?", "b": "Does your team track fleet-level cost curves?"}'
    p = planner(chat=chat)
    text = ("I build planning software. The bottleneck is planning, not power. "
            "It cut GPU hours in replay. Want details?")
    variants = p.beat_variants(STRUCTURES[0], text)
    labels = [v["label"] for v in variants]
    assert labels[1:3] == ["request_a", "request_b"]
    assert all(l.startswith("drop_") for l in labels[3:])
    assert variants[1]["text"].endswith("Want the one-pager?")


def test_gate_pool_prefers_strictly_clean_over_flagged_high_score(monkeypatch):
    """Run-1 forensic: near-miss admission let a flagged 0.95 candidate beat an unflagged one."""
    p = planner()
    monkeypatch.setattr(p, "truth", lambda t: {"ok": True, "violations": []})
    verdicts = {"clean": LanguageVerdict(ok=True, score=0.7),
                "flagged": LanguageVerdict(ok=False, score=0.95,
                                           flags=[{"sentence": "x", "problem": "two numbers"}])}
    p.language = lambda t: verdicts[t]
    pool = p._gate_pool([{"label": "a", "text": "flagged"}, {"label": "b", "text": "clean"}])
    assert [c["text"] for c in pool] == ["clean"]          # strict beats flagged despite 0.95
    pool2 = p._gate_pool([{"label": "a", "text": "flagged"}])
    assert [c["text"] for c in pool2] == ["flagged"]       # near-miss only when nothing strict


def test_repair_language_turns_flags_into_edits(monkeypatch):
    """A flagged finalist gets ONE targeted revision; the repair is accepted only if it comes
    back strictly clean under both gates."""
    def chat(q, **k):
        assert "A language judge flagged these problems" in q
        assert "two big numbers" in q
        return "Peter, one clean sentence with one number: +724% in replay. Want the one-pager?"
    p = planner(chat=chat)
    monkeypatch.setattr(p, "truth", lambda t: {"ok": True, "violations": []})
    p.language = lambda t: (LanguageVerdict(ok=True, score=0.95)
                            if "clean sentence" in t else
                            LanguageVerdict(ok=False, score=0.6,
                                            flags=[{"sentence": "s", "problem": "two big numbers"}]))
    cand = {"label": "polished", "text": "old text with ~1.5M and 724% competing.",
            "gates": {"truth": True, "language": False, "language_score": 0.6},
            "_lang_flags": [{"sentence": "s", "problem": "two big numbers"}]}
    fixed = p.repair_language(cand)
    assert fixed["label"] == "polished+lang_repair"
    assert "clean sentence" in fixed["text"]
    assert fixed["gates"]["language"] and not fixed["_lang_flags"]
    # a repair that fails the gates is rejected: original returned untouched
    p2 = planner(chat=lambda q, **k: "still bad text")
    monkeypatch.setattr(p2, "truth", lambda t: {"ok": True, "violations": []})
    p2.language = lambda t: LanguageVerdict(ok=False, score=0.4,
                                            flags=[{"sentence": "s", "problem": "still bad"}])
    assert p2.repair_language(dict(cand)) == cand


def test_identity_window_frees_structure_search_but_keeps_debate_bait_dead():
    """Run-2 forensic: the v3 'identity in the first two sentences' rule contract-killed every
    structure that places the identity beat later, collapsing the structure search to the
    baseline. The planner now validates with identity_window=None: position is the outcome
    judge's question, PRESENCE is still a hard rule."""
    from swm.decision.outreach_contract import validate
    late_identity = ("Most people think AI infrastructure is power-constrained. "
                     "It's actually planning-constrained: schedulers optimize the next placement. "
                     "I'm Beckett, building Aurelius to fix exactly that. "
                     "Want the one-pager?")
    assert not validate(late_identity, BRIEF).ok                        # v3 rule: too late
    assert validate(late_identity, BRIEF, identity_window=None).ok      # v4 rule: present -> ok
    no_identity = ("Everyone says AI is power-bound. That is wrong and provably so. "
                   "The bottleneck is planning. Which assumption of yours breaks first?")
    v = validate(no_identity, BRIEF, identity_window=None)
    assert not v.ok and any("identity" in m for m in v.missing)         # debate-bait stays dead
    p = planner()
    assert p.truth(late_identity)["ok"]                                 # planner uses window=None


# --------------------------------------------------------------------------- pipeline default
def test_pipeline_defaults_to_reply_first_and_wraps_single_output():
    import inspect

    from swm.decision.message_pipeline import optimize_cold_outreach
    assert inspect.signature(optimize_cold_outreach).parameters["method"].default == "reply_first"

    class _R:
        contact_id = "peter_thiel"
        name = "Peter Thiel"
        label = "Peter Thiel"
        variables = {}
    res = optimize_cold_outreach(_R(), sender_brief=BRIEF, chat_fn=None,
                                 dossier=DOSSIER, hypotheses=[])
    assert res.winner == "reply_first_winner"
    assert "best-supported candidate" in res.honesty
    assert "%" not in res.honesty
    assert "probability" not in json.dumps(res.summary())
