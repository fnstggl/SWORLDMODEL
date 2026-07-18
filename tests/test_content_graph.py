"""Offline tests for the message-content-graph stage (search over ideas/information before language).

The invariants under test are the architectural promises of content_graph.py + its integration into
the reply-first planner:
  * BUILD quarantines LOUDLY — a unit whose number is not in the sender facts, whose text is too
    long, or whose identity/credibility claim is untraceable to a fact never enters the graph;
  * the OFFLINE fallback derives units mechanically from the facts alone (no invention);
  * PLAN checks are deterministic — exactly one request unit, at most one number-bearing evidence
    unit (the one-number rule), a relevance unit unless its omission is stated;
  * ADVERSARIAL DELETION preserves the signature, probes every sentence exactly once, and never
    offers a truth-gate-violating variant as a candidate (it never self-certifies a winner);
  * INTEGRATION — a live planner's content-graph seeds flow through the SAME truth + language gates
    and the SAME blind outcome judge, still yielding ONE output with the no-percent label;
  * with the flag OFF the offline planner is byte-identical to the pre-content-graph default.
"""
import json

from swm.decision.content_graph import (ContentUnit, SemanticPlan, adversarial_deletion,
                                         build_content_graph, plan_semantics, validate_plan,
                                         verbalize)
from swm.decision.language_judge import LanguageVerdict
from swm.decision.llm_moves import allowed_numbers, number_violations
from swm.decision.outreach_contract import validate
from swm.decision.persona_response import PersonaDossier
from swm.decision.reply_first import _NO_PERCENT_LABEL, ReplyFirstPlanner
from tests.test_reply_first import BRIEF, DOSSIER, HYPS


# --------------------------------------------------------------------------- scripted live backend
_UNITS = {"units": [
    {"kind": "identity_fact", "text": "17 years old, starting Princeton in the fall",
     "belief_target": "a real, specific person", "numbers_used": ["17"]},
    {"kind": "credibility_fact", "text": "results are simulated replay, not a production deployment",
     "belief_target": "honest about limits", "numbers_used": []},
    {"kind": "evidence", "text": "+724% SLA-safe goodput per dollar in simulated replay",
     "belief_target": "measured, not asserted", "numbers_used": ["724"]},
    {"kind": "problem_formulation",
     "text": "AI infrastructure has a planning problem disguised as a power problem",
     "belief_target": "the problem is mis-framed", "numbers_used": []},
    {"kind": "insight", "text": "The bottleneck is planning, not power",
     "belief_target": "a non-obvious claim", "numbers_used": []},
    {"kind": "recipient_relevance",
     "text": "You are skeptical of consensus, and this is a contrarian claim",
     "belief_target": "this is for you specifically", "numbers_used": []},
    {"kind": "request", "text": "Want the one-page memo?", "belief_target": "a cheap keystroke",
     "numbers_used": []},
]}
_PLANS = {"plans": [
    {"carrying_idea": "u4", "included": ["u0", "u4", "u2", "u5", "u6"],
     "omitted": [{"unit_id": "u1", "reason": "brevity"}],
     "belief_targets": {"u4": "non-obvious"}, "predicted_shortest_form": "planning not power; memo?"},
    {"carrying_idea": "u3", "included": ["u0", "u3", "u2", "u6"],
     "omitted": [{"unit_id": "u5", "reason": "no strong personal hook here"}],
     "belief_targets": {}, "predicted_shortest_form": "planning problem; memo?"},
]}
_EMAILS = json.dumps([
    "Peter, I'm Beckett, 17, building Aurelius. The real bottleneck is planning, not power. In "
    "replay it lifted goodput sharply. Want the one-page memo? Beckett",
    "Peter, I'm Beckett, starting Princeton, building Aurelius. Planning, not power, is the "
    "bottleneck. Want the one-pager? Beckett",
])


def scripted_chat(prompt, **kw):
    """A deterministic backend that answers each content-graph + planner generation stage by the
    marker in its prompt; returns '' for anything else (gates/outcome are monkeypatched)."""
    if "most valuable POSITIVE replies" in prompt:
        return json.dumps({"replies": [
            {"reply": "Interesting. Send it.", "outcome": "requests_material"},
            {"reply": "What's the insight?", "outcome": "curious_reply"}]})
    if "Work BACKWARD" in prompt:
        return json.dumps({"worthwhile": "w", "surprising": "s", "believable": "b",
                           "noticed": "n", "effortless": "e"})
    if "TRUE INGREDIENTS" in prompt:
        return json.dumps(_UNITS)
    if "PLANNING a cold message at the level of IDEAS" in prompt:
        return json.dumps(_PLANS)
    if "ALREADY-CHOSEN plan" in prompt:
        return _EMAILS
    if "beat sequence" in prompt:
        return ("Peter, I'm Beckett building Aurelius. Planning is the real bottleneck. "
                "Want the memo? Beckett")
    if "TWO alternative CLOSING lines" in prompt:
        return json.dumps({"a": "Want the one-pager?", "b": "What breaks first at fleet scale?"})
    if "STRONGEST reason" in prompt:
        return "It reads like every other cold pitch, so I'd never open it."
    if "defuse exactly that reason" in prompt:
        return ("Peter, I'm Beckett, 17, building Aurelius. Planning, not power, is the bottleneck. "
                "Want the one-pager? Beckett")
    return ""


def _gate_passes(text, brief):
    """The deterministic core of the truth gate (numeric fact guard + cold-outreach contract)."""
    allowed = allowed_numbers(brief.to_prompt())
    return not number_violations(text, allowed) and validate(text, brief, identity_window=None).ok


# --------------------------------------------------------------------------- (A) unit validation
def test_build_drops_number_violating_units_loudly():
    bad = {"units": [
        {"kind": "evidence", "text": "we grew 9999% overnight versus the baseline",
         "belief_target": "x", "numbers_used": ["9999"]},
        {"kind": "insight", "text": "The bottleneck is planning, not power", "numbers_used": []},
    ]}
    g = build_content_graph(lambda p, **k: json.dumps(bad), BRIEF, DOSSIER)
    kept = [u.text for u in g.units]
    assert "The bottleneck is planning, not power" in kept
    assert all("9999" not in t for t in kept)                 # the fabricated-number unit is gone
    rej = [r for r in g.rejected_units if "9999" in json.dumps(r)]
    assert rej, "number-violating unit was dropped SILENTLY"
    assert "9999" in rej[0]["reason"] and "not in sender facts" in rej[0]["reason"]


def test_build_rejects_overlong_and_untraceable_and_declared_number():
    payload = {"units": [
        {"kind": "insight", "text": "x" * 240, "numbers_used": []},                    # too long
        {"kind": "identity_fact", "text": "a Princeton admit featured in the NYT",      # untraceable
         "numbers_used": []},
        {"kind": "credibility_fact", "text": "we shipped to 4200 enterprise customers",  # number in text
         "numbers_used": []},
        {"kind": "evidence", "text": "goodput improved a lot in replay",     # declared number bad
         "numbers_used": ["5000"]},
        {"kind": "insight", "text": "The bottleneck is planning, not power", "numbers_used": []},
    ]}
    g = build_content_graph(lambda p, **k: json.dumps(payload), BRIEF, DOSSIER)
    assert [u.text for u in g.units] == ["The bottleneck is planning, not power"]
    reasons = " ".join(r["reason"] for r in g.rejected_units)
    assert "220 limit" in reasons                              # length quarantine
    assert "not traceable" in reasons                          # invented credential blocked
    assert "4200" in reasons and "5000" in reasons             # both numeric guards fired


def test_offline_fallback_builds_units_from_facts_only():
    g = build_content_graph(None, BRIEF, DOSSIER)
    assert g.source == "offline"
    assert g.units and not g.rejected_units
    assert all(u.source in ("derived_offline", "dossier") for u in g.units)
    # every number used is drawn from the sender facts, and nothing invents an out-of-facts number
    allowed = allowed_numbers(BRIEF.to_prompt())
    for u in g.units:
        assert not number_violations(u.text, allowed)
        assert all(not number_violations(str(n), allowed) for n in u.numbers_used)
    kinds = g.kinds()
    assert {"identity_fact", "evidence", "request"} <= kinds   # the load-bearing ingredients exist
    # the identity/credibility units are literally the sender's facts (traceable by construction)
    for u in g.units:
        if u.kind in ("identity_fact", "credibility_fact"):
            assert any(u.text in f or f in u.text for f in BRIEF.facts)


# --------------------------------------------------------------------------- (C) plan checks
def _units(*specs):
    us = [ContentUnit(kind=k, text=t, numbers_used=list(n)) for k, t, n in specs]
    for i, u in enumerate(us):
        u.unit_id = f"u{i}"
    return us, {u.unit_id: u for u in us}


def test_plan_check_requires_exactly_one_request():
    us, by = _units(("insight", "planning not power", []), ("request", "Want the memo?", []),
                    ("request", "Free next week?", []), ("recipient_relevance", "for you", []))
    two_req = SemanticPlan("p", "u0", ["u0", "u1", "u2", "u3"])
    ok, reason = validate_plan(two_req, by)
    assert not ok and "exactly one request" in reason
    one_req = SemanticPlan("p", "u0", ["u0", "u1", "u3"])
    assert validate_plan(one_req, by)[0]


def test_plan_check_enforces_one_number_rule():
    us, by = _units(("insight", "planning not power", []),
                    ("evidence", "+724% goodput in replay", ["724"]),
                    ("evidence", "1.5M requests replayed", ["1.5"]),
                    ("recipient_relevance", "for you", []), ("request", "Want the memo?", []))
    two_num = SemanticPlan("p", "u0", ["u0", "u1", "u2", "u3", "u4"])
    ok, reason = validate_plan(two_num, by)
    assert not ok and "one-number rule" in reason
    one_num = SemanticPlan("p", "u0", ["u0", "u1", "u3", "u4"])
    assert validate_plan(one_num, by)[0]


def test_plan_check_requires_relevance_unless_omission_stated():
    us, by = _units(("insight", "planning not power", []), ("request", "Want the memo?", []),
                    ("recipient_relevance", "you are contrarian", []))
    kinds = {u.kind for u in us}
    no_rel = SemanticPlan("p", "u0", ["u0", "u1"], omitted=[])
    ok, reason = validate_plan(no_rel, by, graph_kinds=kinds)
    assert not ok and "recipient_relevance" in reason
    # explicitly omitting the relevance unit WITH a reason is allowed
    stated = SemanticPlan("p", "u0", ["u0", "u1"],
                          omitted=[{"unit_id": "u2", "reason": "no strong personal hook"}])
    assert validate_plan(stated, by, graph_kinds=kinds)[0]
    # the deliberately-minimal brevity plan is exempt from the relevance requirement
    minimal = SemanticPlan("minimal", "u0", ["u0", "u1"])
    assert validate_plan(minimal, by, graph_kinds=kinds)[0]


def test_plan_semantics_filters_invalid_and_adds_minimal_baseline():
    g = build_content_graph(None, BRIEF, DOSSIER)          # offline units (u0..)
    ids = [u.unit_id for u in g.units]
    req = next(u.unit_id for u in g.units if u.kind == "request")
    idea = next(u.unit_id for u in g.units if u.kind == "insight")
    rel = next(u.unit_id for u in g.units if u.kind == "recipient_relevance")
    good = {"carrying_idea": idea, "included": [idea, rel, req], "omitted": [],
            "belief_targets": {}, "predicted_shortest_form": "s"}
    twin_request = {"carrying_idea": idea, "included": [idea, req, req], "omitted": []}
    chat = lambda p, **k: json.dumps({"plans": [good, twin_request]})
    plans = plan_semantics(chat, g.units, [{"reply": "x", "outcome": "curious_reply"}])
    by = {u.unit_id: u for u in g.units}
    assert all(validate_plan(p, by, graph_kinds=g.kinds())[0] for p in plans)  # only valid survive
    assert any(p.plan_id == "minimal" for p in plans)                          # brevity baseline added
    assert not any(sorted(p.included) == sorted([idea, req, req]) for p in plans)


def test_verbalize_offline_returns_nothing():
    g = build_content_graph(None, BRIEF, DOSSIER)
    plans = plan_semantics(None, g.units, [])
    assert verbalize(None, plans[0], g.units, BRIEF) == []      # no writer -> no language


# --------------------------------------------------------------------------- (E) adversarial deletion
_SIGNED = ("Peter, I'm Beckett, 17, building Aurelius. Planning is the bottleneck, not power. "
           "In replay it lifted goodput. Want the one-page memo? Beckett")


def test_adversarial_deletion_probes_cover_every_sentence_exactly_once():
    ad = adversarial_deletion(None, _SIGNED, BRIEF)
    removed = [d["removed_sentence"] for d in ad["deletions"]]
    body = ["Peter, I'm Beckett, 17, building Aurelius.", "Planning is the bottleneck, not power.",
            "In replay it lifted goodput.", "Want the one-page memo?"]
    assert removed == body                                      # every body sentence, in order, once
    assert len(removed) == len(set(removed))                    # no sentence probed twice
    assert "Beckett" not in removed                             # the bare signature is NOT a probe


def test_adversarial_deletion_preserves_signature_and_emits_no_violating_candidate():
    ad = adversarial_deletion(None, _SIGNED, BRIEF)
    # the bare trailing signature is held out and re-appended to EVERY variant
    assert all(d["text"].rstrip().endswith("Beckett") for d in ad["deletions"])
    # every variant OFFERED AS A CANDIDATE (truth_ok) actually passes the truth gate
    for d in ad["deletions"]:
        if d["truth_ok"]:
            assert _gate_passes(d["text"], BRIEF)
    # removing the ask breaks the contract -> that variant is NOT a candidate
    drop_request = next(d for d in ad["deletions"] if "one-page memo" in d["removed_sentence"])
    assert not drop_request["truth_ok"]
    # a caller-supplied truth judge is honored: a fail-closed judge admits nothing, yet coverage holds
    hostile = adversarial_deletion(None, _SIGNED, BRIEF, judge=lambda t: {"ok": False})
    assert len(hostile["deletions"]) == len(ad["deletions"])
    assert all(not d["truth_ok"] for d in hostile["deletions"])


def test_adversarial_deletion_repair_must_pass_the_gate():
    # a repair that injects an out-of-facts number is refused (repaired_ok stays False)
    bad_repair = adversarial_deletion(
        lambda p, **k: ("Peter, I'm Beckett building Aurelius. We serve 8800 customers now. "
                        "Want the one-pager? Beckett") if "defuse" in p else "reason",
        _SIGNED, BRIEF)
    assert not bad_repair["repaired_ok"]
    # a clean, contract-valid repair is admitted (and never self-certified as the winner)
    good_repair = adversarial_deletion(
        lambda p, **k: ("Peter, I'm Beckett, 17, building Aurelius. Planning, not power, is the "
                        "bottleneck. Want the one-pager? Beckett") if "defuse" in p else "reason",
        _SIGNED, BRIEF)
    assert good_repair["repaired_ok"] and _gate_passes(good_repair["repaired"], BRIEF)


# --------------------------------------------------------------------------- integration
def _blind_ensemble(monkeypatch):
    class _Ens:
        counts = {"assistant_screens": {"no_response": 3}}
        n_draws = 3
        failures = 0

        def expected_utility(self, *a, **k):
            return 0.0
    import swm.decision.persona_response as pr
    monkeypatch.setattr(pr, "ensemble_evaluate", lambda *a, **k: _Ens())


def test_integration_content_graph_seeds_flow_through_same_gates_and_one_output(monkeypatch):
    _blind_ensemble(monkeypatch)
    p = ReplyFirstPlanner(scripted_chat, sender_brief=BRIEF, dossier=DOSSIER, hypotheses=HYPS,
                          seed=1, use_content_graph=True)
    # focus on the CG integration: deterministic truth/language/wording (as the reply_first tests do)
    monkeypatch.setattr(p, "truth", lambda t: {"ok": True, "violations": []})
    p.language = lambda t: LanguageVerdict(ok=True, score=0.9)
    monkeypatch.setattr(p, "wording_pass", lambda t: t)

    seen = []
    real_gate = p._gate_pool
    monkeypatch.setattr(p, "_gate_pool", lambda cands: (seen.extend(cands), real_gate(cands))[1])

    result = p.run()

    cg = [c for c in seen if c["label"].startswith("CG:")]
    assert cg, "no content-graph seed candidates were produced"
    # they went through the SAME gate machinery every candidate uses (gate fields attached)
    assert all(set(c["gates"]) >= {"truth", "language", "language_score", "problems"} for c in cg)
    assert any(c["label"].startswith("AD:") for c in seen), "no adversarial-deletion variants ran"

    # exactly ONE output, with the no-percent honesty label and no simulated percentages upward
    assert isinstance(result.winner_text, str) and result.winner_text
    assert sum(1 for f in result.finalists if f["ordinal_note"] == "selected") == 1
    assert result.label == _NO_PERCENT_LABEL
    meta = {k: v for k, v in result.summary().items() if k != "recommended_message"}
    assert "%" not in json.dumps(meta) and "expected_utility" not in json.dumps(meta)


def test_integration_cg_candidate_can_win_and_is_labeled(monkeypatch):
    """When the semantic-planning seeds carry the day, the single winner's origin is CG-labeled and
    it still passed the same gates — the whole point of searching over ideas before language."""
    _blind_ensemble(monkeypatch)
    p = ReplyFirstPlanner(scripted_chat, sender_brief=BRIEF, dossier=DOSSIER, hypotheses=HYPS,
                          seed=1, use_content_graph=True)
    monkeypatch.setattr(p, "truth", lambda t: {"ok": True, "violations": []})
    # only content-graph verbalizations score a clean language pass; structures/baseline are flagged
    cg_texts = {e for e in json.loads(_EMAILS)}
    p.language = lambda t: (LanguageVerdict(ok=True, score=0.95) if t in cg_texts
                            else LanguageVerdict(ok=False, score=0.4,
                                                 flags=[{"sentence": "s", "problem": "flagged"}]))
    monkeypatch.setattr(p, "wording_pass", lambda t: t)
    result = p.run()
    assert result.winner_origin.startswith("CG:")
    assert result.winner_text in cg_texts


def test_flag_off_is_byte_identical_offline():
    on = ReplyFirstPlanner(None, sender_brief=BRIEF, dossier=DOSSIER, use_content_graph=True).run()
    off = ReplyFirstPlanner(None, sender_brief=BRIEF, dossier=DOSSIER, use_content_graph=False).run()
    assert on.winner_text == off.winner_text
    assert on.winner_origin == off.winner_origin
    assert on.label == off.label


def test_trace_uses_the_content_graph_stage_names(tmp_path, monkeypatch):
    _blind_ensemble(monkeypatch)
    trace = str(tmp_path / "cg.jsonl")
    p = ReplyFirstPlanner(scripted_chat, sender_brief=BRIEF, dossier=DOSSIER, hypotheses=HYPS,
                          seed=1, trace_path=trace, use_content_graph=True)
    monkeypatch.setattr(p, "truth", lambda t: {"ok": True, "violations": []})
    p.language = lambda t: LanguageVerdict(ok=True, score=0.9)
    monkeypatch.setattr(p, "wording_pass", lambda t: t)
    p.run()
    stages = {json.loads(l).get("stage") for l in open(trace) if l.strip()}
    assert {"content_graph", "semantic_plan", "verbalize", "adversarial_deletion"} <= stages
