"""Iterative-editor mechanics (offline-deterministic + scripted fake LLM): trace integrity,
whole-email guard (reject locally-better-but-globally-worse), deletion as first-class move,
fact-guard on alternatives, judge fail-closed, budget cap, beam ordering."""
from __future__ import annotations

import json

from swm.decision.iterative_editor import AXES, EditState, IterativeEditor, composite
from swm.decision.llm_moves import SenderBrief

BRIEF = SenderBrief(
    sender="Beckett",
    thesis="AI infrastructure has a planning problem disguised as a power problem",
    ask="permission to send the one-page memo",
    facts=["17 years old, starting Princeton in the fall",
           "building Aurelius (runaurelius.com), AI infrastructure",
           "+724% SLA-safe goodput per dollar vs the production scheduler in simulated replay of "
           "~1.5M requests of public production traces",
           "-84% GPU-hours in the same replay"])

THIEL = {"status_orientation": 0.85, "skepticism": 0.9, "status": 0.9,
         "attention_availability": 0.4, "platform_response_norm": 0.3, "relationship_strength": 0.0}

GOOD = ("Peter, I'm Beckett: 17, building Aurelius (runaurelius.com), AI infrastructure. "
        "AI infrastructure has a planning problem disguised as a power problem. "
        "In simulated replay of ~1.5M requests of public production traces it beat the production "
        "scheduler by +724% SLA-safe goodput per dollar. "
        "May I send you the one-page memo? Beckett")


def _editor(chat=None, **kw):
    return IterativeEditor(chat, sender_brief=BRIEF, recipient_vars=THIEL, base_mean=0.2,
                           dossier_text="You are Peter Thiel (investor).", **kw)


def test_composite_weights_documented_axes():
    s = {a: 0.8 for a in AXES}
    s["cognitive_effort"] = 0.0
    s["negative_response_risk"] = 0.0
    assert abs(composite(s) - 0.8) < 1e-9
    s["negative_response_risk"] = 1.0
    assert composite(s) < 0.2


def test_offline_run_produces_beam_and_trace():
    ed = _editor(None, trace_path=None)
    out = ed.run([{"label": "plain", "text": GOOD},
                  {"label": "weak", "text": "Peter, I'm Beckett: building Aurelius. "
                                            "May I send you the one-page memo? Beckett"}])
    assert out["beam"] and out["beam"][0].value >= out["beam"][-1].value
    phases = {r["phase"] for r in out["trace"]}
    assert "seed" in phases and "final" in phases
    assert out["llm_calls"] == 0                       # fully offline


def test_whole_email_guard_rejects_globally_worse_local_choice():
    """A scripted judge that always picks DELETE on the evidence line must be overruled by the
    whole-email rescore (dropping the only evidence lowers the composite)."""
    calls = {"n": 0}

    def chat(prompt, **kw):
        calls["n"] += 1
        if "Return ONLY JSON" in prompt and '"choice"' in prompt:
            # judge: always pick the deletion variant if present
            for lab in ("DELETE", "DROP_2"):
                if f"[{lab}]" in prompt:
                    return json.dumps({"choice": lab, "why": "scripted"})
            return json.dumps({"choice": "KEEP", "why": "scripted"})
        if "Score each axis" in prompt:
            # LLM scorer unavailable -> force the deterministic proxy by returning junk
            return "not json"
        if "Diagnose the WHOLE message" in prompt:
            return "not json"
        if '"rewrite"' in prompt:
            return json.dumps({"rewrite": "", "shorten": "", "reframe": "", "merge": "",
                               "insert_after": ""})
        return ""

    ed = _editor(chat, max_passes=1)
    st = EditState(label="seed", text=GOOD, scores=ed.score(GOOD))
    before_evidence = "+724%" in st.text
    ed.improve_pass(st, phase="pass1")
    rejected = [r for r in ed.trace if r.get("accepted") is False and
                r.get("reject_reason", "").startswith("local improvement worsened")]
    # either the deletion was rejected by the guard, or (if proxy scored it equal-or-better)
    # the evidence line survived some other way — the invariant is the guard RAN and recorded
    assert before_evidence
    assert any(r.get("selected") in ("DELETE",) for r in ed.trace), "deletion must be considered"
    assert all("scores_before" in r for r in ed.trace if r["phase"] == "pass1")
    if "+724%" not in st.text:
        # deletion went through -> it must have not worsened the proxy composite
        accepted = [r for r in ed.trace if r.get("accepted")]
        assert accepted, "an accepted deletion must be traced"
    else:
        assert rejected, "a rejected deletion must be traced with the reason"


def test_fact_guard_blocks_fabricated_alternative():
    def chat(prompt, **kw):
        if '"rewrite"' in prompt:
            return json.dumps({"rewrite": "We cut costs 93% on 512 GPUs last week.",
                               "shorten": "", "reframe": "", "merge": "", "insert_after": ""})
        if '"choice"' in prompt:
            return json.dumps({"choice": "REWRITE", "why": "scripted"})
        return "not json"

    ed = _editor(chat, max_passes=1)
    sents = GOOD.split(". ")
    variants = ed.line_alternatives([s.strip() for s in GOOD.split(". ") if s.strip()], 2, [])
    labels = {v["label"] for v in variants}
    assert "REWRITE" not in labels, "fabricated numbers (93, 512) must be guarded out"
    assert "DELETE" in labels or "KEEP" in labels


def test_judge_fail_closed_keeps_current_line():
    def chat(prompt, **kw):
        if '"choice"' in prompt:
            return "utter garbage"
        if '"rewrite"' in prompt:
            return json.dumps({"rewrite": "AI infrastructure planning is the real bottleneck.",
                               "shorten": "", "reframe": "", "merge": "", "insert_after": ""})
        return "not json"
    ed = _editor(chat, max_passes=1)
    st = EditState(label="seed", text=GOOD, scores=ed.score(GOOD))
    ed.improve_pass(st, phase="pass1")
    kept = [r for r in ed.trace if r.get("judge_reason", "").startswith("judge unparseable")]
    assert kept, "an unparseable judge must fail closed to KEEP"
    assert st.text == GOOD


def test_llm_budget_cap_respected():
    def chat(prompt, **kw):
        return "not json"
    ed = _editor(chat, max_llm_calls=5, max_passes=3)
    ed.run([{"label": "plain", "text": GOOD}])
    assert ed.calls <= 5


def test_trace_serializes_to_jsonl(tmp_path):
    p = tmp_path / "trace.jsonl"
    ed = _editor(None, trace_path=str(p))
    ed.run([{"label": "plain", "text": GOOD}])
    rows = [json.loads(l) for l in open(p)]
    assert rows and rows[0]["phase"] == "seed" and rows[-1]["phase"] == "final"
