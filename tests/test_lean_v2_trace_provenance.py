"""D18 — self-contained traces + persistent-cache provenance. Universal machinery only.

Locks: every call keeps its exact prompt+reply, truncation flag, and cache source/key/run; the
trace is SELF-CONTAINED (no dangling id references, no call missing its text); the fidelity
artifacts are carried so the run is auditable end to end from the trace alone."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.trace_provenance import (
    CallTraceRecord, build_self_contained_trace, call_trace_from_row, verify_self_contained)


# ============================================================ 65 — exact prompt+reply kept
def test_65_call_keeps_exact_prompt_and_reply():
    c = call_trace_from_row(0, {"stage": "grounding", "prompt": "P-text", "reply": "R-text",
                                "model": "claude-x", "tier": "strong"})
    assert c.prompt == "P-text" and c.reply == "R-text"
    assert c.model == "claude-x" and c.tier == "strong"
    assert c.is_self_contained()


# ============================================================ 66 — truncation flagged
def test_66_truncation_is_flagged():
    assert call_trace_from_row(0, {"prompt": "p", "reply": "r", "finish_reason": "length"}).truncated
    assert call_trace_from_row(1, {"prompt": "p", "reply": "r", "truncated": True}).truncated
    assert not call_trace_from_row(2, {"prompt": "p", "reply": "r", "finish_reason": "stop"}).truncated


# ============================================================ 67 — cache provenance carried
def test_67_cache_hit_carries_source_run_and_key():
    c = call_trace_from_row(0, {"prompt": "p", "reply": "r", "cache_source": "cache",
                                "cache_key": "k123", "source_run": "run_42", "source_call_id": "c9"})
    assert c.cache_source == "cache" and c.cache_key == "k123"
    assert c.source_run == "run_42" and c.source_call_id == "c9"


# ============================================================ 68 — dangling reference caught
def test_68_dangling_id_reference_is_not_self_contained():
    trace = {"calls": [{"call_id": 0, "prompt": "p", "reply": "r"}],
             "packet": {"cited_facts": ["f_abc123def456"]}}    # id referenced, no content record
    v = verify_self_contained(trace)
    assert not v["ok"] and "f_abc123def456" in v["dangling_references"]


# ============================================================ 69 — resolvable id is self-contained
def test_69_resolvable_id_is_self_contained():
    trace = {"calls": [{"call_id": 0, "prompt": "p", "reply": "r"}],
             "store": {"facts": [{"fact_id": "f_abc123def456", "content": "inflation rose to 3.5%"}]},
             "reference": ["f_abc123def456"]}
    v = verify_self_contained(trace)
    assert v["ok"] and v["n_resolvable_ids"] >= 1


# ============================================================ 70 — a call missing text fails
def test_70_call_missing_its_text_is_not_self_contained():
    v = verify_self_contained({"calls": [{"call_id": 0, "prompt": "p", "reply": ""}]})
    assert not v["ok"] and 0 in v["calls_missing_content"]


# ============================================================ 71 — full trace is auditable
def test_71_build_full_self_contained_trace():
    gateway_rows = [{"stage": "grounding", "prompt": "P1", "reply": "R1", "model": "m",
                     "finish_reason": "stop"},
                    {"stage": "state_generation", "prompt": "P2", "reply": "R2", "model": "m"}]
    prov = {"lean_v2": {
        "evidence_store": {"n_facts": 3, "as_of": "2025-06-01"},
        "institution_terminal": {"representation": {"total_voting_power": 9, "threshold": 5},
                                 "per_combo": [{"combo": {}, "p_yes": 0.6}]},
        "knowledge_packets": {"n": 2, "actors": ["gov", "deputy"]},
        "structural_fidelity": {"verdict": "ready"},
        "forecast_decomposition": {"headline_forecast": 0.6}}}
    trace = build_self_contained_trace(gateway_rows, prov)
    assert len(trace["calls"]) == 2
    assert all(c["prompt"] and c["reply"] for c in trace["calls"])   # every call has its text
    assert trace["representation"]["threshold"] == 5                 # artifact carried
    assert trace["structural_fidelity"]["verdict"] == "ready"
    assert trace["self_contained"]["ok"]                            # auditable end to end
