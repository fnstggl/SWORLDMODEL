"""D18 — self-contained traces + persistent-cache provenance. The entire run must be auditable
from the trace ALONE: every LLM call keeps its exact prompt and reply, model/tier, lengths, a
truncation flag, and — when it came from cache — the cache key and the source run/call it was
recorded in. No decision, fact, or artifact is a dangling hash that needs an external lookup.

The EXP-113 failure this eliminates: traces referenced facts and decisions by id/hash, and cached
artifacts carried no provenance, so a run could not be reconstructed or audited end to end.

`CallTraceRecord` is one self-contained call. `SelfContainedTrace` aggregates the calls with the
world artifacts (representation, knowledge packets, deliberation, evidence store, outcome
mechanism, structural fidelity) and verifies SELF-CONTAINMENT: every id referenced in the
artifacts resolves to real content inside the trace, and every call carries its actual text.

Universal: the verifier checks structure, never question content."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TRACE_PROVENANCE_VERSION = "lean_v2.trace_provenance.v1"

#: an opaque reference that would need an external lookup if its content were absent from the trace
_ID_LIKE = re.compile(r"^(f_[0-9a-f]{6,}|[0-9a-f]{16,}|vc_\d+|case_\d+|obs_[0-9a-z]+)$", re.I)


@dataclass
class CallTraceRecord:
    call_id: int
    stage: str = ""
    model: str = ""
    tier: str = ""
    prompt: str = ""                    # the EXACT text sent
    reply: str = ""                     # the EXACT text returned
    prompt_chars: int = 0
    reply_chars: int = 0
    truncated: bool = False
    cache_source: str = "live"          # live | cache
    cache_key: str = ""
    source_run: str = ""                # for a cache hit: the run that first recorded it
    source_call_id: str = ""
    parsed_ok: bool = True
    validation: dict = field(default_factory=dict)
    repairs: list = field(default_factory=list)

    def is_self_contained(self) -> bool:
        # a live or cached call must carry its real prompt+reply text, not a placeholder/hash
        return bool(self.prompt) and bool(self.reply) \
            and not _ID_LIKE.match(self.prompt.strip()) and not _ID_LIKE.match(self.reply.strip())

    def as_dict(self) -> dict:
        return {"call_id": self.call_id, "stage": self.stage, "model": self.model,
                "tier": self.tier, "prompt": self.prompt, "reply": self.reply,
                "prompt_chars": self.prompt_chars, "reply_chars": self.reply_chars,
                "truncated": self.truncated, "cache_source": self.cache_source,
                "cache_key": self.cache_key, "source_run": self.source_run,
                "source_call_id": self.source_call_id, "parsed_ok": self.parsed_ok,
                "validation": self.validation, "repairs": self.repairs}


def call_trace_from_row(call_id: int, row: dict) -> CallTraceRecord:
    """Build a self-contained call record from a gateway row, flagging truncation and carrying any
    cache provenance the gateway recorded."""
    prompt, reply = row.get("prompt") or "", row.get("reply") or ""
    pc = row.get("prompt_chars") if row.get("prompt_chars") is not None else len(prompt)
    rc = row.get("reply_chars") if row.get("reply_chars") is not None else len(reply)
    truncated = bool(row.get("truncated")) or bool(row.get("max_tokens_hit")) \
        or (row.get("finish_reason") == "length")
    return CallTraceRecord(
        call_id=call_id, stage=row.get("stage", ""), model=row.get("model", ""),
        tier=row.get("tier", ""), prompt=prompt, reply=reply, prompt_chars=pc, reply_chars=rc,
        truncated=truncated, cache_source=row.get("cache_source", "live"),
        cache_key=row.get("cache_key", ""), source_run=row.get("source_run", ""),
        source_call_id=row.get("source_call_id", ""), parsed_ok=bool(row.get("parsed_ok", True)),
        validation=row.get("validation") or {}, repairs=row.get("repairs") or [])


def _collect_ids(obj, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_ids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_ids(v, out)
    elif isinstance(obj, str) and _ID_LIKE.match(obj.strip()):
        out.add(obj.strip())


def _collect_resolvable(obj, out: set):
    """Every id that IS defined with content somewhere (a fact/case/unit record carrying an id)."""
    if isinstance(obj, dict):
        for key in ("fact_id", "case_id", "unit_id", "state_id", "obs_id"):
            if obj.get(key) and (obj.get("content") or obj.get("description") or obj.get("claim")
                                 or obj.get("represents_label") or obj.get("render") or len(obj) > 2):
                out.add(str(obj[key]).strip())
        for v in obj.values():
            _collect_resolvable(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_resolvable(v, out)


def verify_self_contained(trace: dict) -> dict:
    """A trace is self-contained when (1) every LLM call carries its real prompt+reply text and
    (2) every id-like reference in the artifacts resolves to a record with content inside the same
    trace. Returns {ok, calls_missing_content, dangling_references}."""
    calls = trace.get("calls") or trace.get("llm_calls") or []
    missing = [c.get("call_id", i) for i, c in enumerate(calls)
               if not (c.get("prompt") and c.get("reply"))]
    referenced, resolvable = set(), set()
    artifacts = {k: v for k, v in trace.items() if k not in ("calls", "llm_calls")}
    _collect_ids(artifacts, referenced)
    _collect_resolvable(artifacts, resolvable)
    dangling = sorted(referenced - resolvable)
    ok = not missing and not dangling
    return {"ok": ok, "calls_missing_content": missing, "dangling_references": dangling,
            "n_calls": len(calls), "n_referenced_ids": len(referenced),
            "n_resolvable_ids": len(resolvable), "version": TRACE_PROVENANCE_VERSION}


def build_self_contained_trace(gateway_rows: list, lean_v2_prov: dict) -> dict:
    """Assemble the full self-contained trace: every call (with truncation + cache provenance) plus
    the fidelity artifacts, then verify self-containment. Uncapped — a human-facing sample may cap
    separately, but the audit trace keeps everything."""
    calls = [call_trace_from_row(i, r).as_dict() for i, r in enumerate(gateway_rows or [])]
    prov = lean_v2_prov.get("lean_v2") or lean_v2_prov
    artifacts = {
        "calls": calls,
        "representation": (prov.get("institution_terminal") or {}).get("representation"),
        "knowledge_packets": prov.get("knowledge_packets"),
        "deliberation": (prov.get("institution_terminal") or {}).get("per_combo"),
        "evidence_store": prov.get("evidence_store"),
        "shared_condition_graph": prov.get("shared_condition_graph"),
        "mindset_separation": prov.get("mindset_separation"),
        "outcome_mechanism_dimensions": prov.get("outcome_mechanism_dimensions"),
        "structural_fidelity": prov.get("structural_fidelity"),
        "actor_states": prov.get("actor_states"),
        "forecast_decomposition": prov.get("forecast_decomposition")}
    artifacts["self_contained"] = verify_self_contained(artifacts)
    return artifacts
