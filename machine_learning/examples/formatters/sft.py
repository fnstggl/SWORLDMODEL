"""Format canonical records into supervised-fine-tuning text with a target span.

The prompt renders only pre-cutoff information (the canonical record already guarantees
this). Training loss is applied ONLY to the target section — the formatter returns the
exact character offset where the target begins so the collator can mask everything before
it. Both natural-language message targets and structured (JSON) targets are supported.

Rendered layout (target-only loss on everything after "TARGET:\\n"):

    TASK: <task_type>

    ACTOR:
    <role/id/profile>

    PRIVATE STATE BEFORE:
    <self-reported goal/state, or (none recorded)>

    KNOWN HISTORY:
    <prior events, one per line>

    CURRENT OBSERVATION:
    <what the actor sees now>

    AVAILABLE ACTIONS:
    <the recoverable option set, or (unknown)>

    TARGET:
    <label>
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ...tasks import UNKNOWN_ACTION_SPACE

TARGET_HEADER = "TARGET:\n"


@dataclass
class FormattedExample:
    prompt: str            # everything up to and including "TARGET:\n"
    completion: str        # the target text (loss applies here)
    text: str              # prompt + completion
    target_char_start: int  # == len(prompt); mask everything before this
    task_type: str
    record_id: str

    def as_dict(self) -> dict:
        return {"record_id": self.record_id, "task_type": self.task_type,
                "prompt": self.prompt, "completion": self.completion,
                "target_char_start": self.target_char_start}


def _render_history(events: list[dict], max_events: int) -> str:
    if not events:
        return "(no prior events)"
    ev = events[-max_events:] if max_events and len(events) > max_events else events
    lines = []
    if max_events and len(events) > max_events:
        lines.append(f"... ({len(events) - max_events} earlier events elided)")
    for e in ev:
        who = e.get("actor_id", "?")
        if e.get("kind") == "action":
            lines.append(f"[{who}] ACTION {e.get('action_type')}: {json.dumps(e.get('action_content', {}), ensure_ascii=False)}")
        else:
            txt = e.get("text")
            lines.append(f"[{who}] {txt if txt is not None else '(' + str(e.get('kind', 'event')) + ')'}")
    return "\n".join(lines)


def _compact(obj, limit: int = 800) -> str:
    if obj in (None, {}, [], ""):
        return "(none recorded)"
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= limit else s[:limit] + "…"


def target_to_text(rec: dict) -> str:
    """Serialize a record's target to the completion string.

    Message targets render as raw text; everything else renders as compact JSON so the
    structure (action type + content, distributions, effects) is learnable + parseable.
    """
    task = rec["task_type"]
    tgt = rec["payload"]["target"]
    if task == "PREDICT_NEXT_MESSAGE":
        return tgt.get("message_text", "")
    if task == "PREDICT_NEXT_SPEAKER":
        return str(tgt.get("speaker_id", ""))
    return json.dumps(tgt, ensure_ascii=False, sort_keys=True)


def format_record(rec: dict, *, max_history_events: int = 40) -> FormattedExample:
    ctx = rec.get("context", {})
    du = rec.get("decision_unit", {})
    inp = rec["payload"].get("input", {})

    actor_bits = []
    if du.get("actor_role"):
        actor_bits.append(f"role={du['actor_role']}")
    if du.get("actor_id"):
        actor_bits.append(f"id={du['actor_id']}")
    profile = ctx.get("actor_profile") or {}
    actor_str = ", ".join(actor_bits) if actor_bits else "(anonymous actor)"
    if profile:
        actor_str += "\n" + _compact(profile)

    private = ctx.get("private_state_before") or inp.get("private_goal") or {}
    history = ctx.get("known_history") or inp.get("dialogue_history") or inp.get("history") or []
    observation = ctx.get("current_observation") or inp.get("observation") or inp.get("current_observation") or {}
    avail = ctx.get("available_actions")
    if avail is None:
        avail = inp.get("available_actions")
    avail_str = UNKNOWN_ACTION_SPACE if avail is None else _compact(avail, limit=600)

    obs_str = _compact(observation.get("text") if isinstance(observation, dict) and observation.get("text") else observation)

    prompt = (
        f"TASK: {rec['task_type']}\n\n"
        f"ACTOR:\n{actor_str}\n\n"
        f"PRIVATE STATE BEFORE:\n{_compact(private)}\n\n"
        f"KNOWN HISTORY:\n{_render_history(history, max_history_events)}\n\n"
        f"CURRENT OBSERVATION:\n{obs_str}\n\n"
        f"AVAILABLE ACTIONS:\n{avail_str}\n\n"
        f"{TARGET_HEADER}"
    )
    completion = target_to_text(rec)
    return FormattedExample(
        prompt=prompt, completion=completion, text=prompt + completion,
        target_char_start=len(prompt), task_type=rec["task_type"], record_id=rec["record_id"],
    )
