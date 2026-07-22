"""Deal or No Deal — end-to-end-negotiator human negotiation dialogues.

Real source (verified from data/negotiate/*.txt): one dialogue per line, tag-delimited::

    <input> c0 v0 c1 v1 c2 v2 </input>
    <dialogue> YOU: ... <eos> THEM: ... <eos> ... <selection> </dialogue>
    <output> item0=a item1=b item2=c item0=d item1=e item2=f </output>
    <partner_input> c0 v0 c1 v1 c2 v2 </partner_input>

``<input>`` = 3 item counts + the PERSPECTIVE agent's PRIVATE values (v). ``<output>`` =
the agreed allocation (first 3 = YOU get, next 3 = THEM get) or <no_agreement>/<disagree>.
Each scenario appears twice (once per perspective).

Emits (from the YOU perspective, whose private values we know):
  PREDICT_NEXT_MESSAGE  — each YOU utterance
  PREDICT_NEXT_ACTION   — the terminal selection / 'deal' action
  PREDICT_FINAL_OUTCOME — the agreed allocation + agreement flag
"""
from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before, observation_at

_ITEMS = ["book", "hat", "ball"]
_TAG = re.compile(r"<(input|dialogue|output|partner_input)>(.*?)</\1>", re.DOTALL)


def _parse_line(line: str) -> dict | None:
    parts = {m.group(1): m.group(2).strip() for m in _TAG.finditer(line)}
    if "input" not in parts or "dialogue" not in parts:
        return None
    return parts


def _counts_values(seg: str) -> dict:
    nums = [int(x) for x in seg.split()]
    if len(nums) < 6:
        return {"counts": {}, "values": {}}
    return {"counts": {_ITEMS[i]: nums[2 * i] for i in range(3)},
            "values": {_ITEMS[i]: nums[2 * i + 1] for i in range(3)}}


def _turns(dialogue: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for chunk in dialogue.split("<eos>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("YOU:"):
            out.append(("YOU", chunk[4:].strip()))
        elif chunk.startswith("THEM:"):
            out.append(("THEM", chunk[5:].strip()))
    return out


def _parse_output(seg: str) -> dict:
    if "<no_agreement>" in seg or "<disagree>" in seg:
        return {"deal_reached": False, "you_get": {}, "them_get": {}}
    nums = re.findall(r"item\d+=(\d+)", seg)
    if len(nums) >= 6:
        n = [int(x) for x in nums[:6]]
        return {"deal_reached": True,
                "you_get": {_ITEMS[i]: n[i] for i in range(3)},
                "them_get": {_ITEMS[i]: n[3 + i] for i in range(3)}}
    return {"deal_reached": False, "you_get": {}, "them_get": {}}


class Converter(BaseConverter):
    DATASET_ID = "dealornodeal"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "dealornodeal"
    DOC = {
        "dataset_id": "dealornodeal",
        "original_fields": [
            {"name": "<input>", "meaning": "item counts + the perspective agent's private values"},
            {"name": "<dialogue>", "meaning": "YOU/THEM turns ending in <selection>"},
            {"name": "<output>", "meaning": "agreed allocation or <no_agreement>"},
            {"name": "<partner_input>", "meaning": "partner counts+values"},
        ],
        "canonical_mapping": [
            {"source_field": "<input> values", "canonical_path": "context.private_state_before.item_values"},
            {"source_field": "<dialogue> YOU turns", "canonical_path": "payload.target.message_text"},
            {"source_field": "<output>", "canonical_path": "payload.target.outcome (FINAL_OUTCOME)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_ACTION", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": ["timestamps", "stable worker id", "partner's private values (from YOU's view)"],
        "chronology_rules": "Predict YOU turns from prior turns only; FINAL_OUTCOME cuts off before <selection>.",
        "split_key": "session (scenario+perspective)",
        "leakage_risks": ["each scenario appears twice (two perspectives) — both copies MUST land in the same split (keyed by scenario id)"],
        "known_limitations": ["item names book/hat/ball assumed (standard DND ordering)"],
        "license_implications": "CC-BY-NC: NON-COMMERCIAL training only.",
        "training_suitability": "train",
        "assumptions": ["3 items, 6-number input encodes count,value pairs"],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        files = sorted(glob.glob(str(raw_dir / "**" / "negotiate" / "*.txt"), recursive=True))
        files = [f for f in files if Path(f).name in ("train.txt", "val.txt", "test.txt")]
        if not files:
            files = sorted(glob.glob(str(raw_dir / "**" / "*.txt"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no Deal-or-No-Deal txt found under {raw_dir}")
        for fpath in files:
            split_name = Path(fpath).stem
            with open(fpath, encoding="utf-8") as fh:
                for li, line in enumerate(fh):
                    parsed = _parse_line(line)
                    if not parsed:
                        continue
                    yield from self._one(parsed, split_name, li, Path(fpath).name)

    def _one(self, parsed: dict, split_name: str, li: int, fname: str) -> Iterator[dict]:
        me = _counts_values(parsed["input"])
        # scenario id keyed on the (sorted) item counts so both perspectives co-locate.
        counts_key = "-".join(str(me["counts"].get(it, 0)) for it in _ITEMS)
        episode_id = f"dealornodeal-{split_name}-{li}"
        scenario_id = f"dnd-scn-{counts_key}-{parsed.get('partner_input','').replace(' ','')}"
        turns = _turns(parsed["dialogue"])
        events = []
        for k, (who, text) in enumerate(turns):
            actor = self.pseudonym("participant", f"{episode_id}:{who}")
            events.append(history_event(k, actor, "message", text=text, meta={"role": who}))
        me_actor = self.pseudonym("participant", f"{episode_id}:YOU")
        participants = [me_actor, self.pseudonym("participant", f"{episode_id}:THEM")]
        loc = {"files": [f"data/negotiate/{fname}"], "indices": [li], "ids": [episode_id]}
        private = {"item_counts": me["counts"], "item_values": me["values"]}

        for k, (who, text) in enumerate(turns):
            if who != "YOU":
                continue
            hist = history_before(events, k)
            obs = observation_at(events, k)
            is_selection = text.strip() in ("<selection>", "deal", "<no_agreement>")
            ctx = {"private_state_before": private, "known_history": hist,
                   "current_observation": obs, "world_state": {"item_counts": me["counts"]},
                   "available_actions": None, "language": "en"}
            if is_selection:
                payload = {"input": {"history": hist, "observation": obs,
                                     "available_actions": ["propose", "accept", "<selection>"]},
                           "target": {"action_type": "selection" if "<selection>" in text else text,
                                      "acted": True, "action_content": {"text": text}}}
                yield self.make(task_type="PREDICT_NEXT_ACTION", payload=payload, episode_id=episode_id,
                                sequence_index=k, cutoff_sequence_index=k, group_id=scenario_id,
                                participant_ids=participants, actor_id=me_actor, actor_role="negotiator",
                                context=ctx, raw_locator=loc,
                                transformation_steps=["parse tags", "split turns", f"cutoff before turn {k}"],
                                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                                              "target_verified": True, "license_verified": True, "confidence": "high"})
            else:
                payload = {"input": {"dialogue_history": hist, "private_goal": private,
                                     "current_observation": obs},
                           "target": {"message_text": text, "dialogue_act": None, "strategy": None}}
                yield self.make(task_type="PREDICT_NEXT_MESSAGE", payload=payload, episode_id=episode_id,
                                sequence_index=k, cutoff_sequence_index=k, group_id=scenario_id,
                                participant_ids=participants, actor_id=me_actor, actor_role="negotiator",
                                context=ctx, raw_locator=loc,
                                transformation_steps=["parse tags", "split turns", f"cutoff before turn {k}"],
                                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                                              "target_verified": True, "license_verified": True, "confidence": "high"})

        # FINAL_OUTCOME
        if "output" in parsed:
            outcome = _parse_output(parsed["output"])
            sel_k = next((k for k, (w, t) in enumerate(turns) if "<selection>" in t or t.strip() == "deal"), len(turns))
            hist = history_before(events, sel_k)
            payload = {"input": {"history": hist, "state": {"item_counts": me["counts"], "my_values": me["values"]}},
                       "target": {"outcome": outcome, "outcome_type": "allocation"}}
            yield self.make(task_type="PREDICT_FINAL_OUTCOME", payload=payload, episode_id=episode_id,
                            sequence_index=sel_k, cutoff_sequence_index=sel_k, group_id=scenario_id,
                            participant_ids=participants, actor_role="negotiator",
                            context={"known_history": hist, "private_state_before": private,
                                     "world_state": {"item_counts": me["counts"]}, "language": "en",
                                     "available_actions": None},
                            raw_locator=loc,
                            transformation_steps=["parse <output>", "cutoff before selection"],
                            data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                                          "target_verified": True, "license_verified": True, "confidence": "high"})
