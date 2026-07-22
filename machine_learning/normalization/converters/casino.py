"""CaSiNo — Campsite Negotiation Dialogues.

Real source (verified from kchawla123/casino parquet):
  row = {chat_logs: [{text, task_data:{data, issue2youget, issue2theyget}, id}],
         participant_info: {mturk_agent_1: {value2issue, value2reason, outcomes:
             {points_scored, satisfaction, opponent_likeness}, demographics, personality:
             {svo, big-five}}, mturk_agent_2: {...}},
         annotations: [[text, "strategy1,strategy2"], ...] | None}
Special utterance texts: Submit-Deal (task_data holds the proposed split), Accept-Deal,
Reject-Deal, Walk-Away.

Emits:
  PREDICT_NEXT_MESSAGE  — each natural-language turn; target text + strategy annotations
  PREDICT_NEXT_ACTION   — each deal action (submit/accept/reject/walk) with the split
  PREDICT_FINAL_OUTCOME — negotiation result (deal reached + points) from the messages

Honesty notes: CaSiNo records satisfaction/opponent_likeness only POST-negotiation (one
measurement), so PREDICT_PRIVATE_STATE_UPDATE is NOT produced (no pre-measurement to
predict a change from — inventing one would violate the no-fabrication rule). Workers have
no stable cross-dialogue identifier, so isolation is by conversation only.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator

from ...tasks import MISSING_TIMESTAMP
from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before, observation_at

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
_AGENTS = ("mturk_agent_1", "mturk_agent_2")


def _load_dialogues(raw_dir: Path) -> list[dict]:
    """Load CaSiNo dialogues from a parquet snapshot or a JSON fixture."""
    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "*.parquet"), recursive=True)
                if ".cache" not in f]
    if pq_files:
        import pyarrow.parquet as pq
        rows: list[dict] = []
        for f in sorted(pq_files):
            rows.extend(pq.read_table(f).to_pylist())
        return rows
    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"no CaSiNo parquet/json found under {raw_dir}")


class Converter(BaseConverter):
    DATASET_ID = "casino"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "casino"
    DOC = {
        "dataset_id": "casino",
        "original_fields": [
            {"name": "chat_logs", "meaning": "ordered turns; id=mturk_agent_1/2; deal actions in text + task_data"},
            {"name": "participant_info", "meaning": "per-agent private prefs (value2issue), personality, demographics, post-hoc outcomes"},
            {"name": "annotations", "meaning": "utterance-level negotiation-strategy labels (subset of dialogues)"},
        ],
        "canonical_mapping": [
            {"source_field": "chat_logs[k].text", "canonical_path": "payload.target.message_text | payload.input.dialogue_history"},
            {"source_field": "participant_info[agent].value2issue", "canonical_path": "context.private_state_before.preference_order"},
            {"source_field": "participant_info[agent].personality", "canonical_path": "context.actor_profile.personality"},
            {"source_field": "chat_logs[k].task_data.issue2youget", "canonical_path": "payload.target.action_content.proposer_gets (Submit-Deal)"},
            {"source_field": "participant_info[agent].outcomes.points_scored", "canonical_path": "payload.target.outcome.points (FINAL_OUTCOME)"},
            {"source_field": "annotations[k][1]", "canonical_path": "payload.target.strategy"},
        ],
        "tasks_produced": ["PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_ACTION", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": ["per-turn timestamps", "stable cross-dialogue worker id",
                               "pre-negotiation satisfaction (only post measured)"],
        "chronology_rules": "For a decision at turn k, only turns 0..k-1 are exposed. FINAL_OUTCOME cuts off before the deal-action turns.",
        "split_key": "conversation (episode_id)",
        "leakage_risks": ["the same worker may appear in multiple dialogues but has no linkable id, so cross-dialogue leakage cannot be detected/prevented — documented"],
        "known_limitations": ["strategy labels only on ~396/1030 dialogues",
                              "satisfaction/liking are single post-hoc self-reports"],
        "license_implications": "CC-BY-4.0: training + commercial use permitted with attribution.",
        "training_suitability": "train",
        "assumptions": ["deal actions are exactly {Submit-Deal, Accept-Deal, Reject-Deal, Walk-Away}"],
    }

    def _profile(self, pinfo: dict) -> dict:
        return {
            "personality": pinfo.get("personality", {}),
            "demographics": pinfo.get("demographics", {}),
        }

    def _private(self, pinfo: dict) -> dict:
        return {
            "preference_order": pinfo.get("value2issue", {}),
            "reasons": pinfo.get("value2reason", {}),
        }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        dialogues = _load_dialogues(raw_dir)
        for di, row in enumerate(dialogues):
            yield from self._one_dialogue(di, row)

    def _one_dialogue(self, di: int, row: dict) -> Iterator[dict]:
        episode_id = f"casino-dialogue-{di}"
        chat = row.get("chat_logs") or []
        pinfo = row.get("participant_info") or {}
        ann = row.get("annotations") or []
        strat_by_text = {a[0]: a[1] for a in ann if isinstance(a, (list, tuple)) and len(a) >= 2}

        # Build the full ordered event list for the episode.
        events: list[dict] = []
        for k, turn in enumerate(chat):
            speaker = turn.get("id")
            actor = self.pseudonym("participant", f"{episode_id}:{speaker}")
            text = turn.get("text", "")
            if text in DEAL_ACTIONS:
                td = turn.get("task_data", {}) or {}
                events.append(history_event(
                    k, actor, "action", action_type=text,
                    action_content={"proposer_gets": td.get("issue2youget", {}),
                                    "responder_gets": td.get("issue2theyget", {})},
                    meta={"role": speaker}))
            else:
                events.append(history_event(k, actor, "message", text=text, meta={"role": speaker}))

        participant_ids = [self.pseudonym("participant", f"{episode_id}:{a}")
                           for a in _AGENTS if a in pinfo]

        loc_base = {"files": ["data/train.parquet"], "indices": [di], "ids": [episode_id]}

        # ---- per-turn NEXT_MESSAGE / NEXT_ACTION -------------------------------------
        for k, turn in enumerate(chat):
            speaker = turn.get("id")
            if speaker not in pinfo:
                continue
            actor = self.pseudonym("participant", f"{episode_id}:{speaker}")
            text = turn.get("text", "")
            hist = history_before(events, k)
            obs = observation_at(events, k)
            ctx = {
                "actor_profile": self._profile(pinfo[speaker]),
                "private_state_before": self._private(pinfo[speaker]),
                "known_history": hist,
                "current_observation": obs,
                "world_state": {"issues": ["Firewood", "Water", "Food"]},
                "available_actions": None,
                "language": "en",
            }
            if text in DEAL_ACTIONS:
                td = turn.get("task_data", {}) or {}
                payload = {
                    "input": {"history": hist, "observation": obs,
                              "available_actions": sorted(DEAL_ACTIONS) + ["continue-negotiating"]},
                    "target": {"action_type": text, "acted": True,
                               "action_content": {"proposer_gets": td.get("issue2youget", {}),
                                                  "responder_gets": td.get("issue2theyget", {})}},
                }
                ctx["available_actions"] = sorted(DEAL_ACTIONS) + ["continue-negotiating"]
                yield self.make(
                    task_type="PREDICT_NEXT_ACTION", payload=payload, episode_id=episode_id,
                    sequence_index=k, cutoff_sequence_index=k, participant_ids=participant_ids,
                    actor_id=actor, actor_role="negotiator", context=ctx, raw_locator=loc_base,
                    transformation_steps=["load casino row", "order turns", f"cutoff before turn {k}",
                                          "extract deal action + proposed split"],
                    data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                                  "target_verified": True, "license_verified": True, "confidence": "high",
                                  "inferred_fields": ["sequence_index"]},
                )
            else:
                strat = strat_by_text.get(text)
                payload = {
                    "input": {"dialogue_history": hist,
                              "private_goal": self._private(pinfo[speaker]),
                              "current_observation": obs},
                    "target": {"message_text": text, "dialogue_act": None,
                               "strategy": strat.split(",") if strat else None},
                }
                dq_missing = ["timestamps"] + ([] if strat else ["strategy_annotation"])
                yield self.make(
                    task_type="PREDICT_NEXT_MESSAGE", payload=payload, episode_id=episode_id,
                    sequence_index=k, cutoff_sequence_index=k, participant_ids=participant_ids,
                    actor_id=actor, actor_role="negotiator", context=ctx, raw_locator=loc_base,
                    transformation_steps=["load casino row", "order turns", f"cutoff before turn {k}",
                                          "align strategy annotation by text"],
                    data_quality={"missing_fields": dq_missing, "chronology_verified": True,
                                  "target_verified": True, "license_verified": True,
                                  "confidence": "high", "inferred_fields": ["sequence_index"]},
                )

        # ---- FINAL_OUTCOME (cutoff before deal actions) ------------------------------
        first_action_k = next((k for k, t in enumerate(chat) if t.get("text") in DEAL_ACTIONS), len(chat))
        deal_reached = any(t.get("text") == "Accept-Deal" for t in chat)
        outcomes = {a: (pinfo.get(a, {}) or {}).get("outcomes", {}) for a in _AGENTS if a in pinfo}
        if outcomes:
            hist = history_before(events, first_action_k)
            payload = {
                "input": {"history": hist, "state": {"issues": ["Firewood", "Water", "Food"]}},
                "target": {"outcome": {"deal_reached": deal_reached,
                                       "points": {a: o.get("points_scored") for a, o in outcomes.items()},
                                       "satisfaction": {a: o.get("satisfaction") for a, o in outcomes.items()},
                                       "opponent_likeness": {a: o.get("opponent_likeness") for a, o in outcomes.items()}},
                           "outcome_type": "negotiation_result"},
            }
            yield self.make(
                task_type="PREDICT_FINAL_OUTCOME", payload=payload, episode_id=episode_id,
                sequence_index=first_action_k, cutoff_sequence_index=first_action_k,
                participant_ids=participant_ids, actor_role="negotiator",
                context={"known_history": hist, "world_state": {"issues": ["Firewood", "Water", "Food"]},
                         "language": "en", "available_actions": None},
                raw_locator=loc_base,
                transformation_steps=["load casino row", "cutoff before first deal action",
                                      "collect post-hoc outcomes as target"],
                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high",
                              "warnings": ["outcome (points/satisfaction) is post-negotiation ground truth"]},
            )
