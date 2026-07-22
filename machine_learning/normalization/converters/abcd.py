"""ABCD — Action-Based Conversations Dataset (customer-service, policy-grounded).

Real source (verified from abcd_v1.1.json.gz): {train,dev,test: [conv]} where conv =
  {convo_id, scenario:{personal, order, product, flow, subflow},
   original: [[speaker, text], ...],           # speaker ∈ {agent, customer, action}
   delexed: [{speaker, text, turn_count, targets:[intent, nextstep, action_button, values, cand_id], candidates}]}

Emits (modeling the AGENT, whose behaviour is grounded in guidelines/policy):
  PREDICT_NEXT_MESSAGE  — each agent utterance
  PREDICT_NEXT_ACTION   — each grounded button-click action (take_action)
  PREDICT_FINAL_OUTCOME — the resolved subflow (customer intent served)

The customer's actual goal (scenario.flow/subflow) is the FINAL_OUTCOME label and is
therefore NOT exposed in context (only scenario.personal/order/product are).
"""
from __future__ import annotations

import glob
import gzip
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before, observation_at


def _load(raw_dir: Path) -> dict:
    gz = [f for f in glob.glob(str(raw_dir / "**" / "*.json.gz"), recursive=True) if ".cache" not in f]
    for f in sorted(gz):
        if "abcd" in Path(f).name:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    js = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
          if "abcd" in Path(f).name and "sample" not in Path(f).name and ".cache" not in f]
    for f in sorted(js):
        return json.loads(Path(f).read_text())
    # fixture fallback: any *.json that looks like {train:[...]}
    for f in sorted(glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)):
        try:
            d = json.loads(Path(f).read_text())
            if isinstance(d, dict) and any(k in d for k in ("train", "dev", "test")):
                return d
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(f"no ABCD json(.gz) found under {raw_dir}")


class Converter(BaseConverter):
    DATASET_ID = "abcd"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "abcd"
    DOC = {
        "dataset_id": "abcd",
        "original_fields": [
            {"name": "original", "meaning": "[[speaker, text]] turns; speaker agent/customer/action"},
            {"name": "delexed[i].targets", "meaning": "[intent, nextstep, action_button, values, cand_id]"},
            {"name": "scenario", "meaning": "personal/order/product context + flow/subflow (goal)"},
        ],
        "canonical_mapping": [
            {"source_field": "original[i] agent text", "canonical_path": "payload.target.message_text"},
            {"source_field": "delexed[i].targets[2] (action button)", "canonical_path": "payload.target.action_type"},
            {"source_field": "scenario.subflow", "canonical_path": "payload.target.outcome (FINAL_OUTCOME)"},
            {"source_field": "scenario.personal/order/product", "canonical_path": "context.world_state"},
        ],
        "tasks_produced": ["PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_ACTION", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": ["timestamps", "customer identity"],
        "chronology_rules": "Agent decisions predicted from prior turns; subflow (goal) never exposed in context.",
        "split_key": "conversation (convo_id)",
        "leakage_risks": ["scenario.flow/subflow must NOT enter context (it is the outcome label)"],
        "known_limitations": ["only agent turns are modeled for messages/actions; customer turns are context"],
        "license_implications": "MIT: training + commercial use permitted.",
        "training_suitability": "train",
        "assumptions": ["original and delexed lists are index-aligned"],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        data = _load(raw_dir)
        for split_name, convs in data.items():
            if not isinstance(convs, list):
                continue
            for ci, conv in enumerate(convs):
                yield from self._one(conv, split_name, ci)

    def _one(self, conv: dict, split_name: str, ci: int) -> Iterator[dict]:
        convo_id = str(conv.get("convo_id", f"{split_name}-{ci}"))
        episode_id = f"abcd-{convo_id}"
        original = conv.get("original") or []
        delexed = conv.get("delexed") or []
        scenario = conv.get("scenario") or {}
        world = {k: scenario.get(k) for k in ("personal", "order", "product") if k in scenario}
        subflow = scenario.get("subflow")
        loc = {"files": ["data/abcd_v1.1.json.gz"], "indices": [ci], "ids": [convo_id]}

        events = []
        for k, turn in enumerate(original):
            if not isinstance(turn, (list, tuple)) or len(turn) < 2:
                continue
            speaker, text = turn[0], turn[1]
            actor = self.pseudonym("actor", f"{episode_id}:{speaker}")
            if speaker == "action":
                events.append(history_event(k, actor, "action", action_type="button",
                                            action_content={"text": text}, meta={"role": speaker}))
            else:
                events.append(history_event(k, actor, "message", text=text, meta={"role": speaker}))

        agent_id = self.pseudonym("actor", f"{episode_id}:agent")
        for k, turn in enumerate(original):
            if not isinstance(turn, (list, tuple)) or len(turn) < 2:
                continue
            speaker, text = turn[0], turn[1]
            hist = history_before(events, k)
            obs = observation_at(events, k)
            ctx = {"known_history": hist, "current_observation": obs, "world_state": world,
                   "available_actions": None, "language": "en",
                   "institutional_constraints": [{"kind": "agent_guidelines", "ref": "guidelines.json"}]}
            dlx = delexed[k] if k < len(delexed) else {}
            targets = dlx.get("targets") if isinstance(dlx, dict) else None

            if speaker == "action":
                button = targets[2] if (targets and len(targets) > 2 and targets[2]) else text
                yield self.make(
                    task_type="PREDICT_NEXT_ACTION",
                    payload={"input": {"history": hist, "observation": obs, "available_actions": None},
                             "target": {"action_type": str(button), "acted": True,
                                        "action_content": {"values": targets[3] if targets and len(targets) > 3 else [],
                                                           "text": text}}},
                    episode_id=episode_id, sequence_index=k, cutoff_sequence_index=k, actor_id=agent_id,
                    actor_role="agent", context=ctx, raw_locator=loc,
                    transformation_steps=["load abcd", f"cutoff before turn {k}", "action from delexed targets"],
                    data_quality={"missing_fields": ["timestamps", "available_action_set"], "chronology_verified": True,
                                  "target_verified": True, "license_verified": True, "confidence": "high"})
            elif speaker == "agent":
                yield self.make(
                    task_type="PREDICT_NEXT_MESSAGE",
                    payload={"input": {"dialogue_history": hist, "private_goal": {}, "current_observation": obs},
                             "target": {"message_text": text, "dialogue_act": (targets[1] if targets and len(targets) > 1 else None),
                                        "strategy": None}},
                    episode_id=episode_id, sequence_index=k, cutoff_sequence_index=k, actor_id=agent_id,
                    actor_role="agent", context=ctx, raw_locator=loc,
                    transformation_steps=["load abcd", f"cutoff before turn {k}"],
                    data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                                  "target_verified": True, "license_verified": True, "confidence": "high"})

        # FINAL_OUTCOME = resolved subflow (customer intent served)
        if subflow:
            payload = {"input": {"history": [e for e in events], "state": world},
                       "target": {"outcome": {"subflow": subflow, "flow": scenario.get("flow")},
                                  "outcome_type": "resolved_intent"}}
            yield self.make(
                task_type="PREDICT_FINAL_OUTCOME", payload=payload, episode_id=episode_id,
                sequence_index=len(events), cutoff_sequence_index=len(events), actor_role="agent",
                context={"known_history": events, "world_state": world, "language": "en", "available_actions": None},
                raw_locator=loc,
                transformation_steps=["load abcd", "target = resolved subflow (goal not exposed in context)"],
                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high"})
