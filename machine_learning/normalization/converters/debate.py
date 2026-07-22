"""DEBATE — real human opinion-dynamics debates (debatellm/DEBATE).

Real source (verified from the HF snapshot): one CSV per conversation under
``golden/<study>/<topic>/*.csv`` (study ∈ {depth, breadth}; topic = the claim). Columns:
chat_round_order, event_order, event_type, worker_id, sender_id, recipient_id, field,
text, sliderValue, agreement_level, validity. Event types:
  Initial Opinion  — a participant's PRE-discussion belief (sliderValue) + reasoning text
  tweet            — a public message (sender_id -> recipient_id) during a round
  Post Opinion     — the participant's POST-discussion belief (sliderValue)
  message_sent/received, exit_survey — auxiliary (not used for behaviour targets)

This is the ONLY source with a measured belief BEFORE and AFTER, so it is the substrate
for PREDICT_BELIEF_CHANGE. Emits:
  PREDICT_BELIEF_CHANGE  — per participant: belief_before (initial slider) + observed
                           tweets -> belief_after (post slider) + delta
  PREDICT_NEXT_MESSAGE   — each tweet given prior tweets + the sender's initial opinion
  PREDICT_NEXT_SPEAKER   — who tweets next
  PREDICT_TRAJECTORY_CONTINUATION — next K tweets from a prefix
  PREDICT_FINAL_OUTCOME  — aggregate opinion shift of the group (cut off before post opinions)

Uses the curated `golden/` split. Holds out whole conversations + topics + participants
(worker ids are conversation-scoped, so there is no cross-conversation identity to link).
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before, observation_at


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _conversations(raw_dir: Path, prefer_golden: bool = True) -> list[tuple[str, str, str, list[dict]]]:
    """Return (study, topic, conv_id, rows) for each conversation CSV."""
    root = raw_dir / "golden"
    if not root.exists() or prefer_golden is False:
        root = raw_dir  # fixtures may drop CSVs directly under the dataset dir
    files = [f for f in glob.glob(str(root / "**" / "*.csv"), recursive=True) if ".cache" not in f]
    out = []
    for f in sorted(files):
        p = Path(f)
        parts = p.relative_to(root).parts if str(p).startswith(str(root)) else p.parts
        study = parts[0] if len(parts) >= 3 else "unknown"
        topic = parts[-2] if len(parts) >= 2 else "unknown"
        conv_id = p.stem[-16:]  # the trailing ULID uniquely ids the conversation
        try:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
        out.append((study, topic, conv_id, rows))
    return out


class Converter(BaseConverter):
    DATASET_ID = "debate"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "debate"
    DOC = {
        "dataset_id": "debate",
        "original_fields": [
            {"name": "event_type", "meaning": "Initial Opinion | tweet | Post Opinion | message_* | exit_survey"},
            {"name": "worker_id", "meaning": "participant (conversation-scoped id)"},
            {"name": "sender_id/recipient_id", "meaning": "tweet direction"},
            {"name": "text", "meaning": "opinion reasoning or tweet content"},
            {"name": "sliderValue", "meaning": "belief rating (pre on Initial Opinion, post on Post Opinion)"},
            {"name": "chat_round_order/event_order", "meaning": "ordering"},
        ],
        "canonical_mapping": [
            {"source_field": "Initial Opinion.sliderValue", "canonical_path": "payload.input.belief_before.value"},
            {"source_field": "Post Opinion.sliderValue", "canonical_path": "payload.target.belief_after.value"},
            {"source_field": "tweet.text", "canonical_path": "payload.target.message_text | observed_messages"},
            {"source_field": "sender_id", "canonical_path": "payload.target.speaker_id"},
            {"source_field": "topic (dir)", "canonical_path": "episode.topic_id"},
        ],
        "tasks_produced": ["PREDICT_BELIEF_CHANGE", "PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_SPEAKER",
                           "PREDICT_TRAJECTORY_CONTINUATION", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": ["per-tweet timestamps (only ordering)", "cross-conversation participant identity"],
        "chronology_rules": "BELIEF_CHANGE inputs = initial opinion + observed tweets; the POST slider is target-only. Tweet tasks cut off before the target tweet. FINAL_OUTCOME cuts off before all Post Opinions.",
        "split_key": "conversation (episode) + topic + participant",
        "leakage_risks": ["Post Opinion sliderValue is the belief-change LABEL and must never appear in any input",
                          "the target tweet must not appear in observed history"],
        "known_limitations": ["uses curated golden/ split", "belief is a 1-7 self-reported slider"],
        "license_implications": "Research-Only Non-Commercial v1.0: non-commercial research training only; not redistributable.",
        "training_suitability": "train",
        "assumptions": ["golden/<study>/<topic>/*.csv layout; trailing ULID identifies the conversation"],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        convs = _conversations(raw_dir)
        if not convs:
            raise FileNotFoundError(f"no DEBATE conversation CSVs under {raw_dir}")
        for study, topic, conv_id, rows in convs:
            yield from self._one(study, topic, conv_id, rows)

    def _one(self, study, topic, conv_id, rows) -> Iterator[dict]:
        episode_id = f"debate-{conv_id}"
        topic_id = self.pseudonym("topic", topic)
        study_id = self.pseudonym("study", study)
        initial = {r["worker_id"]: r for r in rows if r["event_type"] == "Initial Opinion" and r.get("worker_id")}
        post = {r["worker_id"]: r for r in rows if r["event_type"] == "Post Opinion" and r.get("worker_id")}
        tweets = [r for r in rows if r["event_type"] == "tweet" and (r.get("text") or "").strip()]
        tweets.sort(key=lambda r: _num(r.get("event_order")) or 0)
        participant_ids = [self.pseudonym("participant", w) for w in sorted(initial)]
        loc = {"files": [f"golden/{study}/{topic}/{conv_id}.csv"], "indices": [], "ids": [conv_id]}

        events = []
        for k, tw in enumerate(tweets):
            actor = self.pseudonym("participant", tw.get("sender_id") or tw.get("worker_id"))
            events.append(history_event(k, actor, "message", text=tw.get("text", ""),
                                        meta={"round": tw.get("chat_round_order")}))

        # ---- BELIEF_CHANGE ----
        for w in sorted(set(initial) & set(post)):
            b0, b1 = _num(initial[w].get("sliderValue")), _num(post[w].get("sliderValue"))
            if b0 is None or b1 is None:
                continue
            actor = self.pseudonym("participant", w)
            payload = {
                "input": {"belief_before": {"value": b0, "text": initial[w].get("text", "")},
                          "observed_messages": [{"actor_id": e["actor_id"], "text": e["text"]} for e in events]},
                "target": {"belief_after": {"value": b1}, "belief_delta": {"value": round(b1 - b0, 3)}},
            }
            yield self.make(
                task_type="PREDICT_BELIEF_CHANGE", payload=payload, episode_id=episode_id,
                topic_id=topic_id, group_id=episode_id, participant_ids=participant_ids,
                actor_id=actor, actor_role="participant", experiment_id=study_id,
                context={"private_state_before": {"initial_opinion": b0}, "known_history": events,
                         "world_state": {"claim": topic}, "language": "en", "available_actions": None},
                raw_locator=loc,
                transformation_steps=["load conversation csv", "link initial+post opinion by worker",
                                      "observed tweets = input; post slider = target"],
                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high"})

        # ---- tweet tasks ----
        for k, tw in enumerate(tweets):
            sender = tw.get("sender_id") or tw.get("worker_id")
            actor = self.pseudonym("participant", sender)
            hist = history_before(events, k)
            obs = observation_at(events, k)
            init_op = initial.get(sender, {})
            hist_msgs = [{"actor_id": e["actor_id"], "text": e["text"]} for e in hist]
            yield self.make(
                task_type="PREDICT_NEXT_MESSAGE",
                payload={"input": {"dialogue_history": hist_msgs,
                                   "private_goal": {"initial_opinion_text": init_op.get("text", ""),
                                                    "initial_opinion_value": _num(init_op.get("sliderValue"))},
                                   "current_observation": obs},
                         "target": {"message_text": tw.get("text", ""), "dialogue_act": None, "strategy": None}},
                episode_id=episode_id, sequence_index=k, cutoff_sequence_index=k, topic_id=topic_id,
                group_id=episode_id, participant_ids=participant_ids, actor_id=actor, actor_role="participant",
                experiment_id=study_id,
                context={"private_state_before": {"initial_opinion": _num(init_op.get("sliderValue"))},
                         "known_history": hist, "current_observation": obs, "world_state": {"claim": topic},
                         "language": "en", "available_actions": None},
                raw_locator=loc, transformation_steps=["order tweets", f"cutoff before tweet {k}"],
                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high"})
            if k > 0:
                yield self.make(
                    task_type="PREDICT_NEXT_SPEAKER",
                    payload={"input": {"dialogue_history": hist_msgs, "participants": participant_ids},
                             "target": {"speaker_id": actor}},
                    episode_id=episode_id, sequence_index=k, cutoff_sequence_index=k, topic_id=topic_id,
                    group_id=episode_id, participant_ids=participant_ids, actor_role="participant",
                    experiment_id=study_id,
                    context={"known_history": hist, "world_state": {"claim": topic}, "language": "en",
                             "available_actions": participant_ids},
                    raw_locator=loc, transformation_steps=[f"cutoff before tweet {k}", "target = sender"],
                    data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                                  "target_verified": True, "license_verified": True, "confidence": "high"})

        # ---- TRAJECTORY_CONTINUATION ----
        if len(tweets) >= 4:
            cut = len(tweets) // 2
            hist = history_before(events, cut)
            cont = [{"actor_id": e["actor_id"], "text": e["text"]} for e in events[cut:cut + 3]]
            yield self.make(
                task_type="PREDICT_TRAJECTORY_CONTINUATION",
                payload={"input": {"history": [{"actor_id": e["actor_id"], "text": e["text"]} for e in hist],
                                   "horizon": len(cont)}, "target": {"continuation": cont}},
                episode_id=episode_id, sequence_index=cut, cutoff_sequence_index=cut, topic_id=topic_id,
                group_id=episode_id, participant_ids=participant_ids, actor_role="participant",
                experiment_id=study_id,
                context={"known_history": hist, "world_state": {"claim": topic}, "language": "en",
                         "available_actions": None},
                raw_locator=loc, transformation_steps=[f"cutoff at tweet {cut}", "predict next 3"],
                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high"})

        # ---- FINAL_OUTCOME (aggregate opinion shift) ----
        both = sorted(set(initial) & set(post))
        inits = [_num(initial[w].get("sliderValue")) for w in both if _num(initial[w].get("sliderValue")) is not None]
        posts = [_num(post[w].get("sliderValue")) for w in both if _num(post[w].get("sliderValue")) is not None]
        deltas = [(_num(post[w].get("sliderValue")) or 0) - (_num(initial[w].get("sliderValue")) or 0) for w in both]
        if both and inits and posts:
            payload = {"input": {"history": [{"actor_id": e["actor_id"], "text": e["text"]} for e in events],
                                 "state": {"claim": topic, "n_participants": len(both)}},
                       "target": {"outcome": {"mean_initial": round(sum(inits) / len(inits), 3),
                                              "mean_post": round(sum(posts) / len(posts), 3),
                                              "mean_shift": round(sum(deltas) / len(deltas), 3),
                                              "n_shifted": sum(1 for d in deltas if abs(d) > 1e-9)},
                                  "outcome_type": "group_opinion_shift"}}
            yield self.make(
                task_type="PREDICT_FINAL_OUTCOME", payload=payload, episode_id=episode_id, topic_id=topic_id,
                group_id=episode_id, participant_ids=participant_ids, actor_role="participant",
                experiment_id=study_id,
                context={"known_history": events, "world_state": {"claim": topic}, "language": "en",
                         "available_actions": None},
                raw_locator=loc, transformation_steps=["cutoff before post opinions", "aggregate opinion shift"],
                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high"})
