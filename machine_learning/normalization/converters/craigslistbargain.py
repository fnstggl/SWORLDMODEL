"""CraigslistBargain — buyer/seller price negotiations (He et al., ACL 2018; COCOA format).

Real source (verified from craigslist_train.json / craigslist_val.json — a list of parsed
COCOA dialogues). Each example:
  {uuid, scenario:{category, kbs:[per-agent {personal:{Role: buyer|seller, Target: price,
      Bottomline}, item:{Category, Title, Description, Price (list price), Images}}]},
   agents:{"0": human|bot, "1": ...},
   events:[{agent:0|1, action: message|offer|accept|reject|quit, data: text | {price,sides} |
      null, time, start_time, metadata:{intent, price}}],
   outcome:{reward, offer:{price, sides}}}.
kbs[a] is agent a's private knowledge base; both agents see the same listing (item), but
each has a private Role + Target price.

Emits:
  PREDICT_NEXT_MESSAGE           — each message event; target text + intent label
  PREDICT_NEXT_ACTION            — each offer/accept/reject/quit event, with the price
  PREDICT_TRAJECTORY_CONTINUATION — next K events given a prefix
  PREDICT_FINAL_OUTCOME          — agreement + final price (from the accept event / outcome)

private_state_before for an agent = its Role + Target price (both known before the dialogue).
This dataset is CROSS_DATASET_EVAL_ONLY: the dataset itself has NO stated license (the HF
card says "More Information Needed"; only the cocoa *code* is MIT). It is normalized
correctly but marked license_class=unknown_unstated / training_suitability=eval_only, and
data_quality.license_verified=False on every record.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before, observation_at

_TRAJ_HORIZON = 5
_ACTIONS = ("message", "offer", "accept", "reject", "quit")
_TERMINAL = {"offer", "accept", "reject", "quit"}


def _load_examples(raw_dir: Path) -> list[dict]:
    """Load COCOA examples from streamed parquet, the craigslist_*.json files, or a fixture."""
    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "stream_shard_*.parquet"), recursive=True)
                if ".cache" not in f]
    if pq_files:
        import pyarrow.parquet as pq
        rows: list[dict] = []
        for f in sorted(pq_files):
            rows.extend(pq.read_table(f).to_pylist())
        return rows

    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    rows = []
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            rows.extend(data)
    if not rows:
        raise FileNotFoundError(f"no CraigslistBargain json/parquet found under {raw_dir}")
    return rows


def _price(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Converter(BaseConverter):
    DATASET_ID = "craigslistbargain"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "craigslistbargain"
    DOC = {
        "dataset_id": "craigslistbargain",
        "original_fields": [
            {"name": "scenario.kbs[a].personal.Role", "meaning": "agent a's role: buyer or seller"},
            {"name": "scenario.kbs[a].personal.Target", "meaning": "agent a's private target price"},
            {"name": "scenario.kbs[a].personal.Bottomline", "meaning": "agent a's walk-away price (usually null)"},
            {"name": "scenario.kbs[a].item", "meaning": "shared listing: Category, Title, Description, Price (list price)"},
            {"name": "events[].action", "meaning": "message | offer | accept | reject | quit"},
            {"name": "events[].data", "meaning": "message text, or {price,sides} for an offer, or null"},
            {"name": "events[].metadata.intent", "meaning": "dialogue-act / intent label of the event"},
            {"name": "outcome", "meaning": "{reward, offer:{price,sides}} — final agreed offer"},
            {"name": "agents", "meaning": "per-agent-index actor type (human/bot)"},
        ],
        "canonical_mapping": [
            {"source_field": "events[].data (message)", "canonical_path": "payload.target.message_text | context.known_history[].text"},
            {"source_field": "events[].metadata.intent", "canonical_path": "payload.target.dialogue_act (NEXT_MESSAGE)"},
            {"source_field": "events[].action (offer/accept/reject/quit)", "canonical_path": "payload.target.action_type (NEXT_ACTION)"},
            {"source_field": "events[].data.price / metadata.price", "canonical_path": "payload.target.action_content.price"},
            {"source_field": "kbs[a].personal.{Role,Target}", "canonical_path": "context.private_state_before / payload.input.private_goal"},
            {"source_field": "kbs[a].item", "canonical_path": "context.world_state (public listing)"},
            {"source_field": "outcome.offer.price / accept event", "canonical_path": "payload.target.outcome (FINAL_OUTCOME)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_ACTION",
                           "PREDICT_TRAJECTORY_CONTINUATION", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": [
            "stable cross-conversation agent id (agents are only indexed 0/1 per dialogue)",
            "Bottomline is null in the source for essentially all agents",
        ],
        "chronology_rules": "For an event at index k only events 0..k-1 are exposed. FINAL_OUTCOME cuts off before the first terminal action (offer/accept/reject/quit) so the literal final offer object is never in the input; the agreement + price are the target.",
        "split_key": "conversation (episode_id = example uuid)",
        "leakage_risks": [
            "an agent's own Target price is a legitimate private input; the OTHER agent's Target is never exposed to it (kept out of shared context)",
        ],
        "known_limitations": [
            "no stated dataset license — cross-dataset EVAL ONLY (see license_implications)",
            "reward is the source's scalar payoff; agreement is derived from the presence of an accept event",
        ],
        "license_implications": "Dataset has NO explicit license (HF card: 'More Information Needed'); only the cocoa code is MIT. Treated as unknown_unstated: NOT used for training — cross-dataset evaluation only. data_quality.license_verified is False on every record.",
        "training_suitability": "eval_only",
        "assumptions": ["kbs[a] corresponds to events with agent==a",
                        "an 'accept' event denotes a reached agreement"],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        for ei, ex in enumerate(_load_examples(raw_dir)):
            yield from self._one(ei, ex)

    def _dq(self, extra_missing=None, warnings=None) -> dict:
        return {"missing_fields": list(extra_missing or []),
                "chronology_verified": True, "target_verified": True,
                "license_verified": False, "confidence": "high",
                "warnings": (warnings or []) + ["dataset license unstated — eval-only"]}

    def _one(self, ei: int, ex: dict) -> Iterator[dict]:
        uuid = ex.get("uuid") or f"idx{ei}"
        episode_id = f"craigslistbargain-{uuid}"
        scenario = ex.get("scenario") or {}
        kbs = scenario.get("kbs") or []
        agents = ex.get("agents") or {}
        events_raw = ex.get("events") or []
        outcome = ex.get("outcome") or {}

        def kb(a):
            return kbs[a] if isinstance(kbs, list) and 0 <= a < len(kbs) else {}

        def personal(a):
            p = (kb(a) or {}).get("personal") or {}
            return {"role": p.get("Role"), "target_price": _price(p.get("Target")),
                    "bottomline": _price(p.get("Bottomline"))}

        def role_of(a):
            return (personal(a).get("role")) or f"agent{a}"

        item = (kb(0) or {}).get("item") or (kb(1) or {}).get("item") or {}
        world = {
            "category": scenario.get("category") or item.get("Category"),
            "title": item.get("Title"),
            "description": item.get("Description"),
            "list_price": _price(item.get("Price")),
        }

        def actor(a):
            return self.pseudonym("participant", f"{episode_id}:{a}")

        participant_ids = [actor(0), actor(1)]

        # ordered canonical event list
        events: list[dict] = []
        for k, ev in enumerate(events_raw):
            a = ev.get("agent")
            action = ev.get("action")
            meta = ev.get("metadata") or {}
            common_meta = {"intent": meta.get("intent"), "agent": a, "role": role_of(a)}
            if action == "message":
                events.append(history_event(k, actor(a), "message",
                                            text=ev.get("data") if isinstance(ev.get("data"), str) else "",
                                            t=ev.get("time"), meta=common_meta))
            else:
                data = ev.get("data")
                pr = _price(data.get("price")) if isinstance(data, dict) else _price(meta.get("price"))
                sides = data.get("sides") if isinstance(data, dict) else None
                events.append(history_event(k, actor(a), "action", action_type=action or "unknown",
                                            action_content={"price": pr, "sides": sides},
                                            t=ev.get("time"), meta=common_meta))
        n = len(events)
        loc = {"files": ["craigslist_train.json", "craigslist_val.json"],
               "indices": [ei], "ids": [episode_id]}

        # ---- per-event NEXT_MESSAGE / NEXT_ACTION --------------------------------------
        for k, ev in enumerate(events_raw):
            a = ev.get("agent")
            action = ev.get("action")
            meta = ev.get("metadata") or {}
            hist = history_before(events, k)
            obs = observation_at(events, k)
            priv = personal(a)
            ctx = {
                "actor_profile": {"agent_type": agents.get(str(a))},
                "private_state_before": priv,
                "known_history": hist,
                "current_observation": obs,
                "world_state": world,
                "available_actions": list(_ACTIONS),
                "language": "en",
            }
            if action == "message":
                text = ev.get("data") if isinstance(ev.get("data"), str) else ""
                payload = {
                    "input": {"dialogue_history": hist,
                              "private_goal": {"role": priv["role"], "target_price": priv["target_price"]},
                              "current_observation": obs},
                    "target": {"message_text": text, "dialogue_act": meta.get("intent"), "strategy": None},
                }
                miss = [] if priv["bottomline"] is not None else ["bottomline"]
                yield self.make(
                    task_type="PREDICT_NEXT_MESSAGE", payload=payload, episode_id=episode_id,
                    sequence_index=k, cutoff_sequence_index=k, participant_ids=participant_ids,
                    actor_id=actor(a), actor_role=role_of(a), context=ctx, raw_locator=loc,
                    transformation_steps=["load COCOA example", "order events",
                                          f"cutoff before event {k}"],
                    data_quality=self._dq(extra_missing=miss))
            else:
                data = ev.get("data")
                pr = _price(data.get("price")) if isinstance(data, dict) else _price(meta.get("price"))
                sides = data.get("sides") if isinstance(data, dict) else None
                payload = {
                    "input": {"history": hist, "observation": obs, "available_actions": list(_ACTIONS)},
                    "target": {"action_type": action or "unknown", "acted": True,
                               "action_content": {"price": pr, "sides": sides}},
                }
                miss = [] if pr is not None or action != "offer" else ["offer_price"]
                yield self.make(
                    task_type="PREDICT_NEXT_ACTION", payload=payload, episode_id=episode_id,
                    sequence_index=k, cutoff_sequence_index=k, participant_ids=participant_ids,
                    actor_id=actor(a), actor_role=role_of(a), context=ctx, raw_locator=loc,
                    transformation_steps=["load COCOA example", "order events",
                                          f"cutoff before event {k}", "extract action + price"],
                    data_quality=self._dq(extra_missing=miss))

        # ---- PREDICT_TRAJECTORY_CONTINUATION --------------------------------------------
        for k in range(1, n):
            cont = events[k:k + _TRAJ_HORIZON]
            if not cont:
                continue
            hist = history_before(events, k)
            payload = {"input": {"history": hist, "horizon": len(cont)},
                       "target": {"continuation": cont}}
            yield self.make(
                task_type="PREDICT_TRAJECTORY_CONTINUATION", payload=payload, episode_id=episode_id,
                sequence_index=k, cutoff_sequence_index=k, participant_ids=participant_ids,
                actor_role="negotiation",
                context={"known_history": hist, "world_state": world,
                         "available_actions": list(_ACTIONS), "language": "en"},
                raw_locator=loc,
                transformation_steps=["order events", f"cutoff before event {k}",
                                      f"target = next {len(cont)} events"],
                data_quality=self._dq())

        # ---- PREDICT_FINAL_OUTCOME (agreement + final price) ----------------------------
        agreement = any((ev.get("action") == "accept") for ev in events_raw)
        final_price = _price((outcome.get("offer") or {}).get("price"))
        if final_price is None:
            # fall back to the last offer's price
            for ev in reversed(events_raw):
                if ev.get("action") == "offer" and isinstance(ev.get("data"), dict):
                    final_price = _price(ev["data"].get("price"))
                    break
        terminal_k = next((k for k, ev in enumerate(events_raw) if ev.get("action") in _TERMINAL), n)
        hist = history_before(events, terminal_k)
        out_obj = {
            "agreement": agreement,
            "final_price": final_price,
            "reward": outcome.get("reward"),
            "list_price": world["list_price"],
        }
        miss = [] if final_price is not None else ["final_price"]
        payload = {"input": {"history": hist, "state": world},
                   "target": {"outcome": out_obj, "outcome_type": "negotiation_result"}}
        yield self.make(
            task_type="PREDICT_FINAL_OUTCOME", payload=payload, episode_id=episode_id,
            sequence_index=terminal_k, cutoff_sequence_index=terminal_k,
            participant_ids=participant_ids, actor_role="negotiation",
            context={"known_history": hist, "world_state": world,
                     "available_actions": list(_ACTIONS), "language": "en"},
            raw_locator=loc,
            transformation_steps=["order events", "cutoff before first terminal action",
                                  "target = agreement + final price"],
            data_quality=self._dq(extra_missing=miss))
