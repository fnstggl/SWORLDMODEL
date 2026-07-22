"""OmniBehavior (jiawei-ucas/OmniBehavior) — real Kuaishou long-horizon behavior traces
with a persistent user identity across five+ scenarios over a ~90-day span.

Real source (verified by streaming the HF repo + downloading raw_user_data/en/user_046.json):
  Each raw file / streamed row is a SINGLE-KEY dict ``{"<user_id>": {...}}`` where the value
  holds::
      user_profile   : str  — natural-language demographic sketch of the user
      action_history : list — ordered heterogeneous behavior events, each::
          {type, timestamp, context?, action}
        * type       — the scenario, one of: "Advertisement", "Live Streaming",
                       "Video Browsing", "E-commerce", "Customer Service", "Search Behavior"
        * timestamp  — string "YYYY-MM-DD HH:MM:SS" (real datetime; parseable)
        * context    — dict describing the STIMULUS shown (video/live/product/ad metadata;
                       mostly-null wide schema; ABSENT for "Search Behavior")
        * action     — list of sub-action dicts encoding the user's BEHAVIOR, e.g.
                       [{type:"watch", play_duration, is_complete_play, ...}],
                       [{type:"cart", is_add_to_cart}, {type:"purchase", is_pay}],
                       [{type:"search", keyword, query_category}, {type:"show", show_cnt}],
                       [{type:"dialogue", content:[{role,content},...]}], ...

Persistent identity: one user == one long cross-scenario trace. ``actor_id`` is the
pseudonymized user id, stable across all of that user's records; persistent_identity_available
=True; split by platform_user (isolation key = actor_id).

Emits:
  PREDICT_NEXT_ACTION             — target = the whole next behavior event (scenario + stimulus
                                    context + behavior sub-actions); history = prior events only.
  PREDICT_RESPONSE_OR_NONRESPONSE — for events carrying an explicit binary engagement/conversion
                                    flag (purchase.is_pay, cart.is_add_to_cart, watch.is_complete_play,
                                    click_cart.is_click_cart_action, comment.is_comment); observation
                                    = the stimulus at the cutoff; target.responded = that real flag.
  PREDICT_TIME_TO_ACTION          — inter-event gap in seconds (from parsed timestamps) between
                                    consecutive actions; plus one right-CENSORED record at the end
                                    of each trace (no further action observed -> seconds=null).
  PREDICT_TRAJECTORY_CONTINUATION — from a mid-trace cutoff, the remaining (capped) events.

NOT produced: PREDICT_FINAL_OUTCOME. OmniBehavior records a raw behavior stream with NO explicit
per-user session/terminal outcome label; inventing one (e.g. total spend) would fabricate a target
and duplicate the trajectory task. Skipped and listed in DOC.unavailable_fields.

Bounds: to keep records finite for the very long full-scale traces (~8k actions/user), at most CAP
per-step examples are emitted per user (documented; surfaced via a data_quality warning when capped).

License: CC-BY-NC-SA-4.0 (NON-COMMERCIAL). training_suitability="train" but every record carries a
non-commercial warning; exclude from any commercial training view.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before, observation_at

#: Documented per-user bound on per-step examples (traces can hold thousands of actions).
CAP = 100
#: Longest string kept in a stimulus-context summary (OCR/ASR/live-cover fields can be huge).
_STR_CAP = 500

SCENARIO_TYPES = ["Advertisement", "Customer Service", "E-commerce", "Live Streaming",
                  "Search Behavior", "Video Browsing"]
_NONCOMMERCIAL = "CC-BY-NC-SA-4.0 non-commercial license: exclude from any commercial training view"


def _load_users(raw_dir: Path) -> list[tuple[str, dict]]:
    """Load (user_id, {user_profile, action_history}) pairs from parquet or JSON.

    Handles BOTH the streamed parquet shard (rows are single-key ``{user_id: {...}}`` dicts)
    and JSON fixtures/real per-user files (a list of such dicts, or one dict per file)."""
    items: list[dict] = []
    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "*.parquet"), recursive=True)
                if ".cache" not in f]
    if pq_files:
        import pyarrow.parquet as pq
        for f in sorted(pq_files):
            items.extend(pq.read_table(f).to_pylist())
    else:
        json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                      if ".cache" not in f and "dataset_info" not in f]
        if not json_files:
            raise FileNotFoundError(f"no OmniBehavior parquet/json found under {raw_dir}")
        for f in sorted(json_files):
            data = json.loads(Path(f).read_text())
            if isinstance(data, list):
                items.extend(data)
            elif isinstance(data, dict):
                items.append(data)

    out: list[tuple[str, dict]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        # inner form already flattened
        if "action_history" in item and "user_profile" in item:
            out.append((str(item.get("user_id") or f"user_{idx}"), item))
            continue
        # single-key {user_id: inner} form (streamed / raw / fixture)
        for uid, inner in item.items():
            if isinstance(inner, dict) and "action_history" in inner:
                out.append((str(uid), inner))
    return out


def _parse_ts(s):
    if not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _nonnull(ctx: dict) -> dict:
    """Drop null context fields (the wide schema is mostly null) and cap long strings."""
    out = {}
    for k, v in (ctx or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and len(v) > _STR_CAP:
            v = v[:_STR_CAP] + "...<trunc>"
        out[k] = v
    return out


def _response_flag(ev: dict):
    """Return (responded_bool, signal_name) if the event carries an explicit binary
    engagement/conversion flag, else None. Never fabricates: only reads real booleans."""
    subs = [s for s in (ev.get("action") or []) if isinstance(s, dict)]
    for s in subs:  # strongest conversion signal first
        if s.get("type") == "purchase" and "is_pay" in s:
            return bool(s["is_pay"]), "purchase.is_pay"
        if s.get("type") == "purchase" and "paid" in s:
            return bool(s["paid"]), "purchase.paid"
    for s in subs:
        if s.get("type") == "cart" and "is_add_to_cart" in s:
            return bool(s["is_add_to_cart"]), "cart.is_add_to_cart"
        if s.get("type") == "click_cart" and "is_click_cart_action" in s:
            return bool(s["is_click_cart_action"]), "click_cart.is_click_cart_action"
        if s.get("type") == "watch" and "is_complete_play" in s:
            return bool(s["is_complete_play"]), "watch.is_complete_play"
        if s.get("type") == "comment" and "is_comment" in s:
            return bool(s["is_comment"]), "comment.is_comment"
    return None


class Converter(BaseConverter):
    DATASET_ID = "omnibehavior"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "omnibehavior"
    DOC = {
        "dataset_id": "omnibehavior",
        "original_fields": [
            {"name": "<user_id> (outer key)", "meaning": "persistent user identity; the trace owner"},
            {"name": "user_profile", "meaning": "natural-language demographic sketch of the user"},
            {"name": "action_history", "meaning": "ordered heterogeneous behavior events across scenarios"},
            {"name": "action_history[].type", "meaning": "scenario: Advertisement/Live Streaming/Video Browsing/"
                                                         "E-commerce/Customer Service/Search Behavior"},
            {"name": "action_history[].timestamp", "meaning": "event time, string 'YYYY-MM-DD HH:MM:SS'"},
            {"name": "action_history[].context", "meaning": "stimulus shown (video/live/product/ad metadata); "
                                                            "absent for Search Behavior; wide mostly-null schema"},
            {"name": "action_history[].action", "meaning": "list of sub-actions = the user's behavior "
                                                           "(watch/cart/purchase/search/dialogue/click_cart/...)"},
        ],
        "canonical_mapping": [
            {"source_field": "<user_id>", "canonical_path": "decision_unit.actor_id (pseudonymized, persistent) / episode_id"},
            {"source_field": "user_profile", "canonical_path": "context.actor_profile.description"},
            {"source_field": "action_history[k].type", "canonical_path": "payload.target.action_type / meta.scenario"},
            {"source_field": "action_history[k].context", "canonical_path": "payload.target.action_content.context "
                                                                           "(NEXT_ACTION) | context.current_observation (RESPONSE)"},
            {"source_field": "action_history[k].action", "canonical_path": "payload.target.action_content.behaviors"},
            {"source_field": "action_history[k].timestamp", "canonical_path": "history_event.t / cutoff_time; "
                                                                             "consecutive diffs -> TIME_TO_ACTION"},
            {"source_field": "action[].is_pay/is_add_to_cart/is_complete_play/is_click_cart_action/is_comment",
             "canonical_path": "payload.target.responded (RESPONSE_OR_NONRESPONSE)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_ACTION", "PREDICT_RESPONSE_OR_NONRESPONSE",
                           "PREDICT_TIME_TO_ACTION", "PREDICT_TRAJECTORY_CONTINUATION"],
        "unavailable_fields": [
            "PREDICT_FINAL_OUTCOME: no explicit per-user session/terminal outcome label exists "
            "(raw behavior stream only); not fabricated",
            "verified private beliefs/goals/motivation (none recorded)",
            "explicit available-action set below the scenario level (behaviors are open-vocabulary)",
            "sub-second action latency (only event-level timestamps)",
        ],
        "chronology_rules": "Events are sorted by parsed timestamp into the true cross-scenario timeline. "
                            "For the decision at step k, only events 0..k-1 are in context/input; event k "
                            "(scenario + context + behaviors) is ONLY in payload.target. TIME_TO_ACTION uses "
                            "current_time=ts[k-1] and target=ts[k]-ts[k-1]; the terminal record is right-censored.",
        "split_key": "platform_user (actor_id = pseudonymized user id; all of a user's records held out together)",
        "leakage_risks": [
            "the stimulus context at step k is exposed ONLY for RESPONSE_OR_NONRESPONSE (it precedes the "
            "engagement flag, which is the target); for NEXT_ACTION the whole event is target-side",
            "a user's records must never be split across train/eval (enforced via actor_id isolation)",
        ],
        "known_limitations": [
            f"at most CAP={CAP} per-step examples emitted per user (full traces can hold thousands of actions); "
            "surfaced via a data_quality warning when capped",
            "PREDICT_TRAJECTORY_CONTINUATION continuation capped to CAP events",
            "wide stimulus-context schema is mostly null; null fields dropped and long strings truncated to "
            f"{_STR_CAP} chars in summaries",
            "RESPONSE_OR_NONRESPONSE 'responded' is a scenario-specific engagement/conversion flag, not a generic reply",
        ],
        "license_implications": "CC-BY-NC-SA-4.0: NON-COMMERCIAL. Derivatives + redistribution allowed with "
                                "attribution + share-alike, but NO commercial use. Every record warns; exclude "
                                "from commercial training views.",
        "training_suitability": "train",
        "assumptions": [
            "the outer single key of each record is the user id",
            "timestamps are 'YYYY-MM-DD HH:MM:SS'; events with parseable timestamps are chronologically ordered",
            "engagement booleans (is_pay/is_add_to_cart/is_complete_play/...) are genuine recorded outcomes",
        ],
    }

    # ---- per-event compact representation (used in history/continuation) --------------
    def _event_repr(self, ev: dict, index: int, actor: str) -> dict:
        return history_event(
            index, actor, "action",
            t=ev.get("timestamp"),
            action_type=ev.get("type"),
            action_content={"behaviors": ev.get("action") or []},
            meta={"scenario": ev.get("type"), "context": _nonnull(ev.get("context") or {})},
        )

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        for ui, (uid, inner) in enumerate(_load_users(raw_dir)):
            yield from self._one_user(ui, uid, inner)

    def _one_user(self, ui: int, uid: str, inner: dict) -> Iterator[dict]:
        profile = inner.get("user_profile") or ""
        history = inner.get("action_history") or []
        if not history:
            return

        # Order into the true cross-scenario timeline (stable on original index).
        indexed = list(enumerate(history))
        indexed.sort(key=lambda p: (_parse_ts(p[1].get("timestamp")) or datetime.min, p[0]))
        events = [ev for _, ev in indexed]
        n = len(events)

        actor = self.pseudonym("actor", uid)
        episode_id = f"omnibehavior-{uid}"
        loc_ids = [episode_id]
        reprs = [self._event_repr(ev, k, actor) for k, ev in enumerate(events)]
        actor_profile = {"description": profile}
        capped = n > CAP

        def base_loc(orig_indices):
            return {"files": [f"raw_user_data/en/{uid}.json"], "indices": list(orig_indices), "ids": loc_ids}

        def dq(missing, warnings=None, confidence="high", **extra):
            w = ["non-commercial license (CC-BY-NC-SA-4.0)"]
            if warnings:
                w += warnings
            base = {"missing_fields": missing, "warnings": w, "confidence": confidence,
                    "chronology_verified": True, "target_verified": True,
                    "possible_leakage": False, "license_verified": True,
                    "inferred_fields": ["sequence_index (from timestamp ordering)"]}
            base.update(extra)
            return base

        # ------------------------------------------------ NEXT_ACTION (first CAP) ------
        for k in range(min(n, CAP)):
            ev = events[k]
            hist = history_before(reprs, k)
            obs = observation_at(reprs, k)
            ctx = {"actor_profile": actor_profile, "known_history": hist,
                   "current_observation": obs, "world_state": {"platform": "Kuaishou"},
                   "available_actions": SCENARIO_TYPES, "language": "en"}
            payload = {
                "input": {"history": hist, "observation": obs, "available_actions": SCENARIO_TYPES},
                "target": {"action_type": ev.get("type") or "<UNKNOWN>", "acted": True,
                           "action_content": {"scenario": ev.get("type"),
                                              "behaviors": ev.get("action") or [],
                                              "context": _nonnull(ev.get("context") or {}),
                                              "timestamp": ev.get("timestamp")}},
            }
            warn = [f"per-user examples capped at {CAP}"] if capped else None
            yield self.make(
                task_type="PREDICT_NEXT_ACTION", payload=payload, episode_id=episode_id,
                sequence_index=k, cutoff_sequence_index=k, cutoff_time=(events[k - 1].get("timestamp") if k else None),
                actor_id=actor, actor_role="platform_user", participant_ids=[actor],
                group_id=actor, persistent_identity_available=True, context=ctx,
                raw_locator=base_loc([indexed[k][0]]),
                transformation_steps=["load omnibehavior user", "sort events by timestamp",
                                      f"cutoff before event {k}", "whole event -> target"],
                data_quality=dq(["available_action_set_below_scenario"], warn))

        # -------------------------------------- RESPONSE_OR_NONRESPONSE (first CAP) -----
        emitted = 0
        for k in range(n):
            if emitted >= CAP:
                break
            ev = events[k]
            flag = _response_flag(ev)
            if flag is None:
                continue
            responded, signal = flag
            hist = history_before(reprs, k)
            stimulus = {"scenario": ev.get("type"), "context": _nonnull(ev.get("context") or {}),
                        "kind": "stimulus", "timestamp": ev.get("timestamp")}
            ctx = {"actor_profile": actor_profile, "known_history": hist,
                   "current_observation": stimulus, "world_state": {"platform": "Kuaishou"},
                   "available_actions": None, "language": "en"}
            payload = {"input": {"history": hist, "observation": stimulus},
                       "target": {"responded": responded, "latency_seconds": None}}
            yield self.make(
                task_type="PREDICT_RESPONSE_OR_NONRESPONSE", payload=payload, episode_id=episode_id,
                sequence_index=k, cutoff_sequence_index=k, cutoff_time=(events[k - 1].get("timestamp") if k else None),
                actor_id=actor, actor_role="platform_user", participant_ids=[actor],
                group_id=actor, persistent_identity_available=True, context=ctx,
                raw_locator=base_loc([indexed[k][0]]),
                transformation_steps=["load omnibehavior user", "sort by timestamp",
                                      f"cutoff before event {k}", f"responded = {signal}"],
                data_quality=dq(["sub_second_latency"],
                                [f"'responded' is the scenario-specific engagement/conversion flag {signal!r}, "
                                 "not a generic reply"]))
            emitted += 1

        # ------------------------------------------------ TIME_TO_ACTION ---------------
        for k in range(1, min(n, CAP + 1)):
            prev_ts, cur_ts = _parse_ts(events[k - 1].get("timestamp")), _parse_ts(events[k].get("timestamp"))
            if prev_ts is None or cur_ts is None:
                continue
            delta = (cur_ts - prev_ts).total_seconds()
            if delta < 0:
                continue
            hist = history_before(reprs, k)
            ctx = {"actor_profile": actor_profile, "known_history": hist,
                   "current_observation": observation_at(reprs, k),
                   "world_state": {"platform": "Kuaishou"}, "available_actions": None, "language": "en"}
            payload = {
                "input": {"history": hist, "current_time": events[k - 1].get("timestamp")},
                "target": {"acted": True, "time_to_action_seconds": delta,
                           "censoring": {"censored": False, "observation_window_seconds": None,
                                         "reason": "next action observed within the trace"}},
            }
            yield self.make(
                task_type="PREDICT_TIME_TO_ACTION", payload=payload, episode_id=episode_id,
                sequence_index=k, cutoff_sequence_index=k, cutoff_time=events[k - 1].get("timestamp"),
                actor_id=actor, actor_role="platform_user", participant_ids=[actor],
                group_id=actor, persistent_identity_available=True, context=ctx,
                raw_locator=base_loc([indexed[k - 1][0], indexed[k][0]]),
                transformation_steps=["load omnibehavior user", "sort by timestamp",
                                      f"gap ts[{k}]-ts[{k-1}] seconds"],
                data_quality=dq([], confidence="high"))

        # terminal right-censored record: no further action observed after the last event.
        last_ts, first_ts = _parse_ts(events[-1].get("timestamp")), _parse_ts(events[0].get("timestamp"))
        window = (last_ts - first_ts).total_seconds() if (last_ts and first_ts) else None
        hist_all = reprs[:CAP]
        payload = {
            "input": {"history": hist_all, "current_time": events[-1].get("timestamp")},
            "target": {"acted": False, "time_to_action_seconds": None,
                       "censoring": {"censored": True, "observation_window_seconds": window,
                                     "reason": "no further action observed after the last event in the trace "
                                               "(right-censored)"}},
        }
        yield self.make(
            task_type="PREDICT_TIME_TO_ACTION", payload=payload, episode_id=episode_id,
            sequence_index=n, cutoff_sequence_index=n, cutoff_time=events[-1].get("timestamp"),
            actor_id=actor, actor_role="platform_user", participant_ids=[actor],
            group_id=actor, persistent_identity_available=True,
            context={"actor_profile": actor_profile, "known_history": hist_all,
                     "world_state": {"platform": "Kuaishou"}, "available_actions": None, "language": "en"},
            raw_locator=base_loc([i for i, _ in indexed]),
            transformation_steps=["load omnibehavior user", "cutoff after last event",
                                  "no successor -> right-censored"],
            data_quality=dq(["time_of_next_action"],
                            ["right-censored: the observation window ends with the trace"], confidence="high"))

        # ------------------------------------------------ TRAJECTORY_CONTINUATION ------
        if n >= 2:
            cutoff = max(1, n // 2)
            hist = reprs[:cutoff]
            cont = reprs[cutoff:cutoff + CAP]
            reaches_end = cutoff + CAP >= n
            warn = None if reaches_end else [f"continuation capped to {CAP} events (trace has {n})"]
            payload = {"input": {"history": hist, "horizon": len(cont)},
                       "target": {"continuation": cont}}
            yield self.make(
                task_type="PREDICT_TRAJECTORY_CONTINUATION", payload=payload, episode_id=episode_id,
                sequence_index=cutoff, cutoff_sequence_index=cutoff,
                cutoff_time=events[cutoff - 1].get("timestamp"),
                actor_id=actor, actor_role="platform_user", participant_ids=[actor],
                group_id=actor, persistent_identity_available=True,
                context={"actor_profile": actor_profile, "known_history": hist,
                         "world_state": {"platform": "Kuaishou"}, "available_actions": None, "language": "en"},
                raw_locator=base_loc([i for i, _ in indexed]),
                transformation_steps=["load omnibehavior user", "sort by timestamp",
                                      f"cutoff after {cutoff} events", "remaining events as continuation"],
                data_quality=dq([], warn))
