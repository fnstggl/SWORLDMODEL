"""Werewolf Among Us — social-deduction game dialogues (Lai et al., ACL Findings 2023).

Real source (verified from Youtube/split/*.json and Ego4D/split/*.json — the integrated
annotation files the dataset card recommends loading). One JSON list per split; each element
is one game:
  {YT_ID | EG_ID, video_name (Youtube only), Game_ID,
   Dialogue: [{Rec_Id, speaker, timestamp ("mm:ss" video time), utterance,
               annotation: [persuasion-strategy label(s)]}],
   playerNames: [names left-to-right], votingOutcome: [per-player vote = index of the
       player voted for, or "N/A"/"NA"], startRoles: [role per player], endRoles: [role
       per player], startTime, endTime, warning}.
Ego4D/split/avalon.json games (Avalon, not One-Night-Werewolf) carry ONLY {EG_ID, Game_ID,
Dialogue} — no players / votes / roles.

Emits:
  PREDICT_NEXT_MESSAGE           — each utterance; target text + persuasion-strategy label
  PREDICT_NEXT_ACTION            — each player's final vote (a discrete recorded action)
  PREDICT_TRAJECTORY_CONTINUATION — next K utterances given a prefix
  PREDICT_FINAL_OUTCOME          — revealed start/end roles + full voting outcome

Deception discipline: a player's hidden role is the *deception*, so roles are NEVER placed
in any task INPUT/context. They appear ONLY inside the FINAL_OUTCOME target (the revealed
end state). PREDICT_BELIEF_CHANGE is NOT emitted: the dataset records no player's private
belief before-and-after — only observed votes and revealed roles — so a belief-change label
cannot be produced without fabrication. Video/audio/MViT numpy features are skipped.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before, observation_at

_TRAJ_HORIZON = 5
_NA_VOTES = {"N/A", "NA", "na", "n/a", None, ""}


def _short(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]


def _load_games(raw_dir: Path) -> list[dict]:
    """Load per-game rows from streamed parquet, the split JSON files, or a JSON fixture."""
    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "stream_shard_*.parquet"), recursive=True)
                if ".cache" not in f]
    if pq_files:
        import pyarrow.parquet as pq
        rows: list[dict] = []
        for f in sorted(pq_files):
            rows.extend(pq.read_table(f).to_pylist())
        return rows

    split_files = [f for f in glob.glob(str(raw_dir / "**" / "split" / "*.json"), recursive=True)
                   if ".cache" not in f]
    if split_files:
        games: list[dict] = []
        for f in sorted(split_files):
            data = json.loads(Path(f).read_text())
            if isinstance(data, list):
                games.extend(data)
        return games

    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"no Werewolf split json / parquet / fixture found under {raw_dir}")


class Converter(BaseConverter):
    DATASET_ID = "werewolf"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "werewolf"
    DOC = {
        "dataset_id": "werewolf",
        "original_fields": [
            {"name": "Dialogue[].utterance", "meaning": "transcribed utterance text"},
            {"name": "Dialogue[].speaker", "meaning": "player name who spoke (pseudonymized)"},
            {"name": "Dialogue[].timestamp", "meaning": "mm:ss video time of the utterance"},
            {"name": "Dialogue[].annotation", "meaning": "persuasion-strategy label(s) for the utterance (6 categories + No Strategy)"},
            {"name": "playerNames", "meaning": "players left-to-right in the video"},
            {"name": "votingOutcome", "meaning": "per-player vote = index of the player they voted for, or N/A (center/unrecorded)"},
            {"name": "startRoles / endRoles", "meaning": "each player's role at game start / end (hidden during play)"},
            {"name": "YT_ID / EG_ID / video_name / Game_ID", "meaning": "game/segment identifiers"},
        ],
        "canonical_mapping": [
            {"source_field": "Dialogue[].utterance", "canonical_path": "payload.target.message_text | context.known_history[].text"},
            {"source_field": "Dialogue[].annotation", "canonical_path": "payload.target.strategy (NEXT_MESSAGE)"},
            {"source_field": "Dialogue[].speaker", "canonical_path": "decision_unit.actor_id (pseudonymized)"},
            {"source_field": "votingOutcome[i]", "canonical_path": "payload.target.action_content.voted_for (NEXT_ACTION) & payload.target.outcome.voting (FINAL_OUTCOME)"},
            {"source_field": "startRoles / endRoles", "canonical_path": "payload.target.outcome.{start_roles,end_roles} (FINAL_OUTCOME ONLY — never in input)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_ACTION",
                           "PREDICT_TRAJECTORY_CONTINUATION", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": [
            "PREDICT_BELIEF_CHANGE: private beliefs are not recorded (no before/after belief measurement); only observed votes and revealed roles exist — a belief-change label would have to be fabricated, so it is NOT emitted",
            "explicit game winner label (only revealed roles + votes are recorded; a winner is derivable via game rules but not provided, so it is not fabricated)",
            "a player's own start role in INPUT (deliberately withheld: roles are deception ground truth, exposed only in the FINAL_OUTCOME target)",
            "N/A votes (voted center / unrecorded) are not emitted as discrete vote actions",
            "video / audio / MViT numpy feature files (skipped)",
        ],
        "chronology_rules": "For an utterance at index k only utterances 0..k-1 are exposed. Votes are simultaneous end-of-game actions: NEXT_ACTION exposes the full discussion (all utterances) and never exposes any other player's vote. FINAL_OUTCOME exposes the full discussion and reveals roles + votes only in the target.",
        "split_key": "group (whole game): group_id == episode_id == a stable per-game id",
        "leakage_risks": [
            "player names appear verbatim inside utterance text ('James is the werewolf'); we pseudonymize actor_ids but never rewrite utterance text, so names remain in content (matches the public-video source)",
            "the same first name can denote different people across videos, so actor pseudonyms are game-scoped (persistent_identity_available=False)",
        ],
        "known_limitations": [
            "Avalon games (Ego4D/split/avalon.json) carry only dialogue — no players/votes/roles — so they yield only NEXT_MESSAGE + TRAJECTORY_CONTINUATION",
            "timestamps are video mm:ss strings (kept as event.t), not absolute wall-clock times",
        ],
        "license_implications": "Apache-2.0 for text + annotations: training and commercial use permitted. The Ego4D VIDEO subset is under a separate Ego4D license, but only text/annotations are normalized here.",
        "training_suitability": "train",
        "assumptions": ["votingOutcome[i] is the index into playerNames of the player that player i voted for",
                        "a game with no startRoles is an Avalon game"],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        for gi, game in enumerate(_load_games(raw_dir)):
            yield from self._one_game(gi, game)

    def _one_game(self, gi: int, game: dict) -> Iterator[dict]:
        subset = "youtube" if game.get("YT_ID") is not None else "ego4d"
        seg = game.get("YT_ID") or game.get("EG_ID") or ""
        video_name = game.get("video_name", "")
        game_field = game.get("Game_ID", "")
        episode_id = f"werewolf-{subset}-{_short(subset, video_name, seg, game_field, gi)}"

        dialogue = game.get("Dialogue") or []
        player_names = game.get("playerNames") or []
        voting = game.get("votingOutcome")
        start_roles = game.get("startRoles")
        end_roles = game.get("endRoles")
        is_avalon = not start_roles and not player_names
        game_type = "Avalon" if is_avalon else "One Night Ultimate Werewolf"

        def player_actor(name) -> str:
            return self.pseudonym("actor", f"{episode_id}:{name}")

        participant_ids = [player_actor(n) for n in player_names]
        world = {"game": game_type, "players": participant_ids, "num_players": len(player_names) or None}

        # ordered event list (discussion only; roles/votes are NOT part of the observable history)
        events: list[dict] = []
        for k, u in enumerate(dialogue):
            speaker = u.get("speaker")
            strat = u.get("annotation") or None
            events.append(history_event(
                k, player_actor(speaker), "message", text=u.get("utterance", ""),
                t=u.get("timestamp"),
                meta={"strategy": strat, "rec_id": u.get("Rec_Id")}))
        n = len(events)
        loc = {"files": [f"{'Youtube' if subset=='youtube' else 'Ego4D'}/split/*.json"],
               "indices": [gi], "ids": [episode_id]}

        # ---- PREDICT_NEXT_MESSAGE -------------------------------------------------------
        for k, u in enumerate(dialogue):
            actor = player_actor(u.get("speaker"))
            strat = u.get("annotation") or None
            hist = history_before(events, k)
            obs = observation_at(events, k)
            payload = {
                "input": {"dialogue_history": hist, "private_goal": {}, "current_observation": obs},
                "target": {"message_text": u.get("utterance", ""), "dialogue_act": None, "strategy": strat},
            }
            missing = ["private_goal_hidden_role", "video_features"]
            if strat is None:
                missing.append("strategy_annotation")
            yield self.make(
                task_type="PREDICT_NEXT_MESSAGE", payload=payload, episode_id=episode_id,
                group_id=episode_id, sequence_index=k, cutoff_sequence_index=k,
                participant_ids=participant_ids, actor_id=actor, actor_role="player",
                context={"known_history": hist, "current_observation": obs, "world_state": world,
                         "available_actions": None, "language": "en"},
                raw_locator=loc,
                transformation_steps=["load integrated split json", "order utterances",
                                      f"cutoff before utterance {k}", "withhold hidden roles from input"],
                data_quality={"missing_fields": missing, "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high"})

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
                group_id=episode_id, sequence_index=k, cutoff_sequence_index=k,
                participant_ids=participant_ids, actor_role="game",
                context={"known_history": hist, "world_state": world, "available_actions": None,
                         "language": "en"},
                raw_locator=loc,
                transformation_steps=["order utterances", f"cutoff before utterance {k}",
                                      f"target = next {len(cont)} utterances"],
                data_quality={"missing_fields": ["video_features"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high"})

        # ---- PREDICT_NEXT_ACTION (each player's final vote) ------------------------------
        if isinstance(voting, list) and player_names:
            for i, vote in enumerate(voting):
                if i >= len(player_names):
                    break
                if vote in _NA_VOTES:
                    continue  # center/unrecorded — not a cleanly recorded discrete action
                try:
                    tgt_idx = int(vote)
                except (TypeError, ValueError):
                    continue
                if not (0 <= tgt_idx < len(player_names)):
                    continue
                voter = player_actor(player_names[i])
                voted_for = player_actor(player_names[tgt_idx])
                hist = list(events)  # votes follow the whole discussion
                payload = {
                    "input": {"history": hist, "observation": observation_at(events, n),
                              "available_actions": participant_ids},
                    # voter is encoded in the target so each player's simultaneous vote is a
                    # distinct record (the dedup/record_id key is derived from the target, not
                    # the actor_id — without this, two players voting the same target collapse).
                    "target": {"action_type": "vote", "acted": True,
                               "action_content": {"voter": voter, "voted_for": voted_for}},
                }
                yield self.make(
                    task_type="PREDICT_NEXT_ACTION", payload=payload, episode_id=episode_id,
                    group_id=episode_id, sequence_index=n, cutoff_sequence_index=n,
                    participant_ids=participant_ids, actor_id=voter, actor_role="player",
                    context={"known_history": hist, "world_state": world,
                             "available_actions": participant_ids, "language": "en"},
                    raw_locator=loc,
                    transformation_steps=["expose full discussion",
                                          "target = this player's end-of-game vote",
                                          "other players' votes withheld"],
                    data_quality={"missing_fields": ["video_features"], "chronology_verified": True,
                                  "target_verified": True, "license_verified": True, "confidence": "high"})

        # ---- PREDICT_FINAL_OUTCOME (revealed roles + votes) -----------------------------
        if player_names and (start_roles or end_roles or isinstance(voting, list)):
            def role_map(roles):
                if not isinstance(roles, list):
                    return None
                return {player_actor(player_names[i]): roles[i]
                        for i in range(min(len(player_names), len(roles)))}

            def vote_map(votes):
                if not isinstance(votes, list):
                    return None
                out = {}
                for i in range(min(len(player_names), len(votes))):
                    v = votes[i]
                    if v in _NA_VOTES:
                        out[player_actor(player_names[i])] = None
                    else:
                        try:
                            vi = int(v)
                            out[player_actor(player_names[i])] = (
                                player_actor(player_names[vi]) if 0 <= vi < len(player_names) else None)
                        except (TypeError, ValueError):
                            out[player_actor(player_names[i])] = None
                return out

            outcome = {
                "game": game_type,
                "start_roles": role_map(start_roles),
                "end_roles": role_map(end_roles),
                "voting": vote_map(voting),
                "num_players": len(player_names),
            }
            missing = ["video_features", "explicit_winner_label"]
            if not start_roles:
                missing.append("start_roles")
            if not isinstance(voting, list):
                missing.append("voting_outcome")
            hist = list(events)
            payload = {"input": {"history": hist, "state": world},
                       "target": {"outcome": outcome, "outcome_type": "game_result"}}
            yield self.make(
                task_type="PREDICT_FINAL_OUTCOME", payload=payload, episode_id=episode_id,
                group_id=episode_id, sequence_index=n, cutoff_sequence_index=n,
                participant_ids=participant_ids, actor_role="game",
                context={"known_history": hist, "world_state": world, "available_actions": None,
                         "language": "en"},
                raw_locator=loc,
                transformation_steps=["expose full discussion",
                                      "reveal start/end roles + votes in target only"],
                data_quality={"missing_fields": missing, "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high",
                              "warnings": ["roles are revealed ground truth, present only in the target"]})
