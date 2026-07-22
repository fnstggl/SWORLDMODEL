"""PersuasionForGood — persuasive donation dialogues (Wang et al., ACL 2019).

Real source (verified from data/FullData/*.csv):
  full_dialog.csv columns = ["" (row idx), Unit (utterance text), Turn (turn index),
    B4 (role: 0=persuader/ER, 1=persuadee/EE), B2 (dialogue id)]. Rows are in
    chronological order.
  full_info.csv  columns = [B2 (dialogue id), B3 (stable worker/user id), B4 (role),
    B6 (ACTUAL donation made after the task), B7 (number of turns), then per-participant
    background/personality scores with a ".x" suffix: Big-5, Moral Foundations, Schwartz
    values, decision style (rational/intuitive) and demographics].

The task: two MTurk workers chat; the persuader (role 0) tries to convince the persuadee
(role 1) to donate part of their task earnings to the charity "Save the Children". After
the chat each worker privately decides an actual donation (B6).

Emits:
  PREDICT_NEXT_MESSAGE           — each utterance (persuader or persuadee)
  PREDICT_RESPONSE_OR_NONRESPONSE — for each persuader turn, does the persuadee reply next
  PREDICT_TRAJECTORY_CONTINUATION — next K utterances given a prefix
  PREDICT_FINAL_OUTCOME          — the persuadee's (and persuader's) actual donation (B6)

Honesty notes: the FullData release carries NO per-utterance strategy annotations (those
exist only for the 300 annotated dialogues in AnnotatedData/) and NO intended-donation
label (B5, annotated only in AnnotatedData) — both are left null and listed in
missing_fields. There are no per-utterance timestamps, so latency is null. Donation (B6)
is never fabricated: if absent it is null and flagged. RANK_CANDIDATE_ACTIONS is NOT
produced — the source offers no candidate-action set to rank.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before, observation_at

_CHARITY = "Save the Children"
_TRAJ_HORIZON = 5

_BIG5 = ["extrovert.x", "agreeable.x", "conscientious.x", "neurotic.x", "open.x"]
_MFT = ["care.x", "fairness.x", "loyalty.x", "authority.x", "purity.x"]
_SCHWARTZ = ["freedom.x", "conform.x", "tradition.x", "benevolence.x", "universalism.x",
             "self_direction.x", "stimulation.x", "hedonism.x", "achievement.x", "power.x",
             "security.x"]
_DECISION = ["rational.x", "intuitive.x"]
_DEMO = ["age.x", "sex.x", "race.x", "edu.x", "marital.x", "employment.x", "income.x",
         "religion.x", "ideology.x"]


def _num(v):
    """Parse a numeric cell to float, else None (empty / NaN / non-numeric)."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "na", "none", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _demo(v):
    s = None if v is None else str(v).strip()
    return None if not s or s.lower() in ("nan", "na", "none", "null") else s


def _load_dialogues(raw_dir: Path) -> list[dict]:
    """Load pre-grouped dialogue rows.

    Supports: (1) streamed parquet shards, (2) the real FullData CSVs, grouped here into
    one row per dialogue, (3) a JSON fixture that is already a list of dialogue rows.
    """
    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "stream_shard_*.parquet"), recursive=True)
                if ".cache" not in f]
    if pq_files:
        import pyarrow.parquet as pq
        rows: list[dict] = []
        for f in sorted(pq_files):
            rows.extend(pq.read_table(f).to_pylist())
        return rows

    dialog_csv = [f for f in glob.glob(str(raw_dir / "**" / "full_dialog.csv"), recursive=True)
                  if ".cache" not in f]
    info_csv = [f for f in glob.glob(str(raw_dir / "**" / "full_info.csv"), recursive=True)
                if ".cache" not in f]
    if dialog_csv and info_csv:
        return _group_csv(sorted(dialog_csv)[0], sorted(info_csv)[0])

    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"no PersuasionForGood parquet/CSV/json found under {raw_dir}")


def _group_csv(dialog_path: str, info_path: str) -> list[dict]:
    """Group the two flat CSVs into ordered per-dialogue rows (preserving file order)."""
    order: list[str] = []
    turns: dict[str, list[dict]] = {}
    with open(dialog_path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            did = r.get("B2")
            if not did:
                continue
            if did not in turns:
                turns[did] = []
                order.append(did)
            turns[did].append({
                "turn": _num(r.get("Turn")),
                "role": _num(r.get("B4")),
                "text": r.get("Unit") if r.get("Unit") is not None else "",
            })
    info: dict[str, dict] = {}
    with open(info_path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            did = r.get("B2")
            if not did:
                continue
            role = str(int(float(r["B4"]))) if _num(r.get("B4")) is not None else None
            if role is None:
                continue
            info.setdefault(did, {})[role] = dict(r)
    return [{"dialogue_id": did, "turns": turns[did], "info": info.get(did, {})}
            for did in order]


class Converter(BaseConverter):
    DATASET_ID = "persuasionforgood"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "persuasionforgood"
    DOC = {
        "dataset_id": "persuasionforgood",
        "original_fields": [
            {"name": "full_dialog.csv:Unit", "meaning": "utterance text (one sentence per row)"},
            {"name": "full_dialog.csv:Turn", "meaning": "turn index within the dialogue"},
            {"name": "full_dialog.csv:B4", "meaning": "role of the speaker: 0=persuader (ER), 1=persuadee (EE)"},
            {"name": "full_dialog.csv:B2", "meaning": "dialogue id"},
            {"name": "full_info.csv:B3", "meaning": "stable worker/user id (persistent across dialogues)"},
            {"name": "full_info.csv:B6", "meaning": "actual donation made by that participant after the task ended"},
            {"name": "full_info.csv:B7", "meaning": "number of turns in the dialogue"},
            {"name": "full_info.csv:*.x", "meaning": "per-participant Big-5, Moral Foundations, Schwartz values, decision style, demographics"},
        ],
        "canonical_mapping": [
            {"source_field": "full_dialog.csv:Unit", "canonical_path": "payload.target.message_text | context.known_history[].text"},
            {"source_field": "full_dialog.csv:B4", "canonical_path": "decision_unit.actor_role (persuader/persuadee)"},
            {"source_field": "full_info.csv:B3", "canonical_path": "decision_unit.actor_id (pseudonymized, persistent)"},
            {"source_field": "full_info.csv:*.x", "canonical_path": "context.actor_profile.{big_five,moral_foundations,schwartz_values,decision_style,demographics}"},
            {"source_field": "persuader private goal (task design)", "canonical_path": "context.private_state_before.goal / payload.input.private_goal (persuader only)"},
            {"source_field": "full_info.csv:B6", "canonical_path": "payload.target.outcome.donation_amount (FINAL_OUTCOME)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_MESSAGE", "PREDICT_RESPONSE_OR_NONRESPONSE",
                           "PREDICT_TRAJECTORY_CONTINUATION", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": [
            "per-utterance strategy annotation (FullData has none; only the 300-dialogue AnnotatedData subset)",
            "intended donation B5 (annotated only in AnnotatedData)",
            "per-utterance timestamps / reply latency",
            "RANK_CANDIDATE_ACTIONS: no candidate-action set exists in the source",
        ],
        "chronology_rules": "For a decision at turn k only turns 0..k-1 are exposed. RESPONSE_OR_NONRESPONSE exposes the persuader turn k and predicts whether turn k+1 is a persuadee reply. FINAL_OUTCOME exposes the full dialogue (the donation is a private post-task action that never appears in the dialogue text).",
        "split_key": "conversation (episode_id = dialogue id); persistent worker ids (B3) are recorded pseudonymously so cross-dialogue reuse is detectable",
        "leakage_risks": [
            "the same worker (B3) can appear in multiple dialogues; we pseudonymize B3 stably so a stricter worker-level split is possible even though the registered split unit is conversation",
        ],
        "known_limitations": [
            "FullData carries no utterance-level persuasion-strategy labels",
            "donation B6 is a single private post-task self-report; a null value means the participant did not report a donation and is flagged, never imputed",
        ],
        "license_implications": "Apache-2.0: training and commercial use permitted with attribution.",
        "training_suitability": "train",
        "assumptions": ["role code 0=persuader, 1=persuadee (per FullData/readme.md)",
                        "charity is 'Save the Children' (dataset task design)"],
    }

    # ---- profile / private-state builders ------------------------------------------------
    def _profile(self, info_row: dict) -> dict:
        if not info_row:
            return {}
        return {
            "big_five": {k: _num(info_row.get(k)) for k in _BIG5},
            "moral_foundations": {k: _num(info_row.get(k)) for k in _MFT},
            "schwartz_values": {k: _num(info_row.get(k)) for k in _SCHWARTZ},
            "decision_style": {k: _num(info_row.get(k)) for k in _DECISION},
            "demographics": {k: _demo(info_row.get(k)) for k in _DEMO},
        }

    def _private_goal(self, role_name: str) -> dict:
        if role_name == "persuader":
            return {"role": "persuader",
                    "goal": f"persuade the persuadee to donate to {_CHARITY}",
                    "charity": _CHARITY}
        return {"role": "persuadee"}

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        for di, row in enumerate(_load_dialogues(raw_dir)):
            yield from self._one_dialogue(di, row)

    def _one_dialogue(self, di: int, row: dict) -> Iterator[dict]:
        did = row.get("dialogue_id") or f"idx{di}"
        episode_id = f"persuasionforgood-{did}"
        turns = row.get("turns") or []
        info = row.get("info") or {}

        def role_name(role) -> str:
            return "persuader" if (role in (0, 0.0, "0")) else "persuadee"

        def user_id(role) -> str | None:
            r = info.get(str(int(float(role)))) if role is not None else None
            return (r or {}).get("B3")

        # stable, persistent actor pseudonyms keyed on the worker id when available
        def actor_of(role) -> str:
            uid = user_id(role)
            key = uid if uid else f"{episode_id}:role{role}"
            return self.pseudonym("participant", key)

        # ordered event list for the whole dialogue
        events: list[dict] = []
        for k, t in enumerate(turns):
            role = t.get("role")
            events.append(history_event(
                k, actor_of(role), "message", text=t.get("text", ""),
                meta={"role": role_name(role), "turn": t.get("turn")}))

        persuader_uid = user_id(0)
        persuadee_uid = user_id(1)
        participant_ids = [actor_of(0)] + ([actor_of(1)] if info.get("1") is not None or len(turns) else [])
        # de-dup while keeping order
        participant_ids = list(dict.fromkeys(participant_ids))
        persistent = bool(persuader_uid or persuadee_uid)
        loc = {"files": ["data/FullData/full_dialog.csv", "data/FullData/full_info.csv"],
               "indices": [di], "ids": [episode_id]}
        world = {"charity": _CHARITY,
                 "task": "persuader convinces persuadee to donate part of task earnings"}
        n = len(turns)

        # ---- PREDICT_NEXT_MESSAGE -------------------------------------------------------
        for k, t in enumerate(turns):
            role = t.get("role")
            rname = role_name(role)
            actor = actor_of(role)
            hist = history_before(events, k)
            obs = observation_at(events, k)
            goal = self._private_goal(rname)
            ctx = {
                "actor_profile": self._profile(info.get(str(int(float(role)))) if role is not None else {}),
                "private_state_before": goal,
                "known_history": hist,
                "current_observation": obs,
                "world_state": world,
                "available_actions": None,
                "language": "en",
            }
            payload = {
                "input": {"dialogue_history": hist, "private_goal": goal, "current_observation": obs},
                "target": {"message_text": t.get("text", ""), "dialogue_act": None, "strategy": None},
            }
            yield self.make(
                task_type="PREDICT_NEXT_MESSAGE", payload=payload, episode_id=episode_id,
                sequence_index=k, cutoff_sequence_index=k, participant_ids=participant_ids,
                actor_id=actor, actor_role=rname, persistent_identity_available=persistent,
                context=ctx, raw_locator=loc,
                transformation_steps=["group full_dialog.csv by dialogue", "order turns",
                                      f"cutoff before turn {k}"],
                data_quality={"missing_fields": ["utterance_strategy_annotation", "dialogue_act",
                                                 "timestamps"],
                              "chronology_verified": True, "target_verified": True,
                              "license_verified": True, "confidence": "high",
                              "inferred_fields": ["sequence_index"]})

        # ---- PREDICT_RESPONSE_OR_NONRESPONSE (persuadee reply to a persuader turn) -------
        for k, t in enumerate(turns):
            if role_name(t.get("role")) != "persuader":
                continue
            nxt = turns[k + 1] if k + 1 < n else None
            responded = bool(nxt is not None and role_name(nxt.get("role")) == "persuadee")
            hist = history_before(events, k + 1)   # includes the persuader turn at k
            obs = observation_at(events, k + 1)     # = the persuader turn at k
            resp_actor = actor_of(1)
            payload = {
                "input": {"history": hist, "observation": obs},
                "target": {"responded": responded, "latency_seconds": None},
            }
            yield self.make(
                task_type="PREDICT_RESPONSE_OR_NONRESPONSE", payload=payload, episode_id=episode_id,
                sequence_index=k + 1, cutoff_sequence_index=k + 1, participant_ids=participant_ids,
                actor_id=resp_actor, actor_role="persuadee", persistent_identity_available=persistent,
                context={"known_history": hist, "current_observation": obs, "world_state": world,
                         "available_actions": None, "language": "en"},
                raw_locator=loc,
                transformation_steps=["order turns", f"expose persuader turn {k}",
                                      "target = whether persuadee replies at turn k+1"],
                data_quality={"missing_fields": ["timestamps", "latency_seconds"],
                              "chronology_verified": True, "target_verified": True,
                              "license_verified": True, "confidence": "high"})

        # ---- PREDICT_TRAJECTORY_CONTINUATION --------------------------------------------
        for k in range(1, n):
            cont = events[k:k + _TRAJ_HORIZON]
            if not cont:
                continue
            hist = history_before(events, k)
            payload = {
                "input": {"history": hist, "horizon": len(cont)},
                "target": {"continuation": cont},
            }
            yield self.make(
                task_type="PREDICT_TRAJECTORY_CONTINUATION", payload=payload, episode_id=episode_id,
                sequence_index=k, cutoff_sequence_index=k, participant_ids=participant_ids,
                actor_role="dialogue", persistent_identity_available=persistent,
                context={"known_history": hist, "world_state": world, "available_actions": None,
                         "language": "en"},
                raw_locator=loc,
                transformation_steps=["order turns", f"cutoff before turn {k}",
                                      f"target = next {len(cont)} turns"],
                data_quality={"missing_fields": ["timestamps"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high"})

        # ---- PREDICT_FINAL_OUTCOME (actual donation, post-task) --------------------------
        ee_info = info.get("1") or {}
        er_info = info.get("0") or {}
        ee_don = _num(ee_info.get("B6"))
        er_don = _num(er_info.get("B6"))
        n_turns = _num(ee_info.get("B7")) or _num(er_info.get("B7"))
        missing = ["timestamps"]
        if ee_don is None:
            missing.append("persuadee_donation")
        outcome = {
            "persuadee_donated": (ee_don > 0) if ee_don is not None else None,
            "persuadee_donation_amount": ee_don,
            "persuader_donation_amount": er_don,
            "charity": _CHARITY,
            "num_turns": n_turns,
        }
        hist = list(events)
        payload = {
            "input": {"history": hist, "state": world},
            "target": {"outcome": outcome, "outcome_type": "donation"},
        }
        yield self.make(
            task_type="PREDICT_FINAL_OUTCOME", payload=payload, episode_id=episode_id,
            sequence_index=n, cutoff_sequence_index=n, participant_ids=participant_ids,
            actor_role="persuadee", persistent_identity_available=persistent,
            context={"known_history": hist, "world_state": world, "available_actions": None,
                     "language": "en"},
            raw_locator=loc,
            transformation_steps=["expose full dialogue",
                                  "target = actual post-task donation (full_info.csv B6)"],
            data_quality={"missing_fields": missing, "chronology_verified": True,
                          "target_verified": True, "license_verified": True,
                          "confidence": "high",
                          "warnings": ["donation is a private post-task self-report (never appears in dialogue)"]})
