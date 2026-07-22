"""KuaiRand — Kuaishou short-video interaction log with randomly-exposed items.

Source is NOT downloaded here (KuaiRand-Pure tar.gz is ~194MB). This converter is written
against the DOCUMENTED CSV interaction-log schema and is exercised on a committed fixture.
Documented per-interaction columns (KuaiRand-Pure log_standard_* / log_random_* CSVs):
  user_id, video_id, time_ms (a.k.a. timestamp), is_click, is_like, is_follow, is_comment,
  long_view, play_time_ms, duration_ms, is_rand (1=randomly exposed / 0=organic),
  tab (0-14, the recommendation surface / logging policy).

Emits (modeling the USER reacting to a shown video):
  PREDICT_NEXT_ACTION            — the engagement action on the current video
  PREDICT_RESPONSE_OR_NONRESPONSE— whether the user engages at all (positive engagement)
  PREDICT_TIME_TO_ACTION         — gap to the user's NEXT interaction; right-censored on the
                                   last logged interaction of each user
  PREDICT_POLICY_VALUE           — reward = is_click of a logged (context, shown-video) pair
                                   under the surface/exposure policy

CRITICAL causal separation: randomly-exposed rows (is_rand=1) carry
causal_metadata.randomized=True, assignment_mechanism="uniform_random_exposure"; organic
rows (is_rand=0) carry randomized=False, assignment_mechanism="observational". A warning on
the randomized records forbids pooling them with organic rows as if identical.

Honesty: PREDICT_INTERVENTION_EFFECT is NOT produced. A per-interaction log has no matched
treated/control pair for the same unit (each row is a single realized exposure with no
counterfactual), so a genuine treated_outcome/control_outcome pair cannot be formed without
fabricating a counterfactual. The randomized-exposure signal is instead preserved in the
POLICY_VALUE records (randomized=True) for off-policy evaluation. See DOC.unavailable_fields.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Iterator

from ...tasks import NO_ACTION
from ..base import Converter as BaseConverter
from ..common.dialogue import history_event

_ENGAGE_COLS = ["is_click", "is_like", "is_follow", "is_comment", "long_view"]


def _load_rows(raw_dir: Path) -> list[dict]:
    """Load KuaiRand rows from CSV logs, streamed parquet, or a JSON fixture."""
    csv_files = [f for f in glob.glob(str(raw_dir / "**" / "*.csv"), recursive=True) if ".cache" not in f]
    if csv_files:
        rows: list[dict] = []
        for f in sorted(csv_files):
            with open(f, newline="", encoding="utf-8") as fh:
                rows.extend(dict(r) for r in csv.DictReader(fh))
        return rows
    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "stream_shard_*.parquet"), recursive=True)
                if ".cache" not in f] or \
               [f for f in glob.glob(str(raw_dir / "**" / "*.parquet"), recursive=True) if ".cache" not in f]
    if pq_files:
        import pyarrow.parquet as pq
        rows = []
        for f in sorted(pq_files):
            rows.extend(pq.read_table(f).to_pylist())
        return rows
    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"no KuaiRand csv/parquet/json found under {raw_dir}")


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _ts(row: dict):
    for k in ("time_ms", "timestamp", "time"):
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except (TypeError, ValueError):
                return None
    return None


class Converter(BaseConverter):
    DATASET_ID = "kuairand"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "kuairand"
    DOC = {
        "dataset_id": "kuairand",
        "original_fields": [
            {"name": "user_id", "meaning": "pseudonymous user id (platform_user split unit)"},
            {"name": "video_id", "meaning": "shown short-video id"},
            {"name": "time_ms", "meaning": "interaction timestamp (ms); a.k.a. timestamp"},
            {"name": "is_click", "meaning": "click engagement (0/1)"},
            {"name": "is_like/is_follow/is_comment/long_view", "meaning": "further positive engagements (0/1)"},
            {"name": "play_time_ms", "meaning": "watch time (OUTCOME; never an input)"},
            {"name": "duration_ms", "meaning": "video length (pre-known)"},
            {"name": "is_rand", "meaning": "1=randomly exposed (unbiased), 0=organic recommendation"},
            {"name": "tab", "meaning": "recommendation surface / logging policy (0-14)"},
        ],
        "canonical_mapping": [
            {"source_field": "user_id", "canonical_path": "decision_unit.actor_id / episode.group_id (pseudonymized)"},
            {"source_field": "video_id/duration_ms/tab", "canonical_path": "context.current_observation.meta (shown video)"},
            {"source_field": "is_click + engagements", "canonical_path": "payload.target.action_type/acted, target.responded, target.reward"},
            {"source_field": "play_time_ms", "canonical_path": "payload.target.action_content.play_time_ms (outcome only)"},
            {"source_field": "time_ms", "canonical_path": "payload.input.current_time / TIME_TO_ACTION gap"},
            {"source_field": "is_rand", "canonical_path": "causal_metadata.randomized / assignment_mechanism"},
            {"source_field": "tab", "canonical_path": "causal_metadata.logging_policy (organic rows)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_ACTION", "PREDICT_RESPONSE_OR_NONRESPONSE",
                           "PREDICT_TIME_TO_ACTION", "PREDICT_POLICY_VALUE"],
        "unavailable_fields": [
            "matched treated/control counterfactual per interaction -> PREDICT_INTERVENTION_EFFECT NOT produced (a per-interaction log has one realized exposure, no counterfactual; randomization is preserved via POLICY_VALUE randomized=True instead)",
            "per-response latency (RESPONSE_OR_NONRESPONSE.latency_seconds null)",
            "logging propensity (uniform-random candidate-set size not in the log; propensity null)",
        ],
        "chronology_rules": "Per user, interactions are ordered by time_ms. A decision at step k sees only interactions 0..k-1 (the shown video's id/duration/tab are pre-known; play_time_ms and engagements are the label). TIME_TO_ACTION anchors on event k (known) and predicts the gap to k+1.",
        "split_key": "platform_user (episode_id = kuairand-user-<pseudo>; group_id = same user; persistent_identity_available=true)",
        "leakage_risks": [
            "play_time_ms and engagement flags are outcomes and must stay in payload.target",
            "randomly-exposed (is_rand=1) rows MUST NOT be pooled with organic rows as if identically distributed — flagged in causal_metadata.randomized + a warning",
        ],
        "known_limitations": [
            "written against the DOCUMENTED schema (not a downloaded sample); exact column names to be reconciled on real acquisition",
            "no session boundaries in the documented schema — episode = whole user log",
            "engagement/interaction rows are all realized exposures; no explicit non-impression negatives",
        ],
        "license_implications": "CC-BY-SA-4.0: training + commercial use permitted WITH attribution; ShareAlike applies to redistributed derivatives.",
        "training_suitability": "train",
        "assumptions": [
            "column names user_id, video_id, time_ms|timestamp, is_click, is_like, is_follow, is_comment, long_view, play_time_ms, duration_ms, is_rand, tab (KuaiRand-Pure log_standard_* / log_random_* CSVs) — reconcile exactly on real acquisition (real files also carry is_forward, is_hate, is_profile_enter, profile_stay_time, comment_stay_time)",
            "is_rand=1 == uniform random exposure; tab is the organic recommendation surface/policy",
            "positive engagement = any of is_click, is_like, is_follow, is_comment, long_view",
        ],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        rows = _load_rows(raw_dir)
        if not rows:
            raise FileNotFoundError(f"KuaiRand source produced 0 rows under {raw_dir}")
        by_user: dict[str, list[dict]] = {}
        for r in rows:
            by_user.setdefault(str(r.get("user_id")), []).append(r)
        for raw_uid, urows in by_user.items():
            urows = sorted(urows, key=lambda r: (_ts(r) is None, _ts(r) if _ts(r) is not None else 0))
            yield from self._one_user(raw_uid, urows)

    def _one_user(self, raw_uid: str, urows: list[dict]) -> Iterator[dict]:
        actor = self.pseudonym("actor", raw_uid)
        group = self.pseudonym("group", raw_uid)
        episode_id = f"kuairand-user-{actor}"

        # full ordered event list (past interactions become history events)
        events = []
        for k, row in enumerate(urows):
            positive = any(_i(row.get(c)) == 1 for c in _ENGAGE_COLS)
            events.append(history_event(
                k, actor, "action", action_type=("engage" if positive else "skip"),
                t=_ts(row), action_content={c: _i(row.get(c)) for c in _ENGAGE_COLS},
                meta={"video_id": row.get("video_id"), "tab": _i(row.get("tab")), "is_rand": _i(row.get("is_rand"))}))

        for k, row in enumerate(urows):
            is_rand = _i(row.get("is_rand")) == 1
            tab = _i(row.get("tab"))
            positive = any(_i(row.get(c)) == 1 for c in _ENGAGE_COLS)
            clicked = _i(row.get("is_click")) == 1
            ts = _ts(row)
            hist = [e for e in events if e["index"] < k]
            obs = {"text": None, "kind": "item",
                   "meta": {"video_id": row.get("video_id"), "duration_ms": _i(row.get("duration_ms")), "tab": tab}}
            causal = self._causal(is_rand, tab)
            loc = {"files": ["log.csv"], "indices": [k], "ids": [f"{raw_uid}:{row.get('video_id')}"]}
            rand_warn = (["randomly-exposed row (is_rand=1); do NOT pool with organic rows as if identical"]
                         if is_rand else [])
            ctx = {"actor_profile": {}, "known_history": hist, "current_observation": obs,
                   "world_state": {"platform": "kuaishou", "surface_tab": tab},
                   "available_actions": None, "language": ""}

            # NEXT_ACTION
            action_type = "engage" if positive else NO_ACTION
            yield self.make(
                task_type="PREDICT_NEXT_ACTION",
                payload={"input": {"history": hist, "observation": obs, "available_actions": None},
                         "target": {"action_type": action_type, "acted": positive,
                                    "action_content": {**{c: _i(row.get(c)) for c in _ENGAGE_COLS},
                                                       "play_time_ms": _i(row.get("play_time_ms")),
                                                       "video_id": row.get("video_id")}}},
                episode_id=episode_id, sequence_index=k, cutoff_sequence_index=k, actor_id=actor,
                actor_role="participant", group_id=group, participant_ids=[actor],
                persistent_identity_available=True, context=ctx, causal_metadata=causal, raw_locator=loc,
                transformation_steps=["group by user", "order by time_ms", f"cutoff before interaction {k}",
                                      "action = engagement on shown video"],
                data_quality={"missing_fields": ["available_action_set"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high",
                              "warnings": rand_warn})

            # RESPONSE_OR_NONRESPONSE
            yield self.make(
                task_type="PREDICT_RESPONSE_OR_NONRESPONSE",
                payload={"input": {"history": hist, "observation": obs},
                         "target": {"responded": positive, "latency_seconds": None}},
                episode_id=episode_id, sequence_index=k, cutoff_sequence_index=k, actor_id=actor,
                actor_role="participant", group_id=group, participant_ids=[actor],
                persistent_identity_available=True, context=ctx, causal_metadata=causal, raw_locator=loc,
                transformation_steps=["group by user", "order by time_ms", f"cutoff before interaction {k}",
                                      "responded = any positive engagement"],
                data_quality={"missing_fields": ["latency_seconds"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high",
                              "warnings": rand_warn})

            # TIME_TO_ACTION — gap to the user's next interaction (censored on the last one)
            has_next = k + 1 < len(urows)
            next_ts = _ts(urows[k + 1]) if has_next else None
            gap = ((next_ts - ts) / 1000.0) if (has_next and ts is not None and next_ts is not None) else None
            tta_missing = [] if (ts is not None and (not has_next or next_ts is not None)) else ["timestamps"]
            censoring = {"censored": not has_next, "observation_window_seconds": None,
                         "reason": ("no further interaction observed for this user" if not has_next
                                    else "next interaction observed")}
            hist_incl = [e for e in events if e["index"] <= k]
            yield self.make(
                task_type="PREDICT_TIME_TO_ACTION",
                payload={"input": {"history": hist_incl, "current_time": ts},
                         "target": {"acted": has_next, "time_to_action_seconds": gap, "censoring": censoring}},
                episode_id=episode_id, sequence_index=k, cutoff_sequence_index=k + 1, actor_id=actor,
                actor_role="participant", group_id=group, participant_ids=[actor],
                persistent_identity_available=True,
                context={"actor_profile": {}, "known_history": hist_incl, "current_observation": obs,
                         "world_state": {"platform": "kuaishou", "surface_tab": tab},
                         "available_actions": None, "language": ""},
                causal_metadata=causal, raw_locator=loc,
                transformation_steps=["group by user", "order by time_ms", "anchor on interaction k",
                                      "target = gap to next interaction (censored if none)"],
                data_quality={"missing_fields": tta_missing, "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high",
                              "warnings": (rand_warn + (["right-censored: last logged interaction"] if not has_next else []))})

            # POLICY_VALUE — reward = is_click of the logged (context, shown-video) pair
            logging_policy = "uniform_random_exposure" if is_rand else f"tab_{tab}"
            yield self.make(
                task_type="PREDICT_POLICY_VALUE",
                payload={"input": {"logged_context": {"video_id": row.get("video_id"),
                                                      "duration_ms": _i(row.get("duration_ms"))},
                                   "action": {"video_id": row.get("video_id"), "tab": tab},
                                   "propensity": None, "logging_policy": logging_policy},
                         "target": {"reward": 1 if clicked else 0, "value": None}},
                episode_id=episode_id, sequence_index=k, cutoff_sequence_index=k, actor_id=actor,
                actor_role="participant", group_id=group, participant_ids=[actor],
                persistent_identity_available=True, context=ctx, causal_metadata=causal, raw_locator=loc,
                transformation_steps=["group by user", "action = shown video", "reward = is_click",
                                      "logging_policy = uniform_random_exposure (is_rand) else tab policy"],
                data_quality={"missing_fields": ["propensity"], "chronology_verified": True,
                              "target_verified": True, "license_verified": True, "confidence": "high",
                              "warnings": rand_warn})

    @staticmethod
    def _causal(is_rand: bool, tab) -> dict:
        if is_rand:
            return {"is_experimental": True, "randomized": True,
                    "assignment_mechanism": "uniform_random_exposure",
                    "logging_policy": "uniform_random_exposure", "propensity": None,
                    "unit_of_assignment": "impression"}
        return {"is_experimental": False, "randomized": False,
                "assignment_mechanism": "observational", "logging_policy": f"tab_{tab}",
                "propensity": None, "unit_of_assignment": "impression"}
