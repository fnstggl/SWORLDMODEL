"""Open Bandit Dataset (ZOZO) — logged bandit feedback with true propensities.

Real source (verified from the in-repo obd/ sample): CSV per {behavior_policy}/{campaign}::
  columns: (index), timestamp, item_id, position, click, propensity_score,
           user_feature_0..3, user-item_affinity_0..N
behavior_policy ∈ {random (Uniform Random), bts (Bernoulli Thompson Sampling)};
campaign ∈ {all, men, women}. item_context.csv holds per-item features.

Emits:
  PREDICT_POLICY_VALUE     — reward (click) of a logged (context, action) under a known
                             logging policy + propensity (enables IPS/DR off-policy eval)
  PREDICT_NEXT_ACTION      — the item shown (under uniform-random this is unbiased)
  RANK_CANDIDATE_ACTIONS   — chosen item as the positive among the campaign's item set

Honesty: OBD is logged bandit feedback, NOT an A/B treatment/control design, so
PREDICT_INTERVENTION_EFFECT is NOT produced (no matched treated/control outcomes to
compare). causal_metadata records the randomized logging policy + propensity instead.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter

_POLICIES = {"random": ("uniform_random", True), "bts": ("bernoulli_ts", False)}


class Converter(BaseConverter):
    DATASET_ID = "open_bandit"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "open_bandit"
    DOC = {
        "dataset_id": "open_bandit",
        "original_fields": [
            {"name": "item_id", "meaning": "recommended item (action)"},
            {"name": "position", "meaning": "slot the item was shown in"},
            {"name": "click", "meaning": "reward (0/1)"},
            {"name": "propensity_score", "meaning": "logging-policy probability of this action"},
            {"name": "user_feature_0..3", "meaning": "user context"},
            {"name": "user-item_affinity_*", "meaning": "user-item affinity features"},
        ],
        "canonical_mapping": [
            {"source_field": "click", "canonical_path": "payload.target.reward"},
            {"source_field": "propensity_score", "canonical_path": "payload.input.propensity / causal_metadata.propensity"},
            {"source_field": "item_id", "canonical_path": "payload.input.action / payload.target.action_type"},
            {"source_field": "behavior_policy (dir)", "canonical_path": "causal_metadata.logging_policy"},
        ],
        "tasks_produced": ["PREDICT_POLICY_VALUE", "PREDICT_NEXT_ACTION", "RANK_CANDIDATE_ACTIONS"],
        "unavailable_fields": ["matched treatment/control outcomes", "user identity across sessions"],
        "chronology_rules": "Each row is an independent logged decision; context precedes the click.",
        "split_key": "session (per row; time-ordered)",
        "leakage_risks": ["randomized-exposure (random policy) rows kept in causal_metadata.randomized=true and MUST NOT be mixed with bts as if identical"],
        "known_limitations": ["in-repo sample is 10k rows/policy/campaign; full set ~26M rows on research.zozo.com"],
        "license_implications": "In-repo sample Apache-2.0 (train ok). Full-data commercial terms unclear (citation-only).",
        "training_suitability": "train",
        "assumptions": ["dir layout {policy}/{campaign}/{campaign}.csv"],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        found = False
        for policy_dir in sorted(glob.glob(str(raw_dir / "*"))):
            policy = Path(policy_dir).name
            if policy not in _POLICIES:
                continue
            logging_policy, randomized = _POLICIES[policy]
            for camp_dir in sorted(glob.glob(str(Path(policy_dir) / "*"))):
                campaign = Path(camp_dir).name
                csvs = [f for f in glob.glob(str(Path(camp_dir) / "*.csv"))
                        if "item_context" not in Path(f).name]
                for fpath in sorted(csvs):
                    found = True
                    yield from self._one_file(fpath, policy, campaign, logging_policy, randomized)
        if not found:
            raise FileNotFoundError(f"no Open Bandit policy/campaign CSVs under {raw_dir}")

    def _one_file(self, fpath: str, policy: str, campaign: str, logging_policy: str,
                  randomized: bool) -> Iterator[dict]:
        rel = str(Path(fpath).relative_to(Path(fpath).parents[2])) if len(Path(fpath).parents) >= 3 else Path(fpath).name
        with open(fpath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for ri, row in enumerate(reader):
                user_feats = {k: row[k] for k in row if k.startswith("user_feature")}
                affinity = {k: _f(row[k]) for k in row if k.startswith("user-item_affinity")}
                item_id = row.get("item_id", "")
                click = _i(row.get("click", "0"))
                pscore = _f(row.get("propensity_score"))
                position = _i(row.get("position", "0"))
                episode_id = f"open_bandit-{policy}-{campaign}-{ri}"
                loc = {"files": [rel], "indices": [ri], "ids": [episode_id]}
                causal = {"is_experimental": True, "randomized": randomized,
                          "assignment_mechanism": logging_policy, "propensity": pscore,
                          "logging_policy": logging_policy, "unit_of_assignment": "impression"}
                ctx = {"actor_profile": user_feats, "current_observation": {"kind": "state",
                       "meta": {"campaign": campaign, "position": position}},
                       "world_state": {"campaign": campaign, "affinity": affinity},
                       "available_actions": None, "language": ""}

                # POLICY_VALUE
                yield self.make(
                    task_type="PREDICT_POLICY_VALUE",
                    payload={"input": {"logged_context": user_feats, "action": {"item_id": item_id, "position": position},
                                       "propensity": pscore, "logging_policy": logging_policy},
                             "target": {"reward": click, "value": None}},
                    episode_id=episode_id, sequence_index=0, actor_role="population",
                    population_id=f"zozo-{campaign}", context=ctx, causal_metadata=causal,
                    raw_locator=loc, dataset_version=policy,
                    transformation_steps=["read csv row", "attach propensity + logging policy"],
                    data_quality={"missing_fields": [], "chronology_verified": True,
                                  "target_verified": True, "license_verified": True, "confidence": "high"})
                # NEXT_ACTION (item shown)
                yield self.make(
                    task_type="PREDICT_NEXT_ACTION",
                    payload={"input": {"history": [], "observation": ctx["current_observation"],
                                       "available_actions": None},
                             "target": {"action_type": "recommend_item", "acted": True,
                                        "action_content": {"item_id": item_id, "position": position}}},
                    episode_id=episode_id, sequence_index=0, actor_role="population",
                    population_id=f"zozo-{campaign}", context=ctx, causal_metadata=causal,
                    raw_locator=loc, dataset_version=policy,
                    transformation_steps=["read csv row", "action = shown item"],
                    data_quality={"missing_fields": ["available_action_set"], "chronology_verified": True,
                                  "target_verified": True, "license_verified": True, "confidence": "high",
                                  "warnings": ["under bts the action is policy-biased (see causal_metadata)"]})


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None
