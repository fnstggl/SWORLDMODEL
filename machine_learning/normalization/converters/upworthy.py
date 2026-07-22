"""Upworthy Research Archive — randomized headline/image A/B "package" experiments.

Source is NOT downloaded here. This converter is written against the DOCUMENTED CSV schema
(Upworthy Research Archive) and is exercised on a committed fixture. Rows are per-"package"
(one headline/image arm) within an experiment; arms sharing a clickability_test_id are the
randomized variants of one test. Documented columns:
  clickability_test_id (experiment id), headline, lede, slug, eyecatcher_id (image id),
  impressions, clicks, created_at, test_week, first_place, winner, share_text, excerpt.

Emits (all built PER experiment; arms of one test are NEVER split across train/test):
  PREDICT_POPULATION_RESPONSE — per arm: aggregate_metrics = {rate: clicks/impressions,
      n: impressions, successes: clicks}.
  PREDICT_INTERVENTION_EFFECT — within one experiment, each non-baseline arm (treatment)
      vs the baseline arm (control), using their observed CTRs. Legitimate because the arms
      are randomized within the test. Arms across DIFFERENT experiments are never compared.
  RANK_CANDIDATE_ACTIONS — rank the headline arms of ONE experiment by observed CTR;
      chosen_id = the highest-CTR arm.
  PREDICT_POLICY_VALUE — reward = an arm's CTR.

Honesty: the documented randomization-problem window (2013-06-25..2014-01-10) is flagged
in data_quality.warnings on every record from an affected experiment (exclude from
confirmatory analysis). CTR is null when impressions=0. Whole experiments are held together
via group_id/experiment_id = clickability_test_id (split_unit=experiment).
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter

#: Documented window in which Upworthy's randomization was known to be compromised.
_RAND_PROBLEM_START = "2013-06-25"
_RAND_PROBLEM_END = "2014-01-10"


def _load_rows(raw_dir: Path) -> list[dict]:
    """Load Upworthy package rows from CSV (real format) or a JSON fixture (list of dicts)."""
    csv_files = [f for f in glob.glob(str(raw_dir / "**" / "*.csv"), recursive=True) if ".cache" not in f]
    if csv_files:
        rows: list[dict] = []
        for f in sorted(csv_files):
            with open(f, newline="", encoding="utf-8") as fh:
                rows.extend(dict(r) for r in csv.DictReader(fh))
        return rows
    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"no Upworthy csv/json found under {raw_dir}")


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _ctr(clicks, impressions):
    c, n = _i(clicks), _i(impressions)
    if not n:
        return None
    return (c or 0) / n


class Converter(BaseConverter):
    DATASET_ID = "upworthy"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "upworthy"
    DOC = {
        "dataset_id": "upworthy",
        "original_fields": [
            {"name": "clickability_test_id", "meaning": "experiment id; arms sharing it are one randomized test"},
            {"name": "headline", "meaning": "the arm's headline text (the treatment content)"},
            {"name": "lede/excerpt/share_text", "meaning": "further arm copy"},
            {"name": "slug", "meaning": "per-package identifier (used as arm id)"},
            {"name": "eyecatcher_id", "meaning": "image variant id"},
            {"name": "impressions", "meaning": "n shown for this arm"},
            {"name": "clicks", "meaning": "n clicks for this arm"},
            {"name": "created_at", "meaning": "arm creation timestamp (used for the randomization-problem window)"},
            {"name": "first_place/winner", "meaning": "arm-level flags recorded by Upworthy"},
        ],
        "canonical_mapping": [
            {"source_field": "clickability_test_id", "canonical_path": "episode.experiment_id / episode.group_id"},
            {"source_field": "headline/lede/excerpt/eyecatcher_id", "canonical_path": "payload.input candidates/intervention content"},
            {"source_field": "clicks/impressions", "canonical_path": "payload.target.aggregate_metrics / treated_or_control_outcome / relevance / reward (CTR)"},
            {"source_field": "created_at", "canonical_path": "episode.start_time + randomization-window warning"},
        ],
        "tasks_produced": ["PREDICT_POPULATION_RESPONSE", "PREDICT_INTERVENTION_EFFECT",
                           "RANK_CANDIDATE_ACTIONS", "PREDICT_POLICY_VALUE"],
        "unavailable_fields": [
            "per-impression features / user identity (data is arm-level aggregate counts only)",
            "CTR when impressions=0 (null, not fabricated)",
        ],
        "chronology_rules": "The outcome (clicks/impressions/CTR) is the label and lives only in payload.target; inputs carry only pre-outcome arm content (headline, lede, image id). Intervention/ranking examples are built strictly within a single experiment.",
        "split_key": "experiment (group_id=experiment_id=clickability_test_id); whole experiments held together",
        "leakage_risks": [
            "arms of one experiment must never be split across train/test (kept together via experiment_id/group_id)",
            "clicks/impressions/CTR must not appear in any input (they are the outcome)",
            "do NOT compare arms across different experiments as treatment/control",
        ],
        "known_limitations": [
            "written against the DOCUMENTED schema (not a downloaded sample); reconcile exact columns on real acquisition",
            "no designated control arm — baseline chosen as the first arm in stable row order",
            "randomization-problem window 2013-06-25..2014-01-10 flagged; exclude from confirmatory splits",
        ],
        "license_implications": "CC-BY-4.0: training + commercial use permitted with attribution.",
        "training_suitability": "train",
        "assumptions": [
            "columns clickability_test_id, headline, lede, slug, eyecatcher_id, impressions, clicks, created_at, test_week, first_place, winner, share_text, excerpt",
            "arm id = slug (fallback: within-experiment index)",
            "created_at is a parseable date; first 10 chars = YYYY-MM-DD",
        ],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        rows = _load_rows(raw_dir)
        if not rows:
            raise FileNotFoundError(f"Upworthy source produced 0 rows under {raw_dir}")
        by_test: dict[str, list[dict]] = {}
        for r in rows:
            by_test.setdefault(str(r.get("clickability_test_id")), []).append(r)
        for test_id, arm_rows in by_test.items():
            yield from self._one_experiment(test_id, arm_rows)

    def _one_experiment(self, test_id: str, arm_rows: list[dict]) -> Iterator[dict]:
        experiment_id = f"upworthy-{test_id}"
        arms = []
        for idx, row in enumerate(arm_rows):
            arm_id = str(row.get("slug") or f"arm{idx}")
            arms.append((arm_id, row))
        created = next((str(r.get("created_at")) for _, r in arms if r.get("created_at")), None)
        in_bad_window = bool(created) and (_RAND_PROBLEM_START <= created[:10] <= _RAND_PROBLEM_END)
        window_warn = ([f"created_at {created[:10]} falls in the documented randomization-problem "
                        f"window ({_RAND_PROBLEM_START}..{_RAND_PROBLEM_END}); exclude from confirmatory analysis"]
                       if in_bad_window else [])

        def _content(arm_id, row):
            return {"arm_id": arm_id, "headline": row.get("headline"), "lede": row.get("lede"),
                    "excerpt": row.get("excerpt"), "eyecatcher_id": row.get("eyecatcher_id")}

        def _causal(treatment_arm, control_arm):
            return {"is_experimental": True, "randomized": True, "assignment_mechanism": "rct",
                    "treatment_arm": treatment_arm, "control_arm": control_arm,
                    "propensity": None, "unit_of_assignment": "impression"}

        loc = {"files": ["upworthy-archive.csv"], "indices": list(range(len(arms))),
               "ids": [experiment_id]}

        # ---- POPULATION_RESPONSE + POLICY_VALUE per arm ---------------------------------
        for arm_id, row in arms:
            n, c = _i(row.get("impressions")), _i(row.get("clicks"))
            ctr = _ctr(c, n)
            dq = {"missing_fields": ([] if ctr is not None else ["ctr(impressions=0)"]),
                  "chronology_verified": True, "target_verified": True, "license_verified": True,
                  "confidence": "high", "warnings": window_warn}
            yield self.make(
                task_type="PREDICT_POPULATION_RESPONSE",
                payload={"input": {"population_features": {"platform": "upworthy"},
                                   "intervention": _content(arm_id, row)},
                         "target": {"aggregate_metrics": {"rate": ctr, "n": n, "successes": c}}},
                episode_id=f"{experiment_id}-arm-{arm_id}-pop", experiment_id=experiment_id,
                group_id=experiment_id, actor_role="population", population_id=f"{experiment_id}-{arm_id}",
                start_time=created, causal_metadata=_causal(arm_id, None),
                context={"world_state": {"headline": row.get("headline"), "eyecatcher_id": row.get("eyecatcher_id")},
                         "available_actions": None, "language": "en"},
                raw_locator=loc,
                transformation_steps=["group by clickability_test_id", "per-arm CTR = clicks/impressions"],
                data_quality=dq)

            yield self.make(
                task_type="PREDICT_POLICY_VALUE",
                payload={"input": {"logged_context": {"experiment_id": experiment_id},
                                   "action": _content(arm_id, row),
                                   "propensity": None, "logging_policy": "upworthy_ab_test"},
                         "target": {"reward": ctr if ctr is not None else 0, "value": None}},
                episode_id=f"{experiment_id}-arm-{arm_id}-pv", experiment_id=experiment_id,
                group_id=experiment_id, actor_role="population", population_id=f"{experiment_id}-{arm_id}",
                start_time=created, causal_metadata=_causal(arm_id, None),
                context={"world_state": {"headline": row.get("headline")}, "available_actions": None, "language": "en"},
                raw_locator=loc,
                transformation_steps=["reward = arm CTR"],
                data_quality={"missing_fields": ([] if ctr is not None else ["ctr(impressions=0)"]),
                              "chronology_verified": True, "target_verified": True, "license_verified": True,
                              "confidence": "high", "warnings": window_warn})

        # ---- INTERVENTION_EFFECT: each non-baseline arm vs the baseline arm -------------
        if len(arms) >= 2:
            base_id, base_row = arms[0]
            base_ctr = _ctr(base_row.get("clicks"), base_row.get("impressions"))
            control_outcome = {"observed": True, "arm": base_id, "ctr": base_ctr,
                               "clicks": _i(base_row.get("clicks")), "impressions": _i(base_row.get("impressions"))}
            for arm_id, row in arms[1:]:
                t_ctr = _ctr(row.get("clicks"), row.get("impressions"))
                treated_outcome = {"observed": True, "arm": arm_id, "ctr": t_ctr,
                                   "clicks": _i(row.get("clicks")), "impressions": _i(row.get("impressions"))}
                eff = {"ctr_difference": (t_ctr - base_ctr) if (t_ctr is not None and base_ctr is not None) else None,
                       "metric": "ctr"}
                yield self.make(
                    task_type="PREDICT_INTERVENTION_EFFECT",
                    payload={"input": {
                        "population_or_actor_features": {"platform": "upworthy", "experiment_id": experiment_id},
                        "treatment": _content(arm_id, row),
                        "control": _content(base_id, base_row),
                        "assignment_mechanism": {"type": "rct", "randomized": True, "unit": "impression",
                                                 "note": "arms randomized within one clickability test"}},
                        "target": {"treated_outcome": treated_outcome, "control_outcome": control_outcome,
                                   "estimated_effect": eff}},
                    episode_id=f"{experiment_id}-ie-{arm_id}-vs-{base_id}", experiment_id=experiment_id,
                    group_id=experiment_id, actor_role="population", population_id=experiment_id,
                    start_time=created, causal_metadata=_causal(arm_id, base_id),
                    context={"world_state": {"experiment_id": experiment_id}, "available_actions": None, "language": "en"},
                    raw_locator=loc,
                    transformation_steps=["within experiment: arm vs baseline arm",
                                          "both CTRs observed (arms randomized within the test)"],
                    data_quality={"missing_fields": [], "chronology_verified": True, "target_verified": True,
                                  "license_verified": True, "confidence": "high",
                                  "warnings": (window_warn + ["baseline arm = first arm in stable row order (no designated control)"])})

        # ---- RANK_CANDIDATE_ACTIONS: rank the experiment's arms by observed CTR ----------
        if len(arms) >= 2:
            relevance = {arm_id: _ctr(row.get("clicks"), row.get("impressions")) for arm_id, row in arms}
            ranked = sorted(arms, key=lambda ar: (relevance[ar[0]] is None, -(relevance[ar[0]] or 0.0)))
            chosen_id = ranked[0][0]
            candidates = [_content(arm_id, row) for arm_id, row in arms]
            yield self.make(
                task_type="RANK_CANDIDATE_ACTIONS",
                payload={"input": {"context": {"experiment_id": experiment_id, "platform": "upworthy"},
                                   "candidates": candidates},
                         "target": {"chosen_id": chosen_id,
                                    "ranking": [arm_id for arm_id, _ in ranked],
                                    "relevance": relevance}},
                episode_id=f"{experiment_id}-rank", experiment_id=experiment_id, group_id=experiment_id,
                actor_role="population", population_id=experiment_id, start_time=created,
                causal_metadata=_causal(chosen_id, None),
                context={"world_state": {"experiment_id": experiment_id}, "available_actions": None, "language": "en"},
                raw_locator=loc,
                transformation_steps=["rank arms of ONE experiment by observed CTR", "chosen = highest CTR"],
                data_quality={"missing_fields": [], "chronology_verified": True, "target_verified": True,
                              "license_verified": True, "confidence": "high",
                              "warnings": (window_warn + ["ranking is by observed CTR within a single experiment only"])})
