"""SocSci210 (socratesft/SocSci210) — persona-conditioned survey responses across
social-science studies and experimental conditions.

Real source (verified from the streamed parquet shard):
  row = {sample_id: int, participant: int, demographic: dict, stimuli: str, response: int,
         condition_num: int, task_num: int, prompt: str, reasoning: str, study_id: str}
  * prompt     — full instruction: a persona profile ("You are a survey respondent ...")
                 followed by the stimulus/question and response format.
  * stimuli    — the question shown (the persona-free part of the prompt).
  * response   — the answer given (an integer; e.g. a 1-7 Likert rating).
  * condition_num / task_num — experimental condition and task within a study.
  * demographic — the respondent profile dict (age, gender, ideology, ...).
  * reasoning  — a self-report rationale generated alongside the response.
  * study_id   — the source study.

Emits:
  PREDICT_NEXT_CHOICE          — one per row: observation = prompt; target.choice = response.
  PREDICT_POPULATION_RESPONSE  — per (study_id, condition_num): response distribution + mean + n.
  PREDICT_INTERVENTION_EFFECT  — per study, each non-baseline condition vs the baseline
                                 (lowest condition_num) condition: treated vs control mean.

LICENSE (critical): SocSci210 declares NO license on its official card -> treated as
LICENSE_RESTRICTED_EVAL_ONLY. Every record sets data_quality.license_verified=False and
DOC.training_suitability="eval_only".

Honesty notes:
  * `reasoning` is a self-report generated with the response; it is NEVER placed in the
    model input (would leak the target) and NEVER treated as ground-truth private state.
    It is stored under payload.target.meta.self_report_reasoning (label-side) with a warning.
  * The prompt is a persona-SIMULATION instruction; whether `response` is a real human
    answer or a model-simulated one is not determinable from the streamed columns (flagged).
  * available_actions is recovered ONLY when the stimulus explicitly states an integer range
    ("integer from A to B"); otherwise None (never guessed).
  * PREDICT_INTERVENTION_EFFECT is skipped for any study with a single condition in the
    streamed rows (noted). condition_num ordering does NOT encode treatment/control
    semantics — the lowest condition_num is used as the control reference by convention.
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter

_RANGE_RE = re.compile(r"integer from (\d+) to (\d+)", re.IGNORECASE)


def _load_rows(raw_dir: Path) -> list[tuple[dict, str]]:
    """Load (row, source_basename) pairs from a parquet snapshot or a JSON fixture."""
    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "*.parquet"), recursive=True)
                if ".cache" not in f]
    if pq_files:
        import pyarrow.parquet as pq
        out: list[tuple[dict, str]] = []
        for f in sorted(pq_files):
            base = Path(f).name
            for row in pq.read_table(f).to_pylist():
                out.append((row, base))
        return out
    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            base = Path(f).name
            return [(row, base) for row in data]
    raise FileNotFoundError(f"no SocSci210 parquet/json found under {raw_dir}")


def _recover_actions(stimuli: str, prompt: str) -> list | None:
    """Recover an explicit integer option set, or None. Only fires on the exact phrase."""
    for src in (stimuli or "", prompt or ""):
        m = _RANGE_RE.search(src)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 <= hi - lo <= 100:
                return list(range(lo, hi + 1))
    return None


def _stats(responses: list) -> dict:
    """Distribution + mean + n over a list of (numeric) responses."""
    n = len(responses)
    counts = Counter(responses)
    dist = {str(k): counts[k] / n for k in sorted(counts, key=lambda x: str(x))}
    nums = [r for r in responses if isinstance(r, (int, float))]
    mean = sum(nums) / len(nums) if nums else None
    return {"distribution": dist, "mean": mean, "n": n,
            "counts": {str(k): counts[k] for k in sorted(counts, key=lambda x: str(x))}}


class Converter(BaseConverter):
    DATASET_ID = "socsci210"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "socsci210"
    DOC = {
        "dataset_id": "socsci210",
        "original_fields": [
            {"name": "sample_id", "meaning": "row id (unique per streamed row)"},
            {"name": "participant", "meaning": "participant id (appears once per task within a condition)"},
            {"name": "demographic", "meaning": "respondent profile dict (age, gender, ideology, ...)"},
            {"name": "stimuli", "meaning": "the question/scenario shown, incl. response-format instruction"},
            {"name": "response", "meaning": "the answer given (integer; e.g. 1-7 Likert)"},
            {"name": "condition_num", "meaning": "experimental condition index within the study"},
            {"name": "task_num", "meaning": "task index within the study/condition"},
            {"name": "prompt", "meaning": "full persona-simulation instruction (profile + stimuli + format)"},
            {"name": "reasoning", "meaning": "self-report rationale generated alongside the response"},
            {"name": "study_id", "meaning": "source study id"},
        ],
        "canonical_mapping": [
            {"source_field": "prompt", "canonical_path": "payload.input.observation.text"},
            {"source_field": "stimuli", "canonical_path": "payload.input.observation.stimuli / context.world_state.stimuli"},
            {"source_field": "response", "canonical_path": "payload.target.choice (NEXT_CHOICE) / aggregated (POPULATION/INTERVENTION)"},
            {"source_field": "demographic", "canonical_path": "context.actor_profile"},
            {"source_field": "condition_num,task_num", "canonical_path": "context.world_state"},
            {"source_field": "study_id", "canonical_path": "episode.experiment_id (pseudonymized)"},
            {"source_field": "participant", "canonical_path": "episode.participant_ids (pseudonymized) / decision_unit.actor_id"},
            {"source_field": "reasoning", "canonical_path": "payload.target.meta.self_report_reasoning (label-side meta only; NOT input, NOT ground truth)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_CHOICE", "PREDICT_POPULATION_RESPONSE", "PREDICT_INTERVENTION_EFFECT"],
        "unavailable_fields": [
            "verified private state (reasoning is a self-report, not ground truth)",
            "per-response timestamps",
            "assignment randomization flag (not stated in the streamed columns)",
            "RANK_CANDIDATE_ACTIONS (no candidate action set with a ranking to recover)",
        ],
        "chronology_rules": "NEXT_CHOICE: the prompt/stimuli precede the response; reasoning is excluded from input. "
                            "POPULATION_RESPONSE / INTERVENTION_EFFECT aggregate independent rows (no temporal cutoff).",
        "split_key": "participant (also study_id + condition_num); episodes never split a participant/study/condition",
        "leakage_risks": [
            "reasoning is generated with the response and would leak the target -> kept out of input entirely",
            "demographics appear both in the prompt text and in actor_profile (both are legitimate input, no leakage)",
        ],
        "known_limitations": [
            "no license declared -> eval-only",
            "response may be a real human answer or a model-simulated persona answer; not determinable from columns",
            "POPULATION_RESPONSE groups by (study_id, condition_num) per spec, which mixes task_nums that use "
            "different stimuli within a condition (flagged per record)",
            "INTERVENTION_EFFECT uses the lowest condition_num as the control reference by convention; condition "
            "ordering does not encode treatment/control semantics; skipped for single-condition studies",
        ],
        "license_implications": "no license declared on official card; eval-only until clarified.",
        "training_suitability": "eval_only",
        "assumptions": [
            "an explicit 'integer from A to B' phrase reliably recovers the option set; otherwise unknown",
            "rows with the same (study_id, condition_num) share the experimental condition",
        ],
    }

    # ------------------------------------------------------------------ NEXT_CHOICE
    def _next_choice(self, ri: int, row: dict, base: str) -> dict:
        study = row.get("study_id")
        participant = row.get("participant")
        condition = row.get("condition_num")
        task = row.get("task_num")
        prompt = row.get("prompt") or ""
        stimuli = row.get("stimuli") or ""
        response = row.get("response")
        demographic = row.get("demographic") or {}
        reasoning = row.get("reasoning")

        actions = _recover_actions(stimuli, prompt)
        choice_index = actions.index(response) if (actions and response in actions) else None
        actor = self.pseudonym("participant", participant)
        experiment_id = self.pseudonym("group", study)
        episode_id = f"socsci210-{study}-c{condition}-t{task}-p{participant}"
        loc = {"files": [base], "indices": [ri], "ids": [str(row.get("sample_id"))]}

        observation = {"text": prompt, "stimuli": stimuli}
        ctx = {
            "actor_profile": demographic,
            "current_observation": observation,
            "world_state": {"study_id": study, "condition_num": condition, "task_num": task,
                            "stimuli": stimuli},
            "available_actions": actions,
            "language": "en",
        }
        payload = {
            "input": {"observation": observation, "history": [], "available_actions": actions},
            "target": {"choice": response, "choice_index": choice_index, "acted": True,
                       "meta": {"self_report_reasoning": reasoning}},
        }
        missing = ["timestamps"]
        if actions is None:
            missing.append("available_actions")
        return self.make(
            task_type="PREDICT_NEXT_CHOICE", payload=payload, episode_id=episode_id,
            sequence_index=0, cutoff_sequence_index=0, participant_ids=[actor],
            experiment_id=experiment_id, actor_id=actor, actor_role="survey_respondent",
            context=ctx, raw_locator=loc,
            transformation_steps=["read socsci210 row", "prompt->observation",
                                  "recover option set if explicit", "response->choice"],
            data_quality={
                "missing_fields": missing, "weak_label_fields": ["reasoning"],
                "warnings": ["reasoning is a self-report generated with the response; stored as meta only, "
                             "excluded from input, not treated as ground-truth private state",
                             "prompt is a persona-simulation instruction; whether response is human or "
                             "model-simulated is not determinable from the streamed columns"],
                "chronology_verified": True, "target_verified": True, "possible_leakage": False,
                "license_verified": False, "confidence": "high",
            })

    # ---------------------------------------------------------- POPULATION_RESPONSE
    def _population(self, study: str, condition, rows: list[tuple[int, dict]], base: str) -> dict:
        responses = [r.get("response") for _, r in rows]
        st = _stats(responses)
        tasks = sorted({r.get("task_num") for _, r in rows}, key=lambda x: str(x))
        stimuli_by_task = {}
        for _, r in rows:
            stimuli_by_task.setdefault(str(r.get("task_num")), r.get("stimuli"))
        actions = None
        for _, r in rows:
            actions = _recover_actions(r.get("stimuli") or "", r.get("prompt") or "")
            if actions:
                break
        participant_ids = sorted({self.pseudonym("participant", r.get("participant")) for _, r in rows})
        experiment_id = self.pseudonym("group", study)
        group_id = self.pseudonym("group", f"{study}:cond{condition}")
        population_id = group_id
        episode_id = f"socsci210-pop-{study}-c{condition}"
        loc = {"files": [base], "indices": [i for i, _ in rows],
               "ids": [str(r.get("sample_id")) for _, r in rows]}

        payload = {
            "input": {
                "population_features": {"study_id": study, "condition_num": condition,
                                        "task_nums": tasks, "n": st["n"]},
                "intervention": {"condition_num": condition, "stimuli_by_task": stimuli_by_task},
                "historical_context": {},
            },
            "target": {
                "response_distribution": st["distribution"],
                "aggregate_metrics": {"mean": st["mean"], "n": st["n"], "counts": st["counts"]},
            },
        }
        warnings = []
        if len(tasks) > 1:
            warnings.append(f"condition group spans task_nums {tasks} (different stimuli); aggregate mixes them "
                            f"per the (study_id, condition_num) grouping spec")
        return self.make(
            task_type="PREDICT_POPULATION_RESPONSE", payload=payload, episode_id=episode_id,
            participant_ids=participant_ids, experiment_id=experiment_id, group_id=group_id,
            population_id=population_id, actor_role="population",
            context={"world_state": {"study_id": study, "condition_num": condition, "task_nums": tasks,
                                     "stimuli_by_task": stimuli_by_task},
                     "available_actions": actions, "language": "en"},
            raw_locator=loc,
            transformation_steps=["group rows by (study_id, condition_num)",
                                  "aggregate responses -> distribution + mean + n"],
            data_quality={
                "missing_fields": ["timestamps"], "warnings": warnings,
                "chronology_verified": True, "target_verified": True, "possible_leakage": False,
                "license_verified": False, "confidence": "high",
            })

    # --------------------------------------------------------- INTERVENTION_EFFECT
    def _intervention(self, study: str, control, treat, groups: dict, base: str) -> dict:
        c_rows, t_rows = groups[control], groups[treat]
        c_st = _stats([r.get("response") for _, r in c_rows])
        t_st = _stats([r.get("response") for _, r in t_rows])
        c_stim = {str(r.get("task_num")): r.get("stimuli") for _, r in c_rows}
        t_stim = {str(r.get("task_num")): r.get("stimuli") for _, r in t_rows}
        participant_ids = sorted({self.pseudonym("participant", r.get("participant"))
                                  for _, r in c_rows + t_rows})
        experiment_id = self.pseudonym("group", study)
        group_id = self.pseudonym("group", f"{study}:ate")
        episode_id = f"socsci210-ate-{study}-c{treat}-vs-c{control}"
        idxs = [i for i, _ in c_rows] + [i for i, _ in t_rows]
        ids = [str(r.get("sample_id")) for _, r in c_rows] + [str(r.get("sample_id")) for _, r in t_rows]
        loc = {"files": [base], "indices": idxs, "ids": ids}

        est = None
        if t_st["mean"] is not None and c_st["mean"] is not None:
            est = {"mean_difference": t_st["mean"] - c_st["mean"]}
        payload = {
            "input": {
                "population_or_actor_features": {"study_id": study,
                                                 "task_nums": sorted({r.get("task_num") for _, r in c_rows + t_rows},
                                                                     key=lambda x: str(x))},
                "treatment": {"condition_num": treat, "stimuli_by_task": t_stim},
                "control": {"condition_num": control, "stimuli_by_task": c_stim},
                "assignment_mechanism": {"type": "experimental_condition_contrast",
                                         "randomized": None, "unit": "participant"},
            },
            "target": {
                "treated_outcome": {"condition_num": treat, "mean": t_st["mean"], "n": t_st["n"],
                                    "distribution": t_st["distribution"]},
                "control_outcome": {"condition_num": control, "mean": c_st["mean"], "n": c_st["n"],
                                    "distribution": c_st["distribution"]},
                "estimated_effect": est,
            },
        }
        return self.make(
            task_type="PREDICT_INTERVENTION_EFFECT", payload=payload, episode_id=episode_id,
            participant_ids=participant_ids, experiment_id=experiment_id, group_id=group_id,
            actor_role="population",
            context={"world_state": {"study_id": study, "treated_condition": treat,
                                     "control_condition": control},
                     "available_actions": None, "language": "en"},
            causal_metadata={"is_experimental": True, "randomized": None,
                             "assignment_mechanism": "experimental_condition_contrast",
                             "unit_of_assignment": "participant"},
            raw_locator=loc,
            transformation_steps=["group rows by condition within study",
                                  "control = lowest condition_num (convention)",
                                  "compare treated vs control mean response"],
            data_quality={
                "missing_fields": ["timestamps", "assignment_randomization_flag"],
                "inferred_fields": ["control_reference (lowest condition_num, by convention)"],
                "warnings": ["condition_num ordering does not encode treatment/control semantics; "
                             "lowest condition_num used as control reference by convention",
                             "randomization not verifiable from the streamed columns (randomized=None)",
                             "means may mix task_nums (different stimuli) within a condition"],
                "chronology_verified": True, "target_verified": True, "possible_leakage": False,
                "license_verified": False, "confidence": "medium",
            })

    # ---------------------------------------------------------------------- driver
    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        rows = _load_rows(raw_dir)

        # NEXT_CHOICE: one per row.
        for ri, (row, base) in enumerate(rows):
            yield self._next_choice(ri, row, base)

        # Group indexed rows by study -> condition (carry the source basename per group).
        by_study: dict = defaultdict(lambda: defaultdict(list))
        base_by_study: dict = {}
        for ri, (row, base) in enumerate(rows):
            study = row.get("study_id")
            by_study[study][row.get("condition_num")].append((ri, row))
            base_by_study.setdefault(study, base)

        # POPULATION_RESPONSE: per (study_id, condition_num).
        for study, conds in by_study.items():
            base = base_by_study[study]
            for condition in sorted(conds, key=lambda x: str(x)):
                yield self._population(study, condition, conds[condition], base)

        # INTERVENTION_EFFECT: per study, each non-baseline condition vs the baseline.
        for study, conds in by_study.items():
            base = base_by_study[study]
            cond_keys = sorted(conds, key=lambda x: str(x))
            if len(cond_keys) < 2:
                continue  # single-condition study -> skipped (noted in DOC.known_limitations)
            control = cond_keys[0]
            for treat in cond_keys[1:]:
                yield self._intervention(study, control, treat, conds, base)
