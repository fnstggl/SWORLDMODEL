"""SimBench (pitehu/SimBench) — reproducing REAL human response DISTRIBUTIONS.

Real source (verified from SimBenchPop.csv / SimBenchGrouped.csv on HF):
  A test case = one (question, population/demographic-group) pair whose label is the
  AGGREGATED human response distribution compiled from 20 real survey/experiment datasets
  (AfroBarometer, ESS, ISSP, OpinionQA, LatinoBarometro, Choices13k, MoralMachine,
  WisdomOfCrowds, GlobalOpinionQA, ChaosNLI, DICES, Jester, NumberGame, OSPsych*, ...).

  Columns (both splits):
    dataset_name              — the source study (split_unit = study)
    group_prompt_template     — persona/grouping prompt (population for Pop; demographics for Grouped)
    group_prompt_variable_map — dict of placeholder -> value (demographics; often {} for Pop)
    input_template            — the question stem shown to participants
    human_answer              — dict {option_label: PERCENT of respondents} = the real distribution
    group_size                — number of human respondents (n)
    auxiliary                 — dict; may hold correct_answer / task_id / domain_name
  Grouped-only extras: answer_options (label->text), output_format, wave, grouping_keys, ...

Emits:
  PREDICT_POPULATION_RESPONSE — target.response_distribution = the real human distribution
                                (normalized to proportions), target.aggregate_metrics = n +
                                raw percentages + modal option; input = question + persona/
                                demographic population features.

Honesty notes:
  * SimBench ships ONLY aggregate distributions — there are NO individual-level responses,
    so PREDICT_NEXT_CHOICE is NOT produced (would require per-participant rows that do not
    exist). Listed in DOC.unavailable_fields.
  * human_answer values are percentages; response_distribution divides by their sum to give
    proportions (documented as an inferred/derived transformation). The verbatim percentages
    are preserved in aggregate_metrics.raw_distribution_percent.
  * EVAL-ONLY: held-out population-response cross-dataset transfer test; CC-BY-NC-SA.
"""
from __future__ import annotations

import ast
import csv
import glob
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter

# CSV can hold very large question/answer fields.
csv.field_size_limit(10 * 1024 * 1024)


def _as_dict(v) -> dict:
    """A dict field is either already a dict (JSON fixture) or a Python-repr string (CSV)."""
    if isinstance(v, dict):
        return v
    if v is None:
        return {}
    s = str(v).strip()
    if not s or s in ("{}", "nan"):
        return {}
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _as_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _load_rows(raw_dir: Path) -> Iterator[tuple[str, dict]]:
    """Yield (split_label, row_dict). Supports the real CSVs and a JSON fixture list."""
    csvs = [f for f in glob.glob(str(raw_dir / "**" / "*.csv"), recursive=True)
            if ".cache" not in f]
    found = False
    for f in sorted(csvs):
        found = True
        split = Path(f).stem  # SimBenchPop / SimBenchGrouped
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                yield split, row
    if found:
        return
    jsons = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
             if ".cache" not in f and "dataset_info" not in f]
    for f in sorted(jsons):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield row.get("_split", "SimBench"), row
            found = True
    if not found:
        raise FileNotFoundError(f"no SimBench csv/json found under {raw_dir}")


class Converter(BaseConverter):
    DATASET_ID = "simbench"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "simbench"
    DOC = {
        "dataset_id": "simbench",
        "original_fields": [
            {"name": "dataset_name", "meaning": "source study (one of 20); the split_unit"},
            {"name": "group_prompt_template", "meaning": "persona/grouping prompt (population or demographic group)"},
            {"name": "group_prompt_variable_map", "meaning": "dict placeholder->value (demographics; often {} for Pop)"},
            {"name": "input_template", "meaning": "question stem presented to human respondents"},
            {"name": "human_answer", "meaning": "dict option->PERCENT of respondents = the real aggregate distribution"},
            {"name": "group_size", "meaning": "number of human respondents contributing (n)"},
            {"name": "auxiliary", "meaning": "extra metadata (correct_answer, task_id, domain_name, ...)"},
            {"name": "answer_options", "meaning": "(Grouped) dict option_label->option_text"},
            {"name": "wave", "meaning": "(Grouped) survey wave, when present"},
        ],
        "canonical_mapping": [
            {"source_field": "human_answer", "canonical_path": "payload.target.response_distribution (normalized) + aggregate_metrics.raw_distribution_percent",
             "transform": "parse dict; divide percentages by their sum for response_distribution"},
            {"source_field": "group_size", "canonical_path": "payload.target.aggregate_metrics.n"},
            {"source_field": "input_template", "canonical_path": "payload.input.population_features.question / context.current_observation.text"},
            {"source_field": "group_prompt_template + group_prompt_variable_map", "canonical_path": "payload.input.population_features.persona / context.actor_profile"},
            {"source_field": "answer_options / human_answer keys", "canonical_path": "context.available_actions"},
            {"source_field": "dataset_name", "canonical_path": "episode.experiment_id (study) + decision_unit.population_id"},
            {"source_field": "auxiliary.correct_answer", "canonical_path": "payload.input.historical_context.correct_answer (reference only)"},
        ],
        "tasks_produced": ["PREDICT_POPULATION_RESPONSE"],
        "unavailable_fields": [
            "individual-level responses (only aggregate distributions exist) -> PREDICT_NEXT_CHOICE not produced",
            "per-respondent demographics / timestamps (only group persona + wave)",
            "response times",
        ],
        "chronology_rules": "Each test case is a static (question, group) aggregate with no temporal sequence; the human distribution is the label and never appears in input/context.",
        "split_key": "study (dataset_name) -> episode.experiment_id; held out as a cross-dataset transfer test",
        "leakage_risks": [
            "the same underlying question may recur across waves/groups; splitting by study (dataset_name) keeps a source's cases together",
            "correct_answer (when present) is exposed only as a reference feature, never as the human-distribution label",
        ],
        "known_limitations": [
            "human_answer is percentages summing to ~100; normalized to proportions for response_distribution (raw preserved)",
            "SimBenchPop personas are broad-population defaults; SimBenchGrouped adds demographic conditioning",
            "no individual responses -> only group-level simulation can be evaluated",
        ],
        "license_implications": "CC-BY-NC-SA-4.0: non-commercial, share-alike. Reserved as the held-out population-response cross-dataset transfer test; evaluation-only, excluded from training manifests.",
        "training_suitability": "eval_only",
        "assumptions": [
            "human_answer keys are the option labels; values are respondent percentages",
            "dict-valued CSV cells are Python-repr and parsed with ast.literal_eval",
        ],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        for idx, (split, row) in enumerate(_load_rows(raw_dir)):
            rec = self._one(split, idx, row)
            if rec is not None:
                yield rec

    def _one(self, split: str, idx: int, row: dict):
        human = _as_dict(row.get("human_answer"))
        # keep only numeric-valued options
        dist_pct = {}
        for k, v in human.items():
            fv = None
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = None
            if fv is not None:
                dist_pct[str(k)] = fv
        if not dist_pct:
            return None  # no usable human distribution -> cannot form target

        total = sum(dist_pct.values())
        dist_norm = {k: (v / total if total else 0.0) for k, v in dist_pct.items()}
        modal = max(dist_pct, key=dist_pct.get)

        dataset_name = str(row.get("dataset_name") or "unknown_study")
        var_map = _as_dict(row.get("group_prompt_variable_map"))
        template = str(row.get("group_prompt_template") or "")
        try:
            persona = template.format(**var_map) if var_map else template
        except (KeyError, IndexError, ValueError):
            persona = template
        question = str(row.get("input_template") or "")
        answer_options = _as_dict(row.get("answer_options"))
        auxiliary = _as_dict(row.get("auxiliary"))
        group_size = _as_int(row.get("group_size"))
        wave = row.get("wave")

        option_labels = sorted(answer_options.keys()) if answer_options else sorted(dist_pct.keys())

        population_features = {
            "source_study": dataset_name,
            "persona": persona,
            "demographic_variables": var_map,
            "question": question,
            "answer_options": answer_options,
            "n_respondents": group_size,
            "wave": (None if wave in (None, "") else wave),
            "split": split,
        }
        historical_context = {}
        if auxiliary.get("correct_answer") is not None:
            historical_context["correct_answer"] = auxiliary.get("correct_answer")

        aggregate_metrics = {
            "n": group_size,
            "n_options": len(dist_pct),
            "modal_option": modal,
            "modal_share_percent": dist_pct[modal],
            "raw_distribution_percent": dist_pct,
            "units": "proportion (response_distribution); percent (raw_distribution_percent)",
        }

        payload = {
            "input": {
                "population_features": population_features,
                "historical_context": historical_context,
            },
            "target": {
                "aggregate_metrics": aggregate_metrics,
                "response_distribution": dist_norm,
            },
        }

        group_sig = ";".join(f"{k}={var_map[k]}" for k in sorted(var_map)) if var_map else "population"
        population_id = f"{dataset_name}:{group_sig}"
        episode_id = f"simbench-{split}-{idx}"

        ctx = {
            "actor_profile": {"persona": persona, "demographics": var_map,
                              "level": "population", "source_study": dataset_name},
            "current_observation": {"kind": "question", "text": question,
                                    "meta": {"answer_options": answer_options}},
            "world_state": {"source_study": dataset_name, "auxiliary": auxiliary,
                            "wave": (None if wave in (None, "") else wave)},
            "available_actions": option_labels,
            "language": "en",
        }

        missing = ["individual_responses", "response_times"]
        if group_size is None:
            missing.append("group_size")

        return self.make(
            task_type="PREDICT_POPULATION_RESPONSE",
            payload=payload,
            episode_id=episode_id,
            experiment_id=dataset_name,
            population_id=population_id,
            group_id=group_sig,
            actor_role="population",
            context=ctx,
            dataset_version=split,
            source_language="en",
            raw_locator={"files": [f"{split}.csv"], "indices": [idx], "ids": [episode_id]},
            transformation_steps=[
                "read SimBench row",
                "parse human_answer dict (percent)",
                "normalize to proportions for response_distribution; preserve raw percent",
                "build persona/demographic population features",
            ],
            data_quality={
                "missing_fields": missing,
                "inferred_fields": ["response_distribution (normalized from percentages)"],
                "warnings": ["human_answer is an aggregate group distribution, not individual responses"],
                "confidence": "high",
                "chronology_verified": True,
                "target_verified": True,
                "possible_leakage": False,
                "license_verified": True,
            },
        )
