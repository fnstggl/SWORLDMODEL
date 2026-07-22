"""BehaviorBench (befm/BehaviorBench) — an EVALUATION benchmark for behavioral science.

Real source (verified from the HF snapshot; every record is a chat triple):
  {"system": <persona/role instruction>, "user": <item prompt incl. options>,
   "assistant": <recorded reference answer>}   (+ optional "metadata" / paper fields)

Item families (directory = task/experiment; the split_unit):
  big_five/*              — Big-Five self-assessment items; assistant = human Likert rating "[1..5]"
  moblab/game_behavior/*  — one-shot economics games (dictator/ultimatum/trust/public_goods/
                            guessing/bomb/push_pull); assistant = human decision, e.g. "[48]"
  moblab/acrossgame_*     — cross-game behavior; same triple shape
  moblab/multiround_*     — multi-round games; metadata={user_id, game, role, history_rounds,
                            target_round, has_others_info}; assistant = decision at target_round
  moblab/strategic_gameplay/* — strategic variants
  economics_contests/*    — Economics-Olympiad MCQ; assistant = correct option letter(s), e.g. "D"
  workflows/*             — research-workflow generation from real published papers
                            (title/idea/impact/method/outcome prediction); assistant = free text;
                            extra fields paper_id/journal/doi/publication_year/task

Emits (input = the item prompt + embedded options; target = the RECORDED answer):
  PREDICT_NEXT_CHOICE   — bounded-response families (big_five, moblab games, econ MCQ):
                          target.choice = verbatim recorded answer, acted=True
  PREDICT_FINAL_OUTCOME — free-text generation families (workflows):
                          target.outcome = verbatim recorded text, outcome_type = task name

Honesty notes:
  * The option set is embedded in the prompt TEXT, not given as a clean enumerated candidate
    list, so context.available_actions is None (UNKNOWN set) and RANK_CANDIDATE_ACTIONS is
    NOT produced (no explicit chosen_id-among-candidates structure).
  * economics_contests answers are exam keys (correct answers), and workflows answers are the
    real published paper artifacts — both are recorded ground truth, not free behavioral
    distributions; noted in known_limitations.
  * LICENSE: CC-BY-NC-ND-4.0 (No-Derivatives). Evaluation-only; MUST NOT enter training.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter

_WORKFLOW_PREFIX = "workflows"


def _family_of(path: Path, raw_dir: Path) -> str:
    """Task/experiment family = the file's path relative to the dataset root, sans extension."""
    try:
        rel = path.relative_to(raw_dir)
    except ValueError:
        rel = Path(path.name)
    fam = rel.as_posix()
    if fam.endswith(".jsonl"):
        fam = fam[: -len(".jsonl")]
    if fam.endswith("_test"):
        fam = fam[: -len("_test")]
    return fam


def _load_units(raw_dir: Path) -> Iterator[tuple[str, int, dict]]:
    """Yield (family, line_index, row). Supports the real .jsonl tree and a JSON fixture list."""
    jsonls = [f for f in glob.glob(str(raw_dir / "**" / "*.jsonl"), recursive=True)
              if ".cache" not in f]
    found = False
    for f in sorted(jsonls):
        fam = _family_of(Path(f), raw_dir)
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    found = True
                    yield fam, i, row
    if found:
        return
    jsons = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
             if ".cache" not in f and "dataset_info" not in f and "indices" not in Path(f).name]
    for f in sorted(jsons):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            for i, row in enumerate(data):
                if isinstance(row, dict):
                    found = True
                    yield row.get("_family", "fixture"), i, row
    if not found:
        raise FileNotFoundError(f"no BehaviorBench jsonl/json found under {raw_dir}")


class Converter(BaseConverter):
    DATASET_ID = "behaviorbench"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "behaviorbench"
    DOC = {
        "dataset_id": "behaviorbench",
        "original_fields": [
            {"name": "system", "meaning": "persona/role instruction (e.g. respondent demographics or 'you are a player')"},
            {"name": "user", "meaning": "item prompt including the response options / question"},
            {"name": "assistant", "meaning": "recorded reference answer (human decision / Likert rating / exam key / paper artifact)"},
            {"name": "metadata", "meaning": "(multiround games) {user_id, game, role, history_rounds, target_round, has_others_info}"},
            {"name": "task / journal / paper_id / doi / publication_year", "meaning": "(workflows) provenance of the source paper"},
        ],
        "canonical_mapping": [
            {"source_field": "system", "canonical_path": "context.actor_profile.persona"},
            {"source_field": "user", "canonical_path": "context.current_observation.text / payload.input.observation"},
            {"source_field": "assistant", "canonical_path": "payload.target.choice (NEXT_CHOICE) | payload.target.outcome (FINAL_OUTCOME)"},
            {"source_field": "metadata", "canonical_path": "context.world_state.metadata + episode.participant_ids (user_id)"},
            {"source_field": "directory family", "canonical_path": "episode.experiment_id (split_unit=experiment)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_CHOICE", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": [
            "explicit enumerated candidate set (options embedded in prompt text) -> available_actions null; RANK_CANDIDATE_ACTIONS not produced",
            "per-item timestamps / response times",
            "individual respondent identity except multiround user_id",
        ],
        "chronology_rules": "Each item is a single scored decision: the prompt (system+user) is the pre-decision context; the recorded answer (assistant) is the target and never appears in input/context. cutoff_sequence_index=0.",
        "split_key": "experiment/task (directory family) -> episode.experiment_id",
        "leakage_risks": [
            "the same respondent/user_id may recur across items in multiround families; splitting by experiment keeps a task's items together",
            "workflows share source papers -> hold papers within an experiment together",
        ],
        "known_limitations": [
            "economics_contests answers are exam keys (correct answers), workflows answers are real published-paper artifacts -> recorded ground truth, not free behavioral distributions",
            "moblab game answers are numeric decisions in bracket notation, preserved verbatim",
            "option set lives inside the prompt text (not a separate field)",
        ],
        "license_implications": "CC-BY-NC-ND: No-Derivatives forbids training; evaluation-only",
        "training_suitability": "eval_only",
        "assumptions": [
            "every record is a {system, user, assistant} chat triple",
            "workflows/* are free-text generation; all other families are bounded-response choices",
        ],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        for fam, i, row in _load_units(raw_dir):
            rec = self._one(fam, i, row)
            if rec is not None:
                yield rec

    def _one(self, family: str, idx: int, row: dict):
        system = row.get("system")
        user = row.get("user")
        answer = row.get("assistant")
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            return None  # no recorded target -> cannot form a labeled item
        answer = answer if not isinstance(answer, str) else answer.strip()

        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        paper_meta = {k: row[k] for k in ("paper_id", "journal", "doi", "publication_year", "task")
                      if k in row}

        episode_id = f"behaviorbench-{family}-{idx}"
        prompt_text = (str(system) + "\n\n" + str(user)) if system else str(user)

        participant_ids = []
        actor_id = None
        if metadata.get("user_id") is not None:
            actor_id = self.pseudonym("participant", str(metadata["user_id"]))
            participant_ids = [actor_id]

        ctx = {
            "actor_profile": {"persona": system},
            "current_observation": {"kind": "prompt", "text": str(user),
                                    "meta": {"family": family}},
            "world_state": {"family": family, "metadata": metadata, "paper": paper_meta},
            "available_actions": None,  # options are embedded in prompt text, not enumerated
            "language": "en",
        }

        loc = {"files": [f"{family}_test.jsonl"], "indices": [idx], "ids": [episode_id]}
        missing = ["explicit_option_set", "response_time", "item_timestamp"]

        if family.startswith(_WORKFLOW_PREFIX):
            payload = {
                "input": {"history": [], "state": {"prompt": prompt_text, "paper": paper_meta}},
                "target": {"outcome": answer, "outcome_type": (metadata.get("task") or paper_meta.get("task") or family.split("/")[-1])},
            }
            return self.make(
                task_type="PREDICT_FINAL_OUTCOME",
                payload=payload,
                episode_id=episode_id,
                experiment_id=family,
                sequence_index=0,
                cutoff_sequence_index=0,
                participant_ids=participant_ids,
                actor_id=actor_id,
                actor_role="respondent",
                context=ctx,
                source_language="en",
                raw_locator=loc,
                transformation_steps=[
                    "read BehaviorBench chat triple",
                    "prompt = system+user (pre-decision context)",
                    "target = recorded free-text answer (paper artifact)",
                ],
                data_quality={
                    "missing_fields": missing,
                    "warnings": ["recorded answer is a published-paper artifact (reference ground truth)"],
                    "confidence": "high",
                    "chronology_verified": True,
                    "target_verified": True,
                    "possible_leakage": False,
                    "license_verified": True,
                },
            )

        # bounded-response families -> NEXT_CHOICE (verbatim recorded decision)
        payload = {
            "input": {"history": [], "observation": ctx["current_observation"],
                      "available_actions": None},
            "target": {"choice": answer, "acted": True},
        }
        warn = []
        if family.startswith("economics_contests"):
            warn.append("recorded answer is an exam key (correct option), not a behavioral distribution")
        return self.make(
            task_type="PREDICT_NEXT_CHOICE",
            payload=payload,
            episode_id=episode_id,
            experiment_id=family,
            sequence_index=0,
            cutoff_sequence_index=0,
            participant_ids=participant_ids,
            actor_id=actor_id,
            actor_role="respondent",
            context=ctx,
            source_language="en",
            raw_locator=loc,
            transformation_steps=[
                "read BehaviorBench chat triple",
                "prompt = system+user (pre-decision context)",
                "target = recorded answer (verbatim), acted=True",
            ],
            data_quality={
                "missing_fields": missing,
                "warnings": warn,
                "confidence": "high",
                "chronology_verified": True,
                "target_verified": True,
                "possible_leakage": False,
                "license_verified": True,
            },
        )
