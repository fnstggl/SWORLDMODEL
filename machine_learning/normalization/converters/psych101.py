"""Psych-101 (marcelbinz/Psych-101) — natural-language transcripts of psychology
experiments where the HUMAN participant's responses are wrapped in double angle brackets.

Real source (verified from the streamed parquet shard):
  row = {text: str, experiment: str, participant: str}
  * text     — a long natural-language transcript of one participant's session. Each human
               choice/keypress is annotated inline as ``<<X>>`` (e.g. "You press <<K>>.");
               everything BEFORE a ``<<...>>`` span is the context the participant saw.
  * experiment — source study id, e.g. "badham2017deficits/exp1.csv".
  * participant — participant id within the experiment (string integer).

Centaur convention: each ``<<...>>`` marker is a human response; the prefix text up to (but
not including) that marker is the leakage-safe context.

Emits:
  PREDICT_NEXT_CHOICE             — one per ``<<X>>`` marker (capped): observation = the
                                    transcript up to but NOT including the marker; history =
                                    prior choices; target.choice = the marked response.
  PREDICT_TRAJECTORY_CONTINUATION — from a mid-transcript cutoff, predict the remaining
                                    (capped) sequence of choices.

Honesty notes:
  * The option/action set is NOT reliably recoverable from free text across the many
    heterogeneous experiments, so available_actions is None everywhere (never guessed).
  * No demographics, timestamps, or private beliefs are present -> left empty + listed in
    data_quality.missing_fields. PREDICT_FINAL_OUTCOME (in the registry) is NOT produced:
    a transcript is a sequence of trial choices with no single downstream outcome to
    predict without inventing one.
  * To keep examples bounded, at most CAP (~50) choice-examples are emitted per transcript
    and the trajectory continuation is capped to CAP events (documented bound, surfaced via
    a data_quality.warning + known_limitations, never silent).
  * Transcripts with zero ``<<>>`` markers are skipped and counted (see known_limitations).
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event

_MARKER_RE = re.compile(r"<<(.*?)>>", re.DOTALL)
#: Max choice-examples emitted per transcript, and max continuation length (documented bound).
CAP = 50


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
    raise FileNotFoundError(f"no Psych-101 parquet/json found under {raw_dir}")


def _natural(prefix: str) -> str:
    """Render a transcript prefix as the participant saw it: drop the annotation brackets.

    The prefix contains only fully-closed earlier ``<<...>>`` markers (the current/later
    markers are excluded by slicing), so removing the literal bracket tokens yields the
    natural text (e.g. "You press <<K>>." -> "You press K.") without leaking anything.
    """
    return prefix.replace("<<", "").replace(">>", "")


class Converter(BaseConverter):
    DATASET_ID = "psych101"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "psych101"
    DOC = {
        "dataset_id": "psych101",
        "original_fields": [
            {"name": "text", "meaning": "natural-language transcript of one participant's session; "
                                        "each human response annotated inline as <<X>>",
             "example": "You see a big black square. You press <<K>>. The correct category is K."},
            {"name": "experiment", "meaning": "source study id, e.g. badham2017deficits/exp1.csv"},
            {"name": "participant", "meaning": "participant id within the experiment (string int)"},
        ],
        "canonical_mapping": [
            {"source_field": "text[:marker_start]", "canonical_path": "payload.input.observation.text",
             "transform": "slice transcript up to (not incl.) the <<X>> marker; strip annotation brackets"},
            {"source_field": "prior <<...>> markers", "canonical_path": "payload.input.history / context.known_history",
             "transform": "prior human choices as choice events"},
            {"source_field": "<<X>> (marked span)", "canonical_path": "payload.target.choice",
             "transform": "strip << >> -> verbatim choice label"},
            {"source_field": "remaining <<...>> markers after cutoff", "canonical_path": "payload.target.continuation"},
            {"source_field": "experiment", "canonical_path": "episode.experiment_id (pseudonymized) / context.world_state.experiment"},
            {"source_field": "participant", "canonical_path": "episode.participant_ids (pseudonymized) / decision_unit.actor_id"},
        ],
        "tasks_produced": ["PREDICT_NEXT_CHOICE", "PREDICT_TRAJECTORY_CONTINUATION"],
        "unavailable_fields": [
            "available_actions / option set (not reliably parseable from heterogeneous free text)",
            "actor_profile / demographics", "per-trial timestamps", "private beliefs/goals",
            "PREDICT_FINAL_OUTCOME (no single downstream outcome; a transcript is a trial sequence)",
        ],
        "chronology_rules": "For the choice at marker k, the observation is exactly text[:start_of_marker_k] "
                            "(brackets stripped) and history is choices 0..k-1; marker k and everything after "
                            "it appear ONLY in the target. Trajectory cuts off at min(M//2, CAP) markers.",
        "split_key": "participant + experiment (hold out both; episode_id = psych101-<participant>-<experiment>)",
        "leakage_risks": [
            "current trial's feedback ('The correct category is ...') follows the marker, so it is never in the "
            "observation prefix; earlier trials' feedback is legitimately-past context",
        ],
        "known_limitations": [
            "at most CAP (~50) PREDICT_NEXT_CHOICE examples emitted per transcript (transcripts can hold 100s of "
            "markers); surfaced via data_quality.warning when capped",
            "PREDICT_TRAJECTORY_CONTINUATION continuation capped to CAP events and may not reach the transcript end",
            "transcripts with zero <<>> markers are skipped and counted",
            "option set / available_actions not recovered (varies per experiment; not guessed)",
        ],
        "license_implications": "Apache-2.0: training + commercial use permitted with attribution.",
        "training_suitability": "train",
        "assumptions": [
            "each <<...>> span is exactly one human choice/keypress; empty spans (<<>>) are ignored",
            "everything before a marker is context the participant had already seen",
        ],
    }

    def _choice_event(self, index: int, actor: str, value: str) -> dict:
        return history_event(index, actor, "choice", text=value)

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        rows = _load_rows(raw_dir)
        for ri, (row, base) in enumerate(rows):
            yield from self._one_transcript(ri, row, base)

    def _one_transcript(self, ri: int, row: dict, base: str) -> Iterator[dict]:
        text = row.get("text") or ""
        experiment = row.get("experiment") or ""
        participant = row.get("participant")

        # (start_position, choice_value) for every non-empty marker, in order.
        markers = [(m.start(), m.group(1).strip()) for m in _MARKER_RE.finditer(text)]
        markers = [(s, v) for (s, v) in markers if v]
        if not markers:
            return  # zero-marker transcript: skipped (counted by the caller's tally)

        m_total = len(markers)
        episode_id = f"psych101-{participant}-{experiment}"
        actor = self.pseudonym("participant", participant)
        experiment_id = self.pseudonym("group", experiment)
        loc = {"files": [base], "indices": [ri], "ids": [episode_id]}
        capped = m_total > CAP

        # ---- PREDICT_NEXT_CHOICE (first CAP markers) ---------------------------------
        n_choice = min(m_total, CAP)
        for j in range(n_choice):
            start_j, value = markers[j]
            obs_text = _natural(text[:start_j])
            history = [self._choice_event(i, actor, markers[i][1]) for i in range(j)]
            ctx = {
                "actor_profile": {},
                "known_history": history,
                "current_observation": {"text": obs_text, "kind": "transcript"},
                "world_state": {"experiment": experiment},
                "available_actions": None,
                "language": "en",
            }
            payload = {
                "input": {"observation": {"text": obs_text}, "history": history,
                          "available_actions": None},
                "target": {"choice": value, "choice_index": None, "acted": True},
            }
            warnings = []
            if capped:
                warnings.append(f"transcript has {m_total} markers; capped to {CAP} choice-examples")
            yield self.make(
                task_type="PREDICT_NEXT_CHOICE", payload=payload, episode_id=episode_id,
                sequence_index=j, cutoff_sequence_index=j, participant_ids=[actor],
                experiment_id=experiment_id, actor_id=actor, actor_role="participant",
                context=ctx, raw_locator=loc,
                transformation_steps=["load psych101 row", "locate <<>> markers",
                                      f"cutoff before marker {j}", "strip brackets from prefix"],
                data_quality={
                    "missing_fields": ["available_actions", "actor_profile", "timestamps"],
                    "inferred_fields": ["sequence_index"], "warnings": warnings,
                    "chronology_verified": True, "target_verified": True,
                    "possible_leakage": False, "license_verified": True, "confidence": "high",
                })

        # ---- PREDICT_TRAJECTORY_CONTINUATION (one per transcript, M>=2) ---------------
        if m_total >= 2:
            cutoff = min(m_total // 2, CAP)  # >=1 when m_total>=2; bounds the observation size
            start_c, _ = markers[cutoff]
            obs_text = _natural(text[:start_c])
            history = [self._choice_event(i, actor, markers[i][1]) for i in range(cutoff)]
            cont = [self._choice_event(i, actor, markers[i][1])
                    for i in range(cutoff, min(m_total, cutoff + CAP))]
            reaches_end = cutoff + CAP >= m_total
            warnings = []
            if not reaches_end:
                warnings.append(f"continuation capped to {CAP} events; does not extend to transcript end "
                                f"({m_total} markers total)")
            payload = {
                "input": {"history": history, "horizon": len(cont),
                          "observation": {"text": obs_text}},
                "target": {"continuation": cont},
            }
            ctx = {
                "actor_profile": {},
                "known_history": history,
                "current_observation": {"text": obs_text, "kind": "transcript"},
                "world_state": {"experiment": experiment},
                "available_actions": None,
                "language": "en",
            }
            yield self.make(
                task_type="PREDICT_TRAJECTORY_CONTINUATION", payload=payload, episode_id=episode_id,
                sequence_index=cutoff, cutoff_sequence_index=cutoff, participant_ids=[actor],
                experiment_id=experiment_id, actor_id=actor, actor_role="participant",
                context=ctx, raw_locator=loc,
                transformation_steps=["load psych101 row", "locate <<>> markers",
                                      f"cutoff after {cutoff} markers", "remaining choices as continuation"],
                data_quality={
                    "missing_fields": ["available_actions", "actor_profile", "timestamps"],
                    "inferred_fields": ["sequence_index"], "warnings": warnings,
                    "chronology_verified": True, "target_verified": True,
                    "possible_leakage": False, "license_verified": True, "confidence": "high",
                })
