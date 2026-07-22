"""DEBATE — Opinion Dynamics in Real US Human Debates (ACCESS-BLOCKED stub).

Data status: NOT publicly released as of last verification (reviewer-gated at submission;
authors state a public release is planned but no stable URL is confirmed). This module
therefore raises SourceNotAvailable and produces no records. The DOC below is the INTENDED
canonical mapping for when the data becomes available, so the converter can be finished the
moment a release appears (watch arXiv 2510.25110 / OpenReview rMnZbCOhSS).

Intended real structure (from the paper): real US human debates in which each participant's
private opinion on a topic is measured on a 6-point Likert scale BEFORE and AFTER a
multi-round group discussion, alongside the public chat transcript.
  ~2,788 participants / 697 groups / 107 topics / 5,302 pre/post opinion ratings.

Intended tasks (the pre/post belief measurement genuinely supports BELIEF_CHANGE):
  PREDICT_BELIEF_CHANGE          — input.belief_before = pre-discussion 6-point rating +
                                   observed_messages; target.belief_after = post-discussion
                                   rating (only emitted because BOTH are measured).
  PREDICT_NEXT_MESSAGE           — each participant chat turn from prior turns + private opinion.
  PREDICT_NEXT_SPEAKER           — turn-taking within a group.
  PREDICT_TRAJECTORY_CONTINUATION — next K turns of a group discussion.
  PREDICT_FINAL_OUTCOME          — group's terminal opinion state / consensus.
Isolation: hold out by participant AND group AND topic (a participant/topic must not leak
across splits).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter, SourceNotAvailable


class Converter(BaseConverter):
    DATASET_ID = "debate"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = None  # no data released -> no fixture
    DOC = {
        "dataset_id": "debate",
        "original_fields": [
            {"name": "participant_id", "meaning": "real human participant (pseudonymize on release)"},
            {"name": "group_id", "meaning": "discussion group"},
            {"name": "topic", "meaning": "debate topic (107 topics)"},
            {"name": "opinion_pre", "meaning": "private 6-point Likert opinion measured BEFORE discussion"},
            {"name": "opinion_post", "meaning": "private 6-point Likert opinion measured AFTER discussion"},
            {"name": "messages", "meaning": "public chat transcript (ordered turns per participant)"},
        ],
        "canonical_mapping": [
            {"source_field": "opinion_pre", "canonical_path": "payload.input.belief_before (BELIEF_CHANGE) / context.private_state_before"},
            {"source_field": "opinion_post", "canonical_path": "payload.target.belief_after (BELIEF_CHANGE)"},
            {"source_field": "messages[k].text", "canonical_path": "payload.target.message_text (NEXT_MESSAGE) / context.known_history"},
            {"source_field": "messages[k].speaker", "canonical_path": "payload.target.speaker_id (NEXT_SPEAKER)"},
            {"source_field": "participant_id", "canonical_path": "decision_unit.actor_id (pseudonymized)"},
            {"source_field": "group_id / topic", "canonical_path": "episode.group_id / episode.topic_id (isolation keys)"},
        ],
        "tasks_produced": [
            "PREDICT_BELIEF_CHANGE", "PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_SPEAKER",
            "PREDICT_TRAJECTORY_CONTINUATION", "PREDICT_FINAL_OUTCOME",
        ],
        "unavailable_fields": [
            "ALL fields (dataset not publicly released) — no records are produced until release",
            "on release: precise per-turn timestamps to confirm",
        ],
        "chronology_rules": "BELIEF_CHANGE uses pre-discussion opinion (measured before) as input and post-discussion opinion (measured after) as target; message/speaker tasks expose only turns 0..k-1 for a decision at turn k.",
        "split_key": "participant + group + topic (hold all three out jointly)",
        "leakage_risks": [
            "a participant appears across turns and possibly topics -> isolate by participant",
            "post-discussion opinion (belief_after) must never enter context/input",
        ],
        "known_limitations": [
            "dataset not publicly released; this is a documented intended mapping, not a live converter",
            "BELIEF_CHANGE is valid ONLY because both pre and post opinions are measured (real longitudinal design)",
        ],
        "license_implications": "License not stated (data not yet released). Human belief ratings + chat are personal data; handle accordingly on release. Do NOT train until license is verified.",
        "training_suitability": "blocked",
        "assumptions": [
            "on release: per-participant pre/post 6-point Likert ratings are linkable to the group chat transcript",
        ],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        raise SourceNotAvailable(
            "debate data not publicly released; see registry blockers "
            "(watch arXiv 2510.25110 / OpenReview rMnZbCOhSS for the promised code+data release)."
        )
