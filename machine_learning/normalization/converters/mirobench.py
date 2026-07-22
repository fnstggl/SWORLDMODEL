"""MiroBench — Realism in Agentic Simulation of Real-world Discussions (ACCESS-BLOCKED stub).

Data status: no public release located as of last verification (appears unreleased or
forthcoming). This module raises SourceNotAvailable and produces no records. The DOC below
is the INTENDED canonical mapping for when the data becomes available (watch arXiv
2606.14715; verify Reddit-ToS compliance before any redistribution — local conversion only).

Intended real structure (from the paper): 4,292 real Reddit threads (875 seed posts across
5 product domains) used as ground truth for agentic discussion simulation. Each thread is a
reply TREE rooted at a seed post, with a seed-post cutoff.

Intended tasks (reply-tree structure genuinely supports discussion-tree growth):
  PREDICT_DISCUSSION_TREE        — input.seed_post + partial_tree observed before cutoff;
                                   target.tree = the continued reply tree (nodes/edges).
  PREDICT_NEXT_MESSAGE           — each reply from its ancestor path (parent-conditioned).
  PREDICT_POPULATION_RESPONSE    — aggregate thread reaction (n_replies, depth, branching,
                                   score distribution) to a seed post.
  PREDICT_TRAJECTORY_CONTINUATION — next K events of a thread given its prefix.
Isolation: hold out by thread / seed post (a seed's whole tree stays together; also hold out
product domains for cross-domain transfer).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter, SourceNotAvailable


class Converter(BaseConverter):
    DATASET_ID = "mirobench"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = None  # no data released -> no fixture
    DOC = {
        "dataset_id": "mirobench",
        "original_fields": [
            {"name": "seed_post", "meaning": "root Reddit post starting a thread (875 seeds)"},
            {"name": "domain", "meaning": "product domain (5 domains) — cross-domain hold-out"},
            {"name": "reply", "meaning": "a comment node: id, parent_id, author, text, score, created_utc"},
            {"name": "tree", "meaning": "the reply-tree structure (parent->child edges) per thread"},
        ],
        "canonical_mapping": [
            {"source_field": "seed_post", "canonical_path": "payload.input.seed_post (DISCUSSION_TREE) / context.current_observation"},
            {"source_field": "reply.parent_id/id (before cutoff)", "canonical_path": "payload.input.partial_tree.edges/nodes"},
            {"source_field": "reply.parent_id/id (after cutoff)", "canonical_path": "payload.target.tree (DISCUSSION_TREE)"},
            {"source_field": "reply.text", "canonical_path": "payload.target.message_text (NEXT_MESSAGE) / context.known_history"},
            {"source_field": "thread aggregate (n_replies, depth, branching)", "canonical_path": "payload.target.aggregate_metrics (POPULATION_RESPONSE)"},
            {"source_field": "seed_post id / domain", "canonical_path": "episode.topic_id (thread) / experiment_id (domain) — isolation keys"},
        ],
        "tasks_produced": [
            "PREDICT_DISCUSSION_TREE", "PREDICT_NEXT_MESSAGE", "PREDICT_POPULATION_RESPONSE",
            "PREDICT_TRAJECTORY_CONTINUATION",
        ],
        "unavailable_fields": [
            "ALL fields (dataset not publicly released) — no records are produced until release",
            "on release: full node text may be Reddit-ToS restricted for redistribution",
        ],
        "chronology_rules": "Seed-post cutoff: the observed partial tree (replies up to the cutoff, ordered by created_utc) is input; later replies are the target. NEXT_MESSAGE for a reply exposes only its ancestor path + earlier siblings.",
        "split_key": "thread / seed post (topic_id); also hold out product domains (experiment_id)",
        "leakage_risks": [
            "replies within a thread are correlated -> keep a seed's whole tree in one split",
            "post-cutoff replies must never appear in the observed partial_tree/history",
        ],
        "known_limitations": [
            "dataset not publicly released; this is a documented intended mapping, not a live converter",
            "underlying Reddit content is ToS-restricted -> local conversion only, no redistribution on release",
        ],
        "license_implications": "License not stated; underlying Reddit content under Reddit ToS (no redistribution). Convert locally only; do NOT train/redistribute until terms are verified.",
        "training_suitability": "blocked",
        "assumptions": [
            "on release: each reply carries a parent_id enabling reply-tree reconstruction with a seed-post cutoff",
        ],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        raise SourceNotAvailable(
            "mirobench data not publicly released; see registry blockers "
            "(watch arXiv 2606.14715; verify Reddit-ToS compliance before any redistribution)."
        )
