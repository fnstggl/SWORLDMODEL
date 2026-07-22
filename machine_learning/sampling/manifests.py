"""Build versioned, balanced training-view manifests.

A training view is a named, reproducible mixture (e.g. ``actor_choice_v1``) defined by a
config in ``configs/sampling/<view>.yaml``. Building it:

1. selects datasets that (a) produce the view's tasks and (b) are training-eligible
   (registry role + license + human approval) — or, for the cross-dataset-eval view, the
   eval-only datasets;
2. reads their split tables, restricts to the view's split (``train`` by default) and
   tasks, and applies per-dataset / per-participant caps;
3. computes anti-dominance sampling weights (see balanced_sampler);
4. writes the full record list to working storage (gitignored) and a small, committed
   SUMMARY (datasets in/out + why, counts, est. tokens, weights, license notes).

``preview=True`` ignores human approval so the readiness report can show what a view WOULD
contain; the record list it emits is still marked ``pending_approval`` and is never used
for a real run.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

from ..config import ARTIFACTS_DIR, CONFIGS_DIR, MANIFESTS_DIR, READINESS_DIR
from ..io_utils import write_json, write_jsonl
from ..registry_io import get_dataset, load_datasets, training_eligibility
from ..splitting.policies import load_split_table
from .balanced_sampler import SamplingConfig, apply_caps, compute_weights

VIEW_DIR = CONFIGS_DIR / "sampling"
_TOKENS_PER_RECORD_EST = 320  # rough SFT token estimate per example (prompt+completion)


def _load_view(view: str) -> dict:
    p = VIEW_DIR / f"{view}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no sampling view config: {p}")
    return yaml.safe_load(p.read_text())


def _cap_by_participant(rows: list[dict], cap: int | None) -> list[dict]:
    if not cap:
        return rows
    seen: dict[str, int] = defaultdict(int)
    out = []
    for r in sorted(rows, key=lambda r: r["record_id"]):  # deterministic
        key = r.get("actor_id") or r.get("episode_id")
        if seen[key] < cap:
            out.append(r)
            seen[key] += 1
    return out


def build_view(view: str, *, preview: bool = False) -> dict:
    cfg_doc = _load_view(view)
    tasks = set(cfg_doc.get("tasks", []))
    is_eval_view = bool(cfg_doc.get("eval_only", False))
    scfg = SamplingConfig(
        temperature=cfg_doc.get("temperature", 0.7),
        task_weights=cfg_doc.get("task_weights", {}) or {},
        dataset_weights=cfg_doc.get("dataset_weights", {}) or {},
        per_dataset_cap=cfg_doc.get("per_dataset_cap"),
        per_participant_cap=cfg_doc.get("per_participant_cap"),
        rare_task_min_fraction=cfg_doc.get("rare_task_min_fraction", 0.0),
        max_dataset_dominance=cfg_doc.get("max_dataset_dominance", 0.6),
        target_examples=cfg_doc.get("target_examples"),
    )
    want_split = cfg_doc.get("split", "test_cross_dataset" if is_eval_view else "train")

    include = cfg_doc.get("include_datasets")
    exclude = set(cfg_doc.get("exclude_datasets", []) or [])

    included: list[str] = []
    excluded: list[dict] = []
    selected_rows: list[dict] = []
    counts: dict = {}

    for did, entry in load_datasets().items():
        if did in exclude:
            excluded.append({"dataset": did, "reason": "explicitly excluded"})
            continue
        if include is not None and did not in include:
            continue
        role = entry.get("dataset_role")
        # eligibility differs for eval vs training views
        if is_eval_view:
            if role not in ("CROSS_DATASET_EVAL_ONLY", "LICENSE_RESTRICTED_EVAL_ONLY"):
                continue
        else:
            elig, reason = training_eligibility(did, require_approval=not preview)
            if not elig:
                excluded.append({"dataset": did, "reason": reason})
                continue

        rows = [r for r in load_split_table(did)
                if r["split"] == want_split and (not tasks or r["task_type"] in tasks)]
        if not rows:
            excluded.append({"dataset": did, "reason": f"no records in split={want_split} for tasks"})
            continue
        rows = _cap_by_participant(rows, scfg.per_participant_cap)
        included.append(did)
        selected_rows.extend(rows)
        for r in rows:
            counts[(did, r["task_type"])] = counts.get((did, r["task_type"]), 0) + 1

    weights = compute_weights(counts, scfg) if not is_eval_view else {
        k: v / max(sum(counts.values()), 1) for k, v in counts.items()}
    capped_counts = apply_caps(counts, scfg)

    # attach a per-record weight (group weight / group size)
    group_size = defaultdict(int)
    for r in selected_rows:
        group_size[(r["dataset_id"], r["task_type"])] += 1
    manifest_records = []
    for r in selected_rows:
        g = (r["dataset_id"], r["task_type"])
        w = weights.get(g, 0.0) / max(group_size[g], 1)
        manifest_records.append({"record_id": r["record_id"], "dataset": r["dataset_id"],
                                 "task": r["task_type"], "split": r["split"], "weight": w})

    n = len(manifest_records)
    summary = {
        "view": view,
        "preview": preview,
        "eval_only_view": is_eval_view,
        "split": want_split,
        "tasks": sorted(tasks),
        "datasets_included": included,
        "datasets_excluded": excluded,
        "n_raw_examples": sum(counts.values()),
        "n_sampled_examples": sum(capped_counts.values()) if not is_eval_view else n,
        "n_manifest_records": n,
        "estimated_tokens": n * _TOKENS_PER_RECORD_EST,
        "per_group_weight": {f"{d}:{t}": round(w, 5) for (d, t), w in sorted(weights.items())},
        "per_group_count": {f"{d}:{t}": c for (d, t), c in sorted(counts.items())},
        "sampling": {"temperature": scfg.temperature, "per_dataset_cap": scfg.per_dataset_cap,
                     "per_participant_cap": scfg.per_participant_cap,
                     "max_dataset_dominance": scfg.max_dataset_dominance,
                     "rare_task_min_fraction": scfg.rare_task_min_fraction},
        "license_notes": _license_notes(included),
        "adapter": cfg_doc.get("adapter"),
        "manifest_hash": _hash_records(manifest_records),
    }

    # full manifest -> working storage (gitignored); small summary -> committed
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(MANIFESTS_DIR / f"{view}.records.jsonl", manifest_records)
    write_json(MANIFESTS_DIR / f"{view}.summary.json", summary)
    (ARTIFACTS_DIR / "manifests").mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACTS_DIR / "manifests" / f"{view}.summary.json", summary)
    READINESS_DIR.mkdir(parents=True, exist_ok=True)
    return summary


def _license_notes(datasets: list[str]) -> dict:
    notes = {}
    for d in datasets:
        e = get_dataset(d)
        notes[d] = {"license": e.get("license"), "license_class": e.get("license_class"),
                    "commercial_use_allowed": e.get("commercial_use_allowed")}
    return notes


def _hash_records(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r["record_id"]):
        h.update(r["record_id"].encode())
    return h.hexdigest()[:16]


def load_manifest_records(view: str) -> list[dict]:
    p = MANIFESTS_DIR / f"{view}.records.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
