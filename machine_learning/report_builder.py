"""Per-dataset + global human-audit reports (Markdown + machine-readable JSON).

Each report is a single place a human can verify a dataset end to end: official source +
license, acquisition status, record/episode/participant counts, distributions (labels,
actions, inactivity, timing, context length, outcomes), split sizes, leakage results, the
converter's stated assumptions + limitations, 50 rendered examples, the 25 most-suspicious
examples, and an explicit train/eval recommendation.
"""
from __future__ import annotations

import json

from .config import AUDIT_DIR, HUMAN_REVIEW_DIR
from .examples.formatters.sft import format_record
from .io_utils import read_json, read_jsonl, write_json
from .normalization.base import load_converter
from .normalization.pipeline import load_normalization_manifest
from .registry_io import get_dataset, training_eligibility
from .acquisition.verify import load_source_manifest
from .splitting.policies import load_split_table
from .validation import distributions as dist_mod
from .validation import leakage as leak_mod


def _split_counts(dataset_id: str) -> dict:
    counts: dict = {}
    for r in load_split_table(dataset_id):
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    return counts


def build_dataset_audit(dataset_id: str, *, limit: int | None = 30000) -> str:
    entry = get_dataset(dataset_id)
    conv = load_converter(dataset_id)
    doc = conv.DOC if conv else {}
    src = load_source_manifest(dataset_id) or {}
    norm = load_normalization_manifest(dataset_id) or {}
    split_counts = _split_counts(dataset_id)
    leak = leak_mod.check_dataset(dataset_id) if split_counts else {"ok": None, "n_records": 0}
    dist = dist_mod.check_dataset(dataset_id, limit=limit).as_dict() if norm.get("n_valid") else {}
    elig, ereason = training_eligibility(dataset_id, require_approval=False)

    sample = list(read_jsonl(HUMAN_REVIEW_DIR / f"{dataset_id}.sample.jsonl")) if (
        HUMAN_REVIEW_DIR / f"{dataset_id}.sample.jsonl").exists() else []
    suspicious = list(read_jsonl(HUMAN_REVIEW_DIR / f"{dataset_id}.suspicious.jsonl")) if (
        HUMAN_REVIEW_DIR / f"{dataset_id}.suspicious.jsonl").exists() else []

    machine = {
        "dataset_id": dataset_id,
        "official_source": entry.get("official_data_source"),
        "official_paper": entry.get("official_paper"),
        "license": entry.get("license"),
        "license_class": entry.get("license_class"),
        "dataset_role": entry.get("dataset_role"),
        "conversion_status": entry.get("conversion_status"),
        "acquisition_status": src.get("status"),
        "raw_files": len(src.get("files", [])),
        "raw_bytes": src.get("total_bytes"),
        "normalized": {"n_valid": norm.get("n_valid"), "n_quarantined": norm.get("n_quarantined"),
                       "n_episodes": norm.get("n_episodes"), "n_actors": norm.get("n_actors"),
                       "task_counts": norm.get("task_counts")},
        "distributions": dist,
        "split_counts": split_counts,
        "leakage": leak,
        "assumptions": doc.get("assumptions", []),
        "known_limitations": doc.get("known_limitations", []),
        "unavailable_fields": doc.get("unavailable_fields", []),
        "training_recommendation": _train_rec(entry, elig, ereason, leak, norm),
        "evaluation_recommendation": _eval_rec(entry),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(AUDIT_DIR / f"{dataset_id}.json", machine)

    md = _render_md(dataset_id, entry, machine, doc, sample, suspicious)
    (AUDIT_DIR / f"{dataset_id}.md").write_text(md)
    return str(AUDIT_DIR / f"{dataset_id}.md")


def _train_rec(entry, elig, ereason, leak, norm) -> str:
    if entry.get("dataset_role") not in ("TRAIN_CANDIDATE", "VALIDATION_CANDIDATE"):
        return f"NOT for training (role={entry.get('dataset_role')})."
    if not elig:
        return f"Blocked from training: {ereason}."
    if leak.get("ok") is False:
        return "Blocked: leakage check failed — do not train until fixed."
    if not norm.get("n_valid"):
        return "Converter ready but not yet normalized at scale (storage-blocked). Normalize, then human-approve."
    return "Eligible for training pending human approval (training_approvals.yaml)."


def _eval_rec(entry) -> str:
    role = entry.get("dataset_role")
    if role in ("CROSS_DATASET_EVAL_ONLY", "LICENSE_RESTRICTED_EVAL_ONLY"):
        return "Reserved as held-out EVALUATION data (never in training manifests)."
    if role == "ACCESS_BLOCKED":
        return "Cannot evaluate — data not accessible."
    return "Usable for in-domain evaluation on its own test split."


def _render_md(dataset_id, entry, m, doc, sample, suspicious) -> str:
    L = [f"# Audit — {entry.get('official_name', dataset_id)}", "",
         f"- **id**: `{dataset_id}`  |  **role**: {entry.get('dataset_role')}  |  "
         f"**status**: {entry.get('conversion_status')}",
         f"- **official source**: {entry.get('official_data_source')}",
         f"- **paper**: {entry.get('official_paper')}",
         f"- **license**: {entry.get('license')} (`{entry.get('license_class')}`) — "
         f"commercial={entry.get('commercial_use_allowed')}, derivatives={entry.get('derivatives_allowed')}",
         f"- **acquisition**: {m['acquisition_status']} ({m['raw_files']} raw files, "
         f"{m['raw_bytes'] or 0} bytes)", ""]
    n = m["normalized"]
    L += ["## Normalized data", "",
          f"- examples: **{n.get('n_valid')}**  |  quarantined: {n.get('n_quarantined')}  |  "
          f"episodes: {n.get('n_episodes')}  |  actors: {n.get('n_actors')}",
          f"- task counts: `{n.get('task_counts')}`",
          f"- split sizes: `{m['split_counts']}`", ""]
    d = m.get("distributions", {})
    if d:
        L += ["## Distributions", "",
              f"- inactivity: `{d.get('inactivity')}`",
              f"- action types: `{dict(list((d.get('action_type_counts') or {}).items())[:8])}`",
              f"- outcomes: `{dict(list((d.get('outcome_counts') or {}).items())[:8])}`",
              f"- response-time (s): `{d.get('response_time_seconds')}`",
              f"- context length (chars): `{d.get('context_length_chars')}`",
              f"- missing fields: `{d.get('missing_field_counts')}`", ""]
    L += ["## Leakage", "", f"- result: `{m['leakage']}`", "",
          "## Converter assumptions", ""]
    for a in doc.get("assumptions", []) or ["(none stated)"]:
        L.append(f"- {a}")
    L += ["", "## Known limitations", ""]
    for a in doc.get("known_limitations", []) or ["(none stated)"]:
        L.append(f"- {a}")
    L += ["", "## Unavailable fields (stored null, never fabricated)", ""]
    for a in doc.get("unavailable_fields", []) or ["(none)"]:
        L.append(f"- {a}")
    L += ["", "## Recommendations", "",
          f"- **training**: {m['training_recommendation']}",
          f"- **evaluation**: {m['evaluation_recommendation']}", ""]
    L += ["## 50 rendered examples (human review)", ""]
    for i, rec in enumerate(sample[:50]):
        fx = format_record(rec, max_history_events=6)
        L += [f"### Example {i+1} — {rec['task_type']} — `{rec['record_id']}`", "```",
              fx.prompt[-900:], "--- TARGET ---", fx.completion[:400], "```", ""]
    if suspicious:
        L += ["## 25 most-suspicious examples (warnings / possible leakage)", ""]
        for i, rec in enumerate(suspicious[:25]):
            L += [f"- `{rec['record_id']}` ({rec['task_type']}): "
                  f"warnings={rec['data_quality'].get('warnings')} "
                  f"possible_leakage={rec['data_quality'].get('possible_leakage')}"]
    return "\n".join(L) + "\n"


def build_global_audit() -> str:
    """One report comparing all datasets."""
    from .registry_io import load_datasets
    rows = []
    for did in load_datasets():
        p = AUDIT_DIR / f"{did}.json"
        if p.exists():
            rows.append(read_json(p))
    L = ["# Global dataset audit", "",
         "| dataset | role | acquired | examples | quarantined | leakage_ok | train_rec |",
         "|---|---|---|---:|---:|---|---|"]
    for r in rows:
        n = r.get("normalized", {})
        L.append(f"| {r['dataset_id']} | {r.get('dataset_role')} | {r.get('acquisition_status')} | "
                 f"{n.get('n_valid') or 0} | {n.get('n_quarantined') or 0} | "
                 f"{r.get('leakage',{}).get('ok')} | {r.get('training_recommendation','')[:40]} |")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "GLOBAL_AUDIT.md").write_text("\n".join(L) + "\n")
    return str(AUDIT_DIR / "GLOBAL_AUDIT.md")
