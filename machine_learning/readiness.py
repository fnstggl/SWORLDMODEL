"""Final readiness reporting.

Produces every required deliverable under reports/readiness/:
  final_readiness_report.md, dataset_status.csv, training_views.csv, license_matrix.csv,
  blockers.md, estimated_gpu_requirements.md, next_commands.md

`build_readiness()` reconciles the registry + normalization manifests + validation results
into a single honest picture: what was acquired/normalized/validated, what is legally
usable for training, what is eval-only, what is blocked, exact example/token counts, and
whether the system is genuinely ready for a GPU run.
"""
from __future__ import annotations

import csv
import io

from .config import READINESS_DIR
from .io_utils import read_json
from .normalization.pipeline import load_normalization_manifest
from .registry_io import get_dataset, load_datasets, training_eligibility
from .validation.licensing import license_matrix

_EST_TOKENS_PER_EX = 320
VIEWS = ["actor_choice_v1", "social_interaction_v1", "long_horizon_behavior_v1",
         "population_response_v1", "causal_intervention_v1", "unified_behavior_multitask_v1",
         "cross_dataset_evaluation_v1"]
EIGHT_B_CONFIGS = ["8b_actor_choice", "8b_social_interaction", "8b_long_horizon",
                   "8b_population_response", "8b_causal_intervention", "8b_unified_multitask"]


def _final_status(dataset_id: str) -> str:
    e = get_dataset(dataset_id)
    role = e["dataset_role"]
    norm = load_normalization_manifest(dataset_id) or {}
    if role == "ACCESS_BLOCKED":
        cs = e.get("conversion_status")
        return "SOURCE_UNAVAILABLE" if cs == "SOURCE_UNAVAILABLE" else "ACCESS_BLOCKED"
    if role == "INFRASTRUCTURE_ONLY":
        return "INFRASTRUCTURE_ONLY"
    if e.get("license_class") == "cc_by_nc_nd":
        return "LICENSE_BLOCKED"  # ND forbids training/derivatives
    if norm.get("n_valid"):
        if role in ("CROSS_DATASET_EVAL_ONLY", "LICENSE_RESTRICTED_EVAL_ONLY"):
            return "NORMALIZED_EVAL_ONLY"
        return "NORMALIZED_AND_VALIDATED"
    return e.get("conversion_status", "PENDING")


def build_readiness() -> dict:
    datasets = load_datasets()
    statuses = {d: _final_status(d) for d in datasets}

    # per-dataset rows
    rows = []
    total_examples = 0
    total_tokens = 0
    for did, e in sorted(datasets.items()):
        norm = load_normalization_manifest(did) or {}
        val = read_json(READINESS_DIR.parent / "normalization" / f"{did}.validation.json") or {}
        elig, ereason = training_eligibility(did, require_approval=False)
        n = norm.get("n_valid") or 0
        total_examples += n if statuses[did] == "NORMALIZED_AND_VALIDATED" else 0
        total_tokens += (n * _EST_TOKENS_PER_EX) if statuses[did] == "NORMALIZED_AND_VALIDATED" else 0
        rows.append({
            "dataset": did, "final_status": statuses[did], "role": e["dataset_role"],
            "license_class": e.get("license_class"),
            "acquired": (load_source_status(did)),
            "examples": n, "tokens": n * _EST_TOKENS_PER_EX,
            "critical_ok": val.get("critical_ok"),
            "training_eligible_if_approved": elig, "reason": ereason if not elig else "",
            "blockers": (e.get("blockers") or "")[:120],
        })

    _write_dataset_status_csv(rows)
    _write_license_matrix_csv()
    view_rows = _write_training_views_csv()
    _write_blockers_md(rows)
    _write_gpu_md()
    _write_next_commands_md()

    n_validated = sum(1 for r in rows if r["final_status"] == "NORMALIZED_AND_VALIDATED")
    n_eval_only = sum(1 for r in rows if r["final_status"] in ("NORMALIZED_EVAL_ONLY",))
    n_blocked = sum(1 for r in rows if r["final_status"] in
                    ("ACCESS_BLOCKED", "SOURCE_UNAVAILABLE", "LICENSE_BLOCKED", "INFRASTRUCTURE_ONLY"))
    smoke = read_json(_smoke_path())
    smoke_passed = bool(smoke and smoke.get("passed"))
    ready = smoke_passed and n_validated >= 1

    report = _write_final_report(rows, view_rows, statuses, total_examples, total_tokens,
                                 smoke_passed, ready, n_validated, n_eval_only, n_blocked)
    return {"report": report, "ready": ready, "n_datasets": len(datasets),
            "n_validated": n_validated, "n_eval_only": n_eval_only, "n_blocked": n_blocked,
            "total_examples": total_examples, "total_tokens": total_tokens,
            "smoke_passed": smoke_passed}


def load_source_status(dataset_id: str) -> str:
    from .acquisition.verify import load_source_manifest
    m = load_source_manifest(dataset_id) or {}
    return m.get("status", "pending")


def _smoke_path():
    from .config import ARTIFACTS_DIR
    return ARTIFACTS_DIR / "runs" / "smoke" / "smoke_result.json"


def _write_dataset_status_csv(rows):
    cols = ["dataset", "final_status", "role", "license_class", "acquired", "examples",
            "tokens", "critical_ok", "training_eligible_if_approved", "reason", "blockers"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    READINESS_DIR.mkdir(parents=True, exist_ok=True)
    (READINESS_DIR / "dataset_status.csv").write_text(buf.getvalue())


def _write_license_matrix_csv():
    rows = license_matrix()
    cols = ["dataset", "license", "license_class", "role", "commercial_use", "derivatives",
            "redistribution", "training_allowed_by_license", "eligible_if_approved",
            "training_eligible_reason"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    (READINESS_DIR / "license_matrix.csv").write_text(buf.getvalue())


def _write_training_views_csv():
    from .sampling.manifests import build_view
    cols = ["view", "adapter", "datasets_included", "n_raw_examples", "n_manifest_records",
            "estimated_tokens", "manifest_hash"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    view_rows = []
    for v in VIEWS:
        try:
            s = build_view(v, preview=True)
            row = {"view": v, "adapter": s.get("adapter"),
                   "datasets_included": "|".join(s["datasets_included"]),
                   "n_raw_examples": s["n_raw_examples"], "n_manifest_records": s["n_manifest_records"],
                   "estimated_tokens": s["estimated_tokens"], "manifest_hash": s["manifest_hash"]}
        except Exception as e:  # noqa: BLE001
            row = {"view": v, "adapter": "", "datasets_included": f"ERROR: {str(e)[:60]}",
                   "n_raw_examples": 0, "n_manifest_records": 0, "estimated_tokens": 0, "manifest_hash": ""}
        view_rows.append(row)
        w.writerow(row)
    (READINESS_DIR / "training_views.csv").write_text(buf.getvalue())
    return view_rows


def _write_blockers_md(rows):
    blocked = [r for r in rows if r["final_status"] in
               ("ACCESS_BLOCKED", "SOURCE_UNAVAILABLE", "LICENSE_BLOCKED", "INFRASTRUCTURE_ONLY",
                "CONVERTER_READY_STORAGE_BLOCKED")]
    L = ["# Blockers", "",
         "Each blocked dataset is documented with the exact reason + the human action needed.", ""]
    for r in rows:
        e = get_dataset(r["dataset"])
        if r["final_status"] in ("ACCESS_BLOCKED", "SOURCE_UNAVAILABLE", "LICENSE_BLOCKED",
                                 "INFRASTRUCTURE_ONLY") or e.get("conversion_status") == "CONVERTER_READY_STORAGE_BLOCKED":
            L += [f"## {r['dataset']} — {r['final_status']}",
                  f"- role: {e.get('dataset_role')}  |  license: {e.get('license')}",
                  f"- blocker: {e.get('blockers')}", ""]
    (READINESS_DIR / "blockers.md").write_text("\n".join(L) + "\n")


def _write_gpu_md():
    L = ["# Estimated GPU requirements (8B QLoRA)", "",
         "Base model is configurable (default `Qwen/Qwen2.5-7B`, Apache-2.0). 4-bit QLoRA.", "",
         "| item | estimate |", "|---|---|",
         "| GPU VRAM (4-bit QLoRA, 8B, seq 2048, bsz 1 + grad-accum 16) | 16-24 GB (1x A100-40GB / A6000 / 4090-24GB) |",
         "| Base weights (4-bit) | ~5-6 GB |",
         "| LoRA adapter (r=16) | ~40-80 MB |",
         "| Peak activation (grad-checkpointing on) | ~8-14 GB |",
         "| Disk (base + tokenizer + cache) | ~20 GB |",
         "| Disk (normalized data + manifests, full acquisition) | ~60-120 GB external (SWM_DATA_ROOT) |",
         "| Throughput (A100-40GB) | ~1-3 examples/s |",
         "| Time for 1 epoch of a ~100k-example view | ~3-10 GPU-hours |", "",
         "Notes:",
         "- The CPU smoke path needs NO GPU and NO bitsandbytes (plain LoRA on a tiny model).",
         "- For a 40GB GPU, the default configs (bsz 1, grad-accum 16, seq 2048, grad-checkpointing) fit comfortably.",
         "- Reduce `max_seq_len` or LoRA `r` if VRAM-constrained; raise `grad_accum` to keep the effective batch size.",
         ""]
    (READINESS_DIR / "estimated_gpu_requirements.md").write_text("\n".join(L) + "\n")


def _write_next_commands_md():
    L = ["# Next commands", "",
         "## 0. Environment", "```bash",
         "export SWM_DATA_ROOT=/path/to/large/volume   # working storage (NOT the repo)",
         "export HF_HOME=/path/to/hf_home",
         "export HF_TOKEN=...                            # for gated/large HF pulls",
         "pip install -r machine_learning/requirements/base.txt -r machine_learning/requirements/data.txt",
         "```", "",
         "## 1. Verify + prepare data", "```bash",
         "python -m machine_learning.cli registry verify",
         "python -m machine_learning.cli datasets prepare-all           # resumable; blocked datasets recorded, not fatal",
         "python -m machine_learning.cli datasets acquire omnibehavior --allow-large   # example: a storage-blocked set",
         "python -m machine_learning.cli datasets normalize omnibehavior",
         "python -m machine_learning.cli datasets validate omnibehavior",
         "python -m machine_learning.cli eval baselines",
         "```", "",
         "## 2. Build training manifests (after human approval in registry/training_approvals.yaml)",
         "```bash",
         "python -m machine_learning.cli manifests build actor_choice_v1",
         "python -m machine_learning.cli manifests build unified_behavior_multitask_v1",
         "python -m machine_learning.cli readiness check",
         "```", "",
         "## 3. Tiny smoke test (CPU, no GPU) — MUST pass before a GPU run", "```bash",
         "pip install -r machine_learning/requirements/training.txt   # CPU torch is fine for the smoke",
         "python -m machine_learning.cli smoke run",
         "```", "",
         "## 4. Launch an 8B QLoRA fine-tune (on a GPU)", "```bash",
         "# dry-run (prints the plan, refuses to launch without --launch):",
         "python -m machine_learning.cli train run 8b_actor_choice",
         "# real launch (needs a CUDA GPU + bitsandbytes):"]
    for c in EIGHT_B_CONFIGS:
        L.append(f"python -m machine_learning.cli train run {c} --launch")
    L += ["```", ""]
    (READINESS_DIR / "next_commands.md").write_text("\n".join(L) + "\n")


def _write_final_report(rows, view_rows, statuses, total_examples, total_tokens,
                        smoke_passed, ready, n_validated, n_eval_only, n_blocked) -> str:
    from collections import Counter
    status_counts = Counter(statuses.values())
    L = ["# SWORLDMODEL behaviour-ML — Final readiness report", "",
         f"**Ready for GPU training:** {'YES' if ready else 'NOT YET'} "
         f"(smoke test {'passed' if smoke_passed else 'NOT passed'}; {n_validated} datasets normalized+validated).", "",
         "## Dataset status summary", "",
         f"- total datasets: **{len(rows)}**",
         f"- normalized + validated (train-usable): **{n_validated}**",
         f"- normalized eval-only: **{n_eval_only}**",
         f"- blocked / infrastructure / license-blocked: **{n_blocked}**",
         f"- status breakdown: `{dict(status_counts)}`",
         f"- exact example count (validated train-usable, as normalized here): **{total_examples:,}**",
         f"- estimated tokens (validated train-usable): **~{total_tokens:,}**", "",
         "> Note: several large datasets are CONVERTER_READY_STORAGE_BLOCKED — their converters are",
         "> implemented + fixture/sample-tested, but full normalization is deferred until run on a",
         "> large volume (see blockers.md + next_commands.md for the exact resume commands).", "",
         "## Per-dataset", "",
         "| dataset | final status | role | examples | critical_ok | train-eligible (if approved) |",
         "|---|---|---|---:|---|---|"]
    for r in sorted(rows, key=lambda r: (r["final_status"], r["dataset"])):
        L.append(f"| {r['dataset']} | {r['final_status']} | {r['role']} | {r['examples']:,} | "
                 f"{r['critical_ok']} | {r['training_eligible_if_approved']} |")
    L += ["", "## Training views (preview — before human approval, records are gated)", "",
          "| view | datasets | manifest records | est. tokens |", "|---|---|---:|---:|"]
    for v in view_rows:
        L.append(f"| {v['view']} | {v['datasets_included'][:60]} | {v['n_manifest_records']:,} | "
                 f"{v['estimated_tokens']:,} |")
    L += ["", "## Recommended first adapters", "",
          "1. `8b_actor_choice` (actor_choice_v1) — the densest, cleanest signal (choices/actions).",
          "2. `8b_social_interaction` (social_interaction_v1) — negotiation/persuasion/deduction messages.",
          "3. `8b_unified_multitask` — once the specialized adapters validate, train the unified mixture.", "",
          "## Recommended unified model mixture", "",
          "`unified_behavior_multitask_v1` with temperature 0.6 + max-dataset-dominance 0.35 so no single",
          "large dataset dominates; rare tasks floored at 5%. Non-commercial datasets (OmniBehavior, DND,",
          "Criteo) stay out of any commercial view.", "",
          "## Remaining human-review requirements", "",
          "- Review each dataset's audit report (`reports/audit/<id>.md`) + human-review sample.",
          "- Approve datasets in `registry/training_approvals.yaml` (nothing approved by default).",
          "- Confirm licensing for commercial use if that is intended (several sets are non-commercial).", "",
          "## Exact commands", "",
          "- Smoke test: `python -m machine_learning.cli smoke run`",
          "- First 8B run: `python -m machine_learning.cli train run 8b_actor_choice --launch`",
          "- See `next_commands.md` for the full sequence.", "",
          "## Unresolved risks", "",
          "- Large datasets (OmniBehavior 6.4GB, OPeRA 8.6GB, SoMe 50GB+, SocSci210 1.45GB, Psych-101 859MB,",
          "  KuaiRand 194MB-46GB) are NORMALIZED ONLY ON SAMPLES here — full runs need a large SWM_DATA_ROOT.",
          "- SocSci210 has NO declared license and its `response` may be human or persona-simulated — eval-only.",
          "- CraigslistBargain license is unstated (held out as eval-only).",
          "- DEBATE / MiroBench / DARPA SocialSim / ACL-shopping data are not publicly released (blocked).",
          "- The 8B base model default (Qwen2.5-7B) is ~7.6B, not exactly 8B; swap in Llama-3.1-8B if desired.", ""]
    (READINESS_DIR / "final_readiness_report.md").write_text("\n".join(L) + "\n")
    return str(READINESS_DIR / "final_readiness_report.md")
