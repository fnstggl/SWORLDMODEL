"""Evaluation reporting: run non-learned baselines over the real test splits and emit a
committed report. This produces the reference numbers the fine-tuned 8B model must beat,
computable now on CPU.
"""
from __future__ import annotations

from ..config import READINESS_DIR, REPORTS_DIR
from ..io_utils import write_json
from ..registry_io import load_datasets
from ..splitting.policies import load_split_table
from .baselines import evaluate_dataset


def run_baselines(dataset_ids: list[str] | None = None) -> dict:
    if dataset_ids is None:
        dataset_ids = [d for d in load_datasets() if load_split_table(d)]
    rows = []
    for d in dataset_ids:
        try:
            rows.extend(evaluate_dataset(d))
        except Exception as e:  # noqa: BLE001 - one dataset must not abort the report
            rows.append({"dataset": d, "task": "*", "error": str(e)[:160]})
    out = {"baselines": rows, "n_datasets": len(dataset_ids)}
    (REPORTS_DIR / "readiness").mkdir(parents=True, exist_ok=True)
    write_json(READINESS_DIR / "baselines.json", out)
    _write_markdown(rows)
    return out


def _write_markdown(rows: list[dict]) -> None:
    lines = ["# Baseline evaluation (non-learned reference numbers)", "",
             "These are the floors the fine-tuned model must beat, computed on the real",
             "in-domain / cross-dataset test splits. Generation tasks (messages, trajectories)",
             "report only a trivial reference — meaningful eval requires the model.", "",
             "| dataset | task | n_test | baseline | key metric |",
             "|---|---|---:|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['dataset']} | {r['task']} | - | ERROR | {r['error']} |")
            continue
        metric = ""
        for k in ("accuracy", "brier", "token_f1", "reward_mae", "rate_mae", "timing_mae", "macro_f1"):
            if k in r:
                metric = f"{k}={r[k]}"
                break
        if not metric and r.get("note"):
            metric = r["note"]
        lines.append(f"| {r.get('dataset')} | {r.get('task')} | {r.get('n_test','-')} | "
                     f"{r.get('baseline','-')} | {metric} |")
    (READINESS_DIR / "baselines.md").write_text("\n".join(lines) + "\n")
