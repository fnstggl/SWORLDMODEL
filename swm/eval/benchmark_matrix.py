"""Benchmark matrix — render (model tier x domain x metric) as one comparable table.

Thin orchestrator over `raw_llm_vs_world_model.run_benchmark`: collect per-domain benchmark results
and format a single matrix so the six tiers are comparable across domains at a glance. Kept
dependency-free; emits a markdown table for the experiment reports.
"""
from __future__ import annotations

from swm.eval.raw_llm_vs_world_model import BenchmarkResult

TIER_ORDER = ["raw_llm", "raw_llm_context", "structured", "calibrated",
              "aggregate_world", "individual_world"]


def matrix(results: dict[str, BenchmarkResult], *, metric: str = "log_loss") -> dict:
    """results: {domain -> BenchmarkResult}. Returns a nested {domain -> {tier -> metric}}."""
    out = {}
    for domain, res in results.items():
        row = {}
        for tier in TIER_ORDER:
            t = res.tiers.get(tier)
            row[tier] = (t.get(metric) if isinstance(t, dict) and metric in t
                         else (t.get("status") if isinstance(t, dict) else None))
        out[domain] = {"base_rate": round(res.base_rate, 4), "n": res.n, **row}
    return out


def to_markdown(results: dict[str, BenchmarkResult], *, metric: str = "log_loss") -> str:
    m = matrix(results, metric=metric)
    tiers = TIER_ORDER
    header = f"| domain (n, base) | " + " | ".join(tiers) + " |"
    sep = "|" + "---|" * (len(tiers) + 1)
    lines = [f"### Benchmark matrix — {metric} (lower=better)", "", header, sep]
    for domain, row in m.items():
        cells = []
        for t in tiers:
            v = row.get(t)
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else str(v or "—"))
        lines.append(f"| {domain} (n={row['n']}, {row['base_rate']}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)
