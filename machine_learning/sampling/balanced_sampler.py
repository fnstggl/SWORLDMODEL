"""Balanced-sampling weight computation (pure functions).

The largest dataset must NOT dominate a training mixture just because it has the most
rows. Given per-(dataset, task) example counts, these functions compute per-group sampling
weights honoring:

* **temperature** over dataset sizes (T=1 proportional; T<1 flattens toward uniform);
* **task weights** (up/down-weight whole tasks);
* **per-dataset caps** and **rare-task minimums**;
* a **max-dominance** ceiling (no dataset may exceed a target fraction of the mixture).

Weights are expected sampling fractions that sum to 1 across all groups. They are separate
from *caps*, which physically bound how many rows a group contributes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingConfig:
    temperature: float = 0.7
    task_weights: dict = field(default_factory=dict)      # task -> multiplier
    dataset_weights: dict = field(default_factory=dict)   # dataset -> multiplier
    per_dataset_cap: int | None = None
    per_participant_cap: int | None = None
    rare_task_min_fraction: float = 0.0                   # each task gets >= this fraction of weight
    max_dataset_dominance: float = 0.6                    # no dataset > this fraction of weight
    target_examples: int | None = None


def apply_caps(counts: dict[tuple[str, str], int], cfg: SamplingConfig) -> dict[tuple[str, str], int]:
    """Cap the effective example count per (dataset, task) by the per-dataset cap.

    The per-dataset cap is shared across that dataset's tasks in proportion to their raw
    counts (so a cap of 100k on a dataset with 3 tasks splits 100k across them by size).
    """
    if not cfg.per_dataset_cap:
        return dict(counts)
    by_dataset: dict[str, int] = {}
    for (ds, _task), n in counts.items():
        by_dataset[ds] = by_dataset.get(ds, 0) + n
    capped: dict[tuple[str, str], int] = {}
    for (ds, task), n in counts.items():
        total = by_dataset[ds]
        if total <= cfg.per_dataset_cap:
            capped[(ds, task)] = n
        else:
            capped[(ds, task)] = max(1, round(n * cfg.per_dataset_cap / total))
    return capped


def compute_weights(counts: dict[tuple[str, str], int], cfg: SamplingConfig) -> dict[tuple[str, str], float]:
    """Return normalized sampling weights per (dataset, task)."""
    if not counts:
        return {}
    capped = apply_caps(counts, cfg)

    # dataset temperature weights
    ds_totals: dict[str, float] = {}
    for (ds, _t), n in capped.items():
        ds_totals[ds] = ds_totals.get(ds, 0.0) + n
    ds_temp = {ds: (tot ** cfg.temperature) * cfg.dataset_weights.get(ds, 1.0)
               for ds, tot in ds_totals.items()}
    z = sum(ds_temp.values()) or 1.0
    ds_weight = {ds: w / z for ds, w in ds_temp.items()}

    # split each dataset's weight across its tasks by (task-weighted) size
    raw: dict[tuple[str, str], float] = {}
    for ds in ds_totals:
        tasks = {t: n for (d, t), n in capped.items() if d == ds}
        tw = {t: n * cfg.task_weights.get(t, 1.0) for t, n in tasks.items()}
        s = sum(tw.values()) or 1.0
        for t, w in tw.items():
            raw[(ds, t)] = ds_weight[ds] * (w / s)

    weights = _enforce_dominance(raw, cfg.max_dataset_dominance)
    weights = _enforce_rare_task_min(weights, cfg.rare_task_min_fraction)
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _enforce_dominance(weights: dict[tuple[str, str], float], ceiling: float) -> dict[tuple[str, str], float]:
    """Cap each dataset's total share at ``ceiling``, redistributing the excess to the
    others proportionally. Mass-conserving, so the sum is unchanged and repeated capping
    converges to <= ceiling for every dataset."""
    if ceiling >= 1.0:
        return dict(weights)
    out = dict(weights)
    for _ in range(200):
        total = sum(out.values()) or 1.0
        by_ds: dict[str, float] = {}
        for (ds, _t), w in out.items():
            by_ds[ds] = by_ds.get(ds, 0.0) + w
        worst = max(by_ds, key=lambda d: by_ds[d])
        if by_ds[worst] / total <= ceiling + 1e-9:
            break
        excess = by_ds[worst] - ceiling * total
        factor = (ceiling * total) / by_ds[worst]
        for k in out:
            if k[0] == worst:
                out[k] *= factor
        others_sum = total - by_ds[worst]
        if others_sum > 0:
            for k in out:
                if k[0] != worst:
                    out[k] += excess * (out[k] / others_sum)
    return out


def _enforce_rare_task_min(weights: dict[tuple[str, str], float], min_frac: float) -> dict[tuple[str, str], float]:
    """Floor each task's total share at ``min_frac``, taking the deficit from the other
    tasks proportionally. Mass-conserving + iterative to convergence."""
    if min_frac <= 0:
        return dict(weights)
    out = dict(weights)
    n_tasks = len({t for _d, t in out})
    if min_frac * n_tasks > 1.0 + 1e-9:  # infeasible; back off to equal shares
        min_frac = 1.0 / n_tasks
    for _ in range(200):
        total = sum(out.values()) or 1.0
        by_task: dict[str, float] = {}
        for (_d, t), w in out.items():
            by_task[t] = by_task.get(t, 0.0) + w
        low = [t for t, s in by_task.items() if s / total < min_frac - 1e-9 and s > 0]
        if not low:
            break
        t0 = min(low, key=lambda t: by_task[t])
        deficit = min_frac * total - by_task[t0]
        factor = (min_frac * total) / by_task[t0]
        for k in out:
            if k[1] == t0:
                out[k] *= factor
        others_sum = total - by_task[t0]
        if others_sum > 0:
            for k in out:
                if k[1] != t0:
                    out[k] -= deficit * (out[k] / others_sum)
    return out
