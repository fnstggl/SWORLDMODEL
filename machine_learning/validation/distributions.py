"""Distribution + imbalance statistics for a normalized dataset.

Feeds the per-dataset audit report and the class-imbalance / dataset-dominance checks:
task mix, action-type frequencies, inactivity frequency, response-time distribution,
context-length distribution, outcome distribution, and missing-field tallies.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..config import normalized_dir
from ..examples.formatters.sft import format_record
from ..normalization.common.parquet_io import iter_records
from ..tasks import NO_ACTION, NO_RESPONSE


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    vs = sorted(values)
    n = len(vs)

    def q(p):
        return vs[min(int(p * n), n - 1)]
    return {"min": vs[0], "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
            "p95": q(0.95), "max": vs[-1], "mean": round(sum(vs) / n, 2), "n": n}


@dataclass
class DistributionReport:
    dataset_id: str
    n_records: int = 0
    task_counts: dict = field(default_factory=dict)
    action_type_counts: dict = field(default_factory=dict)
    outcome_counts: dict = field(default_factory=dict)
    missing_field_counts: dict = field(default_factory=dict)
    inactivity: dict = field(default_factory=dict)
    response_time_seconds: dict = field(default_factory=dict)
    context_length_chars: dict = field(default_factory=dict)
    n_participants: int = 0
    n_episodes: int = 0
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def check_dataset(dataset_id: str, *, limit: int | None = None) -> DistributionReport:
    rep = DistributionReport(dataset_id=dataset_id)
    tasks: Counter = Counter()
    actions: Counter = Counter()
    outcomes: Counter = Counter()
    missing: Counter = Counter()
    ctx_lens: list[float] = []
    resp_times: list[float] = []
    n_inactive = 0
    n_action_or_resp = 0
    participants: set = set()
    episodes: set = set()

    for i, r in enumerate(iter_records(normalized_dir(dataset_id))):
        if limit and i >= limit:
            break
        rep.n_records += 1
        task = r["task_type"]
        tasks[task] += 1
        episodes.add(r["episode"]["episode_id"])
        for p in r["episode"].get("participant_ids", []):
            participants.add(p)
        au = r["decision_unit"].get("actor_id")
        if au:
            participants.add(au)
        for m in r["data_quality"].get("missing_fields", []):
            missing[m] += 1

        tgt = r["payload"].get("target", {})
        if task == "PREDICT_NEXT_ACTION":
            at = tgt.get("action_type", "?")
            actions[at] += 1
            n_action_or_resp += 1
            if at in (NO_ACTION,) or tgt.get("acted") is False:
                n_inactive += 1
        elif task == "PREDICT_RESPONSE_OR_NONRESPONSE":
            n_action_or_resp += 1
            if tgt.get("responded") is False:
                n_inactive += 1
            lat = tgt.get("latency_seconds")
            if isinstance(lat, (int, float)):
                resp_times.append(float(lat))
        elif task == "PREDICT_TIME_TO_ACTION":
            t = tgt.get("time_to_action_seconds")
            if isinstance(t, (int, float)):
                resp_times.append(float(t))
        elif task == "PREDICT_FINAL_OUTCOME":
            oc = tgt.get("outcome")
            key = str(oc.get("outcome_type") if isinstance(oc, dict) else oc)[:40]
            outcomes[key] += 1

        # context length (sampled to bound cost)
        if i % 5 == 0:
            try:
                ctx_lens.append(float(len(format_record(r).prompt)))
            except Exception:  # noqa: BLE001
                pass

    rep.task_counts = dict(tasks)
    rep.action_type_counts = dict(actions.most_common(30))
    rep.outcome_counts = dict(outcomes.most_common(30))
    rep.missing_field_counts = dict(missing)
    rep.context_length_chars = _quantiles(ctx_lens)
    rep.response_time_seconds = _quantiles(resp_times)
    rep.inactivity = {"n_inactive": n_inactive, "n_action_or_response": n_action_or_resp,
                      "inactivity_rate": round(n_inactive / n_action_or_resp, 4) if n_action_or_resp else 0.0}
    rep.n_participants = len(participants)
    rep.n_episodes = len(episodes)

    # imbalance warnings
    if tasks:
        top = max(tasks.values()) / sum(tasks.values())
        if top > 0.9 and len(tasks) > 1:
            rep.warnings.append(f"task imbalance: one task is {top:.0%} of records")
    return rep
