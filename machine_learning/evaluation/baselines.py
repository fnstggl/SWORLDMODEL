"""Non-learned baselines, evaluated on the REAL test splits.

These establish the floor a fine-tuned model must beat — and, unlike the 8B run, they
compute NOW on CPU. Each baseline is fit on the train split and scored on the in-domain
test split, per (dataset, task):

  majority-class          -> NEXT_CHOICE / NEXT_ACTION / NEXT_SPEAKER / FINAL_OUTCOME
  base-rate (Brier/ECE)   -> RESPONSE_OR_NONRESPONSE
  mean/median regressor   -> TIME_TO_ACTION / POLICY_VALUE / POPULATION_RESPONSE
  zero-effect             -> INTERVENTION_EFFECT
  most-frequent-target    -> NEXT_MESSAGE (token-F1; generation tasks need the model)
"""
from __future__ import annotations

from collections import Counter

from ..config import normalized_dir
from ..normalization.common.parquet_io import iter_records
from ..splitting.policies import load_split_table
from . import metrics as M


def _split_records(dataset_id: str, limit: int | None = None):
    table = {r["record_id"]: r["split"] for r in load_split_table(dataset_id)}
    n = 0
    for r in iter_records(normalized_dir(dataset_id)):
        sp = table.get(r["record_id"])
        if sp is None:
            continue
        yield sp, r
        n += 1
        if limit and n >= limit:
            return


def _target_key(r):
    t = r["payload"]["target"]
    task = r["task_type"]
    if task in ("PREDICT_NEXT_CHOICE",):
        return str(t.get("choice"))
    if task == "PREDICT_NEXT_ACTION":
        return str(t.get("action_type"))
    if task == "PREDICT_NEXT_SPEAKER":
        return str(t.get("speaker_id"))
    if task == "PREDICT_FINAL_OUTCOME":
        oc = t.get("outcome")
        return str(oc.get("outcome_type") if isinstance(oc, dict) else oc)
    return None


def evaluate_dataset(dataset_id: str, *, limit: int | None = 40000) -> list[dict]:
    """Return baseline metric rows for every task with a defined baseline."""
    train: dict = {}
    test: dict = {}
    for sp, r in _split_records(dataset_id, limit=limit):
        bucket = train if sp == "train" else (test if sp in ("test_in_domain", "test_cross_dataset") else None)
        if bucket is None:
            continue
        bucket.setdefault(r["task_type"], []).append(r)

    out = []
    for task, test_recs in sorted(test.items()):
        train_recs = train.get(task, [])
        row = _baseline_for(task, train_recs, test_recs)
        if row:
            row.update({"dataset": dataset_id, "task": task, "n_test": len(test_recs),
                        "n_train": len(train_recs)})
            out.append(row)
    return out


def _baseline_for(task, train_recs, test_recs):
    if task in ("PREDICT_NEXT_CHOICE", "PREDICT_NEXT_ACTION", "PREDICT_NEXT_SPEAKER", "PREDICT_FINAL_OUTCOME"):
        counts = Counter(k for k in (_target_key(r) for r in train_recs) if k is not None)
        if not counts:
            return None
        majority = counts.most_common(1)[0][0]
        pairs = [(majority, _target_key(r)) for r in test_recs if _target_key(r) is not None]
        return {"baseline": f"majority-class({majority[:24]})",
                "accuracy": round(M.accuracy(pairs), 4), "macro_f1": round(M.macro_f1(pairs), 4)}

    if task == "PREDICT_RESPONSE_OR_NONRESPONSE":
        rate = _mean([1.0 if r["payload"]["target"].get("responded") else 0.0 for r in train_recs])
        pairs = [(rate, 1.0 if r["payload"]["target"].get("responded") else 0.0) for r in test_recs]
        acc_pairs = [((1.0 if rate >= 0.5 else 0.0), t) for _, t in pairs]
        return {"baseline": f"base-rate({rate:.3f})", "brier": round(M.brier_binary(pairs), 4),
                "ece": round(M.expected_calibration_error(pairs), 4),
                "accuracy": round(M.accuracy(acc_pairs), 4)}

    if task in ("PREDICT_TIME_TO_ACTION",):
        vals = [r["payload"]["target"].get("time_to_action_seconds") for r in train_recs]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return {"baseline": "median-time", "note": "all censored / no timestamps"}
        med = sorted(vals)[len(vals) // 2]
        pairs = [(med, r["payload"]["target"].get("time_to_action_seconds")) for r in test_recs
                 if isinstance(r["payload"]["target"].get("time_to_action_seconds"), (int, float))]
        return {"baseline": f"median-time({med})", "timing_mae": round(M.mae(pairs), 3)}

    if task == "PREDICT_POLICY_VALUE":
        rewards = [_num(r["payload"]["target"].get("reward")) for r in train_recs]
        rewards = [x for x in rewards if x is not None]
        if not rewards:
            return None
        mean = sum(rewards) / len(rewards)
        pairs = [(mean, _num(r["payload"]["target"].get("reward"))) for r in test_recs]
        pairs = [(p, t) for p, t in pairs if t is not None]
        return {"baseline": f"mean-reward({mean:.4f})", "reward_mae": round(M.mae(pairs), 5)}

    if task == "PREDICT_POPULATION_RESPONSE":
        rates = [_num((r["payload"]["target"].get("aggregate_metrics") or {}).get("rate")) for r in train_recs]
        rates = [x for x in rates if x is not None]
        if not rates:
            return {"baseline": "mean-rate", "note": "no scalar rate in aggregate_metrics"}
        mean = sum(rates) / len(rates)
        pairs = [(mean, _num((r["payload"]["target"].get("aggregate_metrics") or {}).get("rate")))
                 for r in test_recs]
        pairs = [(p, t) for p, t in pairs if t is not None]
        return {"baseline": f"mean-rate({mean:.4f})", "rate_mae": round(M.mae(pairs), 5)}

    if task == "PREDICT_INTERVENTION_EFFECT":
        return {"baseline": "zero-effect", "note": "predict no effect; effect MAE requires model estimate vs realized arm"}

    if task == "PREDICT_NEXT_MESSAGE":
        counts = Counter(r["payload"]["target"].get("message_text", "") for r in train_recs)
        if not counts:
            return None
        top = counts.most_common(1)[0][0]
        f1s = [M.token_f1(top, r["payload"]["target"].get("message_text", "")) for r in test_recs]
        return {"baseline": "most-frequent-message",
                "token_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
                "note": "generation task; real eval needs the model"}
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
