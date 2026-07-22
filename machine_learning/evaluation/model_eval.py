"""Model evaluation harness (GPU path — built, not launched in this task).

Compares four predictors on the held-out test splits, per the readiness spec:
  * the base open-weight model (no adapter),
  * the fine-tuned adapter,
  * a prompted API model (e.g. DeepSeek) — interface stub, no key used here,
  * the current SWORLDMODEL actor policy — interface stub (integration is out of scope).

For the base + adapter paths it generates a completion for each test example's prompt,
parses it against the target, and dispatches to the task-family metrics. Heavy imports are
lazy so this module imports without torch.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..examples.formatters.sft import format_record
from ..normalization.common.parquet_io import iter_records
from ..config import normalized_dir
from ..splitting.policies import load_split_table
from . import belief_change, causal, messages, next_action, population, timing, trajectories


def _test_records(dataset_id: str, split: str, limit: int | None):
    table = {r["record_id"]: r["split"] for r in load_split_table(dataset_id)}
    out = []
    for r in iter_records(normalized_dir(dataset_id)):
        if table.get(r["record_id"]) == split:
            out.append(r)
            if limit and len(out) >= limit:
                break
    return out


def generate_predictions(model, tokenizer, records, *, max_new_tokens: int = 64,
                         device: str = "cuda", max_len: int = 2048) -> list[dict]:
    """Greedy-generate a completion per record (base or adapter model)."""
    import torch
    preds = []
    model.eval()
    for r in records:
        fx = format_record(r)
        ids = tokenizer(fx.prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append({"record_id": r["record_id"], "task_type": r["task_type"],
                      "prediction": text.strip(), "target": r["payload"]["target"]})
    return preds


def score_predictions(preds: list[dict]) -> dict:
    """Group predictions by task family and dispatch to the right metric module."""
    by_task = defaultdict(list)
    for p in preds:
        by_task[p["task_type"]].append(p)
    results = {}
    for task, ps in by_task.items():
        results[task] = _score_task(task, ps)
    return results


def _parse_target(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return text


def _score_task(task, ps):
    if task in ("PREDICT_NEXT_ACTION", "PREDICT_NEXT_CHOICE", "PREDICT_NEXT_SPEAKER"):
        key = {"PREDICT_NEXT_ACTION": "action_type", "PREDICT_NEXT_CHOICE": "choice",
               "PREDICT_NEXT_SPEAKER": "speaker_id"}[task]
        pairs = []
        for p in ps:
            pred = _parse_target(p["prediction"])
            pv = pred.get(key) if isinstance(pred, dict) else pred
            pairs.append((str(pv), str(p["target"].get(key))))
        return next_action.evaluate(pairs)
    if task == "PREDICT_NEXT_MESSAGE":
        return messages.evaluate([(p["prediction"], p["target"].get("message_text", "")) for p in ps])
    if task == "PREDICT_RESPONSE_OR_NONRESPONSE":
        pairs = []
        for p in ps:
            pred = _parse_target(p["prediction"])
            pv = 1.0 if (isinstance(pred, dict) and pred.get("responded")) else 0.0
            pairs.append((pv, 1.0 if p["target"].get("responded") else 0.0))
        from . import metrics as M
        return {"n": len(pairs), "brier": M.brier_binary(pairs), "accuracy": M.accuracy(
            [((1.0 if pv >= .5 else 0.0), t) for pv, t in pairs])}
    if task == "PREDICT_TIME_TO_ACTION":
        pairs = []
        for p in ps:
            pred = _parse_target(p["prediction"])
            pv = pred.get("time_to_action_seconds") if isinstance(pred, dict) else None
            tv = p["target"].get("time_to_action_seconds")
            if isinstance(pv, (int, float)) and isinstance(tv, (int, float)):
                pairs.append((pv, tv))
        return timing.evaluate(pairs)
    if task == "PREDICT_FINAL_OUTCOME":
        pairs = [(json.dumps(_parse_target(p["prediction"]), sort_keys=True),
                  json.dumps(p["target"].get("outcome"), sort_keys=True)) for p in ps]
        return next_action.evaluate(pairs)
    if task in ("PREDICT_POPULATION_RESPONSE", "PREDICT_POPULATION_TIME_SERIES"):
        return {"n": len(ps), "note": "population metrics require parsed distributions; see population.evaluate"}
    if task in ("PREDICT_INTERVENTION_EFFECT", "PREDICT_POLICY_VALUE", "RANK_CANDIDATE_ACTIONS"):
        return {"n": len(ps), "note": "causal metrics; see causal.py (effect_error/ips_value/ranking)"}
    if task == "PREDICT_TRAJECTORY_CONTINUATION":
        return {"n": len(ps), "note": "trajectory metrics; see trajectories.evaluate"}
    return {"n": len(ps), "note": "no scorer wired for this task"}


def evaluate_adapter(config, adapter_dir: str, dataset_id: str, *, split: str = "test_in_domain",
                     limit: int | None = 500) -> dict:
    """GPU path: load base+adapter, generate + score on a test split."""
    from ..training.model_registry import ModelSpec, load_tokenizer, cuda_available
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    import yaml
    cfg = yaml.safe_load(Path(config).read_text()) if isinstance(config, str) and Path(config).exists() else config
    spec = ModelSpec.from_config(cfg)
    tok = load_tokenizer(spec)
    device = "cuda" if cuda_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(spec.base_model, revision=spec.revision).to(device)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    recs = _test_records(dataset_id, split, limit)
    preds = generate_predictions(model, tok, recs, device=device)
    return {"dataset": dataset_id, "split": split, "n": len(preds), "scores": score_predictions(preds)}


# ---- comparison predictors (stubs with clear guidance) --------------------------------
def prompted_api_baseline(*_a, **_k):
    raise NotImplementedError(
        "Prompted-API (e.g. DeepSeek) baseline is intentionally NOT run here: it would use a "
        "paid API. Wire a client in evaluation/model_eval.py and pass predictions to "
        "score_predictions(). No API key is used in this task.")


def sworldmodel_actor_baseline(*_a, **_k):
    raise NotImplementedError(
        "SWORLDMODEL actor-policy baseline requires the integration contract "
        "(docs/integration_contract.md predict_actor_behavior). Not wired into the "
        "production runtime during this task.")
