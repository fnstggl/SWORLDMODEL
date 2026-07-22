# Evaluation guide

Evaluation has two layers: **non-learned baselines** that run now on CPU (they establish the floor a
fine-tuned model must beat), and the **model-eval harness** that scores base vs adapter on held-out
test splits (built, ready, not launched in this task). No paid APIs are used anywhere.

## Baselines (run now, CPU)

`evaluation/baselines.py` fits a simple baseline on the `train` split and scores it on the in-domain
(or cross-dataset) test split, per `(dataset, task)`:

| baseline | tasks |
|---|---|
| majority-class | `PREDICT_NEXT_CHOICE` / `NEXT_ACTION` / `NEXT_SPEAKER` / `FINAL_OUTCOME` |
| base-rate (Brier / ECE) | `PREDICT_RESPONSE_OR_NONRESPONSE` |
| median-time | `PREDICT_TIME_TO_ACTION` |
| mean-reward / mean-rate | `PREDICT_POLICY_VALUE` / `PREDICT_POPULATION_RESPONSE` |
| zero-effect | `PREDICT_INTERVENTION_EFFECT` |
| most-frequent-message (token-F1) | `PREDICT_NEXT_MESSAGE` (generation; real eval needs the model) |

```bash
python -m machine_learning.cli eval baselines
```

This writes `reports/readiness/baselines.md` (+ `baselines.json`) with one row per `(dataset, task)`.
Baselines require a split table, so `datasets split` (or `prepare-all`) must have run first.

## Per-task metrics — no single universal metric

A click, a message, a belief shift, a delay, a population rate, and a treatment effect are different
targets, so each task family has its own metric module (dispatched by
`evaluation/metrics.py` + the family files):

- `next_action.py` — accuracy, macro-F1 (choice/action/speaker/outcome)
- `messages.py` — message similarity / token-F1 (`PREDICT_NEXT_MESSAGE`)
- `belief_change.py` — belief-change MAE, direction accuracy, calibration
- `timing.py` — timing MAE, censored likelihood, c-index (`PREDICT_TIME_TO_ACTION`)
- `trajectories.py` — trajectory similarity / edit distance (continuation, discussion tree)
- `population.py` — distribution distance, rate MAE, time-series MAE/DTW
- `causal.py` — effect error, IPS/DR value, ranking (intervention/policy-value/rank)

The primary metrics per task are declared in `registry/task_taxonomy.yaml`.

## The model-eval harness

`evaluation/model_eval.py` is the GPU path (built, lazy torch imports, not launched here). It is a
library API, not a CLI command:

```python
from machine_learning.evaluation.model_eval import evaluate_adapter
res = evaluate_adapter("8b_actor_choice", adapter_dir=".../adapter",
                       dataset_id="casino", split="test_in_domain", limit=500)
```

`generate_predictions()` greedy-generates a completion per test prompt (leakage-safe, from
`format_record`), parses it against the target, and `score_predictions()` dispatches each task to its
family metric.

## The 4-way comparison

Per the readiness spec, the harness compares four predictors on the held-out test splits:

1. **base** open-weight model (no adapter),
2. the **fine-tuned adapter**,
3. a **prompted-API** model (e.g. DeepSeek) — `prompted_api_baseline()` is an interface **stub** that
   raises with guidance; it would use a paid API, so no key is used here,
4. the **current SWORLDMODEL actor policy** — `sworldmodel_actor_baseline()` is an interface **stub**;
   wiring it needs the [`integration_contract.md`](integration_contract.md) and is out of scope for
   this task.

Base and adapter run fully; the two stubs raise with clear instructions rather than calling anything
external. This keeps the whole evaluation self-contained and reproducible on a single GPU box.
