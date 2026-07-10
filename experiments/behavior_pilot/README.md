# Behavior-model pilot (OSim vs DeepSeek) — run on a rented 24 GB GPU

**This does not run in the sandbox (no GPU).** It is the decisive, capped experiment from
`docs/AUDIT_BEHAVIOR_MODELS.md` Part D/E, ready to run where a GPU exists. Per the hard rule, `pilot.py`
**hard-stops with a clear message if no OSim endpoint is present** — it never pretends to have run.

## What it answers

Does replacing/augmenting DeepSeek stakeholder agents with **OSim-8B** (behavior-trained) improve held-out
prediction of **individual human responses / engagement realism** — the class OSim targets — *not* deliberation
forecasting, where EXP-098 already showed the stakeholder layer is net-negative.

Arms (identical dossiers/scenarios/stimuli/as-of/sample count/aggregation):
- **A** grounded direct DeepSeek · **B** DeepSeek stakeholder agents · **C** OSim agents · **D** mixed A+C
- **E** Minitaur-8B on forced-choice items only

Metrics: Brier, log-loss, choice accuracy, calibration (ECE) **plus the OmniBehavior realism metrics** —
action/engagement rate vs real (hyper-activity), inactivity rate, persona heterogeneity (intra/inter
distribution ratio), positivity bias.

## Steps

1. Rent one 24 GB GPU (L4 / A10G / 3090/4090; ≈$0.5–1.0/hr) or a Colab with an L4.
2. Serve OSim — see [`run_osim_server.md`](run_osim_server.md). It exposes an OpenAI-compatible endpoint.
3. `export OSIM_ENDPOINT=http://127.0.0.1:8000/v1  DEEPSEEK_API_KEY=…`
4. `python -m experiments.behavior_pilot.pilot --items data/behavior_pilot_items.jsonl --max-items 50 --max-usd 20`
5. Read the scoreboard; every raw output is cached under `experiments/behavior_pilot/cache/`.

## Caps & stop rule (built in)

`--max-items` (default 50) and `--max-usd` (default 20) are enforced. The harness **aborts if the first 10–15
items show a severe incompatibility** (OSim returns no parseable action on >½ of them). Nothing is promoted to
production from this pilot — it only produces held-out numbers for the keep/reject gate.

## Item set

Needs **labeled** individual-response / engagement items (person dossier · exact stimulus · realized action).
Public options (see `docs/DATASET_REGISTRY.md`): Enron (reply/latency, reconstructable) as a first proxy;
OmniBehavior (the ideal) when it releases. `data/behavior_pilot_items.jsonl` is the expected format (one JSON
object per line: `{"dossier","scenario","stimulus","allowed_actions","outcome"}`).
