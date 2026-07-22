# Persistent GPU server guide

End-to-end on a rented GPU box (RunPod or similar). A CPU pod is enough for all data prep + the smoke
test; rent a GPU pod only for the actual 8B fine-tune.

## 1. Rent + SSH

- **CPU pod** — data prep (`prepare-all`) and the CPU smoke test. No GPU needed.
- **GPU pod** — an **A100-40GB** or **A6000** (both hold the default 8B QLoRA config; also fine:
  4090-24GB). Attach a **persistent volume** large enough for `$SWM_DATA_ROOT` (60–120 GB for full
  normalized data + manifests; ~20 GB for the base model + cache).

```bash
ssh root@<pod-host> -p <port> -i ~/.ssh/id_rsa
```

## 2. Put working storage on the volume

`SWM_DATA_ROOT` must live on the persistent volume so raw data, shards, and checkpoints survive pod
restarts. Never point it at the repo.

```bash
export SWM_DATA_ROOT=/workspace/swm_data       # on the mounted persistent volume
export HF_HOME=/workspace/hf_home
export HF_TOKEN=...                             # gated/large HF pulls, and gated base models
export SWM_DISK_STOP_FRACTION=0.85             # storage guard (default)
```

## 3. Install

```bash
git clone <your-fork-url> && cd SWORLDMODEL
# CPU-side data prep + smoke:
pip install -r machine_learning/requirements/base.txt -r machine_learning/requirements/data.txt
# GPU training (on the GPU pod; install a CUDA torch matching the driver first):
pip install -r machine_learning/requirements/training.txt
```

## 4. Prepare data + smoke test

```bash
python -m machine_learning.cli registry verify
python -m machine_learning.cli datasets prepare-all            # resumable; blocked/deferred recorded, not fatal
python -m machine_learning.cli datasets prepare-all --allow-large --only omnibehavior,kuairand   # big sets, deliberately
python -m machine_learning.cli eval baselines
python -m machine_learning.cli readiness check                 # writes reports/readiness/*

python -m machine_learning.cli smoke run                       # CPU LoRA; MUST print SMOKE PASSED
```

## 5. Build a manifest + launch

```bash
# After human approval in registry/training_approvals.yaml (see licensing_guide.md):
python -m machine_learning.cli manifests build actor_choice_v1
python -m machine_learning.cli train run 8b_actor_choice            # dry-run: prints the plan
python -m machine_learning.cli train run 8b_actor_choice --launch   # real launch (GPU)
```

## 6. Resume-safe checkpoints

Checkpoints are atomic + periodic (`save_steps`) and `train run` resumes automatically from the latest
one in the run dir. A pre-empted or restarted pod loses nothing as long as `SWM_DATA_ROOT` is on the
persistent volume — just re-run the same `train run ... --launch` command.

## 7. Pull the adapter back

The trained LoRA adapter is exported to `$SWM_DATA_ROOT/.../runs/<config>/adapter/` (plus
`run_manifest.json`). Copy it to your machine:

```bash
scp -P <port> -r root@<pod-host>:/workspace/swm_data/artifacts/runs/8b_actor_choice/adapter ./adapter
scp -P <port>    root@<pod-host>:/workspace/swm_data/artifacts/runs/8b_actor_choice/run_manifest.json ./
```

The adapter is small (tens of MB). Keep `run_manifest.json` with it — it records the base + revision,
data-manifest hash, code commit, and seed needed to reproduce or re-load the run (see
[`reproduction_guide.md`](reproduction_guide.md)). Evaluate it with
`evaluation.model_eval.evaluate_adapter`.
