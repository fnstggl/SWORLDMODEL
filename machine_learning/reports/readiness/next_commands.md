# Next commands

## 0. Environment
```bash
export SWM_DATA_ROOT=/path/to/large/volume   # working storage (NOT the repo)
export HF_HOME=/path/to/hf_home
export HF_TOKEN=...                            # for gated/large HF pulls
pip install -r machine_learning/requirements/base.txt -r machine_learning/requirements/data.txt
```

## 1. Verify + prepare data
```bash
python -m machine_learning.cli registry verify
python -m machine_learning.cli datasets prepare-all           # resumable; blocked datasets recorded, not fatal
python -m machine_learning.cli datasets acquire omnibehavior --allow-large   # example: a storage-blocked set
python -m machine_learning.cli datasets normalize omnibehavior
python -m machine_learning.cli datasets validate omnibehavior
python -m machine_learning.cli eval baselines
```

## 2. Build training manifests (after human approval in registry/training_approvals.yaml)
```bash
python -m machine_learning.cli manifests build actor_choice_v1
python -m machine_learning.cli manifests build unified_behavior_multitask_v1
python -m machine_learning.cli readiness check
```

## 3. Tiny smoke test (CPU, no GPU) — MUST pass before a GPU run
```bash
pip install -r machine_learning/requirements/training.txt   # CPU torch is fine for the smoke
python -m machine_learning.cli smoke run
```

## 4. Launch an 8B QLoRA fine-tune (on a GPU)
```bash
# dry-run (prints the plan, refuses to launch without --launch):
python -m machine_learning.cli train run 8b_actor_choice
# real launch (needs a CUDA GPU + bitsandbytes):
python -m machine_learning.cli train run 8b_actor_choice --launch
python -m machine_learning.cli train run 8b_social_interaction --launch
python -m machine_learning.cli train run 8b_long_horizon --launch
python -m machine_learning.cli train run 8b_population_response --launch
python -m machine_learning.cli train run 8b_causal_intervention --launch
python -m machine_learning.cli train run 8b_unified_multitask --launch
```

