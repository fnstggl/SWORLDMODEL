# Colab guide

Running the CPU smoke test and a small QLoRA run on Google Colab. Colab is good for the smoke test
(any runtime) and a small QLoRA proof-of-run on an A100/L4; a free **T4** works with reduced
sequence length and LoRA rank.

## 1. Get the code + mount Drive for `SWM_DATA_ROOT`

Working storage (raw/normalized/checkpoints) must persist across Colab restarts, so point
`SWM_DATA_ROOT` at Drive.

```python
from google.colab import drive; drive.mount('/content/drive')
import os
os.environ['SWM_DATA_ROOT'] = '/content/drive/MyDrive/swm_data'   # persists across restarts
os.environ['HF_HOME']       = '/content/drive/MyDrive/hf_home'
# os.environ['HF_TOKEN']    = '...'                                # for gated/large HF pulls

!git clone <your-fork-url> /content/SWORLDMODEL
%cd /content/SWORLDMODEL
```

## 2. Install cells

```python
# Data prep + smoke (CPU torch is fine for the smoke test):
!pip install -q -r machine_learning/requirements/base.txt \
                  -r machine_learning/requirements/data.txt \
                  -r machine_learning/requirements/training.txt
```

On a GPU runtime, Colab already ships a CUDA torch; `training.txt` adds transformers/peft/trl/
accelerate and (on Linux) bitsandbytes for the 4-bit path.

## 3. Prepare a small dataset + run the smoke test

```python
!python -m machine_learning.cli datasets acquire casino
!python -m machine_learning.cli datasets normalize casino
!python -m machine_learning.cli datasets split casino
!python -m machine_learning.cli smoke run --dataset casino
```

`smoke run` uses the `tiny_smoke_test` config (distilgpt2, plain LoRA) and runs entirely on CPU — no
GPU, no bitsandbytes. It must print `SMOKE PASSED` (12/12 checks) before any GPU run.

## 4. A small QLoRA run on a GPU runtime

Set the runtime to a GPU (Runtime → Change runtime type → A100/L4/T4), then verify and launch:

```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

# After human approval in registry/training_approvals.yaml (see licensing_guide.md):
!python -m machine_learning.cli manifests build actor_choice_v1
!python -m machine_learning.cli train run 8b_actor_choice            # dry-run: prints the plan
!python -m machine_learning.cli train run 8b_actor_choice --launch   # real launch
```

## Caveats

- **T4 (16 GB)** cannot hold the default `max_seq_len: 2048` / `lora.r: 16` for an 8B model
  comfortably. Copy `configs/training/8b_actor_choice.yaml`, lower `train.max_seq_len` (e.g. 512–1024)
  and `lora.r` (e.g. 8), and raise `train.grad_accum` to keep the effective batch size. A100/L4 run
  the defaults.
- Colab sessions time out and reset — because `SWM_DATA_ROOT` is on Drive and checkpoints resume
  automatically (`resume=True`), re-running the same `train run ... --launch` cell continues from the
  last checkpoint.
- Drive I/O is slow; for large datasets prefer a persistent GPU server
  ([`persistent_gpu_server_guide.md`](persistent_gpu_server_guide.md)). Colab is best for the smoke
  test + a single-view proof run.
- Never commit `SWM_DATA_ROOT` contents back to the repo — it is bulk working storage.
