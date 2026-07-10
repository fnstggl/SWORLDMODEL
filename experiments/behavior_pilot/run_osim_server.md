# Serving OSim-8B on a rented 24 GB GPU (exact commands)

OSim-8B (`cmu-lti/osim-8b`, base Qwen3-8B, MIT, ungated) has **no official quantized checkpoint and no hosted
endpoint** — self-host via vLLM. ~16 GB VRAM in BF16 → fits one 24 GB GPU; use `--quantization bitsandbytes`
to fit a 16 GB card.

```bash
# 1. env (CUDA GPU box)
pip install "vllm>=0.6.0"

# 2. serve OSim with an OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
  --model cmu-lti/osim-8b \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
# (add: --quantization bitsandbytes   to fit a 16 GB GPU)

# 3. the pilot talks to it here
export OSIM_ENDPOINT=http://127.0.0.1:8000/v1
export OSIM_MODEL=cmu-lti/osim-8b
```

**Colab (free/L4) alternative:** same `pip install vllm`, run the server in one cell with `nohup … &`, then
`OSIM_ENDPOINT=http://127.0.0.1:8000/v1`. For a 16 GB T4 use the 4B variant `cmu-lti/osim-4b`.

**Minitaur-8B (arm E, forced-choice)** loads differently (not a chat server) — via `unsloth`:

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("marcelbinz/Llama-3.1-Minitaur-8B",
                                               max_seq_length=32768, load_in_4bit=True)
FastLanguageModel.for_inference(model)
# score choice options by token log-likelihood of each option continuation; pick argmax / read P(option)
```

The pilot's OSim runner is a thin OpenAI-client call to `$OSIM_ENDPOINT`; the Minitaur runner is a
log-likelihood scorer. Both plug into `swm/experimental/behavior_models.py` as injected `runner`s. Cache every
response so a re-run costs nothing.
