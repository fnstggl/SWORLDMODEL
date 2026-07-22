# Estimated GPU requirements (8B QLoRA)

Base model is configurable (default `Qwen/Qwen2.5-7B`, Apache-2.0). 4-bit QLoRA.

| item | estimate |
|---|---|
| GPU VRAM (4-bit QLoRA, 8B, seq 2048, bsz 1 + grad-accum 16) | 16-24 GB (1x A100-40GB / A6000 / 4090-24GB) |
| Base weights (4-bit) | ~5-6 GB |
| LoRA adapter (r=16) | ~40-80 MB |
| Peak activation (grad-checkpointing on) | ~8-14 GB |
| Disk (base + tokenizer + cache) | ~20 GB |
| Disk (normalized data + manifests, full acquisition) | ~60-120 GB external (SWM_DATA_ROOT) |
| Throughput (A100-40GB) | ~1-3 examples/s |
| Time for 1 epoch of a ~100k-example view | ~3-10 GPU-hours |

Notes:
- The CPU smoke path needs NO GPU and NO bitsandbytes (plain LoRA on a tiny model).
- For a 40GB GPU, the default configs (bsz 1, grad-accum 16, seq 2048, grad-checkpointing) fit comfortably.
- Reduce `max_seq_len` or LoRA `r` if VRAM-constrained; raise `grad_accum` to keep the effective batch size.

