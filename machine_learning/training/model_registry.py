"""Model + tokenizer resolution for training.

The base model is CONFIGURABLE (never hard-coded to one 8B). A training config names a
base model + revision; this module loads it in the right precision:

* ``quantization: 4bit`` -> QLoRA (bitsandbytes NF4) on a CUDA GPU;
* ``quantization: none`` -> full precision (the CPU smoke path; plain LoRA).

It also stamps a reproducibility record (base + tokenizer revision, package versions,
device) into the run manifest. All heavy imports are lazy so this module imports without
torch/transformers installed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    base_model: str
    revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    quantization: str = "4bit"          # "4bit" | "none"
    dtype: str = "bfloat16"             # compute dtype
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list | None = None  # None -> peft default for the arch
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelSpec":
        m = cfg.get("model", {})
        lora = cfg.get("lora", {})
        return cls(
            base_model=m["base_model"],
            revision=m.get("revision"),
            tokenizer=m.get("tokenizer") or m["base_model"],
            tokenizer_revision=m.get("tokenizer_revision"),
            quantization=m.get("quantization", "4bit"),
            dtype=m.get("dtype", "bfloat16"),
            lora_r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 32),
            lora_dropout=lora.get("dropout", 0.05),
            target_modules=lora.get("target_modules"),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            trust_remote_code=m.get("trust_remote_code", False),
        )


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.tokenizer, revision=spec.tokenizer_revision,
                                        trust_remote_code=spec.trust_remote_code, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def load_model(spec: ModelSpec):
    """Load the base model + attach a LoRA adapter. Returns (model, info)."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    want_4bit = spec.quantization == "4bit"
    if want_4bit and not cuda_available():
        # QLoRA needs CUDA/bitsandbytes; downgrade to full-precision LoRA for CPU (smoke).
        want_4bit = False

    kwargs = dict(trust_remote_code=spec.trust_remote_code)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(spec.dtype, torch.float32)
    if want_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype if cuda_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(spec.base_model, revision=spec.revision, **kwargs)
    if want_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=spec.gradient_checkpointing)
    elif spec.gradient_checkpointing and cuda_available():
        model.gradient_checkpointing_enable()

    lconf = LoraConfig(r=spec.lora_r, lora_alpha=spec.lora_alpha, lora_dropout=spec.lora_dropout,
                       bias="none", task_type="CAUSAL_LM", target_modules=spec.target_modules)
    model = get_peft_model(model, lconf)
    info = {"quantization": "4bit" if want_4bit else "none", "cuda": cuda_available(),
            "base_model": spec.base_model, "revision": spec.revision}
    return model, info


def trainable_parameter_report(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return {"trainable": trainable, "total": total,
            "trainable_fraction": round(trainable / max(total, 1), 6),
            "all_trainable_are_adapters": all(("lora" in n.lower() or "adapter" in n.lower()) for n in names),
            "n_trainable_tensors": len(names)}


def package_versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out
