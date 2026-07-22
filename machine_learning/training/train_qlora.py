"""QLoRA / LoRA SFT trainer for the behaviour model.

A dependency-light custom training loop (plain PyTorch) so the SAME code path runs the
CPU tiny-model smoke test and a real 8B 4-bit QLoRA GPU run — only the config differs.

Features: deterministic seeds, gradient accumulation, mixed precision (CUDA), gradient
checkpointing, TARGET-ONLY loss (labels pre-masked by the collator), periodic evaluation,
early stopping, atomic checkpointing + resume, adapter export, and a full reproducibility
run-manifest (base+tokenizer revision, data-manifest hash, code commit, seed, hyperparams,
package versions, device).

Heavy imports (torch/transformers/peft) are lazy so this module imports without them.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import ARTIFACTS_DIR, CONFIGS_DIR, normalized_dir
from ..io_utils import write_json
from .checkpointing import save_checkpoint
from .collators import BehaviorSFTDataset, PadCollator
from .model_registry import (ModelSpec, cuda_available, load_model, load_tokenizer,
                             package_versions, trainable_parameter_report)
from .resume import latest_checkpoint


def load_train_config(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIGS_DIR / "training" / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no training config: {name_or_path}")
    return yaml.safe_load(p.read_text())


def _code_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    torch.manual_seed(seed)
    if cuda_available():
        torch.cuda.manual_seed_all(seed)


def records_from_manifest(view: str) -> list[dict]:
    """Resolve a training-view manifest's record_ids back to full canonical records."""
    from ..sampling.manifests import load_manifest_records
    from ..normalization.common.parquet_io import iter_records
    man = load_manifest_records(view)
    by_ds: dict[str, set] = {}
    for m in man:
        by_ds.setdefault(m["dataset"], set()).add(m["record_id"])
    out: list[dict] = []
    for ds, ids in by_ds.items():
        for r in iter_records(normalized_dir(ds)):
            if r["record_id"] in ids:
                out.append(r)
    return out


@dataclass
class TrainResult:
    run_dir: str
    steps: int = 0
    train_losses: list = field(default_factory=list)
    eval_losses: list = field(default_factory=list)
    final_train_loss: float | None = None
    best_eval_loss: float | None = None
    trainable_report: dict = field(default_factory=dict)
    manifest_path: str | None = None
    early_stopped: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def train(config: dict | str, *, train_records: list[dict] | None = None,
          eval_records: list[dict] | None = None, resume: bool = True,
          run_name: str | None = None, max_steps_override: int | None = None) -> TrainResult:
    """Run SFT. Data source precedence: explicit train_records > view manifest > dataset/split."""
    import torch
    from torch.utils.data import DataLoader

    cfg = load_train_config(config) if isinstance(config, str) else config
    tcfg = cfg.get("train", {})
    seed = int(tcfg.get("seed", 42))
    set_seed(seed)

    spec = ModelSpec.from_config(cfg)
    tokenizer = load_tokenizer(spec)
    model, model_info = load_model(spec)
    device = "cuda" if cuda_available() else "cpu"
    model.to(device)

    # ---- data ----
    max_len = int(tcfg.get("max_seq_len", 1024))
    if train_records is None:
        view = cfg.get("data", {}).get("view")
        if view:
            train_records = records_from_manifest(view)
        else:
            ds = cfg["data"]["dataset_id"]
            train_records = list(BehaviorSFTDataset(tokenizer, dataset_id=ds, split="train")._records)
    train_ds = BehaviorSFTDataset(tokenizer, records=train_records, max_len=max_len)
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id or 0)
    bsz = int(tcfg.get("batch_size", 1))
    loader = DataLoader(train_ds, batch_size=bsz, shuffle=True, collate_fn=collator)

    eval_ds = BehaviorSFTDataset(tokenizer, records=eval_records, max_len=max_len) if eval_records else None

    # ---- optim ----
    lr = float(tcfg.get("learning_rate", 2e-4))
    accum = int(tcfg.get("grad_accum", 1))
    epochs = int(tcfg.get("epochs", 1))
    max_steps = max_steps_override or tcfg.get("max_steps")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _make_optimizer(params, lr, spec.quantization == "4bit" and cuda_available())
    total_steps = max_steps or (len(loader) // accum) * epochs or 1
    scheduler = _make_scheduler(optimizer, total_steps, int(tcfg.get("warmup_steps", 0)))

    run_dir = Path(ARTIFACTS_DIR) / "runs" / (run_name or cfg.get("name", "run"))
    run_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if resume and latest_checkpoint(run_dir) is not None:
        from .checkpointing import load_checkpoint
        st = load_checkpoint(latest_checkpoint(run_dir), model=model, optimizer=optimizer, scheduler=scheduler)
        start_step = int(st.get("step", 0))

    trep = trainable_parameter_report(model)
    manifest = _build_manifest(cfg, spec, model_info, tokenizer, trep, seed, total_steps, len(train_records))
    write_json(run_dir / "run_manifest.json", manifest)

    # ---- loop ----
    res = TrainResult(run_dir=str(run_dir), trainable_report=trep, manifest_path=str(run_dir / "run_manifest.json"))
    model.train()
    step = start_step
    save_every = int(tcfg.get("save_steps", max(total_steps // 2, 1)))
    eval_every = int(tcfg.get("eval_steps", max(total_steps // 2, 1)))
    patience = int(tcfg.get("early_stopping_patience", 0))
    best = float("inf")
    bad = 0
    accum_loss = 0.0
    done = False
    for _epoch in range(epochs):
        if done:
            break
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            use_amp = device == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullctx():
                out = model(**batch)
                loss = out.loss / accum
            loss.backward()
            accum_loss += out.loss.item()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, float(tcfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                res.train_losses.append(accum_loss / accum)
                accum_loss = 0.0
                if step % eval_every == 0 and eval_ds is not None:
                    ev = _evaluate(model, eval_ds, collator, device)
                    res.eval_losses.append(ev)
                    if ev < best - 1e-4:
                        best = ev
                        bad = 0
                    else:
                        bad += 1
                    model.train()
                    if patience and bad >= patience:
                        res.early_stopped = True
                        done = True
                if step % save_every == 0:
                    save_checkpoint(run_dir, step, model=model, optimizer=optimizer,
                                    scheduler=scheduler, extra={"train_loss": res.train_losses[-1]})
                if max_steps and step >= start_step + max_steps:
                    done = True
                if done:
                    break

    res.steps = step
    res.final_train_loss = res.train_losses[-1] if res.train_losses else None
    res.best_eval_loss = best if best != float("inf") else None
    save_checkpoint(run_dir, step, model=model, optimizer=optimizer, scheduler=scheduler,
                    extra={"final": True})
    export_adapter(model, tokenizer, run_dir / "adapter")
    manifest["result"] = {"steps": res.steps, "final_train_loss": res.final_train_loss,
                          "best_eval_loss": res.best_eval_loss, "early_stopped": res.early_stopped,
                          "finished_at_unix": int(time.time())}
    write_json(run_dir / "run_manifest.json", manifest)
    return res


def _make_optimizer(params, lr, use_paged: bool):
    import torch
    if use_paged:
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(params, lr=lr)
        except Exception:  # noqa: BLE001
            pass
    return torch.optim.AdamW(params, lr=lr)


def _make_scheduler(optimizer, total_steps, warmup):
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup,
                                           num_training_steps=max(total_steps, 1))


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _evaluate(model, eval_ds, collator, device) -> float:
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collator)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
            n += 1
    return total / max(n, 1)


def export_adapter(model, tokenizer, out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return str(out_dir)


def merge_adapter(spec: ModelSpec, adapter_dir: str, out_dir: str) -> str:
    """Merge a trained LoRA adapter into the base weights (optional, GPU-friendly)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(spec.base_model, revision=spec.revision)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    return out_dir


def _build_manifest(cfg, spec, model_info, tokenizer, trep, seed, total_steps, n_records) -> dict:
    from ..sampling.manifests import load_manifest_records
    view = cfg.get("data", {}).get("view")
    data_hash = None
    if view:
        recs = load_manifest_records(view)
        import hashlib
        h = hashlib.sha256()
        for r in sorted(recs, key=lambda r: r["record_id"]):
            h.update(r["record_id"].encode())
        data_hash = h.hexdigest()[:16]
    return {
        "name": cfg.get("name"),
        "config": cfg,
        "base_model": spec.base_model,
        "base_model_revision": spec.revision,
        "tokenizer": spec.tokenizer,
        "tokenizer_revision": spec.tokenizer_revision,
        "quantization_effective": model_info.get("quantization"),
        "data_view": view,
        "data_manifest_hash": data_hash,
        "n_train_records": n_records,
        "code_commit": _code_commit(),
        "seed": seed,
        "planned_total_steps": total_steps,
        "trainable_parameters": trep,
        "package_versions": package_versions(),
        "device": _gpu_name(),
        "cuda": cuda_available(),
    }
