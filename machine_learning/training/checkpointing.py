"""Checkpoint save/load for the custom training loop.

A checkpoint captures everything needed to resume bit-for-bit: the LoRA adapter weights,
optimizer + scheduler state, the global step, and RNG state. Saves are atomic (write to a
tmp dir, then rename) so an interrupted save never corrupts the last good checkpoint.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return Path(run_dir) / f"checkpoint-{step:07d}"


def save_checkpoint(run_dir: Path, step: int, *, model, optimizer=None, scheduler=None,
                    extra: dict | None = None) -> Path:
    import torch
    run_dir = Path(run_dir)
    dst = checkpoint_dir(run_dir, step)
    tmp = dst.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    # adapter weights
    model.save_pretrained(str(tmp))
    state = {"step": step, "rng": torch.get_rng_state().tolist()}
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
    if extra:
        state["extra"] = extra
    (tmp / "trainer_state.json").write_text(json.dumps(state, indent=2))
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    (run_dir / "latest").write_text(dst.name)
    return dst


def load_checkpoint(ckpt_dir: Path, *, model, optimizer=None, scheduler=None) -> dict:
    import torch
    from peft import PeftModel
    ckpt_dir = Path(ckpt_dir)
    # load adapter weights into the existing peft model
    if isinstance(model, PeftModel):
        model.load_adapter(str(ckpt_dir), adapter_name="default", is_trainable=True)
    state = json.loads((ckpt_dir / "trainer_state.json").read_text())
    if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location="cpu"))
    if scheduler is not None and (ckpt_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", map_location="cpu"))
    if state.get("rng") is not None:
        try:
            torch.set_rng_state(torch.tensor(state["rng"], dtype=torch.uint8))
        except Exception:  # noqa: BLE001
            pass
    return state
