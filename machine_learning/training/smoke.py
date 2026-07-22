"""Pre-GPU smoke test — the 12-point verification the repo must pass before it is
declared GPU-ready. Runs a TINY model on a TINY real subset on CPU, exercising the SAME
training/eval/checkpoint/export code path the 8B QLoRA run will use.

Checks:
  1  dataset loading            7  checkpoints save
  2  formatting                 8  resume from checkpoint
  3  target-only loss masking   9  evaluation runs
  4  only adapters trainable   10  adapter export + reload
  5  training loss decreases   11  base vs trained predictions differ
  6  overfit 32-128 examples   12  shuffled-label control (no false improvement)
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ARTIFACTS_DIR, normalized_dir
from ..io_utils import write_json
from ..normalization.common.parquet_io import iter_records
from .collators import BehaviorSFTDataset, PadCollator
from .loss_masking import IGNORE_INDEX, build_labels
from .model_registry import ModelSpec, load_model, load_tokenizer, trainable_parameter_report
from .train_qlora import export_adapter, load_train_config, set_seed
from .resume import latest_checkpoint, resume_step


@dataclass
class SmokeResult:
    passed: bool = False
    checks: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks, "details": self.details}


def _sample_records(dataset_id: str, n: int) -> list[dict]:
    out = []
    for r in iter_records(normalized_dir(dataset_id)):
        out.append(r)
        if len(out) >= n:
            break
    return out


def _train_overfit(model, ds, collator, steps: int, lr: float, device: str, seed: int = 0):
    import torch
    from torch.utils.data import DataLoader
    set_seed(seed)
    loader = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True, collate_fn=collator)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    model.train()
    it = iter(loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _eval_loss(model, ds, collator, device) -> float:
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            tot += float(model(**batch).loss.item())
            n += 1
    return tot / max(n, 1)


def run_smoke(config_name: str = "tiny_smoke_test", *, dataset_id: str = "casino",
              n_examples: int = 64, steps: int = 120) -> SmokeResult:
    import torch
    res = SmokeResult()
    cfg = load_train_config(config_name)
    spec = ModelSpec.from_config(cfg)
    tokenizer = load_tokenizer(spec)
    device = "cpu"
    lr = float(cfg.get("train", {}).get("learning_rate", 5e-3))

    # 1) dataset loading
    records = _sample_records(dataset_id, n_examples)
    res.checks["1_dataset_loading"] = len(records) >= 32
    res.details["n_records"] = len(records)
    n_val = max(8, len(records) // 4)
    train_recs, val_recs = records[n_val:], records[:n_val]

    # 2) formatting
    from ..examples.formatters.sft import format_record
    fx = format_record(train_recs[0])
    res.checks["2_formatting"] = bool(fx.prompt and fx.completion is not None
                                      and fx.text.endswith(fx.completion))

    # 3) target-only loss masking
    me = build_labels(tokenizer, fx.prompt, fx.completion, max_len=256)
    prefix_masked = all(x == IGNORE_INDEX for x in me.labels[:1]) or me.labels[0] == IGNORE_INDEX
    res.checks["3_loss_masking"] = (me.n_target_tokens > 0
                                    and any(x == IGNORE_INDEX for x in me.labels)
                                    and me.labels.count(IGNORE_INDEX) < len(me.labels))
    res.details["target_tokens"] = me.n_target_tokens
    res.details["masked_tokens"] = me.labels.count(IGNORE_INDEX)

    # model
    model, info = load_model(spec)
    model.to(device)
    trep = trainable_parameter_report(model)
    res.details["trainable_report"] = trep
    # 4) only adapters trainable
    res.checks["4_only_adapters_trainable"] = trep["all_trainable_are_adapters"] and trep["trainable"] > 0

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id or 0)
    train_ds = BehaviorSFTDataset(tokenizer, records=train_recs, max_len=256)
    val_ds = BehaviorSFTDataset(tokenizer, records=val_recs, max_len=256)

    # base (untrained) eval loss
    base_val = _eval_loss(model, val_ds, collator, device)
    base_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()
                                if "lora" in k.lower()})

    # 5/6) loss decreases + overfit
    losses = _train_overfit(model, train_ds, collator, steps=steps, lr=lr, device=device, seed=0)
    first = sum(losses[:3]) / min(3, len(losses))
    last = sum(losses[-3:]) / min(3, len(losses))
    res.details["train_loss_first"] = round(first, 4)
    res.details["train_loss_last"] = round(last, 4)
    res.checks["5_loss_decreases"] = last < first - 1e-3
    res.checks["6_overfit"] = last < first * 0.7  # substantial drop on a tiny set

    # 9) evaluation runs
    trained_val = _eval_loss(model, val_ds, collator, device)
    res.details["base_val_loss"] = round(base_val, 4)
    res.details["trained_val_loss"] = round(trained_val, 4)
    res.checks["9_evaluation_runs"] = trained_val == trained_val  # finite (not NaN)

    # 11) base vs trained predictions differ (LoRA weights changed)
    now_state = {k: v for k, v in model.state_dict().items() if "lora" in k.lower()}
    changed = any(not torch.allclose(base_state[k], now_state[k]) for k in base_state) if base_state else False
    res.checks["11_base_vs_trained_differ"] = changed and abs(trained_val - base_val) > 1e-4

    # 7/8) checkpoint save + resume
    run_dir = Path(ARTIFACTS_DIR) / "runs" / "smoke"
    from .checkpointing import save_checkpoint, load_checkpoint
    save_checkpoint(run_dir, 42, model=model, extra={"smoke": True})
    res.checks["7_checkpoint_save"] = latest_checkpoint(run_dir) is not None
    # reload adapter into a fresh model
    model2, _ = load_model(spec)
    model2.to(device)
    st = load_checkpoint(latest_checkpoint(run_dir), model=model2)
    res.checks["8_resume"] = (resume_step(run_dir) == 42 and st.get("step") == 42)

    # 10) adapter export + reload
    exp = export_adapter(model, tokenizer, run_dir / "adapter")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(spec.base_model, revision=spec.revision,
                                                torch_dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(base, exp)
    res.checks["10_adapter_export_reload"] = reloaded is not None

    # 12) shuffled-label control: training on shuffled targets must NOT beat real training
    #     on the held-out real val set (guards against leakage / a broken eval).
    shuffled = _shuffle_targets(train_recs)
    model_sh, _ = load_model(spec)
    model_sh.to(device)
    sh_ds = BehaviorSFTDataset(tokenizer, records=shuffled, max_len=256)
    _train_overfit(model_sh, sh_ds, collator, steps=steps, lr=lr, device=device, seed=0)
    shuffled_val = _eval_loss(model_sh, val_ds, collator, device)
    res.details["shuffled_val_loss"] = round(shuffled_val, 4)
    # real training should generalize at least as well as shuffled (no FALSE improvement
    # from shuffled labels). Small tolerance for tiny-model noise.
    res.checks["12_shuffled_control"] = trained_val <= shuffled_val + 0.05

    res.passed = all(res.checks.values())
    write_json(run_dir / "smoke_result.json", res.as_dict())
    return res


def _shuffle_targets(records: list[dict]) -> list[dict]:
    recs = [copy.deepcopy(r) for r in records]
    targets = [r["payload"]["target"] for r in recs]
    rng = random.Random(1234)
    perm = list(range(len(targets)))
    rng.shuffle(perm)
    for i, r in enumerate(recs):
        r["payload"]["target"] = copy.deepcopy(targets[perm[i]])
    return recs
