"""Data collation: pad masked examples into batched tensors + a manifest-backed dataset.

``BehaviorSFTDataset`` resolves a training-view manifest (or a dataset/split) into
formatted, target-masked, tokenized examples on the fly — so we never duplicate full
history text to disk for every example (the canonical shards stay the single copy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..config import normalized_dir
from ..examples.formatters.sft import format_record
from ..normalization.common.parquet_io import iter_records
from .loss_masking import IGNORE_INDEX, build_labels


class BehaviorSFTDataset:
    """A torch-style map dataset over canonical records -> masked token examples.

    Provide EITHER ``records`` (an in-memory list of canonical records) or
    ``(dataset_id, split)`` to stream from normalized shards + the split table.
    """

    def __init__(self, tokenizer, *, records: list[dict] | None = None,
                 dataset_id: str | None = None, split: str | None = None,
                 max_len: int = 2048, max_history_events: int = 40,
                 train_on_prompt: bool = False, limit: int | None = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history_events = max_history_events
        self.train_on_prompt = train_on_prompt
        if records is not None:
            self._records = records if limit is None else records[:limit]
        else:
            self._records = list(_load_split_records(dataset_id, split, limit))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        rec = self._records[idx]
        fx = format_record(rec, max_history_events=self.max_history_events)
        ex = build_labels(self.tokenizer, fx.prompt, fx.completion, max_len=self.max_len,
                          train_on_prompt=self.train_on_prompt)
        return {"input_ids": ex.input_ids, "attention_mask": ex.attention_mask,
                "labels": ex.labels, "record_id": rec["record_id"], "task_type": rec["task_type"]}


def _load_split_records(dataset_id: str, split: str | None, limit: int | None) -> Iterable[dict]:
    from ..splitting.policies import load_split_table
    wanted = None
    if split:
        wanted = {r["record_id"] for r in load_split_table(dataset_id) if r["split"] == split}
    n = 0
    for r in iter_records(normalized_dir(dataset_id)):
        if wanted is not None and r["record_id"] not in wanted:
            continue
        yield r
        n += 1
        if limit and n >= limit:
            return


@dataclass
class PadCollator:
    """Pad a batch of {input_ids, attention_mask, labels} to the longest sequence."""
    pad_token_id: int = 0
    label_pad: int = IGNORE_INDEX

    def __call__(self, batch: list[dict]):
        import torch
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [self.label_pad] * pad)
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attn),
                "labels": torch.tensor(labels)}
