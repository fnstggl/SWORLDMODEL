"""Target-only loss masking.

For SFT we want the model to learn to PRODUCE the target, not to reproduce the prompt.
``build_labels`` tokenizes prompt + completion and returns ``labels`` where every prompt
token is ``-100`` (ignored by the cross-entropy loss) and only completion tokens carry a
loss. This is the mechanism that makes "loss applies only to the TARGET section" true.

Kept torch-free: it operates on token-id lists so it is unit-testable without a GPU (a
fake tokenizer suffices), and the collator turns the lists into padded tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = -100


@dataclass
class MaskedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]

    @property
    def n_target_tokens(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def build_labels(tokenizer, prompt: str, completion: str, *, max_len: int = 2048,
                 train_on_prompt: bool = False, add_eos: bool = True) -> MaskedExample:
    """Tokenize prompt+completion; mask prompt tokens unless ``train_on_prompt``.

    Truncation keeps the completion intact and drops the OLDEST prompt tokens (so the
    target is never truncated away), matching the SFT formatter's history-elision intent.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos = getattr(tokenizer, "eos_token_id", None)
    if add_eos and eos is not None:
        completion_ids = completion_ids + [eos]

    # never truncate the completion; if over budget, trim the front of the prompt
    budget = max_len - len(completion_ids)
    if budget < 1:
        completion_ids = completion_ids[:max_len]
        prompt_ids = []
    elif len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + completion_ids
    if train_on_prompt:
        labels = list(input_ids)
    else:
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    attention_mask = [1] * len(input_ids)
    return MaskedExample(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
