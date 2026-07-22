"""SFT formatter + target-only loss masking."""
from machine_learning.canonical import make_record
from machine_learning.examples.formatters.sft import TARGET_HEADER, format_record
from machine_learning.training.loss_masking import IGNORE_INDEX, build_labels


class FakeTok:
    """Whitespace tokenizer with a fixed vocab hash. Enough to test masking logic."""
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [abs(hash(t)) % 1000 + 2 for t in text.split()]}


def _rec():
    return make_record(
        dataset_id="d", task_type="PREDICT_NEXT_ACTION", converter="c", converter_version="1",
        license_class="mit", episode_id="e", sequence_index=1, actor_id="a",
        context={"known_history": [{"index": 0, "text": "prior turn"}],
                 "current_observation": {"text": "now"}, "available_actions": ["x", "y"]},
        payload={"input": {"history": [], "observation": {"text": "now"}, "available_actions": ["x", "y"]},
                 "target": {"action_type": "x", "acted": True, "action_content": {}}},
        raw_locator={"files": ["f"], "indices": [0], "ids": ["e"]})


def test_prompt_ends_with_target_header():
    fx = format_record(_rec())
    assert fx.prompt.endswith(TARGET_HEADER)
    assert fx.text == fx.prompt + fx.completion
    assert fx.target_char_start == len(fx.prompt)


def test_loss_masking_masks_prompt_only():
    fx = format_record(_rec())
    me = build_labels(FakeTok(), fx.prompt, fx.completion, max_len=256)
    assert me.n_target_tokens > 0
    # prompt region is masked
    assert me.labels[0] == IGNORE_INDEX
    # completion tokens are not all masked
    assert any(x != IGNORE_INDEX for x in me.labels)
    assert me.labels.count(IGNORE_INDEX) < len(me.labels)


def test_truncation_keeps_completion():
    fx = format_record(_rec())
    me = build_labels(FakeTok(), fx.prompt, fx.completion, max_len=4)
    # completion (+eos) preserved even under a tiny budget
    assert me.n_target_tokens >= 1
