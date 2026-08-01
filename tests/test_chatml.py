"""Label-masking tests.

A masking bug is silent -- the loss curve looks healthy while the model learns
to reproduce prompts -- so the mask is asserted directly rather than inferred
from training behaviour.
"""

from __future__ import annotations

import pytest

from sftlens.data.chatml import IGNORE_INDEX, ChatMLTemplate, describe_masking


class FakeTokenizer:
    """Character-level tokenizer with the ChatML control tokens reserved.

    Deliberately not a real tokenizer: the property under test is which spans
    are masked, and a toy vocabulary makes the assertion exact.
    """

    unk_token_id = 0

    def __init__(self):
        self.vocab = {"<|im_start|>": 1, "<|im_end|>": 2}
        self._next = 3

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def _id(self, ch):
        if ch not in self.vocab:
            self.vocab[ch] = self._next
            self._next += 1
        return self.vocab[ch]

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for tok, tid in (("<|im_start|>", 1), ("<|im_end|>", 2)):
                if text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    break
            else:
                ids.append(self._id(text[i]))
                i += 1
        return {"input_ids": ids}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)


@pytest.fixture
def template():
    return ChatMLTemplate(tokenizer=FakeTokenizer(), system_prompt="SYS")


def _supervised_text(template, out):
    return template.tokenizer.decode(
        [i for i, lab in zip(out["input_ids"], out["labels"], strict=True) if lab != IGNORE_INDEX]
    )


def test_single_turn_supervises_only_the_answer(template):
    out = template.encode([
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ])
    assert _supervised_text(template, out) == "A<|im_end|>\n"
    assert out["n_turns"] == 1


def test_terminator_is_supervised(template):
    """The <|im_end|> that ends the answer must carry loss: it is the only
    token that teaches the model to stop."""
    out = template.encode([
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ])
    im_end = template.im_end_id
    supervised_ids = [
        i for i, lab in zip(out["input_ids"], out["labels"], strict=True)
        if lab != IGNORE_INDEX
    ]
    assert im_end in supervised_ids


def test_assistant_header_is_masked(template):
    """`<|im_start|>assistant\\n` is context the model is given, not text it
    should be trained to emit."""
    out = template.encode([
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ])
    assert "assistant" not in _supervised_text(template, out)


def test_multi_turn_supervises_every_assistant_turn(template):
    out = template.encode([
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ])
    text = _supervised_text(template, out)
    assert text == "A1<|im_end|>\nA2<|im_end|>\n"
    assert out["n_turns"] == 2
    assert "Q1" not in text and "Q2" not in text


def test_labels_align_with_input_ids(template):
    out = template.encode([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ])
    assert len(out["input_ids"]) == len(out["labels"]) == out["length"]
    # Every non-ignored label must equal the token at that position: the labels
    # are the inputs, shifted by the model, not a separate sequence.
    for tok, lab in zip(out["input_ids"], out["labels"], strict=True):
        assert lab in (IGNORE_INDEX, tok)


def test_default_system_prompt_is_prepended_and_masked(template):
    out = template.encode([
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ])
    masked = template.tokenizer.decode(
        [i for i, lab in zip(out["input_ids"], out["labels"], strict=True) if lab == IGNORE_INDEX]
    )
    assert masked.startswith("<|im_start|>system\nSYS<|im_end|>")


def test_existing_system_turn_is_not_duplicated(template):
    """Mixture rows carrying their own system prompt must keep it."""
    out = template.encode([
        {"role": "system", "content": "CUSTOM"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ])
    text = template.tokenizer.decode(out["input_ids"])
    assert "CUSTOM" in text
    assert "SYS" not in text
    assert text.count("<|im_start|>system") == 1


def test_conversation_without_assistant_turn_is_rejected(template):
    with pytest.raises(ValueError, match="no assistant turn"):
        template.encode([{"role": "user", "content": "Q"}])


def test_empty_conversation_is_rejected(template):
    with pytest.raises(ValueError, match="empty conversation"):
        template.encode([])


def test_missing_control_token_is_rejected():
    """A missing <|im_end|> would force an embedding resize, whose randomly
    initialised rows contaminate exactly what the telemetry measures."""
    tok = FakeTokenizer()
    del tok.vocab["<|im_end|>"]
    with pytest.raises(ValueError, match="not in the tokenizer vocabulary"):
        ChatMLTemplate(tokenizer=tok)


def test_render_prompt_ends_at_the_assistant_header(template):
    prompt = template.render_prompt([{"role": "user", "content": "Q"}])
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "SYS" in prompt


def test_describe_masking_rejects_an_unterminated_answer(template):
    out = template.encode([
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ])
    describe_masking(template, out)   # well-formed: must not raise

    # Strip the trailing terminator from the supervised span.
    im_end = template.im_end_id
    for i in range(len(out["labels"]) - 1, -1, -1):
        if out["labels"][i] != IGNORE_INDEX and out["input_ids"][i] == im_end:
            out["labels"][i] = IGNORE_INDEX
            break
    with pytest.raises(AssertionError, match="never learn to stop"):
        describe_masking(template, out)
