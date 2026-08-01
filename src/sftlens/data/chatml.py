"""ChatML rendering and loss masking.

The template is written out here rather than delegated to
`tokenizer.apply_chat_template`, for two reasons:

  1. SmolLM2-1.7B *base* ships no `chat_template` at all, so there is nothing
     to delegate to.
  2. Where a template does exist (the instruct variants, Qwen), it silently
     injects its own default system prompt. For a base-model fine-tune the
     format has to be under our control, because it is the thing being taught.

The rendered form is standard ChatML:

    <|im_start|>system\n{system}<|im_end|>\n
    <|im_start|>user\n{content}<|im_end|>\n
    <|im_start|>assistant\n{content}<|im_end|>\n
    ...

Loss is taken on assistant content and on the `<|im_end|>` that terminates it,
and on nothing else. Supervising the terminator is not optional: it is the only
token that teaches the model to stop.
"""

from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = -100

Message = dict[str, str]


@dataclass(frozen=True)
class ChatMLTemplate:
    """Renders and tokenizes conversations, tracking which tokens carry loss.

    Held frozen and built once per run so that the exact byte-level format is
    fixed for the whole trajectory; a mid-run format change would show up in the
    telemetry as a geometry shift with no cause.
    """

    tokenizer: object
    system_prompt: str = "You are a helpful assistant."
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"

    def __post_init__(self) -> None:
        # A missing control token would force an embedding resize, whose
        # randomly initialised rows have wildly atypical gradient geometry for
        # the first few hundred steps -- contaminating exactly what we measure.
        for token in (self.im_start, self.im_end):
            ids = self.tokenizer.convert_tokens_to_ids(token)
            if ids is None or ids == self.tokenizer.unk_token_id:
                raise ValueError(
                    f"{token!r} is not in the tokenizer vocabulary. Adding it "
                    "would require an embedding resize; pick a base model that "
                    "reserves the ChatML control tokens instead."
                )

    @property
    def im_end_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.im_end)

    # -- rendering ---------------------------------------------------------
    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(self, messages: list[Message]) -> dict[str, list[int]]:
        """Tokenize a conversation into input_ids / labels.

        Segments are tokenized separately and concatenated. This keeps the
        prompt/response boundary exact: tokenizing the joined string and then
        trying to locate the boundary is fragile, because BPE can merge across
        it and shift the mask by a token.
        """
        if not messages:
            raise ValueError("empty conversation")

        input_ids: list[int] = []
        labels: list[int] = []

        def emit(ids: list[int], supervised: bool) -> None:
            input_ids.extend(ids)
            labels.extend(ids if supervised else [IGNORE_INDEX] * len(ids))

        # A system turn is prepended only when the conversation lacks one, so
        # that mixture rows carrying their own system prompt keep it.
        if messages[0].get("role") != "system" and self.system_prompt:
            emit(self._encode(f"{self.im_start}system\n{self.system_prompt}{self.im_end}\n"), False)

        n_supervised_turns = 0
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "assistant":
                # Header is context, content and terminator carry loss.
                emit(self._encode(f"{self.im_start}assistant\n"), False)
                emit(self._encode(f"{content}{self.im_end}\n"), True)
                n_supervised_turns += 1
            else:
                emit(self._encode(f"{self.im_start}{role}\n{content}{self.im_end}\n"), False)

        if n_supervised_turns == 0:
            raise ValueError("conversation has no assistant turn to supervise")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": len(input_ids),
            "n_supervised": sum(1 for x in labels if x != IGNORE_INDEX),
            "n_turns": n_supervised_turns,
        }

    def render_prompt(self, messages: list[Message]) -> str:
        """Render a conversation up to the assistant header, for generation."""
        parts = []
        if messages[0].get("role") != "system" and self.system_prompt:
            parts.append(f"{self.im_start}system\n{self.system_prompt}{self.im_end}\n")
        for msg in messages:
            parts.append(f"{self.im_start}{msg['role']}\n{msg['content']}{self.im_end}\n")
        parts.append(f"{self.im_start}assistant\n")
        return "".join(parts)


def describe_masking(template: ChatMLTemplate, example: dict) -> str:
    """Human-readable dump of what the loss does and does not see.

    Label-masking bugs are silent -- the loss curve looks healthy while the
    model learns to parrot prompts -- so this is printed before every run.
    """
    ids, labels = example["input_ids"], example["labels"]
    if len(ids) != len(labels):
        raise AssertionError(f"length mismatch: {len(ids)} ids vs {len(labels)} labels")

    tok = template.tokenizer
    supervised = tok.decode([i for i, lab in zip(ids, labels, strict=True) if lab != IGNORE_INDEX])
    masked = tok.decode([i for i, lab in zip(ids, labels, strict=True) if lab == IGNORE_INDEX])

    if not supervised.rstrip().endswith(template.im_end):
        raise AssertionError(
            "supervised span does not end with the terminator; the model would "
            "never learn to stop"
        )

    return (
        f"--- masked, no loss ({len(ids) - example['n_supervised']} tokens) ---\n"
        f"{masked}\n"
        f"--- supervised ({example['n_supervised']} tokens, "
        f"{example['n_turns']} assistant turn(s)) ---\n{supervised}\n"
        + "-" * 70
    )
