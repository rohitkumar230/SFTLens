"""Padding collator.

Kept separate from the Trainer because the telemetry needs to build its probe
batch with byte-identical collation. If the probe were padded differently from
training the two would not be comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .chatml import IGNORE_INDEX


@dataclass
class PadCollator:
    pad_token_id: int
    pad_to_multiple_of: int | None = 8   # keeps tensor cores on the fast path

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        maxlen = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            maxlen = ((maxlen + m - 1) // m) * m

        input_ids, labels, attn = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [IGNORE_INDEX] * pad)
            attn.append([1] * len(ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
