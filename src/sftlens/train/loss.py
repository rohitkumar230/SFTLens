"""Token loss, shared by the trainer and the telemetry probe.

Both must use the same reduction. Delta -- the gradient at each module's
output, and the object the whole telemetry is built on -- is proportional to
whatever scaling the loss applies. If the probe used a mean reduction while the
optimizer saw a sum, every gradient-side quantity would be off by a
sequence-length-dependent factor that varies from probe to probe.

SUM vs MEAN
    Tulu 3 sums the per-token cross entropy over the batch instead of averaging
    it, so a batch containing long responses produces a proportionally larger
    gradient. HF's default is the mean. This is a real difference in effective
    step size and it is part of the recipe, not an implementation detail --
    hence `RecipeConfig.loss_reduction` rather than a hardcoded choice.

    Under sum reduction the value is divided only by the gradient accumulation
    count, which is what `Trainer` does to the returned loss. This reproduces
    the open-instruct behaviour SmolTulu inherited.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.chatml import IGNORE_INDEX


def token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "sum",
) -> tuple[torch.Tensor, int]:
    """Causal-LM cross entropy over supervised positions.

    Returns (loss, n_supervised_tokens). The token count is returned so callers
    can normalise consistently and so it can be logged -- under sum reduction
    the loss value is meaningless without it.
    """
    # Standard causal shift: position t predicts token t+1.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    n_supervised = int((flat_labels != IGNORE_INDEX).sum().item())

    # Cross entropy in fp32 regardless of autocast: bf16 logsumexp over a 49k
    # vocabulary loses meaningful precision, and this is the root of every
    # gradient the telemetry measures.
    loss = F.cross_entropy(
        flat_logits.float(),
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction="sum" if reduction == "sum" else "mean",
    )

    if reduction not in {"sum", "mean"}:
        raise ValueError(f"reduction must be sum|mean, got {reduction!r}")

    return loss, n_supervised
