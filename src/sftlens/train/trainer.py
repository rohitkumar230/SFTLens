"""Trainer subclass: recipe-faithful loss, token accounting, joinable log."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer import set_rng_state_for_device
from transformers.training_args import ParallelMode

from ..data.chatml import IGNORE_INDEX
from .loss import token_loss


class SFTTrainer(Trainer):
    """Applies the recipe's loss reduction and counts tokens actually consumed.

    Token accounting exists because the telemetry schedules on tokens rather
    than steps. Deriving it as steps x batch x mean_length would be wrong under
    `group_by_length`, which deliberately makes per-batch length non-uniform.
    """

    def __init__(self, *args, loss_reduction: str = "sum", **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_reduction = loss_reduction
        self.tokens_seen = 0
        self.supervised_tokens_seen = 0

    def _load_rng_state(self, checkpoint) -> None:
        """Restore RNG state on resume, bypassing transformers' broken
        `weights_only=True` load for this file.

        WHY THIS IS OVERRIDDEN
            `Trainer._load_rng_state` calls
            `torch.load(rng_file, weights_only=True)` unconditionally, wrapped
            in transformers' own `safe_globals()` allowlist helper. Two
            separate bugs compound there:

            1. `safe_globals()` no-ops for torch < 2.6, on the (wrong)
               assumption that the strict `weights_only` default doesn't apply
               below that version. `_load_rng_state` passes `weights_only=True`
               explicitly regardless of version, so on torch < 2.6 the call is
               strict with zero allowlisting -- verified failing on this
               project's own pods (torch 2.4.1) with `UnpicklingError:
               Unsupported global: numpy.core.multiarray._reconstruct`.

            2. On torch >= 2.6, where `safe_globals()` does fire, its
               context-manager form does not compose with a permanent
               `torch.serialization.add_safe_globals()` registration: entering
               ANY `safe_globals()` context anywhere in the process resets the
               allowlist to torch's own internal defaults on exit, silently
               dropping any prior permanent registration. Verified directly:
               registering the needed numpy globals at import time worked
               right up until this project's own test suite exercised a
               resume elsewhere in the same process, after which the
               registration was gone.

            A permanent-registration workaround is therefore fragile against
            anything else in the process using that context manager, present
            or future. Bypassing the allowlist for this one call is simpler
            and cannot regress: every RNG-state file this project ever resumes
            from is one it generated itself in a prior step of the same run.
            There is no untrusted-input boundary here to defend.

        This mirrors transformers 4.51.3's `_load_rng_state` exactly, minus
        the XLA/NPU/HPU/MLU/MUSA branches this project's single-GPU CUDA/CPU
        setup never exercises.
        """
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            rng_file = os.path.join(checkpoint, f"rng_state_{self.args.process_index}.pth")
            if not os.path.isfile(rng_file):
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                return

        checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        if torch.cuda.is_available():
            is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
            set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss, n_supervised = token_loss(outputs.logits, labels, self.loss_reduction)

        if model.training:
            self.tokens_seen += int(inputs["attention_mask"].sum().item())
            self.supervised_tokens_seen += n_supervised

        # Put labels back: Trainer reuses the dict for evaluation bookkeeping.
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Evaluate with a mean reduction regardless of the training reduction.

        A sum-reduced eval loss is not comparable across evaluations, because
        its value depends on how many supervised tokens the eval batch happened
        to contain. The number reported as `eval_loss` is per-token so that it
        can be read as a perplexity.
        """
        with torch.no_grad():
            labels = inputs.get("labels")
            outputs = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            total, n = token_loss(outputs.logits, labels, "sum")
            loss = total / max(n, 1)
        return (loss.detach(), None, None)


class TrainLogCallback(TrainerCallback):
    """Mirror the training curve to a jsonl keyed by step AND tokens.

    `report_to="none"` leaves the loss curve nowhere durable, so the telemetry
    parquet cannot be joined against the thing it is supposed to explain. Both
    keys are written because the telemetry is scheduled on tokens while the
    Trainer logs on steps.
    """

    def __init__(self, path: str | Path, trainer_ref):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer_ref = trainer_ref

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return control
        trainer = self.trainer_ref()
        record = {
            "step": int(state.global_step),
            "epoch": state.epoch,
            "tokens_seen": getattr(trainer, "tokens_seen", 0),
            "supervised_tokens_seen": getattr(trainer, "supervised_tokens_seen", 0),
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return control


def count_supervised(dataset) -> int:
    return sum(1 for _ in dataset)


__all__ = ["SFTTrainer", "TrainLogCallback", "IGNORE_INDEX"]
