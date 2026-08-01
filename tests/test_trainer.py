"""Regression test for SFTTrainer's RNG-state resume override.

transformers<=4.51's `Trainer._load_rng_state` calls
`torch.load(rng_file, weights_only=True)` unconditionally. Two bugs compound
against that on this project's actual hardware:

  1. Its own `safe_globals()` allowlist helper no-ops on torch < 2.6, so on the
     pods this project runs on (torch 2.4.1) the call is strict with zero
     allowlisting, and any resume fails with `UnpicklingError: Unsupported
     global: numpy.core.multiarray._reconstruct`.
  2. On torch >= 2.6, where `safe_globals()` does fire, its context-manager
     form does not compose with a permanent `add_safe_globals()` registration:
     entering that context anywhere in the process resets the allowlist to
     torch's own defaults on exit. Verified directly -- a permanent
     registration survived right up until this project's own resume tests ran
     elsewhere in the same process, then vanished.

`SFTTrainer._load_rng_state` (src/sftlens/train/trainer.py) sidesteps both by
overriding the method to call `torch.load(..., weights_only=False)` directly,
since every RNG-state file this project resumes from is one it generated
itself in a prior step of the same run -- there is no untrusted-input boundary
to defend.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from transformers import TrainingArguments

from sftlens.train.trainer import SFTTrainer


def test_load_rng_state_round_trips_via_weights_only_false(tmp_path, monkeypatch):
    """Directly exercises the overridden method against a real checkpoint
    layout, bypassing the (broken) parent implementation entirely."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    ckpt = tmp_path / "checkpoint-6"
    ckpt.mkdir()
    torch.save(state, ckpt / "rng_state.pth")

    # Build a bare SFTTrainer-shaped object: _load_rng_state only needs
    # `self.args`, so a full model/dataset setup is unnecessary here.
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.args = TrainingArguments(output_dir=str(tmp_path), report_to="none")

    # Perturb current RNG state so restoration is actually observable.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    trainer._load_rng_state(str(ckpt))

    assert random.getstate()[1] == state["python"][1]
    np.testing.assert_array_equal(np.random.get_state()[1], state["numpy"][1])
    assert torch.equal(torch.random.get_rng_state(), state["cpu"])


def test_load_rng_state_is_a_noop_for_no_checkpoint():
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.args = TrainingArguments(output_dir="/tmp", report_to="none")
    trainer._load_rng_state(None)  # must not raise


def test_load_rng_state_is_a_noop_when_file_missing(tmp_path):
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    trainer._load_rng_state(str(tmp_path))  # no rng_state.pth here; must not raise
