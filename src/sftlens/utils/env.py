"""Environment reporting and reproducibility.

The original script auto-selected a per-device batch size from detected VRAM.
That is the wrong shape for a research repo: it makes the effective batch --
part of the recipe -- a function of whichever machine happened to be free, and
it hid a memory estimate that was never checked against reality. Sizing is now
an explicit config value, and this module only reports and validates.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess

import numpy as np
import torch
from transformers import set_seed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def describe_environment() -> dict:
    """Everything needed to explain a numerical difference between two runs."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": props.name,
            "gpu_count": torch.cuda.device_count(),
            "gpu_total_gb": round(props.total_memory / 1e9, 1),
            "gpu_capability": f"{props.major}.{props.minor}",
            "cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        info["git_commit"] = None
    info["slurm_or_pod"] = os.environ.get("RUNPOD_POD_ID") or os.environ.get("SLURM_JOB_ID")
    return info


def estimate_training_state_gb(n_params: int, param_dtype: str) -> dict[str, float]:
    """Static memory for a full-parameter fine-tune, before activations.

    Mixed precision holds fp32 master weights, fp32 gradients, and two fp32
    Adam moments: 16 bytes per parameter. This is reported so an OOM is
    diagnosed from the log rather than from a stack trace.
    """
    bytes_per_param = 4 if param_dtype == "float32" else 2
    weights = n_params * bytes_per_param / 1e9
    grads = n_params * bytes_per_param / 1e9
    adam = n_params * 8 / 1e9
    return {
        "weights_gb": round(weights, 2),
        "grads_gb": round(grads, 2),
        "adam_gb": round(adam, 2),
        "total_static_gb": round(weights + grads + adam, 2),
    }


def estimate_probe_accum_gb(targets, n_tokens: int) -> float:
    """Live accumulator for the probe: n_tokens x (D_in + D_out) x 4B per module."""
    total = 0
    for t in targets:
        d_out, d_in = t.module.weight.shape
        total += n_tokens * (d_in + d_out) * 4
    return round(total / 1e9, 2)


def estimate_logits_gb(batch: int, seq: int, vocab: int) -> float:
    """Peak memory in the loss.

    Usually the term people forget, and on a large-vocabulary model it is the
    one that actually binds: bf16 logits, plus the fp32 cast cross entropy
    needs, plus log_softmax's own buffer. Scales linearly in batch AND in
    sequence length, so the longest batch in the run sets the peak.
    """
    return batch * seq * vocab * (2 + 4 + 4) / 1e9


def check_capacity(
    info: dict,
    static_gb: float,
    probe_gb: float,
    logits_gb: float = 0.0,
    logits_note: str = "",
) -> None:
    if not info.get("gpu_total_gb"):
        return
    budget = info["gpu_total_gb"]
    needed = static_gb + probe_gb + logits_gb
    print(
        f"[env] memory budget on {budget:.0f} GB:\n"
        f"         {static_gb:6.1f} GB  training state (fp32 master + grads + Adam)\n"
        f"         {probe_gb:6.1f} GB  telemetry probe accumulator\n"
        f"         {logits_gb:6.1f} GB  logits peak {logits_note}\n"
        f"         {needed:6.1f} GB  total, before activations "
        f"({100 * needed / budget:.0f}% of the card)"
    )
    if needed > 0.80 * budget:
        print(
            "[env] WARNING: little headroom for activations. Lower "
            "telemetry.n_tokens, set telemetry.accum_device=cpu, or reduce "
            "recipe.per_device_batch."
        )
    if not info.get("bf16_supported", True):
        print("[env] WARNING: bf16 unsupported on this GPU; the recipe assumes it")
