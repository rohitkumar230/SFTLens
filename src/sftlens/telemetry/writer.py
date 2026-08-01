"""Persistence for telemetry output.

RESUME SAFETY
    Shard filenames are keyed by the training step they cover, not by a counter
    held in memory. A counter resets to zero when a run resumes from a
    checkpoint, and the first flush after the resume then overwrites the shard
    written before the crash. Step-keyed names make a resumed run additive.

    An existing file is never overwritten: a resumed run that re-probes a step
    it has already written gets a `.dupNN` suffix, so the collision is visible
    in the archive rather than resolved silently in one direction.

FLUSH POLICY
    The buffer is flushed on a row count, on every deep probe, and at train
    end. The original design flushed only at train end or at 4000 rows, which
    put up to 4000 rows at the mercy of the run not crashing -- and a run that
    crashes is exactly the one whose telemetry you want to read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


class TelemetryWriter:
    def __init__(self, out_dir: str | Path, flush_rows: int = 2000):
        self.root = Path(out_dir)
        self.scalars_dir = self.root / "scalars"
        self.deep_dir = self.root / "deep"
        for d in (self.scalars_dir, self.deep_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.flush_rows = flush_rows
        self._buffer: list[dict] = []

    # -- collision-free paths ----------------------------------------------
    def _unique(self, path: Path) -> Path:
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        for i in range(1, 1000):
            candidate = path.with_name(f"{stem}.dup{i:02d}{suffix}")
            if not candidate.exists():
                print(f"[telemetry] {path.name} already exists; writing {candidate.name}")
                return candidate
        raise RuntimeError(f"cannot find a free filename for {path}")

    # -- scalars ------------------------------------------------------------
    def add_rows(self, rows: list[dict]) -> None:
        self._buffer.extend(rows)
        if len(self._buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        import pandas as pd

        df = pd.DataFrame(self._buffer)
        lo, hi = int(df["step"].min()), int(df["step"].max())
        path = self._unique(self.scalars_dir / f"steps_{lo:07d}_{hi:07d}.parquet")
        df.to_parquet(path, index=False)
        print(f"[telemetry] wrote {len(df)} rows -> {path.name}")
        self._buffer.clear()

    # -- deep artifacts -----------------------------------------------------
    def save_deep(self, step: int, payload: dict) -> None:
        if not payload:
            return
        path = self._unique(self.deep_dir / f"step_{step:07d}.npz")
        np.savez_compressed(path, **payload)
        size_mb = path.stat().st_size / 1e6
        print(f"[telemetry] deep dump -> {path.name} ({size_mb:.1f} MB)")

    # -- weight snapshots ---------------------------------------------------
    def save_weight_snapshot(self, step: int, model) -> None:
        """bf16 weights-only copy on the deep-probe grid.

        Deliberately not a Trainer checkpoint: those carry the fp32 master
        weights and both Adam moments (~12 bytes/param), which is 6x the size
        and is only needed to resume. This is the research artifact -- enough
        to re-derive any weight-space quantity after the run, at 2 bytes/param.
        """
        snapshots = self.root / "weights"
        snapshots.mkdir(parents=True, exist_ok=True)
        path = self._unique(snapshots / f"step_{step:07d}.pt")
        state = {k: v.detach().to("cpu", torch.bfloat16)
                 for k, v in model.state_dict().items()}
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        print(f"[telemetry] weight snapshot -> {path.name} "
              f"({path.stat().st_size / 1e9:.1f} GB)")

    # -- run metadata -------------------------------------------------------
    def write_json(self, name: str, payload: dict) -> None:
        (self.root / name).write_text(json.dumps(payload, indent=2, default=str))


class WeightBaseline:
    """Reference weights for measuring dW, persisted so resume stays honest.

    Captured at callback construction the baseline would be whatever the model
    held at that moment -- which after a resume is the checkpoint, not
    initialisation. Every dW would then be measured from an arbitrary mid-run
    origin, and would silently disagree with the pre-crash portion of the same
    run. Writing it to disk on first creation and reloading it thereafter makes
    dW always relative to step 0.

    STORED IN FP32, DELIBERATELY
        An fp16 baseline is 11 bits of mantissa, so W_0 - fp16(W_0) has
        relative magnitude ~3e-4. That is a noise floor under every dW_relnorm,
        and at LR 3.1e-6 the true displacement is comparable to it for the
        first few hundred steps -- exactly the part of the trajectory this
        study is about. Halving the file is not worth fabricating the early
        signal.

        The file is roughly 4 bytes x (tracked parameters), written once. Use
        `telemetry.track_dw_layers` to bound it.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._weights: dict[str, torch.Tensor] = {}

    def capture_or_load(self, named_weights: dict[str, torch.Tensor]) -> None:
        if self.path.exists():
            loaded = torch.load(self.path, map_location="cpu")
            missing = set(named_weights) - set(loaded)
            if missing:
                # Config changed between the original run and the resume; the
                # newly tracked modules have no step-0 reference and must not
                # be reported against a fabricated one.
                print(
                    f"[telemetry] baseline is missing {len(missing)} tracked modules "
                    "(config changed since step 0); dW omitted for those"
                )
            self._weights = loaded
            print(f"[telemetry] loaded step-0 weight baseline from {self.path.name}")
            return

        self._weights = {k: v.detach().to("cpu", torch.float32).clone()
                         for k, v in named_weights.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        torch.save(self._weights, tmp)
        os.replace(tmp, self.path)   # atomic: a partial baseline is unusable
        print(f"[telemetry] captured step-0 weight baseline -> {self.path.name}")

    def delta(self, name: str, weight: torch.Tensor) -> torch.Tensor | None:
        ref = self._weights.get(name)
        if ref is None:
            return None
        return weight.detach().to("cpu", torch.float32) - ref.to(torch.float32)
