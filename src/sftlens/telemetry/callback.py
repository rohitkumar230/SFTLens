"""Trainer integration.

A plain `TrainerCallback` subclass. The original design wrapped the callback in
a shim whose `__getattr__` returned a no-op for every unimplemented event; that
happened to be safe only because `CallbackHandler` ignores a `None` return, and
it would have swallowed any real error in dispatch. Subclassing gets the same
result with none of the ambiguity.

CADENCE IS IN TOKENS
    Probes are scheduled on tokens consumed rather than optimizer steps so that
    arms trained at different batch sizes land on a common x-axis. The SmolTulu
    1207 and 1130 recipes differ by 4x in batch size for the same token budget;
    a step-based cadence would put them on grids that cannot be overlaid.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from transformers import TrainerCallback

from ..config import RunConfig
from ..data.chatml import IGNORE_INDEX
from ..data.mixture import stratified_indices
from ..train.loss import token_loss
from .probe import GramProbe
from .writer import TelemetryWriter, WeightBaseline


def build_probe_batch(eval_dataset, collator, cfg, seed: int) -> list[dict]:
    """A fixed, held-out, source-stratified probe batch, chunked for memory.

    Fixed and identical at every checkpoint: without that you cannot separate
    data drift from network change. Stratified because the tulu-3 mixture spans
    19 sources with very uneven shares, and an unstratified handful of
    sequences measures the geometry of whichever sources it happened to draw.
    """
    n = min(cfg.probe_seqs, len(eval_dataset))
    if cfg.probe_stratify and "source" in eval_dataset.column_names:
        idx = stratified_indices(eval_dataset["source"], n, seed)
    else:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(eval_dataset), size=n, replace=False).tolist())

    features = []
    for i in idx:
        row = dict(eval_dataset[int(i)])
        ids = row["input_ids"][: cfg.probe_max_len]
        labels = row["labels"][: cfg.probe_max_len]
        # Truncation can strip every supervised token from a long-prompt
        # example. Such a row contributes no Delta and would just be dead
        # weight in the probe batch.
        if all(lab == IGNORE_INDEX for lab in labels):
            continue
        features.append({"input_ids": ids, "labels": labels})

    if not features:
        raise RuntimeError(
            f"no probe sequence retained a supervised token within "
            f"probe_max_len={cfg.probe_max_len}; raise it"
        )

    size = cfg.probe_micro_batch
    return [collator(features[i : i + size]) for i in range(0, len(features), size)]


class TelemetryCallback(TrainerCallback):
    def __init__(
        self,
        run_cfg: RunConfig,
        model: nn.Module,
        probe_batch: list[dict],
        token_counter: Callable[[], int],
        optimizer_getter: Callable[[], torch.optim.Optimizer | None],
    ):
        self.run_cfg = run_cfg
        self.cfg = run_cfg.telemetry
        self.model = model
        self.probe_batch = probe_batch
        self.token_counter = token_counter
        self.optimizer_getter = optimizer_getter

        out = run_cfg.resolved_output_dir() / "telemetry"
        self.writer = TelemetryWriter(out, flush_rows=self.cfg.flush_rows)
        self.writer.write_json("config.json", run_cfg.to_dict())

        self.probe = GramProbe(model, self.cfg)
        self._plan_ready = False

        # dW tracking needs an fp32 step-0 copy of every tracked weight, so the
        # default is a three-point depth spread rather than every probed layer:
        # for SmolLM2-1.7B that is ~0.8 GB on disk instead of ~2.1 GB, and the
        # depth profile is already carried by the scalar substrates.
        probed = sorted({t.layer for t in self.probe.targets})
        tracked = self.cfg.track_dw_layers or (probed[0], probed[len(probed) // 2], probed[-1])
        self.dw_targets = [t for t in self.probe.targets if t.layer in set(tracked)]
        self.baseline = WeightBaseline(out / "weight_baseline_step0.pt")

        self._next_light = 0
        self._next_deep = 0
        self._last_probe_step = -1

        print(
            f"[telemetry] {len(self.probe.targets)} modules across "
            f"{len({t.layer for t in self.probe.targets})} layers; "
            f"probe batch {len(probe_batch)} micro-batches"
        )

    # -- scheduling ---------------------------------------------------------
    def _clock(self, state) -> int:
        return self.token_counter() if self.cfg.cadence_unit == "tokens" else int(state.global_step)

    def _due(self, now: int) -> tuple[bool, bool]:
        light = now >= self._next_light
        deep = now >= self._next_deep
        return light or deep, deep

    def _advance(self, now: int) -> None:
        step = self.cfg.light_every
        self._next_light = (now // step + 1) * step
        deep_step = self.cfg.deep_every
        self._next_deep = (now // deep_step + 1) * deep_step

    # -- trainer events -----------------------------------------------------
    def on_train_begin(self, args, state, control, **kwargs):
        if not self.cfg.enabled:
            return control
        self.baseline.capture_or_load(
            {t.name: t.module.weight for t in self.dw_targets}
        )
        if self.cfg.probe_at_step_zero and int(state.global_step) == 0:
            # A baseline measured before any update is the only reference point
            # for everything that follows.
            self._probe(int(state.global_step), deep=True)
        self._advance(self._clock(state))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.cfg.enabled:
            return control
        now = self._clock(state)
        due, deep = self._due(now)
        step = int(state.global_step)
        if not due or step == self._last_probe_step:
            return control
        self._probe(step, deep=deep)
        self._advance(now)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.cfg.enabled:
            self.writer.flush()
        return control

    # -- the probe ----------------------------------------------------------
    def _probe(self, step: int, deep: bool) -> None:
        t0 = time.time()
        try:
            self._run_probe(step, deep)
        except Exception as exc:  # never take the training run down
            import traceback

            print(f"[telemetry] step {step} skipped: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            # Discard probe gradients. The optimizer is never stepped on them.
            self.model.zero_grad(set_to_none=True)
            self.probe.clear()
            self._last_probe_step = step
            if deep:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(
                f"[telemetry] step {step} {'deep' if deep else 'light'} "
                f"{time.time() - t0:.1f}s"
            )

    def _run_probe(self, step: int, deep: bool) -> None:
        device = next(self.model.parameters()).device
        batch = [{k: v.to(device) for k, v in mb.items()} for mb in self.probe_batch]

        if not self._plan_ready:
            plan = self.probe.plan(batch)
            self.writer.write_json(
                "probe_plan.json",
                {
                    "n_tokens": plan["n_tokens"],
                    "total_valid_tokens": plan["total_valid"],
                    "micro_batches": len(batch),
                    "delta_positions": self.cfg.delta_positions,
                    "seed": self.cfg.seed,
                },
            )
            self._plan_ready = True

        def loss_fn(logits, labels):
            return token_loss(logits, labels, self.run_cfg.recipe.loss_reduction)

        stats = self.probe.run(batch, loss_fn)
        rows, artifacts = self.probe.reduce(deep=deep)

        tokens = self.token_counter()
        for r in rows:
            r.update(step=step, tokens_seen=tokens, **stats)
        self.writer.add_rows(rows)

        if deep:
            artifacts.update(self._optimizer_artifacts())
            artifacts.update(self._weight_delta_artifacts())
            self.writer.save_deep(step, artifacts)
            if self.cfg.snapshot_weights_on_deep:
                self.writer.save_weight_snapshot(step, self.model)
            # Flush scalars alongside every deep dump so the parquet and npz
            # halves of the archive never disagree about how far the run got.
            self.writer.flush()

    # -- deep-only extras ---------------------------------------------------
    def _optimizer_artifacts(self) -> dict:
        """Adam moment summaries.

        Needed to ask whether preconditioned updates share the geometry of raw
        gradients. Summaries only -- the raw X/Delta dump lets anything else be
        recomputed after the fact.
        """
        opt = self.optimizer_getter()
        if opt is None:
            return {}
        # Accelerate wraps the optimizer; the moment state lives on the inner one.
        opt = getattr(opt, "optimizer", opt)

        out = {}
        for t in self.probe.targets:
            st = opt.state.get(t.module.weight)
            if not st or "exp_avg" not in st:
                continue
            m, v = st["exp_avg"], st["exp_avg_sq"]
            precond = m / (v.sqrt() + self.run_cfg.recipe.adam_epsilon)
            out[f"{t.name}|adam_m_norm"] = np.float32(m.norm().item())
            out[f"{t.name}|adam_v_norm"] = np.float32(v.norm().item())
            out[f"{t.name}|adam_precond_norm"] = np.float32(precond.norm().item())
            del precond
        return out

    def _weight_delta_artifacts(self) -> dict:
        """Spectrum of the cumulative update, relative to step 0."""
        out = {}
        for t in self.dw_targets:
            dW = self.baseline.delta(t.name, t.module.weight)
            if dW is None:
                continue
            try:
                _, s, _ = torch.svd_lowrank(dW, q=min(self.cfg.dw_rank, min(dW.shape) - 1))
                out[f"{t.name}|dW_svals"] = s.numpy().astype(np.float32)
                current = t.module.weight.detach().float().cpu().norm()
                out[f"{t.name}|dW_relnorm"] = np.float32(
                    (dW.norm() / (current + 1e-12)).item()
                )
            except Exception as exc:
                print(f"[telemetry] dW spectrum failed for {t.name}: {exc}")
            del dW
        return out


def attach_telemetry(
    trainer, run_cfg: RunConfig, eval_dataset, collator
) -> TelemetryCallback | None:
    """Wire the callback to a Trainer. Returns None when telemetry is off."""
    if not run_cfg.telemetry.enabled:
        return None

    probe_batch = build_probe_batch(
        eval_dataset, collator, run_cfg.telemetry, run_cfg.telemetry.seed
    )
    cb = TelemetryCallback(
        run_cfg=run_cfg,
        model=trainer.model,
        probe_batch=probe_batch,
        token_counter=lambda: getattr(trainer, "tokens_seen", 0),
        optimizer_getter=lambda: getattr(trainer, "optimizer", None),
    )
    trainer.add_callback(cb)
    return cb


__all__ = ["TelemetryCallback", "attach_telemetry", "build_probe_batch"]
