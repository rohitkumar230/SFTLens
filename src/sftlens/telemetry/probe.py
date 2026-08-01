"""The Gram probe: hooks, token sampling, and the measurement pass.

DESIGN PRINCIPLE
    The training run is the asset. The probe runs on a separate, fixed,
    held-out batch in its own forward/backward, after optimizer.step() and
    zero_grad(). It never steps the optimizer, never touches the data-order
    RNG, and removes its hooks when not probing. If it raises, the exception is
    caught and training continues.

WHAT THE HOOKS COLLECT
    Forward hooks stash the sampled rows of each module's INPUT (X); backward
    hooks stash the sampled rows of the gradient at each module's OUTPUT
    (Delta). The N x N Grams are formed only after every micro-batch has run,
    because the sample is drawn from the whole probe batch rather than from
    whichever micro-batch happens to be in flight.

    Storing X (N x D) rather than K (N x N) is the cheaper side of the trade
    whenever D < N, which holds for every module except down_proj. At N=8192
    it is the difference between ~6 GB and ~17 GB of live accumulator.

TOKEN SELECTION IS FIXED FOR THE WHOLE RUN
    The generator is reseeded from a constant before every probe, so the same
    token positions of the same sequences are measured at every checkpoint. If
    the sample were redrawn each time, step-to-step variation would mix
    sampling noise into the trajectory, which is the one thing a longitudinal
    measurement cannot tolerate.

    The selection is held in RANDOM order rather than sorted, so that the first
    n rows are themselves a uniform subsample for any n. That is what makes the
    n_tokens sweep free.

GRADIENT CHECKPOINTING IS DISABLED DURING THE PROBE
    With checkpointing on, the forward pass runs twice and every forward hook
    fires twice, silently duplicating the accumulated rows. The probe turns it
    off and restores it afterwards rather than trying to detect recomputation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import TelemetryConfig
from ..data.chatml import IGNORE_INDEX
from .reductions import (
    compute_traces,
    derive_metrics,
    exact_matmul,
    gram,
    shuffle_null,
    top_eigenvalues,
)


@dataclass(frozen=True)
class Target:
    name: str
    module: nn.Linear
    layer: int
    suffix: str


def select_targets(model: nn.Module, cfg: TelemetryConfig) -> list[Target]:
    """Instrument a depth-spread subset of layers, not all of them.

    Cost and storage scale linearly in the number of instrumented modules, and
    it is the depth profile that carries the signal, not every individual
    layer. The first layers are always kept because they behave differently
    from the bulk, and so is the last.
    """
    layers = model.model.layers
    n_layers = len(layers)
    keep = (
        set(cfg.always_layers)
        | set(range(0, n_layers, cfg.layer_stride))
        | {n_layers - 1}
    )
    named = dict(model.named_modules())

    targets: list[Target] = []
    for li in sorted(k for k in keep if 0 <= k < n_layers):
        for suffix in cfg.module_suffixes:
            name = f"model.layers.{li}.{suffix}"
            mod = named.get(name)
            if isinstance(mod, nn.Linear):
                targets.append(Target(name, mod, li, suffix))
    if not targets:
        raise RuntimeError(
            "no modules matched; check telemetry.module_suffixes against "
            f"{type(model).__name__}'s layer naming"
        )
    return targets


class GramProbe:
    """Collects X and Delta for the instrumented modules, then reduces."""

    def __init__(self, model: nn.Module, cfg: TelemetryConfig):
        self.model = model
        self.cfg = cfg
        self.targets = select_targets(model, cfg)
        self.by_name = {t.name: t for t in self.targets}

        self._handles: list = []
        self._plan: dict | None = None       # per-micro-batch gather/scatter idx
        self._chunk = 0                      # which micro-batch is in flight
        self._active = False
        self._X: dict[str, torch.Tensor] = {}
        self._D: dict[str, torch.Tensor] = {}

    # -- hooks --------------------------------------------------------------
    def _forward_hook(self, name: str):
        def hook(_mod, inputs, _output):
            if not self._active:
                return
            self._scatter(self._X, name, inputs[0].detach())
        return hook

    def _backward_hook(self, name: str):
        def hook(_mod, _grad_in, grad_out):
            if not self._active:
                return
            self._scatter(self._D, name, grad_out[0].detach())
        return hook

    def _scatter(self, store: dict, name: str, tensor: torch.Tensor) -> None:
        """Copy this micro-batch's sampled rows into their global slots."""
        plan = self._plan[self._chunk]
        if plan["dest"].numel() == 0:
            return
        flat = tensor.reshape(-1, tensor.shape[-1])
        rows = flat[plan["gather"]].to(torch.float32)
        buf = store.get(name)
        if buf is None:
            buf = torch.zeros(
                (self._plan["n_tokens"], rows.shape[-1]),
                dtype=torch.float32,
                device=rows.device if self.cfg.accum_device == "same" else self.cfg.accum_device,
            )
            store[name] = buf
        buf[plan["dest"]] = rows.to(buf.device)

    def attach(self) -> None:
        if self._handles:
            return
        for t in self.targets:
            self._handles.append(t.module.register_forward_hook(self._forward_hook(t.name)))
            self._handles.append(
                t.module.register_full_backward_hook(self._backward_hook(t.name))
            )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self._X.clear()
        self._D.clear()

    # -- token selection ----------------------------------------------------
    def plan(self, micro_batches: list[dict]) -> dict:
        """Choose which tokens to measure. Deterministic in cfg.seed.

        Built once for the run: the probe batch is fixed, so the plan is too.
        """
        positions = []
        offset = 0
        for mb in micro_batches:
            if self.cfg.delta_positions == "supervised":
                valid = (mb["labels"] != IGNORE_INDEX).reshape(-1)
            else:
                valid = mb["attention_mask"].reshape(-1).bool()
            flat_pos = valid.nonzero(as_tuple=True)[0]
            positions.append(flat_pos)
            offset += flat_pos.numel()

        total = offset
        n_tokens = min(self.cfg.n_tokens, total)
        if n_tokens < self.cfg.n_tokens:
            print(
                f"[telemetry] probe pool holds {total} valid tokens; "
                f"n_tokens reduced from {self.cfg.n_tokens} to {n_tokens}"
            )

        gen = torch.Generator().manual_seed(self.cfg.seed)
        sel = torch.randperm(total, generator=gen)[:n_tokens]   # random ORDER, kept

        plan: dict = {"n_tokens": n_tokens, "total_valid": total}
        offset = 0
        for i, flat_pos in enumerate(positions):
            hi = offset + flat_pos.numel()
            in_chunk = (sel >= offset) & (sel < hi)
            local = sel[in_chunk] - offset
            plan[i] = {
                "gather": flat_pos[local],                     # rows of the flat batch
                "dest": in_chunk.nonzero(as_tuple=True)[0],    # slots in the accumulator
            }
            offset = hi
        self._plan = plan
        return plan

    # -- the measurement pass ----------------------------------------------
    def run(self, micro_batches: list[dict], loss_fn) -> dict:
        """Forward/backward over the probe batch, collecting X and Delta.

        Returns loss bookkeeping. Gradients are not accumulated into
        parameters: all parameters are temporarily frozen and the graph is
        rooted at the input embeddings instead, so a full set of .grad buffers
        (6.8 GB for a 1.7B model) is never allocated. Activation gradients
        still flow, because grad_input of a Linear does not depend on whether
        its weight requires grad.
        """
        model = self.model
        was_training = model.training
        was_cache = model.config.use_cache
        was_ckpt = getattr(model, "is_gradient_checkpointing", False)

        model.eval()
        model.config.use_cache = False
        if was_ckpt:
            model.gradient_checkpointing_disable()

        frozen = [p for p in model.parameters() if p.requires_grad]
        for p in frozen:
            p.requires_grad_(False)

        embed = model.get_input_embeddings()
        total_loss, total_tokens = 0.0, 0

        try:
            self.clear()
            self._active = True
            self.attach()

            for i, mb in enumerate(micro_batches):
                self._chunk = i
                inputs_embeds = embed(mb["input_ids"])
                # Root the graph here: with every parameter frozen this is the
                # only leaf requiring grad, so backward populates one small
                # buffer instead of one per parameter.
                inputs_embeds.requires_grad_(True)

                out = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=mb["attention_mask"],
                )
                loss, n_tok = loss_fn(out.logits, mb["labels"])
                loss.backward()

                total_loss += float(loss.detach().item())
                total_tokens += n_tok
                del out, loss, inputs_embeds
        finally:
            self._active = False
            self.detach()
            for p in frozen:
                p.requires_grad_(True)
            model.config.use_cache = was_cache
            if was_ckpt:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            if was_training:
                model.train()

        return {
            "probe_loss_total": total_loss,
            "probe_tokens": total_tokens,
            "probe_loss_per_token": total_loss / total_tokens if total_tokens else math.nan,
        }

    # -- reduction ----------------------------------------------------------
    def reduce(self, deep: bool) -> tuple[list[dict], dict]:
        """Turn the accumulated X and Delta into scalar rows (+ deep artifacts)."""
        rows: list[dict] = []
        artifacts: dict = {}
        n_full = self._plan["n_tokens"]

        sweep = (
            list(self.cfg.n_tokens_sweep)
            if self.cfg.n_tokens_sweep and (deep or not self.cfg.sweep_on_deep_only)
            else []
        )
        # The full-N measurement always happens; the sweep adds smaller N so
        # the finite-N trend is visible in the same probe.
        n_values = sorted({n_full} | {n for n in sweep if n <= n_full})

        gen = torch.Generator().manual_seed(self.cfg.seed + 1)

        with exact_matmul():
            for t in self.targets:
                X = self._X.get(t.name)
                Delta = self._D.get(t.name)
                if X is None or Delta is None:
                    # A module whose backward hook never fired (input did not
                    # require grad) is skipped rather than silently reported.
                    continue

                for n in n_values:
                    Xn, Dn = X[:n].float(), Delta[:n].float()
                    # Centre once and reuse: the Gram, the eigenvalues and the
                    # raw dump must all refer to the same X_c, or the archived
                    # rows will not reproduce the archived scalars.
                    Xc = Xn - Xn.mean(0, keepdim=True)
                    K = gram(Xc, center=False, dtype=torch.float32)
                    M = gram(Dn, center=False, dtype=torch.float32)

                    metrics = derive_metrics(
                        compute_traces(K, M), n, X.shape[-1], Delta.shape[-1]
                    )
                    metrics.update(shuffle_null(K, M, gen))
                    metrics["R_shuffled"] = (
                        metrics["a_shuffled"] / metrics["c"]
                        if metrics["c"] > 0 else math.nan
                    )

                    if self.cfg.log_uncentered:
                        # Sigma is a covariance and so is centred, but the true
                        # weight gradient is formed from uncentred inputs. On
                        # models with large activation offsets these differ
                        # materially, so both are recorded rather than conflated.
                        Ku = gram(Xn, center=False, dtype=torch.float32)
                        tru = compute_traces(Ku, M)
                        metrics["g2_uncentered"] = tru["trKM"]
                        metrics["tr_Sigma_uncentered"] = tru["trK"] / (n - 1)
                        metrics["c_uncentered"] = (
                            tru["trK2"] / tru["trK"] ** 2 if tru["trK"] > 0 else math.nan
                        )
                        del Ku

                    rows.append({
                        "name": t.name, "layer": t.layer, "module": t.suffix,
                        "is_full_n": n == n_full, **metrics,
                    })

                    if deep and n == n_full:
                        artifacts.update(self._deep_artifacts(t, Xc, Dn))
                    del K, M, Xc

        return rows, artifacts

    def _deep_artifacts(self, t: Target, Xc, Delta) -> dict:
        """Spectra and raw rows. Computed from the FEATURES, not the Gram.

        Top-k eigenvalues of a Gram are the squared singular values of the
        features that formed it, so this avoids an O(N^3) eigendecomposition
        that at N=8192 would dominate the entire probe. See
        `reductions.top_eigenvalues`.
        """
        cfg = self.cfg
        k = cfg.top_eig
        eig = dict(method=cfg.eig_method, oversample=cfg.eig_oversample, niter=cfg.eig_niter)
        out = {
            f"{t.name}|eig_K": top_eigenvalues(Xc, k, **eig).cpu().numpy(),
            f"{t.name}|eig_M": top_eigenvalues(Delta, k, **eig).cpu().numpy(),
        }
        if cfg.dump_raw:
            # The raw rows are what make quantities invented later computable
            # without re-running the probe -- including anything involving
            # Adam-preconditioned gradients.
            m = cfg.n_tokens_deep
            out[f"{t.name}|Xc"] = Xc[:m].half().cpu().numpy()
            out[f"{t.name}|Delta"] = Delta[:m].half().cpu().numpy()
        return out
