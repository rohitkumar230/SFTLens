"""Typed run configuration, loaded from layered YAML.

A run is fully described by one `RunConfig`. It is serialised verbatim into the
output directory so that a telemetry archive is interpretable years later
without reference to this source tree.

Layering, lowest precedence first:
    configs/base.yaml  ->  --config FILE...  ->  --set dotted.key=value
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-1.7B"
    attn_implementation: str = "sdpa"

    # Mixed precision, not pure bf16: parameters are held in fp32 and autocast
    # runs the compute in bf16. Loading directly in bf16 would round every
    # update into an 8-bit mantissa, which at LR 3.1e-6 discards most of the
    # signal -- and the size of that discarded update is one of the things the
    # telemetry is trying to measure.
    param_dtype: str = "float32"
    bf16: bool = True
    gradient_checkpointing: bool = True

    # ChatML control tokens. Asserted present in the base vocabulary at load
    # time; see `sftlens.data.chatml`. A resize here would inject randomly
    # initialised embeddings whose gradient geometry is wildly atypical for the
    # first few hundred steps, contaminating the measurement.
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"


@dataclass
class DataConfig:
    dataset_id: str = "allenai/tulu-3-sft-mixture"
    split: str = "train"
    loader: str = "tulu3"          # {"tulu3", "dolly"} -> sftlens.data.build

    # Proportional stratified sample over the `source` column preserves the
    # mixture proportions that the recipe's learning rate was selected against.
    subset_size: int | None = 50_000
    stratify_column: str | None = "source"

    max_seq_len: int = 4096
    # Drop rather than truncate: a truncated conversation loses its final
    # <|im_end|>, which teaches the model never to stop.
    on_overflow: str = "drop"
    eval_size: int = 1_000
    system_prompt: str = "You are a helpful assistant."

    num_proc: int = 8


@dataclass
class RecipeConfig:
    """Hyperparameters. Every field here is transcribed from a published run.

    Deviating from a published cell means owning a hyperparameter search, which
    is the cost this project is explicitly avoiding. `provenance` is emitted
    into the run manifest so the claim travels with the artifacts.
    """

    provenance: str = (
        "SmolTulu SFT-1207 (Alrashed 2024, arXiv:2412.08347, Table 2): "
        "LR 3.1e-6 @ effective batch 32, LR/BS = 0.097e-6, on SmolLM2-1.7B "
        "over allenai/tulu-3-sft-mixture. Schedule/warmup/decay inherited from "
        "Tulu 3 (Lambert et al. 2024, arXiv:2411.15124)."
    )

    lr: float = 3.1e-6
    effective_batch: int = 32
    epochs: float = 2.0

    lr_scheduler: str = "linear"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch_fused"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Tulu 3 sums the token loss instead of averaging it, so long sequences
    # contribute proportionally more gradient. This changes the effective step
    # size relative to HF's default mean reduction, so it is part of the recipe
    # rather than an implementation detail.
    loss_reduction: str = "sum"

    # Sizing knobs. These do NOT alter the recipe -- per_device_batch x
    # grad_accum is constrained to equal effective_batch -- but they do
    # determine whether the run fits in memory.
    per_device_batch: int = 4
    max_steps: int = -1

    @property
    def grad_accum(self) -> int:
        if self.effective_batch % self.per_device_batch:
            raise ValueError(
                f"effective_batch={self.effective_batch} is not divisible by "
                f"per_device_batch={self.per_device_batch}; the recipe's batch "
                "size cannot be honoured exactly."
            )
        return self.effective_batch // self.per_device_batch

    @property
    def lr_over_bs_e6(self) -> float:
        """The quantity SmolTulu identifies as the controlling ratio."""
        return self.lr / self.effective_batch * 1e6


@dataclass
class TelemetryConfig:
    enabled: bool = True

    # --- cadence ------------------------------------------------------------
    # Measured in optimizer steps consumed, but scheduled on TOKENS so that runs
    # at different batch sizes land on a common x-axis. A step-based cadence
    # would put the BS=8 and BS=32 arms on incomparable grids.
    cadence_unit: str = "tokens"        # {"tokens", "steps"}
    light_every: int = 2_000_000        # tokens between scalar probes
    deep_every: int = 8_000_000         # tokens between spectra + raw dumps
    probe_at_step_zero: bool = True     # baseline before any update

    # --- probe batch --------------------------------------------------------
    # Fixed and held out. Enough sequences to span the 19 mixture sources: with
    # only a handful you measure the geometry of whichever few you sampled, and
    # tokens within one sequence are strongly correlated, so the effective
    # sample size is far below the token count.
    probe_seqs: int = 96
    probe_max_len: int = 1024
    probe_stratify: bool = True
    probe_micro_batch: int = 8          # probe batch is chunked to fit memory

    # --- estimator ----------------------------------------------------------
    # Participation ratio of a D-dimensional covariance estimated from N token
    # samples is capped near N and biased low whenever N is not >> D: an
    # isotropic covariance yields PR = ND/(N+D+1), not D.
    #
    # This directly attacks the down_proj (D_in 8192) vs o_proj (D_in 2048)
    # dimension contrast, whose true ratio is 4.0:
    #
    #     N       PR(8192)   PR(2048)   measured ratio
    #     1024        910        682          1.33
    #     8192       4096       1638          2.50
    #    65536       7285       1985          3.67
    #
    # At N=1024 the 4x contrast reads as 1.33x -- almost entirely an artifact
    # of N. n_tokens=8192 recovers most of it; n_tokens_sweep re-probes the
    # same step at several N so the residual bias can be extrapolated from the
    # measured trend rather than assumed away.
    n_tokens: int = 8192
    n_tokens_sweep: tuple[int, ...] = (1024, 2048, 4096, 8192)
    sweep_on_deep_only: bool = True

    # Gram matrices are formed in fp32 (fp64 matmul is 1/64 rate on Ada and
    # consumer parts); only the scalar reductions are accumulated in fp64,
    # where the tr K vs tr K^2 dynamic range actually matters.
    matmul_dtype: str = "float32"
    reduce_dtype: str = "float64"

    # Where the per-module X and Delta accumulators live between the probe's
    # forward and its reduction. "same" keeps them on the model's device.
    # Live accumulator size is roughly
    #     n_tokens * sum_over_modules(D_in + D_out) * 4 bytes
    # which for SmolLM2-1.7B at n_tokens=8192 over 63 modules is ~10 GB.
    # Set to "cpu" if that does not fit alongside the training state; the probe
    # is infrequent enough that the transfer cost is not material.
    accum_device: str = "same"

    # Sigma is a covariance and so is centred, but the true weight gradient is
    # formed from uncentred inputs. On models with large activation offsets the
    # two differ materially, so both are logged rather than silently conflated.
    log_uncentered: bool = True

    # Restrict Delta to positions carrying supervision. Prompt positions still
    # receive gradient through attention, so both maskings are defensible; this
    # records which one produced the numbers.
    delta_positions: str = "all"        # {"all", "supervised"}

    # --- coverage -----------------------------------------------------------
    layer_stride: int = 4
    always_layers: tuple[int, ...] = (0, 1)
    module_suffixes: tuple[str, ...] = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )

    # --- deep artifacts -----------------------------------------------------
    top_eig: int = 64
    # Top-k eigenvalues come from the FEATURES, not an O(N^3) eigendecomposition
    # of the Gram: at N=8192 the exact route costs ~40 s per matrix and there
    # are two per module. "randomized" lands within ~1e-4 relative on an
    # anisotropic spectrum (~8e-3 on a flat one); "exact" uses eigvalsh on the
    # smaller Gram form. The same method is used for every module, since
    # comparing down_proj against o_proj is the experiment.
    # These are a deep-probe diagnostic; the primary substrates come from the
    # traces, which are exact regardless.
    eig_method: str = "randomized"     # {"randomized", "exact"}
    eig_oversample: int = 192
    eig_niter: int = 8
    dump_raw: bool = True
    n_tokens_deep: int = 256            # tokens kept in the raw X/Delta dump
    # Layers whose cumulative update dW gets an SVD spectrum. Empty means a
    # three-point depth spread (first / middle / last probed layer). Each
    # tracked layer costs 4 bytes x its parameters in the step-0 baseline file,
    # so widening this is a storage decision, not a free one.
    track_dw_layers: tuple[int, ...] = ()
    dw_rank: int = 64

    # bf16 weights-only snapshot alongside each deep dump: 2 bytes/param, so
    # ~3.4 GB per snapshot for a 1.7B model. Off by default because the deep
    # dump already carries dW spectra and the raw X/Delta rows; turn it on when
    # you want to re-derive arbitrary weight-space quantities after the fact
    # without re-running training.
    snapshot_weights_on_deep: bool = False

    seed: int = 1234
    flush_rows: int = 2_000


@dataclass
class RunConfig:
    run_name: str = "smoltulu-sft-1207"
    output_dir: str = "runs/${run_name}"
    seed: int = 42

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    recipe: RecipeConfig = field(default_factory=RecipeConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # --- bookkeeping --------------------------------------------------------
    logging_steps: int = 5
    eval_strategy: str = "steps"
    eval_steps: int = 100
    # "no" disables checkpointing entirely. Correct for short runs: a
    # full-FT checkpoint is ~12 bytes/param (fp32 weights + two fp32 Adam
    # moments), which for a 1.7B model is 20.5 GB to insure a run that only
    # takes 39 minutes to repeat.
    save_strategy: str = "steps"
    save_steps: int = 200
    # Full-FT checkpoints are ~12 bytes/param (fp32 weights + two fp32 Adam
    # moments), so they are for crash recovery, not for archiving. See
    # telemetry.snapshot_weights_on_deep for the research artifact.
    save_total_limit: int | None = 2
    dataloader_num_workers: int = 4
    group_by_length: bool = True
    report_to: str = "none"
    # Force CPU regardless of what accelerate would pick. Used by the test
    # suite so results do not depend on whether the host has MPS.
    use_cpu: bool = False

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir.replace("${run_name}", self.run_name))

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["_derived"] = {
            "grad_accum": self.recipe.grad_accum,
            "lr_over_bs_e6": self.recipe.lr_over_bs_e6,
        }
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_SECTIONS = {f.name: f.type for f in dataclasses.fields(RunConfig)}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(cls, payload: dict) -> Any:
    """Instantiate a dataclass from a dict, rejecting unknown keys.

    Silently dropping a misspelled key is how a run ends up not using the
    setting you thought you set.
    """
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    kwargs = {}
    for name, value in payload.items():
        f = known[name]
        # YAML has no tuple type; dataclass defaults use tuples for immutability.
        if isinstance(f.default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


# YAML 1.1 only recognises a float in exponent form when a decimal point is
# present AND the exponent is signed, so `1e-5` parses as the STRING "1e-5"
# while `1.0e-5` parses as a float. Left alone, `--set recipe.lr=1e-5` would
# reach TrainingArguments as a string. This matches exactly that gap; plain
# integers, decimals and genuine words are handled correctly by YAML already
# and are not touched.
_SCI_NOTATION = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def _parse_scalar(text: str) -> Any:
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, str) and _SCI_NOTATION.match(value.strip()):
        return float(value)
    return value


def load_config(
    paths: list[str | Path] | None = None,
    overrides: list[str] | None = None,
) -> RunConfig:
    """Compose a RunConfig from base.yaml, overlay files, and dotted --set args."""
    payload: dict[str, Any] = {}
    base = CONFIG_ROOT / "base.yaml"
    if base.exists():
        payload = yaml.safe_load(base.read_text()) or {}

    for p in paths or []:
        p = Path(p)
        if not p.exists():
            p = CONFIG_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        payload = _deep_merge(payload, yaml.safe_load(p.read_text()) or {})

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects dotted.key=value, got {item!r}")
        key, _, raw = item.partition("=")
        cursor = payload
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_scalar(raw)

    sections = {}
    for name, cls in (
        ("model", ModelConfig), ("data", DataConfig),
        ("recipe", RecipeConfig), ("telemetry", TelemetryConfig),
    ):
        sections[name] = _coerce(cls, payload.pop(name, {}) or {})

    cfg = _coerce(RunConfig, {**payload, **sections})
    validate(cfg)
    return cfg


def validate(cfg: RunConfig) -> None:
    """Fail fast on combinations that would produce a quietly wrong run."""
    r, t, d = cfg.recipe, cfg.telemetry, cfg.data

    _ = r.grad_accum  # raises if the effective batch cannot be honoured

    if r.loss_reduction not in {"sum", "mean"}:
        raise ValueError(f"recipe.loss_reduction must be sum|mean, got {r.loss_reduction!r}")
    if d.on_overflow not in {"drop", "truncate"}:
        raise ValueError(f"data.on_overflow must be drop|truncate, got {d.on_overflow!r}")
    if t.delta_positions not in {"all", "supervised"}:
        raise ValueError("telemetry.delta_positions must be all|supervised")
    if cfg.save_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"save_strategy must be no|steps|epoch, got {cfg.save_strategy!r}")
    if cfg.eval_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"eval_strategy must be no|steps|epoch, got {cfg.eval_strategy!r}")
    if t.cadence_unit not in {"tokens", "steps"}:
        raise ValueError("telemetry.cadence_unit must be tokens|steps")
    if t.eig_method not in {"randomized", "exact"}:
        raise ValueError(f"telemetry.eig_method must be randomized|exact, got {t.eig_method!r}")

    if t.enabled:
        if t.deep_every % t.light_every:
            raise ValueError(
                "telemetry.deep_every must be a multiple of light_every so that "
                "deep probes land on the light-probe grid"
            )
        if t.n_tokens_sweep and max(t.n_tokens_sweep) > t.n_tokens:
            raise ValueError(
                f"telemetry.n_tokens_sweep max ({max(t.n_tokens_sweep)}) exceeds "
                f"n_tokens ({t.n_tokens}); the sweep subsamples the probe pool"
            )
        if t.probe_max_len > d.max_seq_len:
            raise ValueError("telemetry.probe_max_len exceeds data.max_seq_len")
        # The probe pool must be able to supply n_tokens after padding is dropped.
        pool = t.probe_seqs * t.probe_max_len
        if pool < t.n_tokens:
            raise ValueError(
                f"probe pool is at most {pool} tokens but n_tokens={t.n_tokens}; "
                "raise probe_seqs or lower n_tokens"
            )
