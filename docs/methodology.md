# Methodology

Technical detail behind the summary in the top-level README: the exact
statistics measured, why a naive version of the main one is misleading, the
training recipe, and the dataset used.

## Metrics

For each instrumented linear layer, on a fixed held-out probe batch, at every
checkpoint along the training trajectory:

| | quantity | meaning |
|---|---|---|
| input factor | `tr_Sigma`, `fro_Sigma_sq`, `PR_Sigma`, `c` | spectrum of the input covariance `Sigma = X_c^T X_c / (N-1)` |
| gradient | `g2`, `rho_delta`, `a`, `R`, `mu2` | how the gradient aligns with `Sigma` |
| output factor | `tr_Omega`, `PR_Omega`, `rayleigh_full`, `R_Omega`, `R_Omega_sym` | spectrum of `Omega = Delta^T Delta`, and the error from assuming it is isotropic |
| nulls | `a_shuffled`, `R_shuffled`, `PR_*_null` | the same statistics computed with no alignment, or with an isotropic covariance |

Every quantity reduces to a trace of products of two `N x N` Gram matrices,
independent of the layer width `D`. This is what keeps the probe affordable
at `D_in = 8192`. The identities are defined in
[`reductions.py`](../src/sftlens/telemetry/reductions.py) and each is checked
against a direct dense computation in
[`test_reductions.py`](../tests/test_reductions.py).

## Participation ratio and finite-sample bias

`PR` (participation ratio) is a standard measure of how spread out a
covariance's eigenvalues are: low means the variance concentrates in a few
directions, high means it's spread evenly. It's the main quantity used to
describe layer input structure here.

Estimated from a finite sample of `N` tokens, `PR` is capped near `N`
regardless of the true underlying structure. A covariance with no real
structure at all (isotropic) yields `ND / (N+D+1)`, not the layer's actual
dimension `D`:

| N | PR(D=8192) | PR(D=2048) | measured ratio (true = 4.0) |
|---|---|---|---|
| 1,024 | 910 | 682 | 1.33 |
| 8,192 | 4,096 | 1,638 | 2.50 |
| 65,536 | 7,285 | 1,985 | 3.67 |

At `N=1024`, a real 4x dimension contrast between two layers reads as 1.33x
purely from this sampling artifact. Two mitigations are built into the
pipeline: the null value and the ratio against it are logged on every row, so
the bias is visible rather than implicit, and each probe step can re-sample
at several `N` so the bias can be extrapolated rather than assumed away. A
measured `PR` close to its null value is consistent with no real structure,
not evidence of any.

## Recipe

Training hyperparameters are transcribed without modification from
[SmolTulu](https://arxiv.org/abs/2412.08347) (Alrashed 2024, Table 2) and the
[Tulu 3](https://arxiv.org/abs/2411.15124) pipeline it builds on:

| | SmolTulu SFT-1130 | SmolTulu SFT-1207 (default) | Tulu 3 SFT 8B | Tulu 3 SFT 70B |
|---|---|---|---|---|
| learning rate | 9.0e-5 | 3.1e-6 | 5.0e-6 | 2.0e-6 |
| batch size | 8 | 32 | 128 | 128 |
| LR / BS x 1e6 | 11.25 | 0.097 | 0.039 | 0.016 |

The default configuration (SFT-1207) applies the model
(`HuggingFaceTB/SmolLM2-1.7B`) and dataset (`allenai/tulu-3-sft-mixture`) it
was originally selected on, so no value is extrapolated to a new setting.
Schedule, warmup, weight decay, and the loss reduction are taken from Tulu 3.

`configs/recipe/smoltulu-sft-1130.yaml` is a second arm at a 29x higher
learning rate, also published, giving a controlled learning-rate axis at
fixed model and data.
[`tests/test_config.py::test_arms_differ_only_in_lr_and_batch`](../tests/test_config.py)
asserts the two configurations differ only in learning rate and batch size.

## Dataset

Tokenized with the SmolLM2 tokenizer under this repo's ChatML template.

| | dolly-15k | tulu-3 mixture |
|---|---|---|
| examples | 15,011 | 939,343 |
| mean tokens / example | 205 | 517 |
| median tokens / example | 126 | 404 |
| mean supervised tokens | 84 (41%) | 440 (85%) |
| over 1,024 tokens | 1.5% | 9.1% |
| over 4,096 tokens | 0.07% | 0.18% |
| total tokens | 3.1M | 486M |

The 50k stratified subset of the tulu-3 mixture used for training is 51.7M
tokens over 2 epochs (3,125 optimizer steps at batch size 32). Sequences over
`max_seq_len` are dropped rather than truncated, at a cost of 0.18% of
conversations.

**dolly-15k is a pipeline validation dataset, not a second experimental arm.**
Its supervised-token fraction (41%) differs sharply from the tulu-3 mixture's
(85%), which changes the effective gradient scale under the recipe's loss
reduction. Telemetry collected on dolly-15k confirms the pipeline works; it
does not support a scientific claim on its own.

## Other limitations

- **No output-side dimension contrast.** SmolLM2-1.7B uses standard
  multi-head attention (`num_key_value_heads == num_attention_heads`), so all
  attention projections are square (2048x2048). Any claim about how the
  output factor scales with output dimension cannot be tested on this model.
- **The measured geometry is math- and code-heavy.** The tulu-3 mixture is
  approximately 36% math and 15% code; the stratified subset preserves this
  composition intentionally, since it matches what the learning rate was
  selected against.
- **The probe batch is held out but small.** Tokens within a sequence are
  correlated, so the effective sample size is below the raw token count even
  at 96 probe sequences.

## Implementation details

- **The training run is isolated from the probe.** The probe runs on its own
  batch, in its own forward and backward pass, after the optimizer step.
  Model parameters are frozen during the probe and the backward graph is
  rooted at the embeddings, so no gradient buffers are allocated for the
  trainable parameters. A probe failure is caught and logged; training
  continues.
  [`test_probe_does_not_perturb_the_training_trajectory`](../tests/test_integration.py)
  asserts that attaching telemetry produces bit-identical weights to a run
  without it.
- **The probe batch and token sample are fixed for the whole run.** The token
  selection is reseeded from a constant before every probe, so any change
  between probes reflects the model, not resampling noise.
- **Resume is additive, not overwriting.** Output shards are keyed by step
  range rather than an in-memory counter, and the step-0 weight baseline is
  persisted to disk and reloaded on resume.
- **Reductions run in fp32 with TF32 disabled, and accumulate in fp64.** TF32
  introduces roughly 4e-2 relative error on an N=8192 contraction, and every
  reported quantity is a ratio of terms differing by orders of magnitude.

## Output layout

```
runs/<name>/
  run_config.json      resolved config, including recipe provenance
  environment.json     GPU, library versions, git commit
  train_log.jsonl       loss curve, keyed by both step and tokens_seen
  telemetry/
    scalars/steps_<lo>_<hi>.parquet    per-module scalar metrics per probe
    deep/step_<n>.npz                  spectra, raw activations, Adam state
    weight_baseline_step0.pt           reference weights for weight-delta tracking
  final/                deployable model and generation config
```

Telemetry is scheduled on tokens consumed, not optimizer steps, so runs at
different batch sizes remain comparable on a common axis. `train_log.jsonl`
records both step and token counts so the loss curve can be joined against
the telemetry data either way.
