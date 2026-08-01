# dolly-full dry run: results and telemetry output reference

2026-08-01, 1x H100 SXM. First real (non-throwaway) run of the pipeline: the
actual SmolTulu SFT-1207 recipe, unmodified, applied to the full dolly-15k
dataset. Not the research arm -- dolly's supervised-token fraction (41%) and
token budget differ sharply from the tulu-3 mixture the recipe was selected
against (see README "Known limits") -- but a genuine trajectory, not a
handful of steps, generated with production-scale telemetry (`n_tokens=8192`,
96 stratified probes, full N-sweep). This doc is both: (a) a record of
whether the run was meaningful, and (b) a schema reference for writing
analysis scripts against this or future archives, since the scalar/deep
formats are identical across every run this repo produces.

Local copy: `runpod-dolly-full/dolly-full/` (1.4 GB: 640 MB deep dumps,
484 KB scalars, the rest manifests/logs). Pod terminated; nothing else exists.

**Full analysis, with corrections to the informal read-out below**: see
`analysis/dolly_full_dryrun/FINDINGS.md`. Two of the three findings stated in
this doc's original version did not survive rigorous checking (a dimension
confound in the down_proj/o_proj comparison, and a redundant statistic
double-counted as two findings) -- read `FINDINGS.md` for the corrected
versions before citing anything from this page's results section.

**Structural fact worth knowing before writing any analysis script**:
`q_proj`, `k_proj`, `v_proj` share the exact same input tensor, so every
input-side (`Σ`) statistic -- `PR_Sigma`, `c`, `tr_Sigma`, `rho_iso` -- is
byte-identical across the three, not merely similar. The same holds for
`gate_proj`/`up_proj`. The 7 module types in the schema below therefore
collapse to 4 independent Σ-measurements (`down_proj`, `{gate,up}_proj`,
`{q,k,v}_proj`, `o_proj`); only gradient-side (`Ω`) statistics distinguish
q/k/v from each other.

## Recipe, in short

SmolTulu SFT-1207 (Alrashed 2024, arXiv:2412.08347, Table 2), transcribed
without modification: `lr=3.1e-6`, `effective_batch=32` (4 per-device x 8
grad-accum), 2 epochs, linear schedule, 0.03 warmup, weight decay 0,
sum-reduced token loss (inherited from Tulu 3). Model: `SmolLM2-1.7B` base,
full-parameter fine-tune, fp32 master weights + bf16 compute. Data here:
dolly-15k full (14,280 train / 500 eval after a 1.5% length-filter drop at
`max_seq_len=1024`), ChatML-formatted, loss on the assistant turn only.
Checkpointing off for this run (disk-constrained pod, ~19 min run -- a
restart is cheaper than 41 GB of insurance); the research arm will run with
checkpointing on.

## Did the model learn anything?

Yes -- eval loss fell monotonically for the entire run, no instability:

| step | epoch | eval_loss |
|---|---|---|
| 100 | 0.22 | 2.026 |
| 400 | 0.90 | 1.650 |
| 800 | 1.79 | 1.556 |
| 892 (final) | 2.00 | 1.545 (perplexity 4.69) |

892/892 steps completed, `train_log.jsonl` has an unbroken per-logging-step
record (loss, grad_norm, lr, tokens_seen) plus every eval. This is a
well-behaved SFT run by the ordinary metric.

## Did telemetry log everything, and is it meaningful (not noise)?

Both, confirmed three independent ways (pod, and twice locally with
different scripts) before the pod was terminated: 1,400 scalar rows, all 56
instrumented modules present at every one of 13 probe steps (0 through 813),
full N-sweep (1024/2048/4096/8192) captured, 4 deep dumps, every substrate
finite.

Three findings from the actual numbers, not just "it ran":

- **The core dimension-contrast claim holds, and gets *more* correct as
  training proceeds.** `down_proj` (D_in=8192) vs `o_proj` (D_in=2048) is the
  controlled pair the whole `n_tokens=8192` design exists to resolve (see
  README's finite-N section). Measured `PR_Sigma` ratio: **3.12x at step 0 ->
  4.00x at step 813** -- converging on the true geometric ratio of exactly
  4.0. This is the clearest piece of evidence in the archive that the
  measurement methodology is doing what it was built to do.
- **`PR_Sigma_ratio` (measured PR / isotropic-null PR at N=8192) stays low
  and rises slowly: 0.033 -> 0.050 across training.** Well below 1 throughout
  -- the input covariance is strongly anisotropic (real structure, not
  measurement noise pretending to be structure), and that anisotropy
  increases slightly as training proceeds.
- **Gradient-Sigma alignment (`R`) is noisy and trends down: ~2.0 at step 0
  to ~1.0-1.1 by the end**, with non-monotonic swings in between (2.30 at
  step 202, 1.06 at step 751). The shuffled-null comparison at the final
  step is weak: mean `a/a_shuffled` excess is only 1.07x, and only 17.9% of
  modules clear a 1.2x-over-null bar. **Read this as inconclusive at this
  run's scale, not as evidence of no alignment** -- dolly's short trajectory
  (892 steps) and off-recipe data give this substrate less to work with than
  the 3,125-step tulu3-50k arm will.

None of this is a research claim on its own -- it is evidence the pipeline
produces a coherent, analyzable signal rather than garbage, which is exactly
what a dry run is for.

## Schema: scalar parquet (`telemetry/scalars/*.parquet`)

One row per (probed module, step, N). 1,400 rows x 47 columns in this run.
Load with `pd.concat(pd.read_parquet(p) for p in glob.glob(".../scalars/*.parquet"))`.

**Identity / indexing columns**

| column | dtype | meaning |
|---|---|---|
| `name` | str | full module path, e.g. `model.layers.12.mlp.down_proj` |
| `layer` | int | transformer layer index (0-23 for SmolLM2-1.7B) |
| `module` | str | suffix: one of `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` |
| `step` | int | optimizer step at probe time |
| `tokens_seen` | int | training tokens consumed at probe time (the real x-axis) |
| `N` | int | tokens used for this row's Gram matrices (1024/2048/4096/8192) |
| `is_full_n` | bool | True for the row at the configured `n_tokens` (8192 here); sweep rows are smaller-N re-probes of the same step for finite-N bias diagnosis |
| `D_in`, `D_out` | int | this module's weight shape |

**Input factor Sigma** (`tr_Sigma`, `fro_Sigma_sq`, `c`, `PR_Sigma`,
`PR_Sigma_null`, `PR_Sigma_ratio`, `rho_iso`) -- spectrum of the input
covariance. `PR_Sigma_null` is the isotropic-covariance expectation at this
N and D (`ND/(N+D+1)`); `PR_Sigma_ratio` = measured/null, the number that
tells you whether a PR reading means anything or is just the finite-N floor.

**Gradient / alignment** (`g2`, `rho_delta`, `a`, `R`, `mu2`) -- how the
gradient sits relative to Sigma. `g2` = ||G||^2_F. `R` is the headline
alignment ratio (a/c).

**Output factor Omega** (`tr_Omega`, `fro_Omega_sq`, `PR_Omega`,
`PR_Omega_null`, `PR_Omega_ratio`, `beta_iso`, `rayleigh_full`, `R_Omega`) --
spectrum of Delta^T Delta and the error from assuming it's isotropic.
`R_Omega` isolates that error specifically.

**Raw traces** (`tr_K`, `tr_K2`, `tr_M`, `tr_M2`, `tr_KM`, `tr_K2M`,
`tr_K3M`, `tr_K2M2`) -- the eight quantities every derived metric above is
computed from. Kept so any statistic invented later can be recomputed
without re-running the probe.

**Shuffled null** (`a_shuffled`, `g2_shuffled`, `rho_delta_shuffled`,
`R_shuffled`) -- same quantities with the token-Delta pairing destroyed by
permutation. `a / a_shuffled` is the standard way to ask "is this alignment
real or just what these two marginals would give by chance."

**Uncentered variants** (`g2_uncentered`, `tr_Sigma_uncentered`,
`c_uncentered`) -- Sigma is a covariance (centered), but the true weight
gradient uses uncentered inputs; both are logged since they can differ on
models with large activation offsets.

**Probe bookkeeping** (`probe_loss_total`, `probe_tokens`,
`probe_loss_per_token`) -- loss on the fixed held-out probe batch itself, for
sanity-checking the probe against the training-set eval curve.

Value ranges observed in this run (min, max across all 1,400 rows) are in
`docs/dolly-full-dryrun-ranges.json` alongside this file, generated directly
from the archive.

## Schema: deep dumps (`telemetry/deep/step_NNNNNNN.npz`)

One `.npz` per deep probe (4 in this run: steps 0, 262, 548, 813). 434 keys
each, named `"{module_path}|{field}"`. Load with `np.load(path)`.

| field suffix | shape | dtype | present for | meaning |
|---|---|---|---|---|
| `\|eig_K` | (64,) | float32 | all 56 modules | top-64 eigenvalues of the input Gram, descending |
| `\|eig_M` | (64,) | float32 | all 56 modules | top-64 eigenvalues of the gradient Gram, descending |
| `\|Xc` | (256, D_in) | float16 | all 56 modules | raw centered input activations, subsampled -- lets you recompute anything, including Adam-preconditioned quantities, without re-running training |
| `\|Delta` | (256, D_out) | float16 | all 56 modules | raw output-side gradients, same subsample as `Xc` |
| `\|adam_m_norm` | scalar | float32 | all 56 modules | ‖first Adam moment‖ |
| `\|adam_v_norm` | scalar | float32 | all 56 modules | ‖second Adam moment‖ |
| `\|adam_precond_norm` | scalar | float32 | all 56 modules | ‖m / (sqrt(v) + eps)‖ -- the actual preconditioned update direction |
| `\|dW_svals` | (64,) | float32 | 21 of 56 (depth-spread subset: first/mid/last probed layer) | top-64 singular values of the cumulative weight update since step 0 |
| `\|dW_relnorm` | scalar | float32 | same 21 | ‖ΔW‖ / ‖W‖ at this step |

`dW_*` is a subset because it needs an fp32 step-0 baseline held in memory
per tracked module (`telemetry/weight_baseline_step0.pt`, 21 modules here);
widening `telemetry.track_dw_layers` in config trades archive size for
coverage.

## Other files

- `train_log.jsonl` -- one record per Trainer logging step (every 5 steps)
  and every eval, keyed by both `step` and `tokens_seen`. Join key against
  the telemetry parquet.
- `run_config.json` -- the full resolved config for this run, including
  `recipe.provenance` (the paper citation the hyperparameters came from).
- `environment.json` -- GPU, driver, library versions, git commit.
- `telemetry/probe_plan.json` -- exactly which token positions of which
  sequences were measured (fixed for the whole run).
- `telemetry/config.json` -- the telemetry-specific config (cadence,
  `n_tokens`, module coverage) as it was actually run.

## For the next analysis script

- Join `scalars` to `train_log.jsonl` on `tokens_seen` to overlay any
  substrate against the loss curve.
- Filter `is_full_n == True` for the primary trajectory; the sweep rows
  (`N < 8192`) exist specifically to fit/extrapolate the finite-N bias in
  `PR_Sigma` and `PR_Omega` -- don't average them together with the full-N
  rows without accounting for N.
- `layer` lets you plot any substrate's depth profile at a fixed step;
  `step`/`tokens_seen` lets you plot its trajectory at a fixed layer/module.
- The raw `Xc`/`Delta` arrays in the deep dumps are the escape hatch for
  anything not already a column -- e.g. computing a quantity that needs the
  Adam-preconditioned gradient rather than the raw one.
