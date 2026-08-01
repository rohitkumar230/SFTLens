# Findings: dolly-full dry run telemetry analysis

**Run analyzed:** `dolly-full`, 1x H100 SXM, 2026-08-01. SmolTulu SFT-1207
recipe (unmodified) on the full dolly-15k dataset. 892 optimizer steps, 13
telemetry probes (4 deep), 56 instrumented modules across 8 layers.
Not the research arm -- a dry run to validate the pipeline and get a first
read on the instrument, using off-recipe data (see `docs/dolly-full-dryrun-results.md`
for the schema and run-level facts).

**This document supersedes the informal read-out given right after the run.**
Two of the three findings reported then do not survive the checks below, in
instructive ways. Everything here is reproducible by running
`analysis/dolly_full_dryrun/run_analysis.py` against the archive; every
number has a table in `tables/` and most have a figure in `figures/`.

**Correction, same day:** Section 4.1's original claim --
"`\|log R_Omega\| > \|log R\|` in 100% of rows, output-side correction
decisively dominates" -- **does not hold**. `R` and `R_Omega` were built
against different reference points (`R` normalises against `rho_iso`, Sigma's
own eigenvalue-weighted mean; `R_Omega` normalises against `beta_iso`,
Omega's flat uniform mean), so comparing their raw magnitudes was never
apples-to-apples. `R_Omega`'s magnitude turns out to mostly restate
`PR_Omega`'s anisotropy (Spearman 0.84 against `log(D_out/PR_Omega)`),
which is already measured directly -- not new information. §4.1 below is
rewritten with the corrected, magnitude-comparable statistic `R_Omega_sym`
(verified to 1e-15 against an independent dense computation; now logged
natively by `sftlens.telemetry.reductions` for every future run). The
decisive claim is retracted; what replaces it is much weaker. All other
sections are unaffected by this correction.

**A structural fact to know before reading anything else:** `q_proj`,
`k_proj`, `v_proj` share the exact same input tensor (the pre-attention
residual stream), so their input-side statistics (`PR_Sigma`, `c`, `tr_Sigma`,
and everything else built only from `Σ`) are **byte-identical**, not merely
close -- confirmed to `1e-15`. The same holds for `gate_proj`/`up_proj`
(shared MLP-block input). So the 7 "module types" in this archive collapse
to **4 independent Σ-measurements**: `down_proj`, `{gate,up}_proj`,
`{q,k,v}_proj`, `o_proj`. Gradient-side (`Ω`) statistics do differ between
q/k/v, since each has a distinct output use. Any analysis treating the 7
module types as 7 independent samples on the input side is overcounting by
roughly 1.75x.

---

## 1. Instrument validation

| check | result | verdict |
|---|---|---|
| finite-N cap (`PR_Sigma/N`) | max 0.063 across all modules | **PASS** -- nowhere close to the N=8192 cap |
| N-sweep bias, `PR_Sigma` | median bias 0.996 (need >0.95) | **PASS** |
| N-sweep bias, `PR_Omega` | median bias 0.845 (need >0.95) | **FAIL** |
| centering gap (`g2_uncentered/g2`) | range [0.12x, 30.7x] | **FAIL** -- large and variable |
| probe self-consistency | monotone 3.74 -> 1.60, tracks eval curve | **PASS** |
| trace identities (4 checks) | all exact to 1e-6 | **PASS** |

Three of six checks pass cleanly; two fail in ways that constrain what can be
claimed, and are treated as findings in their own right rather than swept
aside:

- **`PR_Omega`'s finite-N correction is unreliable.** Sigma's bias correction
  is tight (0.996 median, tight spread); Omega's is not (0.845 median, and the
  per-(module,step) bias ranges from 0.01x to 1.98x -- unstable, not just
  offset). Any `PR_Omega` number quoted without the N-sweep behind it should
  be treated as directional, not exact -- and even the sweep-corrected value
  carries more uncertainty than the equivalent Sigma number.
- **The centering choice is not immaterial here.** `g2_uncentered/g2` swings
  from 0.12x to 30.7x depending on module and step, with `mlp.up_proj` and
  `self_attn.q_proj` the worst offenders (means >1.9x, some steps past 27x).
  SmolLM2 uses RMSNorm, which does not center activations, so a non-trivial
  activation mean feeding into `down_proj`/`up_proj`/`gate_proj` is the
  likely mechanism, worth checking directly against the raw `Xc` arrays in
  the deep dumps if this becomes load-bearing. Every quantity built from
  centered `Σ` (`c`, `a`, `R`, `rho_delta`, and everything downstream of
  them) is a *convention choice*, not a unique fact about the gradient. That
  convention is stated in every table (`_uncentered` variants exist
  alongside), so nothing here is silently ambiguous -- but a claim resting on
  one convention and not checked against the other should be treated as
  provisional.

## 2. Two claims reported earlier, reinterpreted

### 2.1 The down_proj/o_proj convergence to 4.0 was mis-framed

**What was reported:** "PR ratio converges from 3.12 to 4.00 -- confirms the
instrument measures dimension correctly."

**What the data actually shows**, checked three ways:

1. The trajectory itself is real -- all 13 points, not two cherry-picked
   endpoints (`tables/2_1_down_o_raw_ratio_trajectory.csv`), and it does land
   almost exactly on 4.0 by the final probed step.
2. But `log(PR_Sigma) ~ log(D_in)` pooled across all 7 module types gives
   **R² = 0.004**. Dimension alone explains essentially none of the
   cross-module variance in this run.
3. Why: `gate_proj`/`up_proj` share `D_in=2048` with `q/k/v_proj` -- the
   *same* dimension -- yet have **3.6x higher** `PR_Sigma/D_in`
   (0.072 vs 0.020, F=91.6, p=2.6e-85, `tables/2_1_PR_normalized_by_module.csv`).
   If PR were dimension-driven, same D_in should mean same normalized PR. It
   does not.

**Read this as:** the down_proj/o_proj ratio landing near 4.0 is a real,
specific fact about how those two modules' spectra evolved during this run --
worth noting, possibly worth watching on the tulu3-50k arm to see if it
replicates -- but it is not evidence the instrument recovers a general
dimension law, and should not be cited as instrument validation. The
`log(D_in)` regression with module/layer fixed effects
(`condition_number=3.9e15`) confirms this is structurally unidentifiable
anyway: with only two distinct `D_in` values in this architecture, one of
which is *exactly* the down_proj indicator, "dimension effect" and
"down_proj-specific effect" cannot be told apart from cross-sectional data
alone.

### 2.2 `R -> 1.0` and the weak shuffle-null result are the same finding, and it's a real positive result

**What was reported as two separate, mixed-strength findings**: "R declines
toward 1.0" (positive-sounding) and "only 17.9% of modules clear 1.2x over
the shuffled null" (negative-sounding, called "inconclusive").

**These are the same statistic.** `Spearman(R, a/a_shuffled) = 0.951`, and
`R / (a/a_shuffled)` concentrates at 1.00 (mean 1.005, IQR [0.947, 1.061],
`tables/2_2_R_vs_shuffled_equivalence.csv`, and see the scatter in
`figures/2_2_R_vs_shuffled_scatter.png` -- points sit tightly on `y=x` across
the full 0.5-5.5 range). `a_shuffled` destroys the token-gradient pairing but
preserves both marginals, so `a_shuffled ≈ c` by construction, making
`a/a_shuffled ≈ a/c = R` almost tautologically. Reporting both was
double-counting one signal as two.

**Corrected reading:** gradient-input alignment starts elevated (`R≈2.0` at
step 0, reflecting the pretrained model's pre-existing structure) and decays
toward the isotropic null (`R≈1.0`) as fine-tuning proceeds. This is a
genuine, single, moderately confident finding -- not a weak one -- confirmed
by two nominally independent statistics that turn out to measure the same
thing, which is itself a useful methodological note for future analysis.

### 2.3 R > 1 does not replicate the CNN's U-shaped depth profile

Checked directly: `figures/2_3_R_depth_profile.png`,
`tables/2_3_R_depth_profile_by_module.csv`. Edge-layer mean R = 0.884,
interior mean = 0.957 -- both close to 1, no structural gap. The CNN result
(interior ≈0.227, edges ≈1.0, a >4x gap) does not appear here. Two most
likely explanations, neither adjudicated by this run alone: (a) SFT from a
converged pretrained checkpoint starts in a different curvature regime than
training a CNN from scratch, or (b) architecture (attention+MLP vs conv)
genuinely produces different depth structure. The `q/k/v` outliers at layers
1 and 23 (R up to 3.48) are noise from a single fixed 8192-token probe batch,
not a depth pattern -- no other layer shows anything like it.

## 3. Core trajectory measurements

- **R's decline survives controlling for gradient norm** in all 7 module
  types (partial Spearman, all p<0.05, most p<0.0001,
  `tables/3_1_R_trend_controlled.csv`), and **survives controlling for
  loss** in 5 of 7 (exceptions: `k_proj`, `v_proj`, both p>0.45 after
  control). The decline is not simply "loss went down and so did R."
- **Cluster-bootstrap on the headline claim** (`R` declines with
  `tokens_seen`), resampling over the 56 module-clusters rather than
  treating 1,400 rows as independent: point estimate -0.357, 95% CI
  **[-0.427, -0.293]**, excludes zero. The decline is robust to the
  non-independence in the data.
- **dR/dt does not plateau before dloss/dt** in any of 7 module types (0/7,
  `tables/3_2_dRdt_vs_dlossdt.csv`) -- a clean negative result against R
  being a leading indicator of loss convergence, though with only 13 probe
  points this is a directional read, not a well-powered test.
- **`spread` (from `mu2`) is moderately, not fully, independent of R**:
  Spearman -0.565 (p<0.0001) -- about 32% shared variance, meaning `spread`
  does carry information beyond `R` but the two are not orthogonal axes
  either.

## 4. The Ω analysis

This is the part of the archive with no prior informal read-out. Its
headline claim was wrong as first stated; the corrected version is much
weaker, and is documented in detail because the mistake itself -- comparing
two ratios normalised against different reference points -- is exactly the
kind of error this whole analysis exercise exists to catch.

### 4.1 The "Ω dominates" claim does not survive an apples-to-apples statistic

**Originally reported:** `|log(R_Omega)|` exceeds `|log(R)|` in 100% of the
728 full-N rows, by 1-2 orders of magnitude, "decisively" showing the
output-side curvature correction dominates the input-side one.

**Why that was wrong:** `R` and `R_Omega` are normalised against different
things. `R = rho_delta/rho_iso`, and `rho_iso = ‖Σ‖²_F/tr Σ` is Σ's own
eigenvalue-*weighted* mean -- close to the top of its spectrum. `R_Omega`
instead divides by `beta_iso = tr Ω/D_out`, the *flat, uniform* mean over all
`D_out` directions. Since `G`'s row space lies inside Ω's range by
construction, `R_Omega`'s numerator samples Ω only where the gradient
actually has mass, while its denominator averages over directions the
gradient never touches. The result: `R_Omega`'s magnitude mostly just
restates Ω's own anisotropy -- confirmed directly, `Spearman(log|R_Omega|,
log(D_out/PR_Omega)) = 0.837` -- not new information beyond
`PR_Omega_ratio`, which already measures that anisotropy.

**The fix:** apply R's own construction to Ω, one level up the K-FAC
decomposition --

```
R_Omega_sym = [tr(Ω G Σ Gᵀ)/tr(G Σ Gᵀ)] / [‖Ω‖²_F/tr Ω]
            = (tr_K2M2 · tr_M) / (tr_K2M · tr_M2)
```

computed here from raw traces already in the archive (no rerun needed) and
now logged natively as `R_Omega_sym` for every future run
(`src/sftlens/telemetry/reductions.py`). Verified to `1e-15` against an
independent dense computation and against the isotropic-Ω edge case (must
equal exactly 1; confirmed) before use.

**Corrected result** (`tables/4_1b_R_vs_ROmegaSym_magnitude.csv`,
`figures/4_1b_correction_comparison.png`): `|log R_Omega_sym| > |log R|` in
only **55.4%** of rows -- essentially a coin flip, not a decisive majority.
The per-row ratio `|log R_Omega_sym|/|log R|` has median **1.17** (near
parity; the mean, 62, is wrecked by rows where `R≈1` exactly and is not
meaningful). `Spearman(log|R_Omega_sym|, log(D_out/PR_Omega))` drops to
**-0.22**, confirming the fix decouples the statistic from simply restating
Ω's anisotropy, as intended.

What's left after the correction is a small, module-dependent split: MLP
modules and `o_proj` lean toward the Ω-side being modestly larger (median
ratio 1.36-1.69x, Ω-side bigger in 65-73% of their rows); attention
`q/k/v_proj` lean the other way or are roughly even (median ratio
0.49-0.90x, Ω-side bigger in only 28-46% of rows). This split is
step-to-step stable (flip rate 0.18, i.e. the sign changes on only 18% of
adjacent-step transitions -- not noise-dominated) but small in effect size
(every module median within ~2x of parity, nothing resembling the original
order-of-magnitude claim) and drawn from only 13 probe steps.

**Verdict: the decisive form of the finding is retracted.** What survives is
a modest, directionally-consistent, but small and thinly-powered
module-family split, worth checking for replication on the tulu3-50k arm's
longer trajectory -- not something to cite as a finding from this run.

- **The PR_Omega -> shuffle-excess prediction failed, in the informative
  direction.** Hypothesis was a positive correlation (modules where the
  shuffled null beats the signal should have low `PR_Omega`). Measured:
  **Spearman = -0.353, p = 0.0077** -- significant, and the *opposite* sign
  (`tables/4_2_PROmega_vs_excess.csv`, `figures/4_2_PROmega_vs_excess.png`).
  This is not "no relationship"; it's a wrong-signed relationship, which is
  more informative than a null: whatever determines where alignment beats
  its null, it moves opposite to what a simple "spread-out Ω dilutes signal"
  story predicts.
- **PR_Omega vs depth is non-monotonic, not decreasing as hypothesized**,
  and the shape differs sharply by module family: MLP modules
  (`gate/up/down_proj`) rise from ~50 at layer 0 to a peak of 400-700 around
  layers 12-16, then fall back to 50-240 by layer 23; attention modules
  (`q/k/v/o_proj`) stay flat and low (20-40) at every depth
  (`figures/4_1_4_2_omega_summary.png`, left panel). The depth structure, to
  the extent there is any, belongs to the MLP path.
- **No GQA on SmolLM2-1.7B** (`D_out=2048` for every attention projection,
  `tables/0_module_shapes.csv`) -- the output-dimension-contrast test from
  the spec is unavailable on this model, as anticipated.
- `beta_iso`-implied per-layer step sizes (§4.4) are purely descriptive here
  -- they span 0.5 to 90 across module types in arbitrary units, with no LR
  sweep run to calibrate them against anything. Not a finding, just logged
  for later use if a calibration run happens.

## 5. Deep-dump analyses

- **Spectral decay exponent (`α`, top-64 eigenvalues) varies by module
  family, not universal**: median 0.72 (`down_proj`, flattest) to 1.05
  (`q/k/v_proj`, steepest), pooled std 0.25, between-module median spread
  0.32 (`tables/5_1_spectral_decay_alpha.csv`,
  `figures/5_1_spectral_decay_alpha.png`). Consistent with §2.1's finding
  that module identity carries real spectral information beyond dimension --
  attention-input activations concentrate their variance in fewer directions
  than MLP-input activations do.
- **Adam-preconditioned R vs raw R** (§5.2): Spearman 0.329, p=0.146 across
  21 tracked modules -- not significant, and explicitly qualitative only
  (the 256-token deep-dump subsample is N-capped against D up to 8192, so
  this recompute fails the §1.1 test by construction; the parquet's own
  `N=8192` columns remain the only quantitative source).
- **ΔW effective rank grows monotonically with training** at 2 of 3 tracked
  depths (layers 12 and 23; layer 0 is flat/slightly declining), and layer 12
  shows both the highest absolute rank and the fastest growth of the three
  (`tables/5_3_dW_vs_geometry.csv`). Only 3 usable steps per module (step 0
  is a structural 0/0 exclusion, not a bug) -- a descriptive pattern, not a
  correlation with enough points to test.

## 6. What would count as a finding -- resolved

| spec's criterion | result |
|---|---|
| module/layer terms significant after `log(D_in)` in §2.1 | **Confirmed, but reframed**: dimension has ~zero pooled explanatory power (R²=0.004); module identity (F=91.6, p=2.6e-85) is what actually predicts normalized PR. The confound cuts the other way from how the spec posed it -- there was never a clean dimension law to break in the first place. |
| `\|log R_Omega\| > \|log R\|` in §4.1 | **Retracted on review**: the comparison was not apples-to-apples (`R` and `R_Omega` normalise against different reference points). Corrected statistic (`R_Omega_sym`) gives 55.4% -- essentially a coin flip -- with only a modest, module-dependent residual split (MLP+`o_proj` lean Ω-heavier ~1.4-1.7x, attention Q/K/V lean the other way or even). See §4.1 for the full correction. |
| `PR_Omega` predicts shuffle-excess in §4.2 | **Rejected in the informative direction**: significant, wrong sign. Unaffected by the §4.1 correction (doesn't use `R_Omega`). |
| U-shaped `R` depth profile matching the CNN | **Not observed**: R is flat with depth here (edges 0.88, interior 0.96), no >4x gap. |
| `dR/dt` plateaus before loss | **Rejected**: 0/7 modules, though underpowered at 13 points. |
| `spread` separates modules with equal `R` | **Partially**: moderate anti-correlation (-0.565), ~32% independent variance -- a real but not orthogonal second axis. |

**Where this leaves the project**, stated the way the spec's own closing
paragraph asked for: this run does not deliver a clean, decisive new result.
The one candidate for that role -- Ω-dominance -- did not survive review; the
corrected version is a small, thinly-powered, directionally-consistent
module-family split, not something to lead with. What the run robustly
establishes instead is the corrected reading of R (real decline toward the
isotropic null, robust to clustering and partial controls, but not evidence
of a dimension law and not U-shaped with depth) and a working, validated
instrument (§1) with two known, quantified limitations (Ω's finite-N
correction, the centering choice). That is closer to the spec's own
"transformer reproduces the CNN's negative results, descriptive paper is the
output" closing scenario than the original read-out suggested -- with one
open thread (the modest Ω/R split by module family) worth a real look on a
longer trajectory before deciding whether it's signal or noise.

## 7. What to check first on the tulu3-50k arm

1. Does the corrected, modest MLP-vs-attention `R_Omega_sym`/`R` split
   replicate and sharpen at ~3,125 steps, or does it stay within noise? This
   is now the open question from §4, not a confirmed finding to replicate --
   treat it as a hypothesis, not an expectation.
2. Re-run §1.3 (centering gap) before trusting any `a`/`R`/`rho_delta` number
   -- it was not immaterial here and there is no reason to expect it will be
   on the next run.
3. With ~3,125 steps instead of 892 and a much longer trajectory, §3.2
   (dR/dt vs dloss/dt) and §5.3 (dW vs geometry) both graduate from
   "exploratory, too sparse" to "adequately powered" -- worth re-running
   properly rather than repeating the caveats.
4. Remember the 7-modules-are-really-4 structural fact when computing any
   cross-module statistic on the input side.
5. `R_Omega_sym` is now logged natively (no post-hoc recompute needed) --
   use the parquet column directly rather than re-deriving it from raw
   traces the way this analysis had to.

---

*Reproduce every number above:* `.venv/bin/python analysis/dolly_full_dryrun/run_analysis.py`
*All tables:* `analysis/dolly_full_dryrun/tables/*.csv`
*All figures:* `analysis/dolly_full_dryrun/figures/*.png`
