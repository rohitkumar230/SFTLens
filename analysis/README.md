# analysis/

Analysis of telemetry archives produced by this repo's training runs. One
subdirectory per run.

## dolly_full_dryrun/

Analysis of the `dolly-full` dry run (SmolTulu SFT-1207 recipe, full
dolly-15k, H100 SXM, 2026-08-01).

- **`FINDINGS.md`** -- start here. Synthesized results, including three
  corrections to the informal read-out given right after the run: a
  dimension-confound reinterpretation, a redundant-statistic merge, and a
  retraction of the headline "output-side correction dominates" claim (`R`
  and `R_Omega` were normalised against different reference points; the
  corrected, apples-to-apples statistic `R_Omega_sym` shows only a small,
  module-dependent residual split, not a decisive result).
- **`run_analysis.py`** -- the script that produced every number in
  `FINDINGS.md`, executed section-by-section against an analysis spec
  written for this archive. Reproduce with:
  ```bash
  .venv/bin/python analysis/dolly_full_dryrun/run_analysis.py
  ```
- **`tables/*.csv`** -- one file per numbered check in the analysis, e.g.
  `2_1_PR_normalized_by_module.csv` is section 2.1's output.
- **`figures/*.png`** -- supporting charts, same numbering.

Depends on the archive at `runpod-dolly-full/dolly-full/` (not committed --
1.4 GB; see `docs/dolly-full-dryrun-results.md` for the schema if you need to
regenerate or re-pull it) and `statsmodels`/`scipy`/`matplotlib` in the
project venv (`pip install -e ".[dev]"` does not include these; install
separately for analysis work).

## Adding a new run's analysis

Same structure: `analysis/<run_name>/{run_analysis.py,FINDINGS.md,tables/,figures/}`.
`docs/<run_name>-results.md` (schema + run-level facts, if the run is
significant enough to warrant one) is the sibling doc this points back to.
