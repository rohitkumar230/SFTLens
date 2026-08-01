#!/usr/bin/env bash
# One paste, start to finish, on a fresh RunPod pod. Designed to be short:
# every minute here is billed.
#
#   bash scripts/runpod_smoke.sh
#
# Get the repo onto the pod FIRST, from your laptop:
#   bash scripts/push_repo.sh <POD_IP> <PORT>
# (there is no git remote on this repo, so `git clone` cannot work)
#
# Run this inside tmux. A dropped SSH session kills it otherwise.
set -euo pipefail

cd "$(dirname "$0")/.."

# Everything heavy goes on the volume disk at /workspace, not the container
# disk: the container disk holds the ~15 GB image and has far less headroom.
export HF_HOME=${HF_HOME:-/workspace/hf}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
OUT=${OUT:-/workspace/runs}
mkdir -p "$OUT" "$HF_HOME"

banner() { echo; echo "############ $* ############"; echo; }

banner "0. install (torch comes from the pod image, do not reinstall it)"
pip install -q --no-deps -e .
pip install -q "transformers==4.51.3" "datasets==3.5.0" "accelerate==1.6.0" \
               "pandas>=2.2" "pyarrow>=15" "pyyaml>=6" pytest

banner "1. environment preflight (no downloads)"
python scripts/preflight.py

banner "2. unit + integration tests (CPU, ~15s, no downloads)"
python -m pytest tests/ -q

banner "3. stage 1 smoke: does the pipeline run? (~3 min + model download)"
python -m sftlens.train.run \
  -c model/smollm2-1.7b.yaml -c data/dolly-smoke.yaml \
  -c recipe/smoltulu-sft-1207.yaml -c telemetry/smoke.yaml -c run/smoke.yaml \
  --no-save-final -s output_dir="$OUT/smoke" 2>&1 | tee "$OUT/smoke.log"

banner "4. stage 2 preflight: does PRODUCTION telemetry sizing fit? (~4 min)"
python -m sftlens.train.run \
  -c model/smollm2-1.7b.yaml -c data/dolly-smoke.yaml \
  -c recipe/smoltulu-sft-1207.yaml -c telemetry/default.yaml -c run/preflight.yaml \
  --no-save-final -s output_dir="$OUT/preflight" 2>&1 | tee "$OUT/preflight.log"

banner "5. verify the archive"
python - "$OUT" <<'PY'
import glob, sys, numpy as np, pandas as pd
out = sys.argv[1]
for run in ("smoke", "preflight"):
    shards = sorted(glob.glob(f"{out}/{run}/telemetry/scalars/*.parquet"))
    deep = sorted(glob.glob(f"{out}/{run}/telemetry/deep/*.npz"))
    assert shards, f"{run}: no scalar shards"
    assert deep, f"{run}: no deep dumps"
    df = pd.concat(pd.read_parquet(p) for p in shards)
    full = df[df.is_full_n]
    bad = [c for c in ("c", "PR_Sigma", "a", "R", "rho_delta", "rayleigh_full",
                       "R_Omega", "mu2") if not np.isfinite(df[c]).all()]
    assert not bad, f"{run}: non-finite {bad}"
    print(f"  {run:<10} {len(df):5d} rows | {full.name.nunique():3d} modules | "
          f"N={sorted(df.N.unique())} | steps={sorted(full.step.unique())} | "
          f"{len(deep)} deep dumps")
    print(f"  {'':10} PR_Sigma/null ratio = "
          f"{full.PR_Sigma_ratio.mean():.3f} (<<1 means real anisotropy, "
          f"~1 means indistinguishable from isotropic)")
print("\n  ARCHIVE OK")
PY

banner "6. sizes"
du -sh "$OUT"/* "$HF_HOME" 2>/dev/null || true

cat <<EOF

================================================================
SMOKE COMPLETE. The pod is still running and still billing.

Next, ON YOUR LAPTOP (not here):

  bash scripts/pull_artifacts.sh <POD_IP> <PORT>

That copies the telemetry home and verifies it. Wait for it to
print "SAFE TO TERMINATE THE POD".

THEN terminate the pod in the RunPod console. Terminate, not Stop
-- a stopped pod still bills for its disk.
================================================================
EOF
