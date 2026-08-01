#!/usr/bin/env bash
# Exercise every code path end to end on dolly-15k in a couple of minutes.
# Run this before committing GPU hours to a real arm.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== unit + integration tests =="
python -m pytest tests/ -q

echo
echo "== dry run: data, model, sizing, no training =="
python -m sftlens.train.run \
  -c model/smollm2-1.7b.yaml \
  -c data/dolly-smoke.yaml \
  -c recipe/smoltulu-sft-1207.yaml \
  -c telemetry/smoke.yaml \
  --dry-run

echo
echo "== 20-step training run with telemetry =="
python -m sftlens.train.run \
  -c model/smollm2-1.7b.yaml \
  -c data/dolly-smoke.yaml \
  -c recipe/smoltulu-sft-1207.yaml \
  -c telemetry/smoke.yaml \
  -s run_name=smoke

echo
echo "== archive =="
find runs/smoke -type f | head -40
python - <<'PY'
import glob
import pandas as pd
shards = sorted(glob.glob("runs/smoke/telemetry/scalars/*.parquet"))
assert shards, "no telemetry written"
df = pd.concat(pd.read_parquet(p) for p in shards)
full = df[df["is_full_n"]]
print(f"\n{len(df)} rows, {full['name'].nunique()} modules, "
      f"steps {sorted(full['step'].unique())}")
print(full.groupby("module")[["PR_Sigma", "PR_Sigma_null", "R", "R_Omega"]].mean())
PY
