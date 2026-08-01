#!/usr/bin/env bash
# Run this on your LAPTOP. Brings the results home and VERIFIES them before you
# terminate the pod.
#
#   bash scripts/pull_artifacts.sh <POD_IP> <PORT> [DEST_DIR]
#
# Pulls telemetry, logs and run manifests. Deliberately does NOT pull
# checkpoints (20.5 GB each) or the base model (a one-minute re-download).
set -euo pipefail

IP=${1:?usage: pull_artifacts.sh <POD_IP> <PORT> [DEST_DIR] [SSH_KEY]}
PORT=${2:?usage: pull_artifacts.sh <POD_IP> <PORT> [DEST_DIR] [SSH_KEY]}
DEST=${3:-./runpod-results}
KEY=${4:-$HOME/.ssh/id_ed25519}
SRC=${SRC:-/workspace/runs}

# Prefer the project's own venv: the finite-value check needs pandas+pyarrow,
# and the ambient `python3` on a laptop that never ran `pip install -e .`
# usually doesn't have them, silently degrading verification to a
# presence-only check right when it matters most (right before terminating
# the pod).
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

mkdir -p "$DEST"
echo "==> pulling root@$IP:$SRC -> $DEST"
rsync -avz --progress \
  -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" \
  --exclude 'checkpoint-*/' \
  --exclude 'final/' \
  "root@$IP:$SRC/" "$DEST/"

echo
echo "==> verifying locally with $PY (do this BEFORE terminating the pod)"
"$PY" - "$DEST" <<'PY'
import glob, json, sys
from pathlib import Path

dest = Path(sys.argv[1])
runs = sorted({Path(p).parents[2] for p in glob.glob(f"{dest}/**/telemetry/scalars/*.parquet", recursive=True)})
if not runs:
    print("  FAIL: no telemetry found. Do NOT terminate the pod yet.")
    raise SystemExit(1)

ok = True
degraded = False
for run in runs:
    tel = run / "telemetry"
    shards = sorted(tel.glob("scalars/*.parquet"))
    deep = sorted(tel.glob("deep/*.npz"))
    print(f"\n  [{run.name}]")
    print(f"    scalars   {len(shards)} shard(s)")
    print(f"    deep      {len(deep)} dump(s)")
    for required in ("config.json", "probe_plan.json"):
        exists = (tel / required).exists()
        print(f"    {required:<16} {'present' if exists else 'MISSING'}")
        ok &= exists
    if (run / "run_config.json").exists():
        cfg = json.loads((run / "run_config.json").read_text())
        print(f"    recipe    lr={cfg['recipe']['lr']} bs={cfg['recipe']['effective_batch']}")

    try:
        import numpy as np, pandas as pd
        df = pd.concat(pd.read_parquet(p) for p in shards)
        bad = [c for c in ("c", "PR_Sigma", "a", "R", "rho_delta", "rayleigh_full",
                           "R_Omega", "mu2") if c in df and not np.isfinite(df[c]).all()]
        full = df[df.is_full_n]
        print(f"    rows      {len(df)} | modules {full.name.nunique()} | "
              f"steps {sorted(full.step.unique())}")
        if bad:
            print(f"    FAIL: non-finite columns {bad}")
            ok = False
        else:
            print(f"    finite    all substrates OK")
    except ImportError:
        print("    WARNING: pandas/pyarrow unavailable to this interpreter -- "
              "finite-value check SKIPPED, this run is presence-checked only")
        degraded = True

print()
if not ok:
    print("  PROBLEMS FOUND -- keep the pod alive")
elif degraded:
    print("  FILES PRESENT, BUT NOT FULLY VERIFIED (pandas/pyarrow missing).")
    print("  Do not terminate on this alone -- rerun with a Python that has "
          "pandas+pyarrow, e.g.:")
    print(f"    .venv/bin/python scripts/verify_archive.py {' '.join(str(r) for r in runs)}")
else:
    print("  SAFE TO TERMINATE THE POD")
raise SystemExit(0 if (ok and not degraded) else 1)
PY

echo
du -sh "$DEST"
