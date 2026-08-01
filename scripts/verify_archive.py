#!/usr/bin/env python
"""Verify a telemetry archive is complete and every substrate is finite.

    python scripts/verify_archive.py /workspace/runs/smoke /workspace/runs/preflight
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def verify(run_dir: str) -> bool:
    from pathlib import Path

    run = Path(run_dir)
    tel = run / "telemetry"
    shards = sorted(tel.glob("scalars/*.parquet"))
    deep = sorted(tel.glob("deep/*.npz"))

    print(f"\n[{run.name}]")
    if not shards:
        print("  FAIL: no scalar shards")
        return False
    if not deep:
        print("  FAIL: no deep dumps")
        return False

    df = pd.concat(pd.read_parquet(p) for p in shards)
    full = df[df["is_full_n"]]
    bad = [
        c for c in ("c", "PR_Sigma", "a", "R", "rho_delta", "rayleigh_full", "R_Omega", "mu2")
        if c in df and not np.isfinite(df[c]).all()
    ]

    print(f"  rows      {len(df)} | modules {full['name'].nunique()} | "
          f"N={sorted(df['N'].unique())} | steps={sorted(full['step'].unique())}")
    print(f"  deep      {len(deep)} dump(s)")
    print(f"  PR_Sigma/null ratio (mean) = {full['PR_Sigma_ratio'].mean():.3f}")

    if bad:
        print(f"  FAIL: non-finite columns {bad}")
        return False
    print("  OK")
    return True


if __name__ == "__main__":
    runs = sys.argv[1:] or ["/workspace/runs/smoke", "/workspace/runs/preflight"]
    ok = all(verify(r) for r in runs)
    print("\nARCHIVE OK" if ok else "\nARCHIVE PROBLEMS FOUND")
    raise SystemExit(0 if ok else 1)
