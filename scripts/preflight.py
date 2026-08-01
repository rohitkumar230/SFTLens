#!/usr/bin/env python
"""Environment check. Downloads nothing, costs seconds, runs first.

Every failure this catches would otherwise be discovered several minutes into a
paid instance, after a 3.4 GB model download.

    python scripts/preflight.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FAIL: list[str] = []
WARN: list[str] = []


def check(label: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f'  ({detail})' if detail else ''}")
    if not ok:
        (FAIL if fatal else WARN).append(label)
    return ok


def main() -> int:
    print("\n=== packages ===")
    try:
        import torch
    except ImportError:
        print("  FAIL  torch not importable")
        return 1
    check("torch", True, torch.__version__)

    import transformers

    check("transformers 4.51.x", transformers.__version__.startswith("4.51"),
          transformers.__version__, fatal=False)
    for mod in ("datasets", "accelerate", "pandas", "pyarrow", "yaml", "numpy"):
        try:
            __import__(mod)
            check(mod, True)
        except ImportError:
            check(mod, False, "missing")

    print("\n=== repo ===")
    try:
        from sftlens.config import load_config

        for name, overlays in {
            "smoke": ["model/smollm2-1.7b.yaml", "data/dolly-smoke.yaml",
                      "recipe/smoltulu-sft-1207.yaml", "telemetry/smoke.yaml",
                      "run/smoke.yaml"],
            "preflight": ["model/smollm2-1.7b.yaml", "data/dolly-smoke.yaml",
                          "recipe/smoltulu-sft-1207.yaml", "telemetry/default.yaml",
                          "run/preflight.yaml"],
            "real": ["model/smollm2-1.7b.yaml", "data/tulu3-50k.yaml",
                     "recipe/smoltulu-sft-1207.yaml", "telemetry/default.yaml"],
        }.items():
            cfg = load_config(overlays)
            check(f"config '{name}' loads", True,
                  f"lr={cfg.recipe.lr:g} bs={cfg.recipe.effective_batch} "
                  f"save={cfg.save_strategy}")
    except Exception as exc:
        check("configs load", False, f"{type(exc).__name__}: {exc}")

    print("\n=== disk ===")
    free = shutil.disk_usage(REPO).free / 1e9
    check("free disk >= 12 GB (smoke)", free >= 12, f"{free:.0f} GB free")
    check("free disk >= 60 GB (real run)", free >= 60, f"{free:.0f} GB free", fatal=False)

    print("\n=== gpu ===")
    if not check("CUDA available", torch.cuda.is_available()):
        print("  (GPU checks skipped; everything above is still authoritative)")
        return 1 if FAIL else 0

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    check("GPU", True, f"{props.name}, {vram:.0f} GB, sm_{props.major}{props.minor}")
    check("VRAM >= 48 GB", vram >= 47,
          f"{vram:.0f} GB; full FT of 1.7B needs 27.4 GB of state alone")
    check("bf16 supported", torch.cuda.is_bf16_supported())

    # fp64 is used only for the scalar reductions, which are memory-bound, so a
    # crippled fp64 rate is fine -- but no fp64 at all is not.
    try:
        (torch.ones(8, device="cuda", dtype=torch.float64) * 2).sum().item()
        check("float64 reductions", True)
    except Exception as exc:
        check("float64 reductions", False, str(exc)[:60])

    print("\n=== a real allocation ===")
    # Estimates are estimates. Actually reserve the training state.
    need = 27.4
    try:
        buf = torch.empty(int(need * 1e9 // 2), dtype=torch.bfloat16, device="cuda")
        del buf
        torch.cuda.empty_cache()
        check(f"can allocate {need:.0f} GB (training state)", True)
    except torch.cuda.OutOfMemoryError:
        check(f"can allocate {need:.0f} GB (training state)", False, "OOM")

    print("\n" + "=" * 62)
    if FAIL:
        print(f"BLOCKED: {len(FAIL)} check(s) failed -> {', '.join(FAIL)}")
        return 1
    if WARN:
        print(f"OK with {len(WARN)} warning(s): {', '.join(WARN)}")
    else:
        print("OK. Environment is ready.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
