"""Print the runtime environment (GPU, compute capability, package versions, disk).

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_vlm.gpu import gpu_info, nvidia_smi_memory  # noqa: E402


def main() -> int:
    info = gpu_info()
    info["nvidia_smi"] = nvidia_smi_memory()
    disks = {}
    for path in (ROOT, Path("/kaggle/working"), Path("/tmp")):
        if path.exists():
            usage = shutil.disk_usage(path)
            disks[str(path)] = {"free_gib": round(usage.free / 1024**3, 1)}
    info["disk"] = disks
    print(json.dumps(info, indent=2, ensure_ascii=False))

    gpus = info.get("gpus") or []
    if not gpus:
        print("\n[WARN] No CUDA GPU visible. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")
        return 1
    if not all(gpu.get("bf16_native", False) for gpu in gpus if "bf16_native" in gpu):
        print("\n[NOTE] GPU has no native bf16 (e.g. T4, sm_75): use dtype=half / fp16 everywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
