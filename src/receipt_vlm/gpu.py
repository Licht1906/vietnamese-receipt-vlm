"""GPU / environment introspection used to make every result file self-describing."""

from __future__ import annotations

import platform
import shutil
import subprocess
from importlib import metadata
from typing import Any

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "vllm",
    "xgrammar",
    "unsloth",
    "unsloth_zoo",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def nvidia_smi_memory() -> list[dict[str, Any]]:
    """Memory usage per GPU as reported by nvidia-smi (works across processes)."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": int(parts[2]),
                    "memory_total_mib": int(parts[3]),
                }
            )
    return rows


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "packages": package_versions()}
    try:
        import torch
    except ImportError:
        info["torch_available"] = False
        info["gpus"] = nvidia_smi_memory()
        return info

    info["torch_available"] = True
    info["cuda_available"] = torch.cuda.is_available()
    info["torch_cuda"] = torch.version.cuda
    gpus = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_total_gib": round(props.total_memory / 1024**3, 2),
                    "bf16_native": props.major >= 8,
                }
            )
    info["gpus"] = gpus
    return info
