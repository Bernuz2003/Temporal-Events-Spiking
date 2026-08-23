from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_is_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(selected_device: torch.device | str | None = None) -> dict[str, Any]:
    """Collect a bounded, non-secret runtime snapshot for experiment provenance.

    Environment variables are intentionally allow-listed. Dumping the complete process environment
    could leak credentials into versioned or shared artifacts.
    """

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": int(properties.total_memory),
                }
            )

    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "description": platform.platform(),
        },
        "packages": {
            name: _distribution_version(name)
            for name in (
                "temporal-event-spiking-research",
                "torch",
                "numpy",
                "PyYAML",
            )
        },
        "torch_runtime": {
            "version": str(torch.__version__),
            "selected_device": None if selected_device is None else str(selected_device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_compiled_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "cuda_devices": cuda_devices,
        },
        "threading": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
        },
        "privacy": {
            "environment_policy": "allow-listed non-secret fields only",
            "full_process_environment_recorded": False,
        },
    }
