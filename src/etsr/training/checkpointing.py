from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    epoch: int,
    score: float,
    config: dict[str, Any],
    num_classes: int,
) -> None:
    _atomic_save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "score": score,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
            "num_classes": num_classes,
        },
        path,
    )


def save_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_score: float,
    best_epoch: int,
    config: dict[str, Any],
    num_classes: int,
    run_id: str,
    artifact_dir: Path,
    peak_cuda_memory_bytes: int,
) -> None:
    """Atomically save the epoch-boundary state needed to resume the same run."""

    random_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        random_state["cuda"] = torch.cuda.get_rng_state_all()
    _atomic_save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "random_state": random_state,
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
            "num_classes": num_classes,
            "run_id": run_id,
            "artifact_dir": str(artifact_dir.resolve()),
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        },
        path,
    )


def load_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "random_state",
        "epoch",
        "best_score",
        "best_epoch",
        "config",
        "num_classes",
        "run_id",
        "artifact_dir",
        "peak_cuda_memory_bytes",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint is incomplete; missing: {sorted(missing)}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    _restore_random_state(checkpoint["random_state"])
    return checkpoint


def load_model_state(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def _atomic_save(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def _restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
