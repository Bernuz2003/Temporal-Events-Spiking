"""Dense-frame training/evaluation engine retained through P0.

The `(frames, targets, indices)` batch contract is an observed limitation, not the canonical API for
the future raw-event DVS-Lip pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from etsr.evaluation.metrics import ClassificationAccumulator, ClassificationResult
from etsr.profiling.activity import FiringRateProfiler
from etsr.profiling.energy import estimate_horowitz_energy
from etsr.profiling.operations import OperationProfiler
from etsr.training.checkpointing import load_model_state


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    kwargs = dict(
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
):
    """Build the epoch scheduler, preserving the legacy no-warmup default."""

    if str(config.get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError(f"Unsupported scheduler: {config.get('scheduler')}")

    epochs = int(config["epochs"])
    warmup_epochs = int(config.get("warmup_epochs", 0))
    minimum_lr = float(config.get("min_learning_rate", 0.0))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=minimum_lr,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(config.get("warmup_start_factor", 0.01)),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=minimum_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def make_criterion(config: dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    gradient_clip_norm: float | None,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    start = time.perf_counter()
    total_batches = len(loader)
    accumulated_samples = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train", leave=False)
    for batch_index, (frames, targets, _indices) in enumerate(progress):
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(frames)
            loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}.")

        batch_size = int(targets.numel())
        accumulated_samples += batch_size
        scaler.scale(loss * batch_size).backward()
        group_complete = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == total_batches
        )
        if group_complete:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0

        loss_sum += float(loss.detach().item()) * batch_size
        correct += int((logits.argmax(1) == targets).sum().item())
        samples += batch_size
        progress.set_postfix(loss=f"{loss_sum / samples:.4f}")

    return {
        "loss": loss_sum / max(1, samples),
        "accuracy": correct / max(1, samples),
        "seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    prefix_fraction: float | None = None,
) -> tuple[ClassificationResult, dict[str, np.ndarray]]:
    model.eval()
    accumulator = ClassificationAccumulator(num_classes)
    for batch_index, (frames, targets, indices) in enumerate(
        tqdm(loader, desc="eval", leave=False)
    ):
        if prefix_fraction is not None:
            prefix = max(1, math.ceil(frames.shape[1] * prefix_fraction))
            frames = frames[:, :prefix]
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(frames)
        loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite evaluation loss at batch {batch_index}.")
        accumulator.update(logits, targets, loss, indices)

    return accumulator.compute(), {
        "indices": np.asarray(accumulator.indices, dtype=np.int64),
        "targets": np.asarray(accumulator.targets, dtype=np.int64),
        "predictions": np.asarray(accumulator.predictions, dtype=np.int64),
        "margins": np.asarray(accumulator.margins, dtype=np.float64),
    }


@torch.no_grad()
def profile_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict:
    model.eval()
    firing = FiringRateProfiler(model)
    operations = OperationProfiler(model)
    samples = 0
    max_batches = int(config.get("max_batches", 8))

    for batch_index, (frames, _targets, _indices) in enumerate(loader):
        if batch_index >= max_batches:
            break
        frames = frames.to(device)
        operations.set_batch_size(frames.shape[0])
        model(frames)
        samples += int(frames.shape[0])

    firing_summary = firing.summary()
    operation_summary = operations.summary(samples)
    firing.close()
    operations.close()

    energy = estimate_horowitz_energy(
        operation_summary["mac_ops_per_sample"],
        operation_summary["ac_ops_per_sample"],
        float(config.get("mac_energy_pj", 4.6)),
        float(config.get("ac_energy_pj", 0.9)),
    )
    return {
        "profiled_samples": samples,
        "firing": firing_summary,
        "operations": operation_summary,
        "energy": energy,
    }


def restore_best_model(
    checkpoint_path: Path, model: nn.Module, device: torch.device, logger: logging.Logger
) -> dict:
    logger.info("Loading best checkpoint: %s", checkpoint_path)
    return load_model_state(checkpoint_path, model, device)
