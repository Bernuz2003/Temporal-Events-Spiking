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


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


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
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    start = time.perf_counter()

    progress = tqdm(loader, desc="train", leave=False)
    for frames, targets, _indices in progress:
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(frames)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(targets.numel())
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
    for frames, targets, indices in tqdm(loader, desc="eval", leave=False):
        if prefix_fraction is not None:
            prefix = max(1, math.ceil(frames.shape[1] * prefix_fraction))
            frames = frames[:, :prefix]
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(frames)
        loss = criterion(logits, targets)
        accumulator.update(logits, targets, loss, indices)

    return accumulator.compute(), {
        "indices": np.asarray(accumulator.indices, dtype=np.int64),
        "targets": np.asarray(accumulator.targets, dtype=np.int64),
        "predictions": np.asarray(accumulator.predictions, dtype=np.int64),
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
