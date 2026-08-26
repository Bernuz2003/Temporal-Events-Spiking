"""Minimal training and validation loop for encoded event tensors."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from etsr.evaluation.metrics import ClassificationAccumulator, ClassificationResult
from etsr.training.checkpointing import load_model_state


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
):
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
) -> dict[str, float | None]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    start = time.perf_counter()
    total_batches = len(loader)
    accumulated_samples = 0
    gradient_norm_sum = 0.0
    finite_gradient_steps = 0
    nonfinite_gradient_steps = 0
    amp_overflow_steps = 0
    clipped_steps = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (frames, targets, _indices) in enumerate(loader):
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
        batches_seen = batch_index + 1
        group_complete = (
            batches_seen % gradient_accumulation_steps == 0 or batches_seen == total_batches
        )
        if group_complete:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            gradient_is_finite = True
            if gradient_clip_norm is not None:
                gradient_norm = float(
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm).item()
                )
                gradient_is_finite = math.isfinite(gradient_norm)
                if gradient_is_finite:
                    gradient_norm_sum += gradient_norm
                    finite_gradient_steps += 1
                    clipped_steps += int(gradient_norm > gradient_clip_norm)
                else:
                    nonfinite_gradient_steps += 1
            scale_before = float(scaler.get_scale()) if amp_enabled else None
            scaler.step(optimizer)
            scaler.update()
            overflow = (
                amp_enabled
                and scale_before is not None
                and float(scaler.get_scale()) < scale_before
            )
            amp_overflow_steps += int(overflow)
            if not gradient_is_finite and not overflow:
                raise FloatingPointError(
                    f"Non-finite gradient at optimizer step {optimizer_steps + 1}."
                )
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
            optimizer_steps += 1

        loss_sum += float(loss.detach().item()) * batch_size
        correct += int((logits.argmax(1) == targets).sum().item())
        samples += batch_size

    return {
        "loss": loss_sum / max(1, samples),
        "accuracy": correct / max(1, samples),
        "seconds": time.perf_counter() - start,
        "gradient_norm_mean": (
            gradient_norm_sum / max(1, finite_gradient_steps)
            if gradient_clip_norm is not None
            else None
        ),
        "gradient_clip_fraction": (
            clipped_steps / max(1, finite_gradient_steps)
            if gradient_clip_norm is not None
            else None
        ),
        "gradient_nonfinite_fraction": (
            nonfinite_gradient_steps / max(1, optimizer_steps)
            if gradient_clip_norm is not None
            else None
        ),
        "amp_overflow_fraction": amp_overflow_steps / max(1, optimizer_steps),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    collect_predictions: bool = False,
    prefix_steps: int | Sequence[int] | None = None,
) -> tuple[ClassificationResult, dict[str, Any] | None]:
    model.eval()
    accumulator = ClassificationAccumulator(num_classes, collect_predictions)
    for batch_index, (frames, targets, indices) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if prefix_steps is None:
            logits = model(frames)
        elif isinstance(prefix_steps, int):
            if not 1 <= prefix_steps <= frames.shape[1]:
                raise ValueError("prefix_steps must fit the encoded time axis.")
            logits = model(frames[:, :prefix_steps])
        else:
            batch_steps = torch.as_tensor(
                [prefix_steps[int(index)] for index in indices],
                dtype=torch.long,
                device=device,
            )
            if bool(((batch_steps < 1) | (batch_steps > frames.shape[1])).any().item()):
                raise ValueError("Every per-sample prefix must fit the encoded time axis.")
            positions: list[torch.Tensor] = []
            outputs: list[torch.Tensor] = []
            for steps in torch.unique(batch_steps, sorted=True):
                selected = torch.nonzero(batch_steps == steps, as_tuple=False).squeeze(1)
                positions.append(selected)
                outputs.append(model(frames.index_select(0, selected)[:, : int(steps.item())]))
            order = torch.cat(positions).argsort()
            logits = torch.cat(outputs).index_select(0, order)
        loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite evaluation loss at batch {batch_index}.")
        accumulator.update(logits, targets, loss, indices)

    predictions = accumulator.prediction_arrays() if collect_predictions else None
    return accumulator.compute(), predictions


def restore_best_model(
    checkpoint_path: Path, model: nn.Module, device: torch.device, logger: logging.Logger
) -> dict:
    logger.info("Loading best checkpoint: %s", checkpoint_path)
    return load_model_state(checkpoint_path, model, device)
