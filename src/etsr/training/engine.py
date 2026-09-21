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

from etsr.data.events import (
    encoded_time_steps,
    move_encoded_input,
    slice_encoded_time,
)
from etsr.evaluation.metrics import ClassificationAccumulator, ClassificationResult
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.training.augmentation import EventMix
from etsr.training.checkpointing import load_model_state
from etsr.training.predictive import (
    PredictiveTrainingObjective,
    freeze_batchnorm_running_statistics,
)


def make_optimizer(
    model: nn.Module,
    config: dict[str, Any],
    *,
    new_parameter_names: set[str] | None = None,
    new_parameter_learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    """Build AdamW, optionally giving newly introduced continuation parameters their own LR."""

    base_learning_rate = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    delay_ids = {
        id(module.delay_centers)
        for module in model.modules()
        if isinstance(module, CausalTemporalChannelMixer) and module.delay_centers is not None
    }
    requested_new_names = set(new_parameter_names or ())
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    known_names = {name for name, _parameter in named_parameters}
    unknown_names = requested_new_names - known_names
    if unknown_names:
        raise ValueError(f"Unknown new continuation parameters: {sorted(unknown_names)}")

    if new_parameter_learning_rate is None or not requested_new_names:
        delay_centers = [parameter for _name, parameter in named_parameters if id(parameter) in delay_ids]
        if delay_centers:
            parameters: Any = [
                {
                    "params": [
                        parameter
                        for _name, parameter in named_parameters
                        if id(parameter) not in delay_ids
                    ],
                    "group_name": "default",
                },
                {"params": delay_centers, "weight_decay": 0.0, "group_name": "delay_centers"},
            ]
        else:
            parameters = [
                {
                    "params": [parameter for _name, parameter in named_parameters],
                    "group_name": "default",
                }
            ]
    else:
        grouped: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, parameter in named_parameters:
            key = (name in requested_new_names, id(parameter) in delay_ids)
            grouped.setdefault(key, []).append(parameter)
        parameters = []
        for (is_new, is_delay), values in grouped.items():
            group_name = "new" if is_new else "inherited"
            if is_delay:
                group_name += "_delay_centers"
            parameters.append(
                {
                    "params": values,
                    "lr": (
                        float(new_parameter_learning_rate)
                        if is_new
                        else base_learning_rate
                    ),
                    "weight_decay": 0.0 if is_delay else weight_decay,
                    "group_name": group_name,
                }
            )
    return torch.optim.AdamW(
        parameters,
        lr=base_learning_rate,
        weight_decay=weight_decay,
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


def _accumulate_routing_statistics(
    accumulators: dict[str, dict[str, Any]],
    modules: tuple[tuple[str, CausalTemporalChannelMixer], ...],
) -> None:
    """Accumulate exact gate moments without retaining an autograd graph or synchronizing CUDA."""

    for name, module in modules:
        statistics = module.last_routing_statistics
        if statistics is None:
            continue
        observations = int(statistics["gate_observation_count"])
        sample_count = int(statistics["gate_sample_count"])
        current = accumulators.setdefault(
            name,
            {
                "delays": module.delays,
                "observation_count": 0,
                "sample_count": 0,
            },
        )
        weighted = {
            "gate_sum": statistics["gate_mean_by_delay"] * observations,
            "gate_second_moment_sum": (
                statistics["gate_second_moment_by_delay"] * observations
            ),
            "sample_mean_sum": statistics["gate_mean_by_delay"] * sample_count,
            "sample_mean_second_moment_sum": (
                statistics["gate_sample_mean_second_moment_by_delay"] * sample_count
            ),
            "within_sample_variance_sum": (
                statistics["gate_within_sample_variance_mean_by_delay"] * sample_count
            ),
        }
        contributions = statistics.get("effective_contribution_mean_abs_by_delay")
        if isinstance(contributions, torch.Tensor):
            weighted["effective_contribution_sum"] = contributions.detach().float() * observations
        for key, value in weighted.items():
            current[key] = current.get(key, torch.zeros_like(value)) + value
        current["observation_count"] += observations
        current["sample_count"] += sample_count


def _finalize_routing_statistics(
    accumulators: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for name, values in accumulators.items():
        observations = int(values["observation_count"])
        sample_count = int(values["sample_count"])
        mean = values["gate_sum"] / observations
        second_moment = values["gate_second_moment_sum"] / observations
        variance = (second_moment - mean.square()).clamp_min(0)
        sample_mean = values["sample_mean_sum"] / sample_count
        sample_mean_second = values["sample_mean_second_moment_sum"] / sample_count
        between_sample_variance = (sample_mean_second - sample_mean.square()).clamp_min(0)
        within_sample_variance = values["within_sample_variance_sum"] / sample_count
        contribution = (
            values["effective_contribution_sum"] / observations
            if "effective_contribution_sum" in values
            else None
        )
        for index, delay in enumerate(values["delays"]):
            gate_mean = float(mean[index].cpu())
            gate_std = float(variance[index].sqrt().cpu())
            row = {
                "module": name,
                "delay": int(delay),
                "gate_mean": gate_mean,
                "gate_std": gate_std,
                "gate_cv": gate_std / max(abs(gate_mean), 1e-12),
                "gate_between_sample_std": float(
                    between_sample_variance[index].sqrt().cpu()
                ),
                "gate_within_sample_std": float(within_sample_variance[index].sqrt().cpu()),
                "gate_observation_count": observations,
                "sample_count": sample_count,
            }
            if contribution is not None:
                row["effective_contribution_mean_abs"] = float(contribution[index].cpu())
            rows.append(row)
    return rows


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
    batch_augmentation: EventMix | None = None,
    predictive_objective: PredictiveTrainingObjective | None = None,
    epoch: int = 1,
    freeze_batchnorm_statistics: bool = False,
) -> dict[str, Any]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    model.train()
    if freeze_batchnorm_statistics:
        freeze_batchnorm_running_statistics(model)
    loss_sum = 0.0
    correct = 0.0
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
    component_sums: dict[str, float] = {}
    delay_modules = tuple(
        module
        for module in model.modules()
        if isinstance(module, CausalTemporalChannelMixer) and module.learnable_delays
    )
    for batch_index, (frames, targets, _indices) in enumerate(loader):
        frames = move_encoded_input(frames, device)
        targets = targets.to(device, non_blocking=True)
        mixed_batch = batch_augmentation(frames, targets) if batch_augmentation else None
        if mixed_batch is not None:
            frames = mixed_batch.frames

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if predictive_objective is not None:
                if mixed_batch is not None:
                    raise ValueError("Predictive continuations do not support batch mixing.")
                batch_result = predictive_objective(model, frames, targets, criterion, epoch)
                logits = batch_result.logits
                loss = batch_result.total_loss
            else:
                logits = model(frames)
                batch_result = None
            if predictive_objective is None and mixed_batch is None:
                loss = criterion(logits, targets)
            elif predictive_objective is None:
                if not isinstance(criterion, nn.CrossEntropyLoss):
                    raise TypeError("EventMix requires CrossEntropyLoss")
                primary_loss = nn.functional.cross_entropy(
                    logits,
                    targets,
                    weight=criterion.weight,
                    ignore_index=criterion.ignore_index,
                    reduction="none",
                    label_smoothing=criterion.label_smoothing,
                )
                secondary_loss = nn.functional.cross_entropy(
                    logits,
                    mixed_batch.secondary_targets,
                    weight=criterion.weight,
                    ignore_index=criterion.ignore_index,
                    reduction="none",
                    label_smoothing=criterion.label_smoothing,
                )
                loss = (
                    mixed_batch.primary_weights * primary_loss
                    + (1.0 - mixed_batch.primary_weights) * secondary_loss
                ).mean()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}.")

        batch_size = int(targets.numel())
        if batch_result is not None:
            for name, value in batch_result.metrics.items():
                component_sums[name] = component_sums.get(name, 0.0) + value * batch_size
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
            if not gradient_is_finite and not amp_enabled:
                raise FloatingPointError(
                    f"Non-finite gradient at optimizer step {optimizer_steps + 1}."
                )
            scale_before = float(scaler.get_scale()) if amp_enabled else None
            scaler.step(optimizer)
            scaler.update()
            for module in delay_modules:
                module.project_delay_centers_()
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
        predictions = logits.argmax(1)
        if mixed_batch is None:
            correct += float((predictions == targets).sum().item())
        else:
            correct += float(
                (
                    mixed_batch.primary_weights * (predictions == targets)
                    + (1.0 - mixed_batch.primary_weights)
                    * (predictions == mixed_batch.secondary_targets)
                )
                .sum()
                .item()
            )
        samples += batch_size

    result = {
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
    result.update({name: value / max(1, samples) for name, value in component_sums.items()})
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    collect_predictions: bool = False,
    prefix_steps: int | Sequence[int] | None = None,
    routing_statistics: list[dict[str, Any]] | None = None,
) -> tuple[ClassificationResult, dict[str, Any] | None]:
    model.eval()
    accumulator = ClassificationAccumulator(num_classes, collect_predictions)
    routing_modules = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, CausalTemporalChannelMixer)
        and (module.content_router is not None or module.surprise_router is not None)
    )
    routing_accumulators: dict[str, dict[str, Any]] = {}
    if routing_statistics is not None and prefix_steps is not None:
        raise ValueError("Routing statistics require full-window evaluation.")
    for batch_index, (frames, targets, indices) in enumerate(loader):
        frames = move_encoded_input(frames, device)
        targets = targets.to(device, non_blocking=True)
        if prefix_steps is None:
            logits = model(frames)
        elif isinstance(prefix_steps, int):
            if not 1 <= prefix_steps <= encoded_time_steps(frames):
                raise ValueError("prefix_steps must fit the encoded time axis.")
            logits = model(slice_encoded_time(frames, prefix_steps))
        else:
            batch_steps = torch.as_tensor(
                [prefix_steps[int(index)] for index in indices],
                dtype=torch.long,
                device=device,
            )
            if bool(((batch_steps < 1) | (batch_steps > encoded_time_steps(frames))).any().item()):
                raise ValueError("Every per-sample prefix must fit the encoded time axis.")
            positions: list[torch.Tensor] = []
            outputs: list[torch.Tensor] = []
            for steps in torch.unique(batch_steps, sorted=True):
                selected = torch.nonzero(batch_steps == steps, as_tuple=False).squeeze(1)
                positions.append(selected)
                if isinstance(frames, torch.Tensor):
                    selected_frames = frames.index_select(0, selected)
                else:
                    selected_frames = {
                        name: value.index_select(0, selected) for name, value in frames.items()
                    }
                outputs.append(model(slice_encoded_time(selected_frames, int(steps.item()))))
            order = torch.cat(positions).argsort()
            logits = torch.cat(outputs).index_select(0, order)
        if routing_statistics is not None:
            _accumulate_routing_statistics(routing_accumulators, routing_modules)
        loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite evaluation loss at batch {batch_index}.")
        accumulator.update(logits, targets, loss, indices)

    predictions = accumulator.prediction_arrays() if collect_predictions else None
    if routing_statistics is not None:
        routing_statistics.extend(_finalize_routing_statistics(routing_accumulators))
    return accumulator.compute(), predictions


def restore_best_model(
    checkpoint_path: Path, model: nn.Module, device: torch.device, logger: logging.Logger
) -> dict:
    logger.info("Loading best checkpoint: %s", checkpoint_path)
    return load_model_state(checkpoint_path, model, device)
