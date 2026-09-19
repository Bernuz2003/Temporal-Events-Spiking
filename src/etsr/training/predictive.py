"""Training-only objectives for the predictive temporal research phase."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from etsr.data.events import EncodedInput, slice_encoded_time
from etsr.models.mini_qkformer import MiniQKFormer


def freeze_batchnorm_running_statistics(module: nn.Module) -> None:
    """Keep checkpoint running statistics fixed while retaining affine gradients."""

    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def coarse_frames(frames: EncodedInput) -> torch.Tensor:
    return frames if isinstance(frames, torch.Tensor) else frames["coarse"]


def last_occupied_steps(frames: EncodedInput) -> torch.Tensor:
    return MiniQKFormer._last_event_steps(frames)


def _balanced_masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_target_steps: torch.Tensor,
    target_offset: int,
) -> tuple[torch.Tensor, float]:
    """Smooth-L1 with equal sample weight over target steps preceding each endpoint."""

    if prediction.shape != target.shape:
        raise ValueError("Predictive target and prediction shapes differ.")
    per_step = nn.functional.smooth_l1_loss(prediction, target.detach(), reduction="none")
    per_step = per_step.flatten(2).mean(2)  # [T, B]
    target_positions = torch.arange(
        target_offset,
        target_offset + prediction.shape[0],
        device=prediction.device,
    ).unsqueeze(1)
    mask = target_positions < valid_target_steps.unsqueeze(0)
    counts = mask.sum(0)
    valid_samples = counts > 0
    if not bool(valid_samples.any().item()):
        raise ValueError("The batch contains no valid predictive target pairs.")
    per_sample = (per_step * mask).sum(0) / counts.clamp_min(1)
    coverage = float(mask.sum().item() / max(1, mask.numel()))
    return per_sample[valid_samples].mean(), coverage


@dataclass
class PredictiveBatchResult:
    logits: torch.Tensor
    total_loss: torch.Tensor
    metrics: dict[str, float]


class PredictiveTrainingObjective:
    """Compute the preregistered auxiliary loss without changing deployment forward."""

    MODES = {"none", "fine_future", "fine_same", "coarse_future", "late_prefix"}

    def __init__(self, config: dict[str, Any], teacher: MiniQKFormer | None = None) -> None:
        self.config = config
        self.mode = str(config.get("mode", "none"))
        if self.mode not in self.MODES:
            raise ValueError(f"Unsupported predictive objective: {self.mode}")
        self.teacher = teacher
        self.weight = float(config.get("weight", 0.0))
        self.ramp_epochs = int(config.get("ramp_epochs", 0))
        self.horizon = int(config.get("horizon_steps", 2))
        self.alignment_horizon = int(config.get("alignment_horizon_steps", self.horizon))
        self.prefix_steps = tuple(int(value) for value in config.get("prefix_steps", (20, 30)))
        self.temperature = float(config.get("temperature", 2.0))
        self.target_mean: torch.Tensor | None = None
        self.target_std: torch.Tensor | None = None
        normalization_report = config.get("normalization_report")
        if normalization_report is not None:
            report = json.loads(Path(normalization_report).read_text(encoding="utf-8"))
            standardization = report["target_standardization"]
            self.target_mean = torch.tensor(
                standardization["mean_by_channel"], dtype=torch.float32
            )
            self.target_std = torch.tensor(
                standardization["std_by_channel"], dtype=torch.float32
            ).clamp_min(float(standardization.get("minimum_std", 1e-5)))
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("predictive objective weight must be in [0, 1]")
        if self.horizon <= 0 or self.alignment_horizon < self.horizon:
            raise ValueError("predictive horizons must be positive and consistently aligned")
        if self.mode != "none" and teacher is None:
            raise ValueError("A frozen teacher is required for a predictive objective")

    def effective_weight(self, epoch: int) -> float:
        if self.ramp_epochs <= 0:
            return self.weight
        if self.ramp_epochs == 1:
            return self.weight
        progress = (epoch - 1) / (self.ramp_epochs - 1)
        return self.weight * min(1.0, max(0.0, progress))

    def __call__(
        self,
        model: MiniQKFormer,
        frames: EncodedInput,
        targets: torch.Tensor,
        criterion: nn.Module,
        epoch: int,
    ) -> PredictiveBatchResult:
        if self.mode == "late_prefix":
            return self._late_prefix(model, frames, targets, criterion, epoch)

        context = model.extract_stage1(frames)
        logits = model.logits_from_stage1(context, frames)
        classification = criterion(logits, targets)
        auxiliary_terms: list[torch.Tensor] = []
        metrics: dict[str, float] = {"classification_loss": float(classification.detach())}

        if self.mode != "none":
            assert self.teacher is not None
            with torch.no_grad():
                if self.mode.startswith("fine_"):
                    if not isinstance(frames, dict):
                        raise ValueError("Fine prediction requires a coarse/fine representation.")
                    teacher_features = self.teacher.extract_fine_pre_lif(frames)
                else:
                    teacher_features = self.teacher.extract_stage1(frames)
            predicted = model.predict_stage1_target(context)
            target_offset = self.horizon if self.mode.endswith("future") else 0
            context_stop = context.shape[0] - self.alignment_horizon
            if context_stop <= 0:
                raise ValueError("Predictive horizon exceeds the encoded sequence.")
            prediction = predicted[:context_stop]
            target = teacher_features[
                target_offset : target_offset + context_stop
            ]
            if self.target_mean is not None and self.target_std is not None:
                if self.target_mean.numel() != prediction.shape[2]:
                    raise ValueError("Target normalization channel count does not match prediction.")
                shape = (1, 1, prediction.shape[2], 1, 1)
                mean = self.target_mean.to(prediction.device, prediction.dtype).reshape(shape)
                std = self.target_std.to(prediction.device, prediction.dtype).reshape(shape)
                prediction = (prediction - mean) / std
                target = (target - mean) / std
            auxiliary, coverage = _balanced_masked_loss(
                prediction,
                target,
                last_occupied_steps(frames),
                target_offset,
            )
            auxiliary_terms.append(auxiliary)
            metrics.update(
                predictive_loss=float(auxiliary.detach()),
                predictive_pair_coverage=coverage,
            )

        temporal_auxiliary, temporal_metrics = model.temporal_auxiliary_statistics(
            last_occupied_steps(frames)
        )
        if temporal_auxiliary is not None:
            auxiliary_terms.append(temporal_auxiliary)
            metrics["temporal_prediction_loss"] = float(temporal_auxiliary.detach())
            metrics.update(temporal_metrics)

        auxiliary_total = (
            torch.stack(auxiliary_terms).sum()
            if auxiliary_terms
            else classification.new_zeros(())
        )
        weight = self.effective_weight(epoch)
        total = classification + weight * auxiliary_total
        metrics.update(auxiliary_loss=float(auxiliary_total.detach()), auxiliary_weight=weight)
        return PredictiveBatchResult(logits, total, metrics)

    def _late_prefix(
        self,
        model: MiniQKFormer,
        frames: EncodedInput,
        targets: torch.Tensor,
        criterion: nn.Module,
        epoch: int,
    ) -> PredictiveBatchResult:
        assert self.teacher is not None
        logits = model(frames)
        classification = criterion(logits, targets)
        with torch.no_grad():
            teacher_logits = self.teacher(frames)
            teacher_probability = nn.functional.softmax(
                teacher_logits / self.temperature, dim=1
            )
        terms = []
        for steps in self.prefix_steps:
            prefix_logits = model(slice_encoded_time(frames, steps))
            terms.append(
                nn.functional.kl_div(
                    nn.functional.log_softmax(prefix_logits / self.temperature, dim=1),
                    teacher_probability,
                    reduction="batchmean",
                )
                * self.temperature**2
            )
        auxiliary = torch.stack(terms).mean()
        weight = self.effective_weight(epoch)
        return PredictiveBatchResult(
            logits,
            classification + weight * auxiliary,
            {
                "classification_loss": float(classification.detach()),
                "auxiliary_loss": float(auxiliary.detach()),
                "auxiliary_weight": weight,
            },
        )
