"""Training-only objectives for the predictive temporal research phase."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from etsr.data.common import DatasetSubset, build_loader
from etsr.data.events import EncodedInput, move_encoded_input, slice_encoded_time
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.reproducibility import capture_random_state, git_commit, restore_random_state
from etsr.training.gradient_diagnostics import objective_gradient_diagnostics
from etsr.utils.io import sha256_file

_PHASE1_AUDIT_SECTIONS = {
    "A1_gradient_authority",
    "A2_discriminative_probes",
    "A3_representation_movement",
    "A4_tail_margin",
}


# Model fields that may differ from the frozen C0 topology in this phase. Everything else must
# match C0 exactly, so that a branch changes only the mechanism it declares.
PREDICTIVE_MODEL_FIELDS = frozenset(
    {
        "predictive_head",
        "predictive_head_spatial_kernel_size",
        "predictive_head_hidden_channels",
        "temporal_channel_mixer_dynamic_routing",
        "temporal_channel_mixer_router_pooling",
        "temporal_channel_mixer_router_hidden_divisor",
        "temporal_channel_mixer_predictive_auxiliary",
        "temporal_channel_mixer_predictor_channel_groups",
        "temporal_channel_mixer_predictor_spatial_kernel_size",
        "temporal_channel_mixer_surprise_routing",
        "temporal_channel_mixer_routing_stages",
        "temporal_channel_mixer_routing_parameterization",
        "temporal_channel_mixer_predictive_stages",
    }
)
# Parameter-name tokens that a C0-topology state dict does not contain.
NEW_MODULE_PARAMETER_TOKENS = (
    "predictive_head.",
    ".content_router.",
    ".predictor_logits",
    ".predictor_spatial.",
    ".predictor_projections.",
    ".surprise_router",
)
PREDICTIVE_SCRATCH_REFERENCE_CONFIG = "configs/dvslip_f_tcap_stage1_dwc3_d8.yaml"
PREDICTION_PREFIX_READOUTS = {"prefix_mean", "fixed_window_denominator"}


def base_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """Return the C0-topology model section obtained by removing registered phase fields."""

    return {
        key: value
        for key, value in model_config.items()
        if key not in PREDICTIVE_MODEL_FIELDS and key != "num_classes"
    }


def load_backbone_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> set[str]:
    """Load a C0-topology state and return the names of the phase-specific new tensors."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not any(token in key for token in NEW_MODULE_PARAMETER_TOKENS)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Model is not topology-compatible with the C0 backbone: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    parameter_names = dict(model.named_parameters())
    return {name for name in incompatible.missing_keys if name in parameter_names}


def dataset_targets(dataset: Dataset) -> list[int]:
    """Targets aligned with ``dataset`` indices, including nested index views."""

    if isinstance(dataset, DatasetSubset):
        parent = dataset_targets(dataset.dataset)
        return [parent[index] for index in dataset.indices]
    targets = getattr(dataset, "targets", None)
    if targets is None or len(targets) != len(dataset):
        raise ValueError("Diagnostic batches require a dataset exposing one target per sample.")
    return [int(target) for target in targets]


def class_stratified_indices(dataset: Dataset, count: int) -> list[int]:
    """Deterministic diagnostic indices spanning as many classes as possible.

    DVS-Lip samples are sorted by class, so the first unshuffled batch contains a single word.
    Gradient diagnostics on such a batch describe one class, not the objective.
    """

    if count <= 0:
        raise ValueError("Diagnostic batch size must be positive.")
    targets = dataset_targets(dataset)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(target, []).append(index)
    classes = sorted(by_class)
    count = min(count, len(targets))
    if count <= len(classes):
        if count == 1:
            chosen = [classes[0]]
        else:
            chosen = [
                classes[round(position * (len(classes) - 1) / (count - 1))]
                for position in range(count)
            ]
        return [by_class[target][0] for target in chosen]
    selected: list[int] = []
    depth = 0
    while len(selected) < count:
        for target in classes:
            if depth < len(by_class[target]) and len(selected) < count:
                selected.append(by_class[target][depth])
        depth += 1
    return selected


def diagnostic_batch(
    dataset: Dataset,
    dataset_config: dict[str, Any],
    device: torch.device,
    count: int | None = None,
) -> tuple[EncodedInput, torch.Tensor]:
    """Load one class-stratified batch with the dataset's own transforms."""

    size = int(count if count is not None else dataset_config.get("batch_size", 16))
    indices = class_stratified_indices(dataset, size)
    loader = build_loader(
        DatasetSubset(dataset, indices),
        {**dataset_config, "batch_size": len(indices), "num_workers": 0},
        shuffle=False,
    )
    frames, targets, _indices = next(iter(loader))
    return move_encoded_input(frames, device), targets.to(device)


def validate_predictive_training_authorization(
    predictive: dict[str, Any] | None,
    continuation: dict[str, Any] | None = None,
) -> Path:
    """Require the phase-1 audit contract before any predictive training entry point."""

    if not isinstance(predictive, dict):
        raise ValueError("Predictive-phase training requires a predictive section.")
    blocked_reason = predictive.get("blocked_reason")
    if blocked_reason:
        raise RuntimeError(
            "Predictive branch is intentionally blocked by the phase-1 audit: "
            f"{blocked_reason}"
        )
    report_value = predictive.get("phase1_audit_report")
    if not isinstance(report_value, str) or not report_value.strip():
        raise ValueError(
            "Predictive training requires predictive.phase1_audit_report from A1-A4."
        )
    audit_path = Path(report_value)
    if not audit_path.is_file():
        raise FileNotFoundError(
            "The mandatory checkpoint-only A1-A4 audit has not been produced: "
            f"{audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checkpoints = audit.get("checkpoints")
    c0 = checkpoints.get("c0") if isinstance(checkpoints, dict) else None
    if (
        audit.get("schema_version") != 1
        or audit.get("complete") is not True
        or set(audit.get("sections", ())) != _PHASE1_AUDIT_SECTIONS
        or audit.get("official_test_used") is not False
        or not isinstance(c0, dict)
        or not isinstance(c0.get("sha256"), str)
    ):
        raise ValueError(f"Incomplete or incompatible phase-1 audit report: {audit_path}")
    if continuation is not None and c0["sha256"] != sha256_file(continuation["parent_checkpoint"]):
        raise ValueError("Phase-1 audit and continuation use different C0 checkpoints.")
    return audit_path


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
    valid_context_steps: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Smooth-L1 with equal sample weight over one matched set of context indices."""

    if prediction.shape != target.shape:
        raise ValueError("Predictive target and prediction shapes differ.")
    per_step = nn.functional.smooth_l1_loss(prediction, target.detach(), reduction="none")
    per_step = per_step.flatten(2).mean(2)  # [T, B]
    context_positions = torch.arange(prediction.shape[0], device=prediction.device).unsqueeze(1)
    mask = context_positions < valid_context_steps.unsqueeze(0)
    counts = mask.sum(0)
    valid_samples = counts > 0
    if not bool(valid_samples.any().item()):
        raise ValueError("The batch contains no valid predictive target pairs.")
    per_sample = (per_step * mask).sum(0) / counts.clamp_min(1)
    coverage = float(mask.sum().item() / max(1, mask.numel()))
    return per_sample[valid_samples].mean(), coverage


def fixed_window_prefix_logits(
    model: MiniQKFormer,
    prefix_logits: torch.Tensor,
    steps: int,
    total_steps: int,
) -> torch.Tensor:
    """Read a prefix on the full-window scale: ``head(sum_{t<steps} f_t / total_steps)``.

    With the mean fixed-window readout, full-window logits decompose exactly as
    ``z_T = (s/T) z_s + ((T-s)/T) z_tail`` up to the head bias. Reading the prefix on the same
    denominator removes the pure readout-dilution difference, so a full-window teacher differs
    from the student prefix only by the contribution of the steps the prefix has not seen.
    """

    if getattr(model, "readout_name", None) != "mean" or getattr(model, "readout_time", None) != "fixed_window":
        raise ValueError("The fixed-window prefix readout is exact only for the mean fixed-window readout.")
    if not 0 < steps <= total_steps:
        raise ValueError("Prefix steps must lie within the encoded window.")
    scale = steps / total_steps
    bias = model.head.bias
    if bias is None:
        return prefix_logits * scale
    return scale * (prefix_logits - bias) + bias


@dataclass
class PredictiveBatchResult:
    logits: torch.Tensor
    total_loss: torch.Tensor
    classification_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    auxiliary_components: dict[str, torch.Tensor]
    metrics: dict[str, float]


class PredictiveTrainingObjective:
    """Compute the preregistered auxiliary loss without changing deployment forward."""

    MODES = {"none", "fine_future", "fine_same", "coarse_future", "late_prefix"}

    def __init__(
        self,
        config: dict[str, Any],
        teacher: MiniQKFormer | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
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
        self.prefix_readout = str(config.get("prefix_readout", "prefix_mean"))
        self.temperature = float(config.get("temperature", 2.0))
        authority = config.get("authority")
        self.authority: dict[str, float] | None = None
        if authority is not None:
            if not isinstance(authority, dict):
                raise ValueError("objective.authority must be a mapping")
            self.authority = {
                "target_ratio": float(authority["target_ratio"]),
                "max_step_factor": float(authority.get("max_step_factor", 2.0)),
                "min_weight": float(authority.get("min_weight", 0.0)),
                "max_weight": float(authority.get("max_weight", 1.0e6)),
            }
        self.calibrated = False
        self.calibration_history: list[dict[str, Any]] = []
        self.temporal_region_weights = {
            str(name): float(value)
            for name, value in config.get(
                "temporal_region_weights", {"active": 1.0, "tail": 0.25}
            ).items()
        }
        self.temporal_stage_weights = {
            str(name): float(value)
            for name, value in config.get(
                "temporal_stage_weights", {"stage1": 1.0, "stage2": 1.0}
            ).items()
        }
        self.target_mean: torch.Tensor | None = None
        self.target_std: torch.Tensor | None = None
        normalization_report = config.get("normalization_report")
        if normalization_report is not None:
            report = json.loads(Path(normalization_report).read_text(encoding="utf-8"))
            if provenance is None or "parent_checkpoint" not in provenance:
                raise ValueError(
                    "Normalization reports are defined only for continuation objectives, whose "
                    "probe student is the continuation parent."
                )
            expected = {
                "schema_version": 2,
                "mode": self.mode,
                "horizon_steps": self.horizon,
                "alignment_horizon_steps": self.alignment_horizon,
                "probe_geometry": "shared_dense_affine_1x1_per_spatial_position",
                "probe_git_commit": git_commit(),
                "student_checkpoint_sha256": sha256_file(provenance["parent_checkpoint"]),
                "teacher_checkpoint_sha256": sha256_file(provenance["teacher_checkpoint"]),
                "representation": provenance["representation"],
            }
            mismatches = {
                key: (report.get(key), value)
                for key, value in expected.items()
                if report.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Normalization report provenance mismatch: {mismatches}")
            standardization = report["target_standardization"]
            self.target_mean = torch.tensor(
                standardization["mean_by_channel"], dtype=torch.float32
            )
            self.target_std = torch.tensor(
                standardization["std_by_channel"], dtype=torch.float32
            ).clamp_min(float(standardization.get("minimum_std", 1e-5)))
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("predictive objective weight must be finite and non-negative")
        if self.authority is not None:
            target = self.authority["target_ratio"]
            factor = self.authority["max_step_factor"]
            low = self.authority["min_weight"]
            high = self.authority["max_weight"]
            if not (math.isfinite(target) and target > 0.0):
                raise ValueError("objective.authority.target_ratio must be positive")
            if not (math.isfinite(factor) and factor >= 1.0):
                raise ValueError("objective.authority.max_step_factor must be at least 1")
            if not (math.isfinite(high) and 0.0 <= low < high):
                raise ValueError("objective.authority requires 0 <= min_weight < max_weight")
        if self.prefix_readout not in PREDICTION_PREFIX_READOUTS:
            raise ValueError(f"Unsupported prefix readout: {self.prefix_readout}")
        default_minimum = 0.05 if self.auxiliary_declared else 0.0
        self.minimum_shared_gradient_ratio = float(
            config.get("minimum_shared_gradient_ratio", default_minimum)
        )
        if not math.isfinite(self.minimum_shared_gradient_ratio) or self.minimum_shared_gradient_ratio < 0:
            raise ValueError("objective.minimum_shared_gradient_ratio must be non-negative")
        if self.horizon <= 0 or self.alignment_horizon < self.horizon:
            raise ValueError("predictive horizons must be positive and consistently aligned")
        if self.mode != "none" and teacher is None:
            raise ValueError("A frozen teacher is required for a predictive objective")
        for label, weights, required in (
            ("temporal_region_weights", self.temporal_region_weights, {"active", "tail"}),
            ("temporal_stage_weights", self.temporal_stage_weights, {"stage1", "stage2"}),
        ):
            if set(weights) != required or any(value < 0 for value in weights.values()):
                raise ValueError(f"{label} must contain non-negative weights for {sorted(required)}")
            if sum(weights.values()) <= 0:
                raise ValueError(f"{label} must retain at least one positive weight")

    def effective_weight(self, epoch: int) -> float:
        if self.ramp_epochs <= 0:
            return self.weight
        if self.ramp_epochs == 1:
            return self.weight
        progress = (epoch - 1) / (self.ramp_epochs - 1)
        return self.weight * min(1.0, max(0.0, progress))

    @property
    def auxiliary_declared(self) -> bool:
        """Whether this objective adds an auxiliary term to the classification loss."""

        return self.mode != "none" or self.weight > 0.0 or self.authority is not None

    def selection_eligible(self, epoch: int) -> bool:
        """Return whether the auxiliary objective is at its full nominal weight.

        The ramp is defined by progress, not by comparing weights, so that a calibrated weight
        that changes between epochs does not affect eligibility.
        """

        return not self.auxiliary_declared or self.ramp_epochs <= 1 or epoch >= self.ramp_epochs

    def state_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
            "calibrated": self.calibrated,
            "calibration_history": list(self.calibration_history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.weight = float(state["weight"])
        self.calibrated = bool(state["calibrated"])
        self.calibration_history = list(state.get("calibration_history", ()))

    def calibrate(
        self,
        model: MiniQKFormer,
        frames: EncodedInput,
        targets: torch.Tensor,
        criterion: nn.Module,
        *,
        epoch: int,
        freeze_batchnorm_statistics: bool,
    ) -> dict[str, Any] | None:
        """Measure shared-backbone authority on a fixed batch and optionally set the weight.

        The measurement is side-effect free: module modes, BatchNorm buffers, gradients and every
        RNG stream are restored, so calibration never perturbs the training trajectory. It runs
        in float32 because the auxiliary gradients can be many orders of magnitude below CE.
        """

        if not self.auxiliary_declared:
            return None
        modes = {module: module.training for module in model.modules()}
        buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        random_state = capture_random_state()
        model.train()
        if freeze_batchnorm_statistics:
            freeze_batchnorm_running_statistics(model)
        try:
            with torch.autocast(device_type=targets.device.type, enabled=False):
                result = self(model, frames, targets, criterion, max(epoch, self.ramp_epochs, 1))
                diagnostics = objective_gradient_diagnostics(
                    model, result.classification_loss, result.auxiliary_loss, 1.0
                )
        finally:
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, buffer in model.named_buffers():
                    buffer.copy_(buffers[name])
            for module, mode in modes.items():
                module.training = mode
            restore_random_state(random_state)

        unit_ratio = diagnostics["gradient_shared_unit_ratio"]
        previous = self.weight
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "unit_ratio": unit_ratio,
            "previous_weight": previous,
            "weight": previous,
            "shared_ratio": unit_ratio * previous,
            "shared_cosine": diagnostics["gradient_shared_cosine"],
            "updated": False,
        }
        if self.authority is not None:
            if not math.isfinite(unit_ratio) or unit_ratio <= 0.0:
                raise RuntimeError(
                    "The auxiliary objective has no measurable gradient on the shared backbone; "
                    "its authority cannot be calibrated."
                )
            weight = self.authority["target_ratio"] / unit_ratio
            if self.calibrated:
                factor = self.authority["max_step_factor"]
                weight = min(max(weight, previous / factor), previous * factor)
            weight = min(max(weight, self.authority["min_weight"]), self.authority["max_weight"])
            self.weight = weight
            self.calibrated = True
            record.update(weight=weight, shared_ratio=unit_ratio * weight, updated=True)
        self.calibration_history.append(record)
        return record

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
        auxiliary_components: dict[str, torch.Tensor] = {}
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
                (last_occupied_steps(frames) - self.alignment_horizon).clamp_min(0),
            )
            auxiliary_terms.append(auxiliary)
            auxiliary_components["active"] = auxiliary
            metrics.update(
                predictive_loss=float(auxiliary.detach()),
                predictive_pair_coverage=coverage,
            )

        temporal_auxiliary, temporal_metrics, temporal_components = (
            model.temporal_auxiliary_statistics(
                last_occupied_steps(frames),
                region_weights=self.temporal_region_weights,
                stage_weights=self.temporal_stage_weights,
            )
        )
        if temporal_auxiliary is not None:
            auxiliary_terms.append(temporal_auxiliary)
            for region, component in temporal_components.items():
                auxiliary_components[region] = (
                    auxiliary_components.get(region, component.new_zeros(())) + component
                )
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
        return PredictiveBatchResult(
            logits=logits,
            total_loss=total,
            classification_loss=classification,
            auxiliary_loss=auxiliary_total,
            auxiliary_components=auxiliary_components,
            metrics=metrics,
        )

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
        total_steps = int(coarse_frames(frames).shape[1])
        terms = []
        prefix_metrics: dict[str, float] = {}
        for steps in self.prefix_steps:
            prefix_logits = model(slice_encoded_time(frames, steps))
            if self.prefix_readout == "fixed_window_denominator":
                prefix_logits = fixed_window_prefix_logits(model, prefix_logits, steps, total_steps)
            term = (
                nn.functional.kl_div(
                    nn.functional.log_softmax(prefix_logits / self.temperature, dim=1),
                    teacher_probability,
                    reduction="batchmean",
                )
                * self.temperature**2
            )
            terms.append(term)
            prefix_metrics[f"prefix_kl_{steps}"] = float(term.detach())
        auxiliary = torch.stack(terms).mean()
        weight = self.effective_weight(epoch)
        return PredictiveBatchResult(
            logits=logits,
            total_loss=classification + weight * auxiliary,
            classification_loss=classification,
            auxiliary_loss=auxiliary,
            auxiliary_components={"active": auxiliary},
            metrics={
                "classification_loss": float(classification.detach()),
                "auxiliary_loss": float(auxiliary.detach()),
                "auxiliary_weight": weight,
                **prefix_metrics,
            },
        )
