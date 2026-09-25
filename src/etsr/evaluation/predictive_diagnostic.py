"""Checkpoint-only probes for cross-resolution temporal predictability."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch

from etsr.config import load_config
from etsr.data.common import DatasetSubset, build_loader
from etsr.data.events import move_encoded_input
from etsr.data.factory import build_dataset_bundle
from etsr.models.factory import build_model
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.reproducibility import git_commit, seed_everything
from etsr.training.checkpointing import load_model_state
from etsr.training.engine import evaluate, make_criterion
from etsr.training.gradient_diagnostics import objective_gradient_diagnostics
from etsr.training.predictive import (
    PredictiveTrainingObjective,
    base_model_config,
    diagnostic_batch,
    freeze_batchnorm_running_statistics,
    last_occupied_steps,
    load_backbone_state,
)
from etsr.utils.io import ensure_dir, sha256_file, write_json


class _DenseMetricAccumulator:
    """Streaming metrics for a spatially shared per-position linear predictor."""

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.error_square_sum = 0.0
        self.target_square_sum = 0.0
        self.target_sum = torch.zeros(channels, dtype=torch.float64)
        self.target_channel_square_sum = torch.zeros(channels, dtype=torch.float64)
        self.cosine_sum = 0.0
        self.rows = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.reshape(-1, self.channels).float()
        target = target.reshape(-1, self.channels).float()
        self.error_square_sum += float((prediction - target).square().sum().item())
        self.target_square_sum += float(target.square().sum().item())
        self.target_sum += target.sum(0).double().cpu()
        self.target_channel_square_sum += target.square().sum(0).double().cpu()
        self.cosine_sum += float(
            torch.nn.functional.cosine_similarity(prediction, target, dim=1).sum().item()
        )
        self.rows += len(target)

    def summary(self) -> dict[str, float]:
        elements = max(1, self.rows * self.channels)
        mse = self.error_square_sum / elements
        variance = float(
            (
                self.target_channel_square_sum
                - self.target_sum.square() / max(1, self.rows)
            ).sum().item()
            / elements
        )
        target_rms = (self.target_square_sum / elements) ** 0.5
        return {
            "r2_variance_normalized": 1.0 - mse / max(variance, 1e-12),
            "nrmse_rms_target": mse**0.5 / max(target_rms, 1e-12),
            "cosine_similarity": self.cosine_sum / max(1, self.rows),
            "mse": mse,
            "dense_position_rows": self.rows,
        }


@torch.no_grad()
def _dense_pair_batches(
    student: MiniQKFormer,
    teacher: MiniQKFormer,
    loader,
    device: torch.device,
    mode: str,
    horizon: int,
    alignment_horizon: int,
    max_samples: int,
):
    """Yield matched dense rows from the exact geometry used by the 1x1 training head."""

    samples = 0
    for frames, _labels, _indices in loader:
        if samples >= max_samples:
            break
        frames = move_encoded_input(frames, device)
        if not isinstance(frames, dict) and mode.startswith("fine_"):
            raise ValueError("Fine probe requires a multi-granular representation.")
        context = student.extract_stage1(frames)
        if mode.startswith("fine_"):
            assert isinstance(frames, dict)
            target_features = teacher.extract_fine_pre_lif(frames)
        else:
            target_features = teacher.extract_stage1(frames)
        endpoint = last_occupied_steps(frames)
        target_offset = horizon if mode.endswith("future") else 0
        context_stop = context.shape[0] - alignment_horizon
        for batch_index in range(context.shape[1]):
            if samples >= max_samples:
                break
            stop = min(
                context_stop,
                int(endpoint[batch_index].item()) - alignment_horizon,
            )
            if stop <= 0:
                continue
            sample_context = context[:stop, batch_index].permute(0, 2, 3, 1).reshape(
                -1, context.shape[2]
            )
            sample_target = target_features[
                target_offset : target_offset + stop, batch_index
            ].permute(0, 2, 3, 1).reshape(-1, target_features.shape[2])
            sample_persistence = target_features[:stop, batch_index].permute(
                0, 2, 3, 1
            ).reshape(-1, target_features.shape[2])
            samples += 1
            yield sample_context, sample_target, sample_persistence, samples


def _diagnostic_split(dataset, fit_samples: int, holdout_samples: int, seed: int):
    required = fit_samples + holdout_samples
    if required > len(dataset):
        raise ValueError(
            f"Predictive probe needs {required} development-train samples, found {len(dataset)}."
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:required].tolist()
    return (
        DatasetSubset(dataset, indices[:fit_samples]),
        DatasetSubset(dataset, indices[fit_samples:]),
    )


def _fit_dense_ridge(
    batches,
    channels: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    gram = torch.zeros(channels + 1, channels + 1, dtype=torch.float64)
    cross = torch.zeros(channels + 1, channels, dtype=torch.float64)
    target_sum = torch.zeros(channels, dtype=torch.float64)
    target_square_sum = torch.zeros(channels, dtype=torch.float64)
    rows = 0
    samples = 0
    for context, target, _persistence, sample_number in batches:
        samples = sample_number
        design = torch.cat((context.float(), torch.ones(len(context), 1, device=context.device)), 1)
        gram += (design.T @ design).double().cpu()
        cross += (design.T @ target.float()).double().cpu()
        target_sum += target.sum(0).double().cpu()
        target_square_sum += target.square().sum(0).double().cpu()
        rows += len(target)
    if rows == 0:
        raise ValueError("No valid dense predictive pairs were extracted.")
    penalty = torch.eye(channels + 1, dtype=torch.float64) * ridge
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(gram / rows + penalty, cross / rows).float()
    return weights, target_sum, target_square_sum, rows, samples


def run_cross_resolution_probe(
    config: dict[str, Any],
    output_path: str | Path,
    *,
    max_train_samples: int = 512,
    max_validation_samples: int = 256,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    """Fit a train-only linear probe and evaluate it on development validation."""

    continuation = config.get("continuation")
    predictive = config.get("predictive")
    if not isinstance(continuation, dict) or not isinstance(predictive, dict):
        raise ValueError("Predictive probe requires continuation and predictive sections.")
    objective = predictive["objective"]
    mode = str(objective["mode"])
    if mode not in {"fine_future", "fine_same", "coarse_future"}:
        raise ValueError("Probe supports fine_future, fine_same or coarse_future configurations.")
    if max_train_samples <= 0 or max_validation_samples <= 0 or ridge <= 0:
        raise ValueError("Probe sample limits and ridge must be positive.")

    probe_config = copy.deepcopy(config)
    probe_config["augmentation"] = {"horizontal_flip_probability": 0.0}
    seed_everything(int(config["experiment"]["seed"]), True)
    bundle = build_dataset_bundle(probe_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_config = load_config(continuation["parent_config"])
    student = build_model(parent_config["model"], len(bundle.classes)).to(device)
    load_model_state(continuation["parent_checkpoint"], student, device)
    teacher_config = load_config(predictive["teacher_config"])
    teacher = build_model(teacher_config["model"], len(bundle.classes)).to(device)
    load_model_state(predictive["teacher_checkpoint"], teacher, device)
    student.eval().requires_grad_(False)
    teacher.eval().requires_grad_(False)
    fit_dataset, holdout_dataset = _diagnostic_split(
        bundle.train,
        max_train_samples,
        max_validation_samples,
        int(config["experiment"]["seed"]),
    )
    train_loader = build_loader(fit_dataset, probe_config["dataset"], shuffle=False)
    validation_loader = build_loader(holdout_dataset, probe_config["dataset"], shuffle=False)
    horizon = int(objective.get("horizon_steps", 2))
    alignment_horizon = int(objective.get("alignment_horizon_steps", horizon))
    channels = int(parent_config["model"].get("embed_dim", 128)) // 2
    weights, target_sum, target_square_sum, target_count, train_samples = _fit_dense_ridge(
        _dense_pair_batches(
            student,
            teacher,
            train_loader,
            device,
            mode,
            horizon,
            alignment_horizon,
            max_train_samples,
        ),
        channels,
        ridge,
    )
    learned_metrics = _DenseMetricAccumulator(channels)
    mean_metrics = _DenseMetricAccumulator(channels)
    persistence_metrics = _DenseMetricAccumulator(channels)
    train_target_mean = (target_sum / target_count).float()
    validation_samples = 0
    for context, target, persistence, sample_number in _dense_pair_batches(
        student,
        teacher,
        validation_loader,
        device,
        mode,
        horizon,
        alignment_horizon,
        max_validation_samples,
    ):
        validation_samples = sample_number
        design = torch.cat(
            (context.float(), torch.ones(len(context), 1, device=context.device)), 1
        )
        prediction = design @ weights.to(context.device)
        learned_metrics.update(prediction, target)
        mean_metrics.update(train_target_mean.to(target.device).expand_as(target), target)
        persistence_metrics.update(persistence, target)
    if validation_samples == 0:
        raise ValueError("No valid dense holdout pairs were extracted.")
    target_mean = target_sum / target_count
    target_variance = target_square_sum / target_count - target_mean.square()
    target_std = target_variance.clamp_min(1e-10).sqrt().clamp_min(1e-5)
    report = {
        "schema_version": 2,
        "mode": mode,
        "horizon_steps": horizon,
        "alignment_horizon_steps": alignment_horizon,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "train_dense_position_rows": target_count,
        "validation_dense_position_rows": learned_metrics.rows,
        "ridge": ridge,
        "probe_geometry": "shared_dense_affine_1x1_per_spatial_position",
        "probe_git_commit": git_commit(),
        "linear_probe": learned_metrics.summary(),
        "train_mean_baseline": mean_metrics.summary(),
        "privileged_target_persistence_baseline": persistence_metrics.summary(),
        "normalization_fit_on": "development_train_only",
        "target_standardization": {
            "mean_by_channel": target_mean.tolist(),
            "std_by_channel": target_std.tolist(),
            "element_count": target_count,
            "minimum_std": 1e-5,
        },
        "diagnostic_partition": {
            "source": "development_train",
            "fit_samples": train_samples,
            "holdout_samples": validation_samples,
            "disjoint_by_sample": True,
            "seed": int(config["experiment"]["seed"]),
        },
        "representation": copy.deepcopy(config["representation"]),
        "student_config": str(Path(continuation["parent_config"]).resolve()),
        "teacher_config": str(Path(predictive["teacher_config"]).resolve()),
        "student_checkpoint": str(Path(continuation["parent_checkpoint"]).resolve()),
        "student_checkpoint_sha256": sha256_file(continuation["parent_checkpoint"]),
        "teacher_checkpoint": str(Path(predictive["teacher_checkpoint"]).resolve()),
        "teacher_checkpoint_sha256": sha256_file(predictive["teacher_checkpoint"]),
        "official_test_used": False,
        "interpretation_limit": (
            "The persistence baseline sees the teacher target at the current step and is a "
            "privileged diagnostic, not a deployable student baseline."
        ),
    }
    output = Path(output_path)
    ensure_dir(output.parent)
    write_json(report, output)
    return report


PREFLIGHT_SCRATCH_WARMUP_STEPS = 2
_NEW_MODULE_FAMILIES = (
    "predictive_head",
    "content_router",
    "predictor_logits",
    "predictor_spatial",
    "predictor_projections",
    "surprise_router",
)


def _load_continuation_model(
    config: dict[str, Any], num_classes: int, device: torch.device
) -> tuple[MiniQKFormer, dict[str, Any]]:
    model = build_model(config["model"], num_classes).to(device)
    checkpoint = torch.load(
        config["continuation"]["parent_checkpoint"],
        map_location=device,
        weights_only=False,
    )
    load_backbone_state(model, checkpoint["model"])
    return model, checkpoint


def _build_scratch_model(
    config: dict[str, Any], num_classes: int, device: torch.device
) -> tuple[MiniQKFormer, bool]:
    """Build a from-scratch branch as the runner does and verify its backbone initialization."""

    base = build_model(base_model_config(config["model"]), num_classes)
    model = build_model(config["model"], num_classes)
    load_backbone_state(model, base.state_dict())
    state = model.state_dict()
    matches = all(torch.equal(state[name], value) for name, value in base.state_dict().items())
    return model.to(device), matches


def _gradient_family_norms(model: torch.nn.Module) -> tuple[dict[str, float | None], dict[str, float], bool]:
    gradient_norms = {
        name: (float(parameter.grad.norm().item()) if parameter.grad is not None else None)
        for name, parameter in model.named_parameters()
        if any(token in name for token in _NEW_MODULE_FAMILIES)
    }
    finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for name, parameter in model.named_parameters()
        if any(token in name for token in _NEW_MODULE_FAMILIES)
    )
    family_norms: dict[str, float] = {}
    for token in _NEW_MODULE_FAMILIES:
        if not any(token in name for name, _parameter in model.named_parameters()):
            continue
        squared = sum(
            float(parameter.grad.double().square().sum().item())
            for name, parameter in model.named_parameters()
            if token in name and parameter.grad is not None
        )
        family_norms[token] = squared**0.5
    return gradient_norms, family_norms, finite


def _perturb_future(frames, cutoff: int):
    if isinstance(frames, dict):
        altered = {name: value.clone() for name, value in frames.items()}
        coarse_steps = altered["coarse"].shape[1]
        ratio = altered["fine"].shape[1] // coarse_steps
        altered["coarse"][:, cutoff:] = torch.randn_like(altered["coarse"][:, cutoff:])
        altered["fine"][:, cutoff * ratio :] = torch.randn_like(
            altered["fine"][:, cutoff * ratio :]
        )
        return altered
    altered = frames.clone()
    altered[:, cutoff:] = torch.randn_like(altered[:, cutoff:])
    return altered


def _prefix_differences(model, frames, altered, cutoff: int) -> tuple[float, float]:
    with torch.no_grad():
        stage1 = model.extract_stage1(frames)
        altered_stage1 = model.extract_stage1(altered)
        encoded = model.encode_from_stage1(stage1)[:cutoff]
        altered_encoded = model.encode_from_stage1(altered_stage1)[:cutoff]
    return (
        float((stage1[:cutoff] - altered_stage1[:cutoff]).abs().max().item()),
        float((encoded - altered_encoded).abs().max().item()),
    )


def run_predictive_preflight(
    config: dict[str, Any], output_path: str | Path
) -> dict[str, Any]:
    """Check initialization, causality and gradient authority on a class-stratified train batch.

    Continuations must reproduce their parent exactly. From-scratch branches must start from the
    C0-topology initialization at the same seed; because TCAP matrices start at zero, routers
    receive gradient only after the matrices move, so new-module gradients are checked after a
    short warm-up on a copy of the model. Auxiliary authority is measured after the same
    calibration the training run applies, and must reach the objective's declared minimum.
    """

    predictive = config.get("predictive")
    if not isinstance(predictive, dict):
        raise ValueError("Predictive preflight requires a predictive section.")
    continuation = config.get("continuation")
    regime = "continuation" if continuation is not None else "from_scratch"
    seed_everything(int(config["experiment"]["seed"]), True)
    no_aug = copy.deepcopy(config)
    no_aug["augmentation"] = {"horizontal_flip_probability": 0.0}
    bundle = build_dataset_bundle(no_aug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(bundle.classes)
    parent = None
    parent_checkpoint = None
    backbone_matches = None
    if continuation is not None:
        candidate, parent_checkpoint = _load_continuation_model(config, num_classes, device)
        parent_config = load_config(continuation["parent_config"])
        parent = build_model(parent_config["model"], num_classes).to(device)
        load_model_state(continuation["parent_checkpoint"], parent, device)
    else:
        candidate, backbone_matches = _build_scratch_model(config, num_classes, device)

    objective_config = predictive["objective"]
    teacher = None
    if str(objective_config.get("mode", "none")) != "none":
        teacher_config = load_config(predictive["teacher_config"])
        teacher = build_model(teacher_config["model"], num_classes).to(device)
        load_model_state(predictive["teacher_checkpoint"], teacher, device)
        teacher.eval().requires_grad_(False)
    objective = PredictiveTrainingObjective(
        objective_config,
        teacher,
        {**predictive, **(continuation or {}), "representation": config["representation"]},
    )
    frames, targets = diagnostic_batch(bundle.train, no_aug["dataset"], device)
    freeze = continuation is not None
    criterion = make_criterion(config["training"])

    initial_difference = None
    if parent is not None:
        parent.eval()
        candidate.eval()
        with torch.no_grad():
            initial_difference = float((candidate(frames) - parent(frames)).abs().max().item())
    initialization_passed = (
        initial_difference is not None and initial_difference <= 1e-6
        if continuation is not None
        else bool(backbone_matches)
    )

    # Architectural causality is checked with BatchNorm on fixed running statistics: that is the
    # deployed computation, and the only one in which a prefix can be independent of its future.
    candidate.train()
    freeze_batchnorm_running_statistics(candidate)
    cutoff = max(1, candidate.extract_stage1(frames).shape[0] // 2)
    altered = _perturb_future(frames, cutoff)
    stage1_causal_difference, encoded_causal_difference = _prefix_differences(
        candidate, frames, altered, cutoff
    )
    # From scratch, training-mode BatchNorm pools statistics over time and batch, so a prefix
    # depends on future steps through per-channel batch moments. Frozen C0 was trained the same
    # way; the dependence is measured and declared, not gated.
    training_batchnorm_difference = None
    if not freeze:
        candidate.train()
        training_batchnorm_difference = max(
            _prefix_differences(candidate, frames, altered, cutoff)
        )

    calibration = objective.calibrate(
        candidate,
        frames,
        targets,
        criterion,
        epoch=0,
        freeze_batchnorm_statistics=freeze,
    )
    full_weight_epoch = max(1, objective.ramp_epochs)
    candidate.zero_grad(set_to_none=True)
    candidate.train()
    if freeze:
        freeze_batchnorm_running_statistics(candidate)
    result = objective(candidate, frames, targets, criterion, epoch=full_weight_epoch)
    gradient_authority: dict[str, dict[str, float]] = {}
    for region, component in result.auxiliary_components.items():
        if component.requires_grad:
            gradient_authority[region] = objective_gradient_diagnostics(
                candidate, result.classification_loss, component, objective.weight
            )
    total_authority = (
        objective_gradient_diagnostics(
            candidate, result.classification_loss, result.auxiliary_loss, objective.weight
        )
        if objective.auxiliary_declared and result.auxiliary_loss.requires_grad
        else None
    )
    shared_ratio = total_authority["gradient_shared_ratio"] if total_authority else None
    shared_gradient_authority_passed = not objective.auxiliary_declared or (
        shared_ratio is not None
        and math.isfinite(shared_ratio)
        and shared_ratio >= objective.minimum_shared_gradient_ratio
    )

    result.total_loss.backward()
    _norms_init, families_at_initialization, _finite_init = _gradient_family_norms(candidate)
    gradient_model: torch.nn.Module = candidate
    if continuation is None:
        # Mixers cache the last forward's graph-attached tensors; they are not deep-copyable.
        for module in candidate.modules():
            if isinstance(module, CausalTemporalChannelMixer):
                module._clear_prediction_diagnostics()
                module.last_routing_statistics = None
                module.last_temporal_variation = None
        gradient_model = copy.deepcopy(candidate)
        gradient_model.train()
        warmup_optimizer = torch.optim.AdamW(
            [parameter for parameter in gradient_model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        for _step in range(PREFLIGHT_SCRATCH_WARMUP_STEPS):
            warmup_optimizer.zero_grad(set_to_none=True)
            objective(gradient_model, frames, targets, criterion, epoch=full_weight_epoch).total_loss.backward()
            warmup_optimizer.step()
        warmup_optimizer.zero_grad(set_to_none=True)
        objective(gradient_model, frames, targets, criterion, epoch=full_weight_epoch).total_loss.backward()
    gradient_norms, gradient_family_norms, finite_gradients = _gradient_family_norms(gradient_model)
    nonzero_gradients = all(value > 1e-12 for value in gradient_family_norms.values())
    teacher_gradients_absent = teacher is None or all(
        parameter.grad is None for parameter in teacher.parameters()
    )
    report = {
        "schema_version": 3,
        "experiment": config["experiment"]["name"],
        "regime": regime,
        "parent_checkpoint": (
            str(Path(continuation["parent_checkpoint"]).resolve()) if continuation else None
        ),
        "parent_checkpoint_sha256": (
            sha256_file(continuation["parent_checkpoint"]) if continuation else None
        ),
        "parent_epoch": int(parent_checkpoint["epoch"]) if parent_checkpoint else None,
        "initial_logits_max_abs_difference": initial_difference,
        "backbone_matches_c0_topology_initialization": backbone_matches,
        "initialization_passed": initialization_passed,
        "diagnostic_batch_samples": int(targets.numel()),
        "diagnostic_batch_classes": len(set(targets.tolist())),
        "stage1_causal_prefix_max_abs_difference": stage1_causal_difference,
        "encoded_causal_prefix_max_abs_difference": encoded_causal_difference,
        "causal_prefix_passed": max(
            stage1_causal_difference, encoded_causal_difference
        ) <= 1e-6,
        "causal_prefix_batchnorm": "fixed_running_statistics",
        "training_batchnorm_prefix_max_abs_difference": training_batchnorm_difference,
        "training_batchnorm_note": (
            None
            if freeze
            else "Training-mode BatchNorm pools per-channel moments over time and batch; the "
            "deployed model uses running statistics and is causal. Shared with frozen C0."
        ),
        "total_loss_finite": bool(torch.isfinite(result.total_loss).item()),
        "objective_metrics": result.metrics,
        "objective_declared": objective.auxiliary_declared,
        "auxiliary_nominal_weight": objective.weight,
        "authority_calibration": calibration,
        "minimum_shared_gradient_ratio": objective.minimum_shared_gradient_ratio,
        "shared_gradient_ratio": shared_ratio,
        "shared_gradient_cosine": (
            total_authority["gradient_shared_cosine"] if total_authority else None
        ),
        "shared_gradient_authority_total": total_authority,
        "shared_gradient_authority_by_region": gradient_authority,
        "shared_gradient_authority_passed": shared_gradient_authority_passed,
        "new_parameter_gradient_measurement": (
            "at_parent" if continuation is not None
            else f"after_{PREFLIGHT_SCRATCH_WARMUP_STEPS}_warmup_steps"
        ),
        "new_parameter_gradient_family_norms_at_initialization": families_at_initialization,
        "new_parameter_gradient_norms": gradient_norms,
        "new_parameter_gradient_family_norms": gradient_family_norms,
        "new_parameter_gradients_finite": finite_gradients,
        "new_parameter_gradient_families_nonzero": nonzero_gradients,
        "teacher_gradients_absent": teacher_gradients_absent,
        "batchnorm_running_statistics": "fixed" if freeze else "training",
        "official_test_used": False,
    }
    report["passed"] = all(
        (
            report["initialization_passed"],
            report["causal_prefix_passed"],
            report["total_loss_finite"],
            report["new_parameter_gradients_finite"],
            report["new_parameter_gradient_families_nonzero"],
            report["shared_gradient_authority_passed"],
            report["teacher_gradients_absent"],
        )
    )
    output = Path(output_path)
    ensure_dir(output.parent)
    write_json(report, output)
    return report


def _delayed(sequence: torch.Tensor, delay: int) -> torch.Tensor:
    output = torch.zeros_like(sequence)
    if delay < sequence.shape[0]:
        output[delay:] = sequence[:-delay]
    return output


def _region_weights(time_steps: int, endpoints: torch.Tensor) -> torch.Tensor:
    weights = torch.zeros(time_steps, len(endpoints), device=endpoints.device)
    for batch_index, endpoint_tensor in enumerate(endpoints):
        endpoint = min(time_steps, max(0, int(endpoint_tensor.item())))
        regions = [(0, endpoint), (endpoint, time_steps)]
        present = [(start, stop) for start, stop in regions if stop > start]
        for start, stop in present:
            weights[start:stop, batch_index] = 1.0 / (len(present) * (stop - start))
    return weights


class _TCAPPredictiveAccumulator:
    def __init__(self, model: MiniQKFormer, *, fit: bool) -> None:
        self.fit = fit
        self.endpoints: torch.Tensor | None = None
        self.coefficients: dict[str, torch.Tensor] = {}
        self.statistics: dict[str, dict[str, torch.Tensor]] = {}
        self.region_errors: dict[str, dict[str, list[torch.Tensor]]] = {}
        self.sample_surprise: dict[str, list[torch.Tensor]] = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, CausalTemporalChannelMixer):
                self._handles.append(module.register_forward_hook(self._hook(name, module)))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def _hook(self, name: str, module: CausalTemporalChannelMixer):
        def collect(_module, inputs, _output) -> None:
            if self.endpoints is None:
                raise RuntimeError("TCAP diagnostic endpoints were not set before forward.")
            sequence = inputs[0].detach().float()
            delays = [_delayed(sequence, delay) for delay in module.delays]
            if self.fit:
                self._accumulate_fit(name, sequence, delays)
            else:
                self._accumulate_validation(name, sequence, delays)

        return collect

    def _accumulate_fit(
        self, name: str, sequence: torch.Tensor, delayed: list[torch.Tensor]
    ) -> None:
        assert self.endpoints is not None
        channels = sequence.shape[2]
        taps = len(delayed)
        stats = self.statistics.setdefault(
            name,
            {
                "gram": torch.zeros(channels, taps, taps, dtype=torch.float64),
                "cross": torch.zeros(channels, taps, dtype=torch.float64),
                "target_sum": torch.zeros(channels, dtype=torch.float64),
                "target_square_sum": torch.zeros(channels, dtype=torch.float64),
                "target_count": torch.zeros((), dtype=torch.float64),
            },
        )
        weights = _region_weights(sequence.shape[0], self.endpoints).to(sequence.device)
        spatial_count = sequence[0, 0, 0].numel()
        weight_shape = (sequence.shape[0], sequence.shape[1], 1, *((1,) * (sequence.ndim - 3)))
        expanded = weights.reshape(weight_shape) / spatial_count
        reduce_dims = (0, 1, *range(3, sequence.ndim))
        for left in range(taps):
            stats["cross"][:, left] += (
                (delayed[left] * sequence * expanded).sum(dim=reduce_dims).double().cpu()
            )
            for right in range(taps):
                stats["gram"][:, left, right] += (
                    (delayed[left] * delayed[right] * expanded)
                    .sum(dim=reduce_dims)
                    .double()
                    .cpu()
                )
        stats["target_sum"] += sequence.sum(dim=reduce_dims).double().cpu()
        stats["target_square_sum"] += sequence.square().sum(dim=reduce_dims).double().cpu()
        stats["target_count"] += sequence.shape[0] * sequence.shape[1] * spatial_count

    def _accumulate_validation(
        self, name: str, sequence: torch.Tensor, delayed: list[torch.Tensor]
    ) -> None:
        assert self.endpoints is not None
        coefficients = self.coefficients[name].to(sequence.device, sequence.dtype)
        coefficient_shape = (1, 1, sequence.shape[2], *((1,) * (sequence.ndim - 3)))
        learned = sum(
            value * coefficients[index].reshape(coefficient_shape)
            for index, value in enumerate(delayed)
        )
        predictions = {
            "learned_convex": learned,
            "persistence": delayed[0],
            "uniform_tap_mean": torch.stack(delayed).mean(0),
        }
        scale = self.statistics[name]["target_std"].to(sequence.device, sequence.dtype)
        scale_shape = (1, 1, sequence.shape[2], *((1,) * (sequence.ndim - 3)))
        layer_errors = self.region_errors.setdefault(
            name,
            {
                f"{method}_{region}": []
                for method in predictions
                for region in ("active", "tail")
            },
        )
        learned_by_step = None
        for method, prediction in predictions.items():
            normalized = ((prediction - sequence) / scale.reshape(scale_shape)).square()
            per_step = normalized.flatten(2).mean(2)
            if method == "learned_convex":
                learned_by_step = per_step
            positions = torch.arange(sequence.shape[0], device=sequence.device).unsqueeze(1)
            active = positions < self.endpoints.unsqueeze(0)
            for region, mask in (("active", active), ("tail", ~active)):
                counts = mask.sum(0)
                present = counts > 0
                if bool(present.any().item()):
                    values = (per_step * mask).sum(0) / counts.clamp_min(1)
                    layer_errors[f"{method}_{region}"].append(values[present].cpu())
        assert learned_by_step is not None
        positions = torch.arange(sequence.shape[0], device=sequence.device).unsqueeze(1)
        active = positions < self.endpoints.unsqueeze(0)
        active_count = active.sum(0).clamp_min(1)
        tail_count = (~active).sum(0)
        surprise = (learned_by_step * active).sum(0) / active_count
        has_tail = tail_count > 0
        if bool(has_tail.any().item()):
            tail = (learned_by_step * ~active).sum(0) / tail_count.clamp_min(1)
            surprise = torch.where(has_tail, (surprise + tail) / 2, surprise)
        self.sample_surprise.setdefault(name, []).append(surprise.cpu())


def _fit_convex_tcap_predictors(
    statistics: dict[str, dict[str, torch.Tensor]], steps: int = 400
) -> dict[str, torch.Tensor]:
    coefficients = {}
    for name, stats in statistics.items():
        gram = stats["gram"]
        cross = stats["cross"]
        logits = torch.zeros(gram.shape[1], gram.shape[0], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.05)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            weights = logits.softmax(dim=0).T
            quadratic = torch.einsum("ci,cij,cj->c", weights, gram, weights)
            linear = torch.einsum("ci,ci->c", weights, cross)
            loss = (quadratic - 2 * linear).mean()
            loss.backward()
            optimizer.step()
        coefficients[name] = logits.detach().softmax(dim=0).float()
        count = stats["target_count"].clamp_min(1)
        mean = stats["target_sum"] / count
        variance = stats["target_square_sum"] / count - mean.square()
        stats["target_std"] = variance.clamp_min(1e-10).sqrt().clamp_min(1e-5).float()
    return coefficients


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum().div(denominator.clamp_min(1e-12)).item())


def _partial_correlation(
    surprise: torch.Tensor,
    damage: torch.Tensor,
    amplitude: torch.Tensor,
    event_rate: torch.Tensor,
) -> float:
    design = torch.stack(
        (torch.ones_like(amplitude), amplitude, event_rate), dim=1
    ).double()
    solution_surprise = torch.linalg.lstsq(design, surprise.double().unsqueeze(1)).solution
    solution_damage = torch.linalg.lstsq(design, damage.double().unsqueeze(1)).solution
    residual_surprise = surprise.double() - (design @ solution_surprise).squeeze(1)
    residual_damage = damage.double() - (design @ solution_damage).squeeze(1)
    return _pearson(residual_surprise, residual_damage)


def run_tcap_predictive_probe(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    fit_samples: int = 256,
    holdout_samples: int = 256,
) -> dict[str, Any]:
    """Diagnose causal TCAP predictability and history-ablation damage checkpoint-only."""

    if fit_samples <= 0 or holdout_samples <= 0:
        raise ValueError("TCAP probe sample limits must be positive.")
    probe_config = copy.deepcopy(config)
    probe_config["augmentation"] = {"horizontal_flip_probability": 0.0}
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, True)
    bundle = build_dataset_bundle(probe_config)
    fit_dataset, holdout_dataset = _diagnostic_split(
        bundle.train, fit_samples, holdout_samples, seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint_path, model, device)
    model.eval().requires_grad_(False)

    fit_accumulator = _TCAPPredictiveAccumulator(model, fit=True)
    with torch.no_grad():
        for frames, _targets, _indices in build_loader(
            fit_dataset, probe_config["dataset"], shuffle=False
        ):
            frames = move_encoded_input(frames, device)
            fit_accumulator.endpoints = last_occupied_steps(frames)
            model(frames)
    fit_accumulator.close()
    coefficients = _fit_convex_tcap_predictors(fit_accumulator.statistics)

    ablated = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint_path, ablated, device)
    with torch.no_grad():
        for module in ablated.modules():
            if isinstance(module, CausalTemporalChannelMixer):
                module.weight.zero_()
    ablated.eval().requires_grad_(False)

    holdout_accumulator = _TCAPPredictiveAccumulator(model, fit=False)
    holdout_accumulator.statistics = fit_accumulator.statistics
    holdout_accumulator.coefficients = coefficients
    damages = []
    amplitudes = []
    event_rates = []
    with torch.no_grad():
        for frames, targets, _indices in build_loader(
            holdout_dataset, probe_config["dataset"], shuffle=False
        ):
            frames = move_encoded_input(frames, device)
            targets = targets.to(device)
            holdout_accumulator.endpoints = last_occupied_steps(frames)
            logits = model(frames)
            ablated_logits = ablated(frames)
            full_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            ablated_loss = torch.nn.functional.cross_entropy(
                ablated_logits, targets, reduction="none"
            )
            damages.append((ablated_loss - full_loss).cpu())
            raw = frames if isinstance(frames, torch.Tensor) else frames["coarse"]
            amplitudes.append(raw.float().abs().flatten(1).mean(1).cpu())
            event_rates.append(raw.ne(0).float().flatten(1).mean(1).cpu())
    holdout_accumulator.close()

    layer_reports = {}
    for name, errors in holdout_accumulator.region_errors.items():
        metrics = {}
        for key, chunks in errors.items():
            values = torch.cat(chunks)
            metrics[f"{key}_normalized_mse"] = float(values.mean().item())
        for region in ("active", "tail"):
            learned = metrics.get(f"learned_convex_{region}_normalized_mse")
            persistence = metrics.get(f"persistence_{region}_normalized_mse")
            uniform = metrics.get(f"uniform_tap_mean_{region}_normalized_mse")
            if learned is not None and persistence is not None:
                metrics[f"learned_skill_vs_persistence_{region}"] = 1.0 - learned / persistence
            if learned is not None and uniform is not None:
                metrics[f"learned_skill_vs_uniform_{region}"] = 1.0 - learned / uniform
        metrics["coefficients_by_delay_and_channel"] = coefficients[name].tolist()
        layer_reports[name] = metrics
    surprise = torch.stack(
        [
            torch.cat(holdout_accumulator.sample_surprise[name])
            for name in sorted(holdout_accumulator.sample_surprise)
        ]
    ).mean(0)
    damage = torch.cat(damages)
    amplitude = torch.cat(amplitudes)
    event_rate = torch.cat(event_rates)
    report = {
        "schema_version": 1,
        "fit_samples": fit_samples,
        "holdout_samples": holdout_samples,
        "partition_source": "development_train",
        "disjoint_by_sample": True,
        "layers": layer_reports,
        "history_damage": {
            "mean_cross_entropy_increase": float(damage.mean().item()),
            "surprise_damage_pearson": _pearson(surprise, damage),
            "surprise_damage_partial_correlation_given_amplitude_event_rate": (
                _partial_correlation(surprise, damage, amplitude, event_rate)
            ),
            "amplitude_damage_pearson": _pearson(amplitude, damage),
            "event_rate_damage_pearson": _pearson(event_rate, damage),
            "interpretation_limit": (
                "Zeroing TCAP history in a co-adapted checkpoint is diagnostic and is not the "
                "counterfactual performance of a retrained architecture."
            ),
        },
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_test_used": False,
    }
    output = Path(output_path)
    ensure_dir(output.parent)
    write_json(report, output)
    return report


@torch.no_grad()
def run_dynamic_routing_diagnostic(
    config: dict[str, Any], checkpoint_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Compare learned content-dependent TCAP gates with train-mean constant gates."""

    if not (
        config["model"].get("temporal_channel_mixer_dynamic_routing", False)
        or config["model"].get("temporal_channel_mixer_surprise_routing", False)
    ):
        raise ValueError("Routing diagnostic requires a conditional TCAP checkpoint.")
    seed_everything(int(config["experiment"]["seed"]), True)
    no_aug = copy.deepcopy(config)
    no_aug["augmentation"] = {"horizontal_flip_probability": 0.0}
    bundle = build_dataset_bundle(no_aug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint_path, model, device)
    model.eval().requires_grad_(False)
    gate_sums: dict[str, torch.Tensor] = {}
    contribution_sums: dict[str, torch.Tensor] = {}
    gate_counts: dict[str, int] = {}
    handles = []

    def hook(name: str):
        def collect(module, _inputs, output) -> None:
            stats = module.last_routing_statistics
            if stats is None:
                return
            observations = int(stats["gate_observation_count"])
            values = stats["gate_mean_by_delay"].double().cpu() * observations
            gate_sums[name] = gate_sums.get(name, torch.zeros_like(values)) + values
            contributions = stats.get("effective_contribution_mean_abs_by_delay")
            if contributions is not None:
                contribution_values = contributions.double().cpu() * observations
                contribution_sums[name] = contribution_sums.get(
                    name, torch.zeros_like(contribution_values)
                ) + contribution_values
            gate_counts[name] = gate_counts.get(name, 0) + observations

        return collect

    for name, module in model.named_modules():
        if isinstance(module, CausalTemporalChannelMixer) and (
            module.content_router is not None or module.surprise_router is not None
        ):
            handles.append(module.register_forward_hook(hook(name)))
    for frames, _targets, _indices in build_loader(
        bundle.train, no_aug["dataset"], shuffle=False
    ):
        model(move_encoded_input(frames, device))
    for handle in handles:
        handle.remove()
    gate_means = {name: gate_sums[name] / gate_counts[name] for name in gate_sums}
    contribution_means = {
        name: contribution_sums[name] / gate_counts[name] for name in contribution_sums
    }

    constant = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint_path, constant, device)
    constant.eval().requires_grad_(False)
    for name, module in constant.named_modules():
        if isinstance(module, CausalTemporalChannelMixer) and (
            module.content_router is not None or module.surprise_router is not None
        ):
            for router in (module.content_router, module.surprise_router):
                if router is not None:
                    for parameter in router.parameters():
                        parameter.zero_()
            selected_router = (
                module.content_router
                if module.content_router is not None
                else module.surprise_router
            )
            final = (
                selected_router[-1]
                if isinstance(selected_router, torch.nn.Sequential)
                else selected_router
            )
            mean = gate_means[name].to(final.bias.device, final.bias.dtype)
            if module.routing_parameterization == "independent":
                mean = mean.clamp(1e-5, 2 - 1e-5)
                bias = torch.log(mean / (2 - mean))
            else:
                amplitude = mean.mean().clamp(1e-5, 2 - 1e-5)
                allocation = (mean / mean.sum().clamp_min(1e-12)).clamp_min(1e-12)
                amplitude_logit = torch.log(amplitude / (2 - amplitude))
                bias = torch.cat((amplitude_logit.reshape(1), allocation.log()))
            final.bias.copy_(bias)

    loader = build_loader(bundle.validation, no_aug["dataset"], shuffle=False)
    criterion = make_criterion(config["training"])
    dynamic_result, _ = evaluate(
        model, loader, criterion, device, len(bundle.classes)
    )
    constant_result, _ = evaluate(
        constant, loader, criterion, device, len(bundle.classes)
    )
    report = {
        "schema_version": 1,
        "dynamic": dynamic_result.to_dict(),
        "train_mean_constant": constant_result.to_dict(),
        "delta_dynamic_minus_constant": {
            "accuracy": dynamic_result.accuracy - constant_result.accuracy,
            "macro_f1": dynamic_result.macro_f1 - constant_result.macro_f1,
        },
        "train_gate_mean_by_module": {
            name: values.tolist() for name, values in gate_means.items()
        },
        "train_effective_contribution_mean_abs_by_module": {
            name: values.tolist() for name, values in contribution_means.items()
        },
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_test_used": False,
        "interpretation_limit": (
            "The constant-gate model is a checkpoint-only intervention, not a separately "
            "retrained ablation."
        ),
    }
    output = Path(output_path)
    ensure_dir(output.parent)
    write_json(report, output)
    return report
