"""Checkpoint-only probes for cross-resolution temporal predictability."""

from __future__ import annotations

import copy
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
from etsr.reproducibility import seed_everything
from etsr.training.checkpointing import load_model_state
from etsr.training.engine import evaluate, make_criterion
from etsr.training.predictive import (
    PredictiveTrainingObjective,
    freeze_batchnorm_running_statistics,
    last_occupied_steps,
)
from etsr.utils.io import ensure_dir, sha256_file, write_json


def _metric_summary(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    mse = error.square().mean()
    variance = (target - target.mean(0, keepdim=True)).square().mean().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=1).mean()
    scale = target.square().mean().sqrt().clamp_min(1e-12)
    return {
        "r2_variance_normalized": float((1.0 - mse / variance).item()),
        "nrmse_rms_target": float((mse.sqrt() / scale).item()),
        "cosine_similarity": float(cosine.item()),
        "mse": float(mse.item()),
    }


@torch.no_grad()
def _extract_pairs(
    student: MiniQKFormer,
    teacher: MiniQKFormer,
    loader,
    device: torch.device,
    mode: str,
    horizon: int,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, int]:
    contexts = []
    targets = []
    persistence = []
    target_sum = None
    target_square_sum = None
    target_count = 0
    samples = 0
    for frames, _labels, _indices in loader:
        if samples >= max_samples:
            break
        frames = move_encoded_input(frames, device)
        if not isinstance(frames, dict) and mode.startswith("fine_"):
            raise ValueError("Fine probe requires a multi-granular representation.")
        context = student.extract_stage1(frames).mean(dim=(3, 4))
        if mode.startswith("fine_"):
            assert isinstance(frames, dict)
            target_features = teacher.extract_fine_pre_lif(frames)
        else:
            target_features = teacher.extract_stage1(frames)
        target_sequence = target_features.mean(dim=(3, 4))
        endpoint = last_occupied_steps(frames)
        for batch_index in range(context.shape[1]):
            if samples >= max_samples:
                break
            stop = min(context.shape[0] - horizon, int(endpoint[batch_index].item()) - horizon)
            if stop <= 0:
                continue
            contexts.append(context[:stop, batch_index].cpu())
            sample_target = target_sequence[horizon : horizon + stop, batch_index]
            targets.append(sample_target.cpu())
            persistence.append(target_sequence[:stop, batch_index].cpu())
            sample_target_features = target_features[
                horizon : horizon + stop, batch_index
            ]
            channel_sum = sample_target_features.sum(dim=(0, 2, 3)).double().cpu()
            channel_square_sum = (
                sample_target_features.square().sum(dim=(0, 2, 3)).double().cpu()
            )
            target_sum = channel_sum if target_sum is None else target_sum + channel_sum
            target_square_sum = (
                channel_square_sum
                if target_square_sum is None
                else target_square_sum + channel_square_sum
            )
            target_count += (
                sample_target_features.shape[0]
                * sample_target_features.shape[2]
                * sample_target_features.shape[3]
            )
            samples += 1
    if not contexts:
        raise ValueError("No valid predictive pairs were extracted.")
    assert target_sum is not None and target_square_sum is not None
    return (
        torch.cat(contexts),
        torch.cat(targets),
        torch.cat(persistence),
        samples,
        target_sum,
        target_square_sum,
        target_count,
    )


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


def _fit_ridge(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    x_mean, x_std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-5)
    y_mean, y_std = train_y.mean(0), train_y.std(0, unbiased=False).clamp_min(1e-5)
    normalized_x = (train_x - x_mean) / x_std
    normalized_y = (train_y - y_mean) / y_std
    design = torch.cat((normalized_x, torch.ones(len(normalized_x), 1)), dim=1).double()
    target = normalized_y.double()
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * ridge
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
    validation_design = torch.cat(
        (((validation_x - x_mean) / x_std), torch.ones(len(validation_x), 1)), dim=1
    ).double()
    prediction = (validation_design @ weights).float() * y_std + y_mean
    normalization = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    return prediction, normalization


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
    if not isinstance(continuation, dict):
        raise ValueError("Predictive probe requires a continuation configuration.")
    objective = continuation["objective"]
    mode = str(objective["mode"])
    if mode not in {"fine_future", "coarse_future"}:
        raise ValueError("Probe supports fine_future or coarse_future configurations.")
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
    teacher_config = load_config(continuation["teacher_config"])
    teacher = build_model(teacher_config["model"], len(bundle.classes)).to(device)
    load_model_state(continuation["teacher_checkpoint"], teacher, device)
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
    (
        train_x,
        train_y,
        _train_persistence,
        train_samples,
        target_sum,
        target_square_sum,
        target_count,
    ) = _extract_pairs(
        student, teacher, train_loader, device, mode, horizon, max_train_samples
    )
    (
        validation_x,
        validation_y,
        validation_persistence,
        validation_samples,
        _holdout_sum,
        _holdout_square_sum,
        _holdout_count,
    ) = _extract_pairs(
        student,
        teacher,
        validation_loader,
        device,
        mode,
        horizon,
        max_validation_samples,
    )
    prediction, normalization = _fit_ridge(train_x, train_y, validation_x, ridge)
    mean_prediction = train_y.mean(0, keepdim=True).expand_as(validation_y)
    target_mean = target_sum / target_count
    target_variance = target_square_sum / target_count - target_mean.square()
    target_std = target_variance.clamp_min(1e-10).sqrt().clamp_min(1e-5)
    report = {
        "schema_version": 1,
        "mode": mode,
        "horizon_steps": horizon,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "train_pairs": len(train_x),
        "validation_pairs": len(validation_x),
        "ridge": ridge,
        "linear_probe": _metric_summary(prediction, validation_y),
        "train_mean_baseline": _metric_summary(mean_prediction, validation_y),
        "privileged_fine_persistence_baseline": _metric_summary(
            validation_persistence, validation_y
        ),
        "normalization_fit_on": "development_train_only",
        "normalization": normalization,
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
        "student_checkpoint": str(Path(continuation["parent_checkpoint"]).resolve()),
        "student_checkpoint_sha256": sha256_file(continuation["parent_checkpoint"]),
        "teacher_checkpoint": str(Path(continuation["teacher_checkpoint"]).resolve()),
        "teacher_checkpoint_sha256": sha256_file(continuation["teacher_checkpoint"]),
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


def _load_continuation_model(
    config: dict[str, Any], num_classes: int, device: torch.device
) -> tuple[MiniQKFormer, dict[str, Any]]:
    model = build_model(config["model"], num_classes).to(device)
    checkpoint = torch.load(
        config["continuation"]["parent_checkpoint"],
        map_location=device,
        weights_only=False,
    )
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = ("predictive_head.", ".content_router.", ".predictor_logits", ".surprise_router")
    invalid = [key for key in incompatible.missing_keys if not any(item in key for item in allowed)]
    if invalid or incompatible.unexpected_keys:
        raise ValueError(
            f"Continuation topology mismatch: missing={invalid}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model, checkpoint


def run_predictive_preflight(
    config: dict[str, Any], output_path: str | Path
) -> dict[str, Any]:
    """Check parent equivalence, causal prefixes and gradients on one validation batch."""

    continuation = config.get("continuation")
    if not isinstance(continuation, dict):
        raise ValueError("Predictive preflight requires a continuation configuration.")
    seed_everything(int(config["experiment"]["seed"]), True)
    no_aug = copy.deepcopy(config)
    no_aug["augmentation"] = {"horizontal_flip_probability": 0.0}
    bundle = build_dataset_bundle(no_aug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate, parent_checkpoint = _load_continuation_model(config, len(bundle.classes), device)
    parent_config = load_config(continuation["parent_config"])
    parent = build_model(parent_config["model"], len(bundle.classes)).to(device)
    load_model_state(continuation["parent_checkpoint"], parent, device)

    teacher = None
    if str(continuation["objective"].get("mode", "none")) != "none":
        teacher_config = load_config(continuation["teacher_config"])
        teacher = build_model(teacher_config["model"], len(bundle.classes)).to(device)
        load_model_state(continuation["teacher_checkpoint"], teacher, device)
        teacher.eval().requires_grad_(False)
    objective = PredictiveTrainingObjective(continuation["objective"], teacher)
    loader = build_loader(bundle.validation, no_aug["dataset"], shuffle=False)
    frames, targets, _indices = next(iter(loader))
    frames = move_encoded_input(frames, device)
    targets = targets.to(device)

    parent.eval()
    candidate.eval()
    with torch.no_grad():
        parent_logits = parent(frames)
        candidate_logits = candidate(frames)
    initial_difference = float((candidate_logits - parent_logits).abs().max().item())

    candidate.train()
    freeze_batchnorm_running_statistics(candidate)
    cutoff = max(1, candidate.extract_stage1(frames).shape[0] // 2)
    if isinstance(frames, dict):
        altered = {name: value.clone() for name, value in frames.items()}
        coarse_steps = altered["coarse"].shape[1]
        ratio = altered["fine"].shape[1] // coarse_steps
        altered["coarse"][:, cutoff:] = torch.randn_like(altered["coarse"][:, cutoff:])
        altered["fine"][:, cutoff * ratio :] = torch.randn_like(
            altered["fine"][:, cutoff * ratio :]
        )
    else:
        altered = frames.clone()
        altered[:, cutoff:] = torch.randn_like(altered[:, cutoff:])
    with torch.no_grad():
        prefix = candidate.extract_stage1(frames)[:cutoff]
        altered_prefix = candidate.extract_stage1(altered)[:cutoff]
    causal_difference = float((prefix - altered_prefix).abs().max().item())

    candidate.zero_grad(set_to_none=True)
    result = objective(
        candidate,
        frames,
        targets,
        torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        epoch=max(1, int(continuation["objective"].get("ramp_epochs", 0))),
    )
    result.total_loss.backward()
    diagnostic_tokens = ("predictive_head", "content_router", "predictor_logits", "surprise_router")
    gradient_norms = {
        name: (float(parameter.grad.norm().item()) if parameter.grad is not None else None)
        for name, parameter in candidate.named_parameters()
        if any(token in name for token in diagnostic_tokens)
    }
    finite_gradients = all(
        value is not None and torch.isfinite(parameter.grad).all().item()
        for name, parameter in candidate.named_parameters()
        if any(token in name for token in diagnostic_tokens)
        for value in [gradient_norms[name]]
    )
    teacher_gradients_absent = teacher is None or all(
        parameter.grad is None for parameter in teacher.parameters()
    )
    report = {
        "schema_version": 1,
        "experiment": config["experiment"]["name"],
        "parent_checkpoint": str(Path(continuation["parent_checkpoint"]).resolve()),
        "parent_checkpoint_sha256": sha256_file(continuation["parent_checkpoint"]),
        "parent_epoch": int(parent_checkpoint["epoch"]),
        "initial_logits_max_abs_difference": initial_difference,
        "initial_function_preserved": initial_difference <= 1e-6,
        "causal_prefix_max_abs_difference": causal_difference,
        "causal_prefix_passed": causal_difference <= 1e-6,
        "total_loss_finite": bool(torch.isfinite(result.total_loss).item()),
        "objective_metrics": result.metrics,
        "new_parameter_gradient_norms": gradient_norms,
        "new_parameter_gradients_finite": finite_gradients,
        "teacher_gradients_absent": teacher_gradients_absent,
        "batchnorm_running_statistics": "fixed",
        "official_test_used": False,
    }
    report["passed"] = all(
        (
            report["initial_function_preserved"],
            report["causal_prefix_passed"],
            report["total_loss_finite"],
            report["new_parameter_gradients_finite"],
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

    if not config["model"].get("temporal_channel_mixer_dynamic_routing", False):
        raise ValueError("Dynamic-routing diagnostic requires a dynamic TCAP checkpoint.")
    seed_everything(int(config["experiment"]["seed"]), True)
    no_aug = copy.deepcopy(config)
    no_aug["augmentation"] = {"horizontal_flip_probability": 0.0}
    bundle = build_dataset_bundle(no_aug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint_path, model, device)
    model.eval().requires_grad_(False)
    gate_sums: dict[str, torch.Tensor] = {}
    gate_counts: dict[str, int] = {}
    handles = []

    def hook(name: str):
        def collect(module, _inputs, output) -> None:
            stats = module.last_routing_statistics
            if stats is None:
                return
            observations = output.shape[0] * output.shape[1]
            values = stats["gate_mean_by_delay"].double().cpu() * observations
            gate_sums[name] = gate_sums.get(name, torch.zeros_like(values)) + values
            gate_counts[name] = gate_counts.get(name, 0) + observations

        return collect

    for name, module in model.named_modules():
        if isinstance(module, CausalTemporalChannelMixer) and module.content_router is not None:
            handles.append(module.register_forward_hook(hook(name)))
    for frames, _targets, _indices in build_loader(
        bundle.train, no_aug["dataset"], shuffle=False
    ):
        model(move_encoded_input(frames, device))
    for handle in handles:
        handle.remove()
    gate_means = {name: gate_sums[name] / gate_counts[name] for name in gate_sums}

    constant = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint_path, constant, device)
    constant.eval().requires_grad_(False)
    for name, module in constant.named_modules():
        if isinstance(module, CausalTemporalChannelMixer) and module.content_router is not None:
            mean = gate_means[name].clamp(1e-5, 2 - 1e-5).to(
                module.content_router.bias.device,
                module.content_router.bias.dtype,
            )
            module.content_router.weight.zero_()
            module.content_router.bias.copy_(torch.log(mean / (2 - mean)))

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
