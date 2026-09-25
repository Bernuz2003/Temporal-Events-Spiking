"""Checkpoint-only evidence bundle required before predictive phase-1 reruns."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from etsr.config import load_config
from etsr.data.common import DatasetSubset, build_loader
from etsr.data.events import move_encoded_input
from etsr.data.factory import build_dataset_bundle
from etsr.evaluation.metrics import classification_metrics
from etsr.models.factory import build_model
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.reproducibility import git_commit, git_is_dirty, seed_everything
from etsr.training.checkpointing import load_model_state
from etsr.training.engine import make_criterion
from etsr.training.gradient_diagnostics import objective_gradient_diagnostics
from etsr.training.predictive import (
    PredictiveTrainingObjective,
    class_stratified_indices,
    freeze_batchnorm_running_statistics,
    last_occupied_steps,
)
from etsr.utils.io import ensure_dir, sha256_file, write_csv, write_json


def _load_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[dict[str, Any], MiniQKFormer, Any]:
    config = load_config(config_path)
    no_aug = copy.deepcopy(config)
    no_aug["augmentation"] = {"horizontal_flip_probability": 0.0}
    bundle = build_dataset_bundle(no_aug)
    model = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint_path, model, device)
    model.eval()
    return config, model, bundle


def _split_indices(length: int, fit_samples: int, holdout_samples: int, seed: int) -> tuple[list[int], list[int]]:
    required = fit_samples + holdout_samples
    if required > length:
        raise ValueError(f"Phase-1 audit needs {required} train samples, found {length}.")
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed))[:required]
    return indices[:fit_samples].tolist(), indices[fit_samples:].tolist()


def _delayed_history(module: CausalTemporalChannelMixer, sequence: torch.Tensor) -> torch.Tensor:
    state = module.reset_state(sequence[0])
    return torch.cat((state, sequence), dim=0)


def _region_pool(
    sequence: torch.Tensor,
    endpoints: torch.Tensor,
    region: str,
) -> torch.Tensor:
    spatial = sequence.flatten(3).mean(3)
    positions = torch.arange(sequence.shape[0], device=sequence.device).unsqueeze(1)
    mask = positions < endpoints.unsqueeze(0)
    if region == "tail":
        mask = ~mask
    counts = mask.sum(0).clamp_min(1)
    return (spatial * mask.unsqueeze(2)).sum(0) / counts.unsqueeze(1)


@torch.no_grad()
def _predictive_feature_rows(
    model: MiniQKFormer,
    loader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows: dict[str, list[torch.Tensor]] = {}
    labels: list[torch.Tensor] = []
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def hook(name: str, module: CausalTemporalChannelMixer):
        def collect(_module, inputs, _output) -> None:
            sequence = inputs[0]
            prediction = module._causal_prediction(sequence, _delayed_history(module, sequence))
            stage = "stage1" if name.startswith("patch_embed1") else "stage2"
            captured[stage] = (sequence.detach(), prediction.detach())

        return collect

    handles = [
        module.register_forward_hook(hook(name, module))
        for name, module in model.named_modules()
        if isinstance(module, CausalTemporalChannelMixer) and module.predictive_auxiliary
    ]
    if not handles:
        raise ValueError("A2 requires a checkpoint with trained TCAP predictors.")
    try:
        for frames, targets, _indices in loader:
            frames = move_encoded_input(frames, device)
            captured.clear()
            model(frames)
            endpoints = last_occupied_steps(frames)
            for stage, (feature, prediction) in captured.items():
                residual = feature - prediction
                for region in ("active", "tail"):
                    values = {
                        "x": _region_pool(feature, endpoints, region),
                        "x_hat": _region_pool(prediction, endpoints, region),
                        "residual": _region_pool(residual, endpoints, region),
                    }
                    values["concat"] = torch.cat((values["x_hat"], values["residual"]), 1)
                    for component, value in values.items():
                        rows.setdefault(f"{stage}_{region}_{component}", []).append(value.cpu())
            labels.append(targets.cpu())
    finally:
        for handle in handles:
            handle.remove()
    return (
        {name: torch.cat(values).numpy() for name, values in rows.items()},
        torch.cat(labels).numpy(),
    )


def _ridge_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    holdout_features: np.ndarray,
    holdout_targets: np.ndarray,
    num_classes: int,
    ridge: float,
) -> dict[str, Any]:
    mean = train_features.mean(0, keepdims=True)
    std = train_features.std(0, keepdims=True).clip(1e-6)
    train = (train_features - mean) / std
    holdout = (holdout_features - mean) / std
    train = np.concatenate((train, np.ones((len(train), 1))), 1)
    holdout = np.concatenate((holdout, np.ones((len(holdout), 1))), 1)
    targets = np.eye(num_classes, dtype=np.float64)[train_targets]
    penalty = np.eye(train.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train.T @ train / len(train) + penalty, train.T @ targets / len(train))
    predictions = (holdout @ weights).argmax(1)
    result = classification_metrics(holdout_targets, predictions, num_classes)
    return {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "feature_dimensions": int(train_features.shape[1]),
        "ridge": ridge,
    }


def _linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double() - first.double().mean(0, keepdim=True)
    second = second.double() - second.double().mean(0, keepdim=True)
    cross = (first.T @ second).square().sum()
    denominator = ((first.T @ first).square().sum() * (second.T @ second).square().sum()).sqrt()
    return float((cross / denominator.clamp_min(1e-12)).item())


@torch.no_grad()
def _shared_feature_rows(
    model: MiniQKFormer,
    loader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {}
    captured: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def collect(_module, inputs, _output) -> None:
            stage = "stage1" if name.startswith("patch_embed1") else "stage2"
            captured[stage] = inputs[0].detach()

        return collect

    handles = [
        module.register_forward_hook(hook(name))
        for name, module in model.named_modules()
        if isinstance(module, CausalTemporalChannelMixer)
    ]
    try:
        for frames, _targets, _indices in loader:
            frames = move_encoded_input(frames, device)
            captured.clear()
            model(frames)
            endpoints = last_occupied_steps(frames)
            for stage, feature in captured.items():
                for region in ("active", "tail"):
                    rows.setdefault(f"{stage}_{region}", []).append(
                        _region_pool(feature, endpoints, region).cpu()
                    )
    finally:
        for handle in handles:
            handle.remove()
    return {stage: torch.cat(values) for stage, values in rows.items()}


A1_DIAGNOSTIC_BATCHES = 4


def _gradient_authority(
    config: dict[str, Any],
    model: MiniQKFormer,
    bundle,
    device: torch.device,
) -> dict[str, Any]:
    """Post-mortem of the archived S0 objective on class-stratified batches.

    DVS-Lip samples are sorted by class: an unshuffled first batch contains a single word, and
    the CE gradient on it describes that word only. A1 therefore averages several disjoint
    batches that together span ``batches x batch_size`` distinct classes.
    """

    continuation = config.get("continuation")
    predictive = config.get("predictive")
    source = predictive if isinstance(predictive, dict) else continuation
    if not isinstance(source, dict) or "objective" not in source:
        raise ValueError("A1 requires the S0 objective configuration.")
    objective_config = copy.deepcopy(source["objective"])
    # Reconstruct the archived S0 objective exactly: active/tail and both stages had equal weight.
    objective_config["temporal_region_weights"] = {"active": 1.0, "tail": 1.0}
    objective_config["temporal_stage_weights"] = {"stage1": 1.0, "stage2": 1.0}
    objective_config.pop("authority", None)
    objective = PredictiveTrainingObjective(
        objective_config,
        None,
        {**source, **(continuation or {}), "representation": config["representation"]},
    )
    batch_size = int(config["dataset"].get("batch_size", 16))
    indices = class_stratified_indices(bundle.train, batch_size * A1_DIAGNOSTIC_BATCHES)
    criterion = make_criterion(config["training"])
    per_batch: list[dict[str, Any]] = []
    classes: set[int] = set()
    for offset in range(0, len(indices), batch_size):
        chunk = indices[offset : offset + batch_size]
        loader = build_loader(
            DatasetSubset(bundle.train, chunk),
            {**config["dataset"], "batch_size": len(chunk), "num_workers": 0},
            shuffle=False,
        )
        frames, targets, _indices = next(iter(loader))
        frames = move_encoded_input(frames, device)
        targets = targets.to(device)
        classes.update(int(value) for value in targets.tolist())
        model.train()
        freeze_batchnorm_running_statistics(model)
        result = objective(model, frames, targets, criterion, max(1, objective.ramp_epochs))
        regions = {
            region: objective_gradient_diagnostics(
                model, result.classification_loss, component, objective.weight
            )
            for region, component in result.auxiliary_components.items()
            if component.requires_grad
        }
        model.zero_grad(set_to_none=True)
        per_batch.append({"samples": int(targets.numel()), "regions": regions})
    model.eval()
    region_names = sorted({region for batch in per_batch for region in batch["regions"]})
    mean_regions: dict[str, dict[str, float]] = {}
    for region in region_names:
        rows = [batch["regions"][region] for batch in per_batch if region in batch["regions"]]
        mean_regions[region] = {
            key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]
        }
    return {
        "batch_selection": "class_stratified_disjoint_batches",
        "batches": len(per_batch),
        "batch_samples": batch_size,
        "distinct_classes": len(classes),
        "nominal_auxiliary_weight": objective.weight,
        "reconstructed_archived_region_weights": objective.temporal_region_weights,
        "reconstructed_archived_stage_weights": objective.temporal_stage_weights,
        "regions": mean_regions,
        "per_batch": per_batch,
        "measured": bool(mean_regions),
    }


def _parameter_and_feature_movement(
    r0: MiniQKFormer,
    s0: MiniQKFormer,
    loader,
    device: torch.device,
) -> dict[str, Any]:
    r0_parameters = dict(r0.named_parameters())
    displacements = {}
    excluded = []
    for name, parameter in s0.named_parameters():
        if name not in r0_parameters or parameter.shape != r0_parameters[name].shape:
            excluded.append(name)
            continue
        reference = r0_parameters[name].detach().double()
        relative = (parameter.detach().double() - reference).norm() / reference.norm().clamp_min(1e-12)
        displacements[name] = float(relative.cpu())
    r0_features = _shared_feature_rows(r0, loader, device)
    s0_features = _shared_feature_rows(s0, loader, device)
    feature_metrics = {}
    for stage in sorted(set(r0_features) & set(s0_features)):
        first = r0_features[stage]
        second = s0_features[stage]
        feature_metrics[stage] = {
            "linear_cka": _linear_cka(first, second),
            "mean_paired_cosine": float(
                nn.functional.cosine_similarity(first.float(), second.float(), dim=1).mean().item()
            ),
            "samples": len(first),
        }
    return {
        "relative_parameter_displacement": displacements,
        "excluded_nonshared_parameters": excluded,
        "features": feature_metrics,
    }


@torch.no_grad()
def _tail_margin_audit(
    model: MiniQKFormer,
    bundle,
    config: dict[str, Any],
    device: torch.device,
    output_rows: Path,
    collect_temporal_diagnostics: bool = False,
) -> dict[str, Any]:
    loader = build_loader(bundle.validation, config["dataset"], shuffle=False)
    class_groups_path = config.get("evaluation", {}).get("class_groups_manifest")
    class_groups = json.loads(Path(class_groups_path).read_text(encoding="utf-8")) if class_groups_path else {}
    acc1 = set(class_groups.get("visually_confusable_words", ()))
    rows = []
    totals: dict[str, dict[str, float]] = {}
    temporal_sums: dict[str, float] = {}
    temporal_samples = 0
    for frames, targets, indices in loader:
        frames = move_encoded_input(frames, device)
        encoded = model._encode(frames)
        if collect_temporal_diagnostics:
            batch_size = int(targets.numel())
            temporal = model.temporal_prediction_statistics(last_occupied_steps(frames))
            for name, value in temporal.items():
                temporal_sums[name] = temporal_sums.get(name, 0.0) + value * batch_size
            temporal_samples += batch_size
        spatial = encoded.flatten(3).mean(3)
        if spatial.shape[0] < 40:
            raise ValueError("A4 requires the 40-step DVS-Lip representation.")
        logits30 = model.head(spatial[:30].mean(0))
        logits40 = model.head(spatial[:40].mean(0))
        logits_tail = model.head(spatial[30:40].mean(0))
        exact = (logits40 - (0.75 * logits30 + 0.25 * logits_tail)).abs().max()
        if float(exact.item()) > 1e-5:
            raise RuntimeError("Tail logit decomposition is not numerically exact.")
        for batch_index, sample_index in enumerate(indices.tolist()):
            target = int(targets[batch_index])
            true30 = float(logits30[batch_index, target])
            true40 = float(logits40[batch_index, target])
            competitor30_tensor = logits30[batch_index].clone()
            competitor40_tensor = logits40[batch_index].clone()
            competitor30_tensor[target] = float("-inf")
            competitor40_tensor[target] = float("-inf")
            competitor30 = float(competitor30_tensor.max())
            competitor40 = float(competitor40_tensor.max())
            prediction30 = int(logits30[batch_index].argmax())
            prediction40 = int(logits40[batch_index].argmax())
            class_name = bundle.classes[target]
            group = "Acc1" if class_name in acc1 else "Acc2"
            row = {
                "index": sample_index,
                "target": target,
                "class": class_name,
                "group": group,
                "prediction_1500ms": prediction30,
                "prediction_2000ms": prediction40,
                "correct_1500ms": int(prediction30 == target),
                "correct_2000ms": int(prediction40 == target),
                "true_logit_1500ms": true30,
                "true_logit_2000ms": true40,
                "delta_true_logit": true40 - true30,
                "best_competitor_1500ms": competitor30,
                "best_competitor_2000ms": competitor40,
                "delta_best_competitor": competitor40 - competitor30,
                "margin_1500ms": true30 - competitor30,
                "margin_2000ms": true40 - competitor40,
                "delta_margin": (true40 - competitor40) - (true30 - competitor30),
            }
            rows.append(row)
            aggregate = totals.setdefault(group, {key: 0.0 for key in (
                "samples", "correct_1500ms", "correct_2000ms", "delta_true_logit",
                "delta_best_competitor", "delta_margin"
            )})
            aggregate["samples"] += 1
            for key in aggregate:
                if key != "samples":
                    aggregate[key] += float(row[key])
    write_csv(rows, output_rows)
    summaries = {}
    for group in ("all", *totals):
        selected = rows if group == "all" else [row for row in rows if row["group"] == group]
        summaries[group] = {
            "samples": len(selected),
            "accuracy_1500ms": sum(row["correct_1500ms"] for row in selected) / max(1, len(selected)),
            "accuracy_2000ms": sum(row["correct_2000ms"] for row in selected) / max(1, len(selected)),
            "mean_delta_true_logit": float(np.mean([row["delta_true_logit"] for row in selected])),
            "mean_delta_best_competitor": float(np.mean([row["delta_best_competitor"] for row in selected])),
            "mean_delta_margin": float(np.mean([row["delta_margin"] for row in selected])),
        }
        if group == "all":
            targets = np.asarray([row["target"] for row in selected])
            predictions30 = np.asarray([row["prediction_1500ms"] for row in selected])
            predictions40 = np.asarray([row["prediction_2000ms"] for row in selected])
            summaries[group]["macro_f1_1500ms"] = classification_metrics(
                targets, predictions30, len(bundle.classes)
            )["macro_f1"]
            summaries[group]["macro_f1_2000ms"] = classification_metrics(
                targets, predictions40, len(bundle.classes)
            )["macro_f1"]
    result: dict[str, Any] = {
        "groups": summaries,
        "per_sample_csv": str(output_rows.resolve()),
    }
    if collect_temporal_diagnostics:
        temporal_diagnostics = {
            name: value / max(1, temporal_samples)
            for name, value in temporal_sums.items()
        }
        result["temporal_diagnostics"] = temporal_diagnostics
        result["temporal_variation"] = {
            name: value
            for name, value in temporal_diagnostics.items()
            if "temporal_variation" in name
        }
    return result


def run_predictive_checkpoint_audit(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Measure A4 tail margins and active-region temporal variation on one checkpoint."""

    output = ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42, True)
    config, model, bundle = _load_model(config_path, checkpoint_path, device)
    mixers = [
        module for module in model.modules() if isinstance(module, CausalTemporalChannelMixer)
    ]
    previous_recording = [module.record_temporal_variation for module in mixers]
    try:
        for module in mixers:
            module.record_temporal_variation = True
        diagnostic = _tail_margin_audit(
            model,
            bundle,
            config,
            device,
            output / "a4_per_sample.csv",
            collect_temporal_diagnostics=True,
        )
    finally:
        for module, enabled in zip(mixers, previous_recording, strict=True):
            module.record_temporal_variation = enabled
    report = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(Path(checkpoint_path).resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "config": str(Path(config_path).resolve()),
        "A4_tail_margin": diagnostic,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "official_test_used": False,
    }
    write_json(report, output / "predictive_checkpoint_audit.json")
    return report


def run_predictive_phase1_audit(
    *,
    c0_config: str | Path,
    c0_checkpoint: str | Path,
    r0_config: str | Path,
    r0_checkpoint: str | Path,
    s0_config: str | Path,
    s0_checkpoint: str | Path,
    output_dir: str | Path,
    fit_samples: int = 8192,
    holdout_samples: int = 2048,
    feature_samples: int = 256,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Produce A1--A4 from frozen checkpoints without touching the official test set."""

    if min(fit_samples, holdout_samples, feature_samples) <= 0 or ridge <= 0:
        raise ValueError("Audit sample counts and ridge must be positive.")
    if fit_samples != 8192 or holdout_samples != 2048:
        raise ValueError("Phase-1 audit schema 2 requires exactly 8192 fit and 2048 holdout samples.")
    output = ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42, True)
    c0_cfg, c0, c0_bundle = _load_model(c0_config, c0_checkpoint, device)
    r0_cfg, r0, r0_bundle = _load_model(r0_config, r0_checkpoint, device)
    s0_cfg, s0, s0_bundle = _load_model(s0_config, s0_checkpoint, device)
    if c0_bundle.classes != r0_bundle.classes or c0_bundle.classes != s0_bundle.classes:
        raise ValueError("Audit checkpoints use different class vocabularies.")

    fit_indices, holdout_indices = _split_indices(
        len(s0_bundle.train), fit_samples, holdout_samples, 42
    )
    fit_loader = build_loader(DatasetSubset(s0_bundle.train, fit_indices), s0_cfg["dataset"], shuffle=False)
    holdout_loader = build_loader(DatasetSubset(s0_bundle.train, holdout_indices), s0_cfg["dataset"], shuffle=False)
    fit_features, fit_targets = _predictive_feature_rows(s0, fit_loader, device)
    holdout_features, holdout_targets = _predictive_feature_rows(s0, holdout_loader, device)
    probes = {
        name: _ridge_probe(
            fit_features[name], fit_targets, holdout_features[name], holdout_targets,
            len(s0_bundle.classes), ridge,
        )
        for name in sorted(set(fit_features) & set(holdout_features))
    }

    movement_indices = fit_indices[: min(feature_samples, len(fit_indices))]
    movement_loader = build_loader(
        DatasetSubset(s0_bundle.train, movement_indices), s0_cfg["dataset"], shuffle=False
    )
    a3 = _parameter_and_feature_movement(r0, s0, movement_loader, device)
    a4 = {
        "r0": _tail_margin_audit(
            r0, r0_bundle, r0_cfg, device, output / "a4_r0_per_sample.csv"
        ),
        "c0": _tail_margin_audit(
            c0, c0_bundle, c0_cfg, device, output / "a4_c0_per_sample.csv"
        ),
    }
    a1 = _gradient_authority(s0_cfg, s0, s0_bundle, device)
    if not a1["measured"]:
        raise RuntimeError("A1 produced no auxiliary gradient family; audit is incomplete.")
    report = {
        "schema_version": 2,
        "complete": True,
        "sections": ["A1_gradient_authority", "A2_discriminative_probes", "A3_representation_movement", "A4_tail_margin"],
        "A1_gradient_authority": a1,
        "A2_discriminative_probes": {
            "fit_partition": "development_train",
            "holdout_partition": "development_train_disjoint_by_sample",
            "fit_samples": fit_samples,
            "holdout_samples": holdout_samples,
            "probes": probes,
        },
        "A3_representation_movement": a3,
        "A4_tail_margin": a4,
        "checkpoints": {
            "c0": {"path": str(Path(c0_checkpoint).resolve()), "sha256": sha256_file(c0_checkpoint)},
            "r0": {"path": str(Path(r0_checkpoint).resolve()), "sha256": sha256_file(r0_checkpoint)},
            "s0": {"path": str(Path(s0_checkpoint).resolve()), "sha256": sha256_file(s0_checkpoint)},
        },
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "official_test_used": False,
    }
    write_json(report, output / "phase1_audit.json")
    return report
