from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from etsr.config import load_config, save_config
from etsr.data.common import DatasetBundle, DatasetSubset, balanced_overfit_bundle, build_loader
from etsr.data.factory import build_dataset_bundle
from etsr.evaluation.metrics import (
    ClassificationResult,
    grouped_accuracies,
    interval_normalized_auc,
    paired_confusions,
    trapezoidal_auc,
)
from etsr.models.factory import build_model
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.reproducibility import (
    collect_environment,
    git_commit,
    git_is_dirty,
    seed_everything,
)
from etsr.training.augmentation import build_batch_augmentation
from etsr.training.checkpointing import (
    load_model_state,
    load_training_state,
    save_checkpoint,
    save_deployment_checkpoint,
    save_training_state,
)
from etsr.training.engine import (
    evaluate,
    make_criterion,
    make_optimizer,
    make_scheduler,
    restore_best_model,
    train_one_epoch,
)
from etsr.training.gates import overfit_gate
from etsr.training.predictive import PredictiveTrainingObjective
from etsr.utils.io import append_csv, ensure_dir, sha256_file, write_csv, write_json
from etsr.utils.logging import configure_logging


def _run_id(config: dict[str, Any], seed: int) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{config['experiment']['name']}__{timestamp}__seed{seed}"


def _delay_trajectory_rows(
    modules: dict[str, CausalTemporalChannelMixer], epoch: int
) -> list[dict[str, float | int | str]]:
    rows = []
    for name, module in modules.items():
        centers = module.delay_centers.detach().cpu()
        hard = module.discrete_delays().cpu()
        for branch, initial_delay in enumerate(module.delays):
            branch_centers = centers[branch]
            rows.append(
                {
                    "epoch": epoch,
                    "module": name,
                    "branch": branch,
                    "temperature": module.delay_temperature,
                    "initial_delay": initial_delay,
                    "mean_center": float(branch_centers.mean().item()),
                    "mean_absolute_shift_bins": float(
                        (branch_centers - initial_delay).abs().mean().item()
                    ),
                    "fraction_hard_changed": float(
                        (hard[branch] != initial_delay).float().mean().item()
                    ),
                    "mean_rounding_distance_bins": float(
                        (branch_centers - hard[branch]).abs().mean().item()
                    ),
                }
            )
    return rows


def _readout_metadata(config: dict[str, Any]) -> dict[str, str]:
    readout_time = str(config["model"].get("readout_time", "fixed_window"))
    return {
        "name": str(config["model"].get("readout", "mean")),
        "time": readout_time,
        "endpoint_knowledge": "none",
        "tail_policy": (
            "latest_event_snapshot" if readout_time == "last_event" else "process_full_window"
        ),
    }


def _prepare_run(
    config: dict[str, Any], seed: int, resume: dict[str, Any] | None
) -> tuple[str, Path, Path, logging.Logger]:
    if resume is not None:
        run_id = str(resume["run_id"])
        artifact_dir = Path(resume["artifact_dir"])
        checkpoint_dir = Path(resume["checkpoint_path"]).parent
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"Resume artifact directory is missing: {artifact_dir}")
        return run_id, artifact_dir, checkpoint_dir, configure_logging(artifact_dir / "run.log")
    run_id = _run_id(config, seed)
    artifact_dir = ensure_dir(Path(config["experiment"]["artifact_root"]) / run_id)
    checkpoint_dir = ensure_dir(Path(config["experiment"]["checkpoint_root"]) / run_id)
    return run_id, artifact_dir, checkpoint_dir, configure_logging(artifact_dir / "run.log")


def _config_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if not key.startswith("_") and key != "runtime"
    }
    contract.get("model", {}).pop("num_classes", None)
    return contract


def _checkpoint_evaluation_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match to evaluate old weights under a current metric definition."""

    contract = {
        key: copy.deepcopy(config[key])
        for key in ("dataset", "representation", "augmentation", "model", "training")
    }
    contract["model"].pop("num_classes", None)
    return contract


def _load_checkpoint_context(
    config: dict[str, Any], checkpoint_path: str | Path
) -> tuple[Path, dict[str, Any], DatasetBundle, nn.Module, torch.device, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model", "epoch", "score", "config", "num_classes"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "Operation requires a validation-selected best.pt checkpoint; "
            f"missing: {sorted(missing)}"
        )
    if _checkpoint_evaluation_contract(config) != _checkpoint_evaluation_contract(
        checkpoint["config"]
    ):
        raise ValueError("Config changes the checkpoint data, model or training contract.")

    seed_everything(
        int(config["experiment"]["seed"]),
        bool(config["experiment"].get("deterministic", True)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataset_bundle(config)
    num_classes = len(bundle.classes)
    if int(checkpoint["num_classes"]) != num_classes:
        raise ValueError("Checkpoint class count differs from the configured dataset.")
    checkpoint_runtime = checkpoint["config"].get("runtime", {})
    requested_run = config.get("runtime", {}).get("run_id")
    if requested_run is not None and checkpoint_runtime.get("run_id") != requested_run:
        raise ValueError("Resolved config and checkpoint belong to different runs.")
    if checkpoint_runtime.get("classes") != bundle.classes:
        raise ValueError("Checkpoint class order differs from the configured dataset.")
    dataset_metadata = getattr(bundle.validation, "runtime_metadata", {})
    for name, value in dataset_metadata.items():
        if name.endswith("_sha256") and checkpoint_runtime.get(name) != value:
            raise ValueError(f"Checkpoint and configured dataset differ in {name}.")
    model = build_model(config["model"], num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint_path, checkpoint, bundle, model, device, dataset_metadata


def _evaluate_prefixes(
    config: dict[str, Any],
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    full_validation: ClassificationResult,
) -> dict[str, tuple[list[dict[str, float | int]], dict[str, Any]]] | None:
    evaluation = config.get("evaluation", {})
    absolute_times_us = evaluation.get("absolute_prefix_times_us")
    relative_fractions = evaluation.get("relative_prefix_fractions")
    if absolute_times_us is None and relative_fractions is None:
        return None

    representation = config["representation"]
    bin_width_us = int(representation["bin_width_us"])
    total_steps = int(representation["window_us"]) // bin_width_us
    outputs: dict[str, tuple[list[dict[str, float | int]], dict[str, Any]]] = {}

    if absolute_times_us is not None:
        observed_steps = [
            max(1, math.ceil(int(requested_us) / bin_width_us))
            for requested_us in absolute_times_us
        ]
        if len(observed_steps) != len(set(observed_steps)):
            raise ValueError("Absolute prefix times resolve to duplicate encoded time steps.")

        rows: list[dict[str, float | int]] = []
        for requested_us, steps in zip(absolute_times_us, observed_steps, strict=True):
            if steps == total_steps:
                result = full_validation
            else:
                result, _ = evaluate(
                    model,
                    loader,
                    criterion,
                    device,
                    num_classes,
                    prefix_steps=steps,
                )
            rows.append(
                {
                    "requested_us": int(requested_us),
                    "observed_us": steps * bin_width_us,
                    "observed_time_steps": steps,
                    "accuracy": result.accuracy,
                    "macro_f1": result.macro_f1,
                    "loss": result.loss,
                }
            )
        points = [float(row["observed_us"]) for row in rows]
        accuracies = [float(row["accuracy"]) for row in rows]
        outputs["absolute_time"] = (
            rows,
            {
                "axis": "physical_time_us",
                "endpoint_knowledge": "none",
                "accuracy_auc_us": trapezoidal_auc(points, accuracies),
                "accuracy_auc_normalized": interval_normalized_auc(points, accuracies),
                "measured_interval_us": [int(points[0]), int(points[-1])],
                "point_count": len(rows),
            },
        )

    if relative_fractions is not None:
        raw_dataset = getattr(loader.dataset, "raw_dataset", None)
        if raw_dataset is None or len(raw_dataset) != len(loader.dataset):
            raise ValueError("Relative prefix evaluation requires an aligned raw event dataset.")
        durations_us = [int(raw_dataset[index].duration_us) for index in range(len(raw_dataset))]

        rows = []
        for requested_fraction in relative_fractions:
            sample_steps = [
                min(
                    total_steps,
                    max(1, math.ceil(duration_us * float(requested_fraction) / bin_width_us)),
                )
                for duration_us in durations_us
            ]
            result, _ = evaluate(
                model,
                loader,
                criterion,
                device,
                num_classes,
                prefix_steps=sample_steps,
            )
            rows.append(
                {
                    "requested_duration_fraction": float(requested_fraction),
                    "minimum_observed_us": min(sample_steps) * bin_width_us,
                    "maximum_observed_us": max(sample_steps) * bin_width_us,
                    "accuracy": result.accuracy,
                    "macro_f1": result.macro_f1,
                    "loss": result.loss,
                }
            )
        points = [float(row["requested_duration_fraction"]) for row in rows]
        accuracies = [float(row["accuracy"]) for row in rows]
        outputs["relative_duration"] = (
            rows,
            {
                "axis": "fraction_of_sample_duration",
                "endpoint_knowledge": "oracle_final_duration",
                "accuracy_auc": trapezoidal_auc(points, accuracies),
                "accuracy_auc_normalized": interval_normalized_auc(points, accuracies),
                "measured_interval": [points[0], points[-1]],
                "point_count": len(rows),
            },
        )

    return outputs


def _dvslip_group_metrics(
    config: dict[str, Any], classes: list[str], confusion_matrix: torch.Tensor
) -> dict[str, Any] | None:
    manifest_path = config.get("evaluation", {}).get("class_groups_manifest")
    if manifest_path is None:
        return None

    from etsr.dvslip.dataset import DvsLipExpectations
    from etsr.dvslip.preflight import load_class_groups_manifest

    manifest = load_class_groups_manifest(manifest_path, classes, DvsLipExpectations())
    with Path(manifest_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    semantics = payload["paper_semantics"]
    groups = {metric: payload[group_name] for metric, group_name in semantics.items()}
    return {
        "manifest_sha256": manifest["sha256"],
        "source_id": manifest["source_id"],
        "paper_semantics": semantics,
        "metrics": grouped_accuracies(confusion_matrix, classes, groups),
        "visually_confusable_pair_errors": paired_confusions(
            confusion_matrix,
            classes,
            payload["visually_confusable_pairs"],
        ),
    }


def _write_final_evaluation(
    config: dict[str, Any],
    model: nn.Module,
    bundle: DatasetBundle,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger,
    *,
    complete: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    num_classes = len(bundle.classes)
    validation, predictions = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        num_classes,
        collect_predictions=complete,
    )
    validation_payload = validation.to_dict()
    if config["dataset"]["name"] == "dvslip":
        group_metrics = _dvslip_group_metrics(config, bundle.classes, validation.confusion_matrix)
        if group_metrics is not None:
            validation_payload["class_group_accuracies"] = group_metrics

    prefix_evaluation = None
    if complete:
        prefix_output = _evaluate_prefixes(
            config,
            model,
            validation_loader,
            criterion,
            device,
            num_classes,
            validation,
        )
        if prefix_output is not None:
            prefix_evaluation = {}
            for curve_name, (prefix_rows, curve_summary) in prefix_output.items():
                write_csv(prefix_rows, output_dir / f"prefix_curve_{curve_name}.csv")
                prefix_evaluation[curve_name] = curve_summary
                logger.info(
                    "%s prefix accuracy AUC normalized: %.4f",
                    curve_name,
                    curve_summary["accuracy_auc_normalized"],
                )

    write_json(validation_payload, output_dir / "validation_metrics.json")
    shortcut_correlations = None
    if predictions is not None and config["dataset"]["name"] == "dvslip":
        from etsr.dvslip.shortcut import align_prediction_shortcuts

        rows, shortcut_correlations = align_prediction_shortcuts(
            bundle.validation,
            predictions,
            bin_width_us=int(config["representation"]["bin_width_us"]),
        )
        write_csv(rows, output_dir / "validation_shortcuts.csv")
    return validation_payload, shortcut_correlations, prefix_evaluation


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate selected weights with current metrics without repeating training."""

    config = copy.deepcopy(config)
    checkpoint_path, checkpoint, bundle, model, device, dataset_metadata = (
        _load_checkpoint_context(config, checkpoint_path)
    )
    output = ensure_dir(output_dir)
    logger = configure_logging(output / "evaluation.log")
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Device: %s", device)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    validation_loader = build_loader(bundle.validation, config["dataset"], shuffle=False)
    criterion = make_criterion(config["training"])
    validation, shortcuts, prefixes = _write_final_evaluation(
        config,
        model,
        bundle,
        validation_loader,
        criterion,
        device,
        output,
        logger,
        complete=True,
    )

    environment_path = output / "environment.json"
    write_json(collect_environment(device), environment_path)
    save_config(config, output / "config_evaluated.yaml")
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_score": float(checkpoint["score"]),
        "checkpoint_git_commit": checkpoint["config"].get("runtime", {}).get("git_commit"),
        "evaluation_git_commit": git_commit(),
        "evaluation_git_dirty": git_is_dirty(),
        "device": str(device),
        "seed": int(config["experiment"]["seed"]),
        "trainable_parameters": parameter_count,
        "readout": _readout_metadata(config),
        "validation": validation,
        "official_test_used": False,
        "environment": str(environment_path.resolve()),
        "environment_sha256": sha256_file(environment_path),
    }
    summary.update(dataset_metadata)
    if shortcuts is not None:
        summary["validation_shortcut_correlations"] = shortcuts
    if prefixes is not None:
        summary["prefix_evaluation"] = prefixes
    write_json(summary, output / "evaluation_summary.json")
    logger.info("Evaluation artifacts: %s", output)
    return summary


def profile_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    max_samples: int = 64,
) -> dict[str, Any]:
    """Profile one selected checkpoint with explicitly bounded arithmetic energy proxies."""

    from etsr.profiling import profile_model
    from etsr.profiling.selection import profile_indices

    config = copy.deepcopy(config)
    checkpoint_path, checkpoint, bundle, model, device, dataset_metadata = (
        _load_checkpoint_context(config, checkpoint_path)
    )
    selected = profile_indices(bundle.validation.targets, max_samples)
    loader = build_loader(DatasetSubset(bundle.validation, selected), config["dataset"], shuffle=False)
    profile = profile_model(model, loader, device, max_samples)
    profile["sampling"]["policy"] = "class_round_robin_seed0_v1"
    profile.update(
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_git_commit": checkpoint["config"].get("runtime", {}).get("git_commit"),
            "profiling_git_commit": git_commit(),
            "profiling_git_dirty": git_is_dirty(),
            "device": str(device),
            "dataset": dataset_metadata,
            "representation": getattr(bundle.validation, "representation_metadata", {}),
            "readout": _readout_metadata(config),
            "official_test_used": False,
        }
    )
    output = Path(output_path)
    ensure_dir(output.parent)
    write_json(profile, output)
    return profile


def train_experiment(
    config: dict[str, Any],
    seed: int | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    overfit = config["training"].get("overfit")
    if overfit is not None:
        config["training"]["amp"] = False
        config["training"]["select_metric"] = "accuracy"
        config["augmentation"]["horizontal_flip_probability"] = 0.0
        for field in (
            "temporal_mask_count",
            "temporal_mask_max_steps",
            "spatial_erasing_count",
            "spatial_erasing_max_pixels",
        ):
            config["augmentation"][field] = 0
        config["augmentation"]["event_mix_probability"] = 0.0

    seed = int(config["experiment"]["seed"] if seed is None else seed)
    config["experiment"]["seed"] = seed
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    resume = None
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        metadata = torch.load(resume_path, map_location="cpu", weights_only=False)
        for field in ("run_id", "artifact_dir", "config"):
            if field not in metadata:
                raise ValueError(f"Resume checkpoint is missing {field!r}: {resume_path}")
        if _config_contract(config) != _config_contract(metadata["config"]):
            raise ValueError("Resume configuration differs from the checkpoint configuration.")
        resume = {
            "run_id": metadata["run_id"],
            "artifact_dir": metadata["artifact_dir"],
            "checkpoint_path": str(resume_path.resolve()),
        }
    run_id, artifact_dir, checkpoint_dir, logger = _prepare_run(config, seed, resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Run ID: %s", run_id)
    logger.info("Device: %s", device)

    bundle = build_dataset_bundle(config)
    if overfit is not None:
        bundle = balanced_overfit_bundle(
            bundle,
            class_count=int(overfit["class_count"]),
            samples_per_class=int(overfit["samples_per_class"]),
        )
        logger.info(
            "Overfit subset: %d classes x %d samples",
            int(overfit["class_count"]),
            int(overfit["samples_per_class"]),
        )

    num_classes = len(bundle.classes)
    model = build_model(config["model"], num_classes).to(device)
    continuation = config.get("continuation")
    parent_metadata = None
    new_parameter_names: set[str] = set()
    if continuation is not None:
        parent_path = Path(continuation["parent_checkpoint"])
        if not parent_path.is_file():
            raise FileNotFoundError(f"Continuation parent checkpoint not found: {parent_path}")
        parent_metadata = torch.load(parent_path, map_location=device, weights_only=False)
        if int(parent_metadata.get("num_classes", -1)) != num_classes:
            raise ValueError("Continuation parent class count differs from the dataset.")
        incompatible = model.load_state_dict(parent_metadata["model"], strict=False)
        allowed_missing = (
            "predictive_head.",
            ".content_router.",
            ".predictor_logits",
            ".predictor_spatial.",
            ".predictor_projections.",
            ".surprise_router",
        )
        invalid_missing = [
            key for key in incompatible.missing_keys if not any(token in key for token in allowed_missing)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Continuation parent is not topology-compatible: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        model_parameter_names = dict(model.named_parameters())
        new_parameter_names = {
            name for name in incompatible.missing_keys if name in model_parameter_names
        }
        logger.info("Initialized continuation from: %s", parent_path)

    teacher = None
    objective_config = (continuation or {}).get("objective", {"mode": "none", "weight": 0.0})
    if str(objective_config.get("mode", "none")) != "none":
        teacher_config = load_config(continuation["teacher_config"])
        teacher = build_model(teacher_config["model"], num_classes).to(device)
        teacher_checkpoint = load_model_state(
            continuation["teacher_checkpoint"], teacher, device
        )
        if int(teacher_checkpoint.get("num_classes", -1)) != num_classes:
            raise ValueError("Predictive teacher class count differs from the dataset.")
        teacher.requires_grad_(False)
        teacher.eval()
        logger.info("Loaded frozen predictive teacher: %s", continuation["teacher_checkpoint"])
    predictive_objective = (
        PredictiveTrainingObjective(
            objective_config,
            teacher,
            {
                **continuation,
                "representation": config["representation"],
            },
        )
        if continuation is not None
        else None
    )
    delay_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, CausalTemporalChannelMixer) and module.learnable_delays
    }
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_breakdown = {
        name: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        for name, module in model.named_children()
    }
    logger.info("Trainable parameters: %d", parameter_count)
    deployment_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("predictive_head.")
        and not (
            any(
                token in name
                for token in (".predictor_logits", ".predictor_spatial.", ".predictor_projections.")
            )
            and not config["model"].get("temporal_channel_mixer_surprise_routing", False)
        )
    )

    if continuation is not None and resume is None:
        # New routers/predictors and optional teachers consume different amounts of RNG during
        # construction. Reset before DataLoader creation so every continuation sees the same
        # shuffle and augmentation streams as R0.
        seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    train_loader = build_loader(bundle.train, config["dataset"], shuffle=True)
    validation_loader = build_loader(bundle.validation, config["dataset"], shuffle=False)
    new_parameter_learning_rate = (
        continuation.get("new_parameter_learning_rate")
        if continuation is not None
        else None
    )
    optimizer = make_optimizer(
        model,
        config["training"],
        new_parameter_names=new_parameter_names,
        new_parameter_learning_rate=new_parameter_learning_rate,
    )
    if new_parameter_learning_rate is not None:
        logger.info(
            "Continuation learning rates | inherited %.3e | new %.3e | new tensors %d",
            float(config["training"]["learning_rate"]),
            float(new_parameter_learning_rate),
            len(new_parameter_names),
        )
    scheduler = make_scheduler(optimizer, config["training"])
    criterion = make_criterion(config["training"])
    initial_validation = None
    initial_validation_path = artifact_dir / "continuation_initial_validation.json"
    if continuation is not None and resume is None:
        initial_result, _ = evaluate(
            model, validation_loader, criterion, device, num_classes
        )
        initial_validation = initial_result.to_dict()
        write_json(initial_validation, initial_validation_path)
        logger.info(
            "Continuation epoch 0 | val %.4f/%.4f",
            initial_result.loss,
            initial_result.macro_f1,
        )
    elif continuation is not None and initial_validation_path.is_file():
        initial_validation = json.loads(initial_validation_path.read_text(encoding="utf-8"))
    batch_augmentation = build_batch_augmentation(config["augmentation"])
    amp_enabled = bool(config["training"].get("amp", False) and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.1/2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    select_metric = str(config["training"].get("select_metric", "macro_f1"))

    commit = git_commit()
    dirty = git_is_dirty()
    resolved_config = copy.deepcopy(config)
    resolved_config["experiment"]["seed"] = seed
    resolved_config["model"]["num_classes"] = num_classes
    runtime = {
        "run_id": run_id,
        "device": str(device),
        "git_commit": commit,
        "git_dirty": dirty,
        "trainable_parameters": parameter_count,
        "deployment_parameters": deployment_parameter_count,
        "parameter_breakdown": parameter_breakdown,
        "classes": bundle.classes,
        "official_test_used": False,
    }
    if continuation is not None:
        runtime["continuation_new_parameter_names"] = sorted(new_parameter_names)
        runtime["continuation_new_parameter_learning_rate"] = new_parameter_learning_rate
    runtime.update(getattr(bundle.train, "runtime_metadata", {}))
    if overfit is not None:
        runtime["overfit_train_indices"] = list(bundle.train.indices)
    representation_metadata = getattr(bundle.train, "representation_metadata", None)
    if representation_metadata is not None:
        runtime["representation_metadata"] = representation_metadata
    environment_path = artifact_dir / "environment.json"
    best_score = float("-inf")
    best_epoch = -1
    start_epoch = 1
    peak_cuda_memory_bytes = 0
    checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"
    if resume is None:
        write_json(collect_environment(device), environment_path)
        runtime["environment_sha256"] = sha256_file(environment_path)
        resolved_config["runtime"] = runtime
        save_config(resolved_config, artifact_dir / "config_resolved.yaml")
    else:
        checkpoint = load_training_state(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if int(checkpoint["num_classes"]) != num_classes:
            raise ValueError("Resume checkpoint class count differs from the dataset.")
        checkpoint_commit = checkpoint["config"].get("runtime", {}).get("git_commit")
        if dirty or checkpoint_commit != commit:
            raise ValueError("Resume requires the same clean Git commit used by the checkpoint.")
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        peak_cuda_memory_bytes = int(checkpoint["peak_cuda_memory_bytes"])
        resolved_config = checkpoint["config"]
        runtime = resolved_config["runtime"]
        environment_path = artifact_dir / "environment.json"
        logger.info("Resuming run %s from epoch %d", run_id, start_epoch)

    total_epochs = int(config["training"]["epochs"])
    delay_anneal_epochs = int(config["training"].get("delay_anneal_epochs", total_epochs))
    if delay_modules and not 1 <= delay_anneal_epochs <= total_epochs:
        raise ValueError("delay_anneal_epochs must fit the training horizon")
    if start_epoch > total_epochs:
        raise ValueError(
            f"Checkpoint already reached epoch {start_epoch - 1}; configured total is {total_epochs}."
        )
    gate_rows = []
    if overfit is not None and resume is not None:
        with (artifact_dir / "history.csv").open(newline="") as handle:
            gate_rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) < start_epoch]
    for epoch in range(start_epoch, total_epochs + 1):
        for module in delay_modules.values():
            module.set_delay_progress(min(epoch, delay_anneal_epochs), delay_anneal_epochs)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        learning_rate = optimizer.param_groups[0]["lr"]
        group_learning_rates = {
            str(group.get("group_name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp_enabled,
            config["training"].get("gradient_clip_norm"),
            int(config["training"].get("gradient_accumulation_steps", 1)),
            batch_augmentation,
            predictive_objective,
            epoch,
            bool((continuation or {}).get("freeze_batchnorm_statistics", False)),
        )
        routing_statistics: list[dict[str, Any]] = []
        temporal_prediction_statistics: list[dict[str, float]] | None = (
            []
            if config["model"].get("temporal_channel_mixer_predictive_auxiliary", False)
            else None
        )
        validation, _ = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            num_classes,
            routing_statistics=routing_statistics,
            temporal_prediction_statistics=temporal_prediction_statistics,
        )
        epoch_peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        if epoch_peak_memory is not None:
            peak_cuda_memory_bytes = max(peak_cuda_memory_bytes, epoch_peak_memory)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation.loss,
            "validation_accuracy": validation.accuracy,
            "validation_macro_f1": validation.macro_f1,
            "epoch_seconds": train_metrics["seconds"],
            "gradient_norm_mean": train_metrics["gradient_norm_mean"],
            "gradient_clip_fraction": train_metrics["gradient_clip_fraction"],
            "gradient_nonfinite_fraction": train_metrics["gradient_nonfinite_fraction"],
            "amp_overflow_fraction": train_metrics["amp_overflow_fraction"],
            "peak_cuda_memory_bytes": epoch_peak_memory,
        }
        if "inherited" in group_learning_rates:
            row["inherited_learning_rate"] = group_learning_rates["inherited"]
        if "new" in group_learning_rates:
            row["new_parameter_learning_rate"] = group_learning_rates["new"]
        for name in (
            "classification_loss",
            "auxiliary_loss",
            "predictive_loss",
            "predictive_pair_coverage",
            "temporal_prediction_loss",
            "temporal_prediction_active_loss",
            "temporal_prediction_tail_loss",
            "temporal_persistence_active_loss",
            "temporal_persistence_tail_loss",
            "temporal_delay_mean_active_loss",
            "temporal_delay_mean_tail_loss",
            "temporal_target_active_variance",
            "temporal_target_tail_variance",
            "temporal_prediction_active_skill_vs_persistence",
            "temporal_prediction_tail_skill_vs_persistence",
            "temporal_prediction_active_skill_vs_delay_mean",
            "temporal_prediction_tail_skill_vs_delay_mean",
            "auxiliary_weight",
            "classification_gradient_norm",
            "auxiliary_gradient_norm",
            "weighted_auxiliary_gradient_norm",
            "classification_auxiliary_gradient_cosine",
        ):
            if name in train_metrics:
                row[f"train_{name}"] = train_metrics[name]
        if temporal_prediction_statistics:
            row.update(
                {
                    f"validation_{name}": value
                    for name, value in temporal_prediction_statistics[0].items()
                }
            )
        if delay_modules:
            row["delay_temperature"] = next(iter(delay_modules.values())).delay_temperature
        append_csv(row, artifact_dir / "history.csv")
        for routing_row in routing_statistics:
            append_csv(
                {"epoch": epoch, "partition": "validation", **routing_row},
                artifact_dir / "routing_gate_trajectory.csv",
            )
        if delay_modules:
            for delay_row in _delay_trajectory_rows(delay_modules, epoch):
                append_csv(delay_row, artifact_dir / "learned_delay_trajectory.csv")
        logger.info(
            "Epoch %03d | train %.4f/%.4f | val %.4f/%.4f | %.1fs",
            epoch,
            row["train_loss"],
            row["train_accuracy"],
            row["validation_loss"],
            row["validation_macro_f1"],
            row["epoch_seconds"],
        )

        score = float(getattr(validation, select_metric))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model,
                epoch,
                score,
                resolved_config,
                num_classes,
            )
        save_training_state(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            config=resolved_config,
            num_classes=num_classes,
            run_id=run_id,
            artifact_dir=artifact_dir,
            peak_cuda_memory_bytes=peak_cuda_memory_bytes,
        )
        if overfit is not None:
            gate_rows.append(row)
            gate = overfit_gate(gate_rows)
            write_json(gate, artifact_dir / "overfit_gate.json")
            if overfit.get("stop_on_pass", False) and gate["passed"]:
                logger.info("Bounded overfit passed for five consecutive epochs; stopping.")
                break

    restore_best_model(checkpoint_path, model, device, logger)
    deployment_checkpoint_path = None
    if continuation is not None:
        deployment_config = copy.deepcopy(resolved_config)
        deployment_config.pop("continuation", None)
        deployment_config["model"].pop("predictive_head", None)
        deployment_config["model"].pop("predictive_head_spatial_kernel_size", None)
        deployment_config["model"].pop("predictive_head_hidden_channels", None)
        if not deployment_config["model"].get(
            "temporal_channel_mixer_surprise_routing", False
        ):
            deployment_config["model"].pop(
                "temporal_channel_mixer_predictive_auxiliary", None
            )
            deployment_config["model"].pop(
                "temporal_channel_mixer_predictor_channel_groups", None
            )
            deployment_config["model"].pop(
                "temporal_channel_mixer_predictor_spatial_kernel_size", None
            )
        parent_config = load_config(continuation["parent_config"])
        if objective_config.get("mode") in {"fine_future", "fine_same"}:
            deployment_config["representation"] = copy.deepcopy(parent_config["representation"])
        deployment_checkpoint_path = checkpoint_dir / "deployment.pt"
        save_config(deployment_config, artifact_dir / "deployment_config_resolved.yaml")
        save_deployment_checkpoint(
            deployment_checkpoint_path,
            model,
            best_epoch,
            best_score,
            deployment_config,
            num_classes,
        )
    validation_payload, shortcut_correlations, prefix_evaluation = _write_final_evaluation(
        config,
        model,
        bundle,
        validation_loader,
        criterion,
        device,
        artifact_dir,
        logger,
        complete=overfit is None,
    )
    summary = {
        "run_id": run_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "best_validation_score": best_score,
        "selection_metric": select_metric,
        "readout": _readout_metadata(config),
        "validation": validation_payload,
        "trainable_parameters": parameter_count,
        "deployment_parameters": deployment_parameter_count,
        "parameter_breakdown": parameter_breakdown,
        "checkpoint": str(checkpoint_path.resolve()),
        "last_checkpoint": str(last_checkpoint_path.resolve()),
        "deployment_checkpoint": (
            str(deployment_checkpoint_path.resolve())
            if deployment_checkpoint_path is not None
            else None
        ),
        "artifact_dir": str(artifact_dir.resolve()),
        "git_commit": commit,
        "git_dirty": dirty,
        "official_test_used": False,
        "environment": str(environment_path.resolve()),
        "environment_sha256": runtime["environment_sha256"],
        "peak_cuda_memory_bytes": (peak_cuda_memory_bytes if device.type == "cuda" else None),
    }
    if overfit is not None:
        summary["overfit_gate"] = overfit_gate(gate_rows)
    if shortcut_correlations is not None:
        summary["validation_shortcut_correlations"] = shortcut_correlations
    if prefix_evaluation is not None:
        summary["prefix_evaluation"] = prefix_evaluation
    if delay_modules:
        summary["learned_delays"] = {
            name: module.learned_delay_summary() for name, module in delay_modules.items()
        }
    if continuation is not None:
        summary["continuation"] = {
            "parent_checkpoint": str(Path(continuation["parent_checkpoint"]).resolve()),
            "parent_checkpoint_sha256": sha256_file(continuation["parent_checkpoint"]),
            "parent_epoch": int(parent_metadata["epoch"]),
            "objective": objective_config,
            "freeze_batchnorm_statistics": bool(
                continuation.get("freeze_batchnorm_statistics", False)
            ),
            "teacher_checkpoint": (
                str(Path(continuation["teacher_checkpoint"]).resolve())
                if teacher is not None
                else None
            ),
            "teacher_checkpoint_sha256": (
                sha256_file(continuation["teacher_checkpoint"])
                if teacher is not None
                else None
            ),
            "normalization_report_sha256": (
                sha256_file(objective_config["normalization_report"])
                if objective_config.get("normalization_report")
                else None
            ),
            "initial_validation": initial_validation,
            "new_parameter_names": sorted(new_parameter_names),
            "new_parameter_learning_rate": new_parameter_learning_rate,
        }
    routing_trajectory_path = artifact_dir / "routing_gate_trajectory.csv"
    if routing_trajectory_path.is_file():
        summary["routing_gate_trajectory"] = str(routing_trajectory_path.resolve())
    write_json(summary, artifact_dir / "summary.json")
    logger.info("Artifacts: %s", artifact_dir)
    logger.info("Checkpoint: %s", checkpoint_path)
    return summary
