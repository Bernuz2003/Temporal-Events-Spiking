from __future__ import annotations

import copy
import datetime as dt
import logging
from pathlib import Path
from typing import Any

import torch

from etsr.config import save_config
from etsr.data.common import balanced_overfit_bundle, build_loader
from etsr.data.factory import build_dataset_bundle
from etsr.models.factory import build_model
from etsr.reproducibility import (
    collect_environment,
    git_commit,
    git_is_dirty,
    seed_everything,
)
from etsr.training.checkpointing import save_checkpoint
from etsr.training.engine import (
    evaluate,
    make_criterion,
    make_optimizer,
    make_scheduler,
    restore_best_model,
    train_one_epoch,
)
from etsr.utils.io import append_csv, ensure_dir, sha256_file, write_csv, write_json
from etsr.utils.logging import configure_logging


def _run_id(config: dict[str, Any], seed: int) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config['experiment']['name']}__{timestamp}__seed{seed}"


def _prepare_run(config: dict[str, Any], seed: int) -> tuple[str, Path, Path, logging.Logger]:
    run_id = _run_id(config, seed)
    artifact_dir = ensure_dir(Path(config["experiment"]["artifact_root"]) / run_id)
    checkpoint_dir = ensure_dir(Path(config["experiment"]["checkpoint_root"]) / run_id)
    return run_id, artifact_dir, checkpoint_dir, configure_logging(artifact_dir / "run.log")


def train_experiment(config: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    config = copy.deepcopy(config)
    overfit = config["training"].get("overfit")
    if overfit is not None:
        config["augmentation"]["horizontal_flip_probability"] = 0.0

    seed = int(config["experiment"]["seed"] if seed is None else seed)
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    run_id, artifact_dir, checkpoint_dir, logger = _prepare_run(config, seed)
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
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_breakdown = {
        name: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        for name, module in model.named_children()
    }
    logger.info("Trainable parameters: %d", parameter_count)

    train_loader = build_loader(bundle.train, config["dataset"], shuffle=True)
    validation_loader = build_loader(bundle.validation, config["dataset"], shuffle=False)
    optimizer = make_optimizer(model, config["training"])
    scheduler = make_scheduler(optimizer, config["training"])
    criterion = make_criterion(config["training"])
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
        "parameter_breakdown": parameter_breakdown,
        "classes": bundle.classes,
        "official_test_used": False,
    }
    runtime.update(getattr(bundle.train, "runtime_metadata", {}))
    representation_metadata = getattr(bundle.train, "representation_metadata", None)
    if representation_metadata is not None:
        runtime["representation_metadata"] = representation_metadata
    environment_path = artifact_dir / "environment.json"
    write_json(collect_environment(device), environment_path)
    runtime["environment_sha256"] = sha256_file(environment_path)
    resolved_config["runtime"] = runtime
    save_config(resolved_config, artifact_dir / "config_resolved.yaml")

    best_score = float("-inf")
    best_epoch = -1
    peak_cuda_memory_bytes = 0
    checkpoint_path = checkpoint_dir / "best.pt"
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        learning_rate = optimizer.param_groups[0]["lr"]
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
        )
        validation, _ = evaluate(model, validation_loader, criterion, device, num_classes)
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
            "peak_cuda_memory_bytes": epoch_peak_memory,
        }
        append_csv(row, artifact_dir / "history.csv")
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

    restore_best_model(checkpoint_path, model, device, logger)
    validation, predictions = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        num_classes,
        collect_predictions=overfit is None,
    )
    write_json(validation.to_dict(), artifact_dir / "validation_metrics.json")
    shortcut_correlations = None
    if predictions is not None and config["dataset"]["name"] == "dvslip":
        from etsr.dvslip.shortcut import align_prediction_shortcuts

        rows, shortcut_correlations = align_prediction_shortcuts(
            bundle.validation,
            predictions,
            bin_width_us=int(config["representation"]["bin_width_us"]),
        )
        write_csv(rows, artifact_dir / "validation_shortcuts.csv")
    summary = {
        "run_id": run_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "selection_metric": select_metric,
        "validation": validation.to_dict(),
        "trainable_parameters": parameter_count,
        "parameter_breakdown": parameter_breakdown,
        "checkpoint": str(checkpoint_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "git_commit": commit,
        "git_dirty": dirty,
        "official_test_used": False,
        "environment": str(environment_path.resolve()),
        "environment_sha256": runtime["environment_sha256"],
        "peak_cuda_memory_bytes": (peak_cuda_memory_bytes if device.type == "cuda" else None),
    }
    if shortcut_correlations is not None:
        summary["validation_shortcut_correlations"] = shortcut_correlations
    write_json(summary, artifact_dir / "summary.json")
    logger.info("Artifacts: %s", artifact_dir)
    logger.info("Checkpoint: %s", checkpoint_path)
    return summary
