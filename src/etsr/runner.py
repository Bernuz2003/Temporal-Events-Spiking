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
from etsr.training.checkpointing import (
    load_training_state,
    save_checkpoint,
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
from etsr.utils.io import append_csv, ensure_dir, sha256_file, write_csv, write_json
from etsr.utils.logging import configure_logging


def _run_id(config: dict[str, Any], seed: int) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config['experiment']['name']}__{timestamp}__seed{seed}"


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


def train_experiment(
    config: dict[str, Any],
    seed: int | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    overfit = config["training"].get("overfit")
    if overfit is not None:
        config["augmentation"]["horizontal_flip_probability"] = 0.0

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
    if start_epoch > total_epochs:
        raise ValueError(
            f"Checkpoint already reached epoch {start_epoch - 1}; configured total is {total_epochs}."
        )
    for epoch in range(start_epoch, total_epochs + 1):
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
            "gradient_nonfinite_fraction": train_metrics["gradient_nonfinite_fraction"],
            "amp_overflow_fraction": train_metrics["amp_overflow_fraction"],
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
        "last_checkpoint": str(last_checkpoint_path.resolve()),
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
