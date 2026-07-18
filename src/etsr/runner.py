from __future__ import annotations

import datetime as dt
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from etsr.config import save_config
from etsr.data.factory import build_dataset_bundle, build_loader
from etsr.data.perturbations import PerturbationSpec, PerturbedDataset
from etsr.evaluation.metrics import (
    normalized_prefix_auc,
    paired_prediction_analysis,
    prefix_auc,
)
from etsr.evaluation.reports import save_confusion_matrix, save_prefix_curve
from etsr.models.factory import build_model
from etsr.reproducibility import seed_everything
from etsr.training.checkpointing import save_checkpoint
from etsr.training.engine import (
    evaluate,
    make_criterion,
    make_optimizer,
    make_scheduler,
    profile_model,
    restore_best_model,
    train_one_epoch,
)
from etsr.utils.io import append_csv, ensure_dir, write_csv, write_json
from etsr.utils.logging import configure_logging


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_id(config: dict[str, Any]) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config['experiment']['name']}__{timestamp}__seed{config['experiment']['seed']}"


def _prepare_run(config: dict[str, Any]) -> tuple[str, Path, Path, logging.Logger]:
    run_id = _run_id(config)
    artifact_dir = ensure_dir(Path(config["experiment"]["artifact_root"]) / run_id)
    checkpoint_dir = ensure_dir(Path(config["experiment"]["checkpoint_root"]) / run_id)
    logger = configure_logging(artifact_dir / "run.log")
    return run_id, artifact_dir, checkpoint_dir, logger


def train_experiment(config: dict[str, Any]) -> dict:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    run_id, artifact_dir, checkpoint_dir, logger = _prepare_run(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Run ID: %s", run_id)
    logger.info("Device: %s", device)

    bundle = build_dataset_bundle(config["dataset"], seed)
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
    test_loader = build_loader(bundle.test, config["dataset"], shuffle=False)

    optimizer = make_optimizer(model, config["training"])
    scheduler = make_scheduler(optimizer, int(config["training"]["epochs"]))
    criterion = make_criterion(config["training"])
    amp_enabled = bool(config["training"].get("amp", False) and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch versions before the unified AMP API
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    select_metric = str(config["training"].get("select_metric", "macro_f1"))

    resolved_config = dict(config)
    resolved_config["model"] = dict(config["model"])
    resolved_config["model"]["num_classes"] = num_classes
    resolved_config["runtime"] = {
        "run_id": run_id,
        "device": str(device),
        "git_commit": _git_commit(),
        "trainable_parameters": parameter_count,
        "parameter_breakdown": parameter_breakdown,
        "classes": bundle.classes,
    }
    save_config(resolved_config, artifact_dir / "config_resolved.yaml")

    best_score = float("-inf")
    best_epoch = -1
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp_enabled,
            config["training"].get("gradient_clip_norm"),
        )
        validation_result, _ = evaluate(
            model, validation_loader, criterion, device, num_classes
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_result.loss,
            "validation_accuracy": validation_result.accuracy,
            "validation_macro_f1": validation_result.macro_f1,
            "epoch_seconds": train_metrics["seconds"],
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

        score = getattr(validation_result, select_metric)
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, epoch, score, resolved_config, num_classes
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch, score, resolved_config, num_classes
            )

    restore_best_model(checkpoint_dir / "best.pt", model, device, logger)
    test_result, test_predictions = evaluate(model, test_loader, criterion, device, num_classes)
    write_json(test_result.to_dict(), artifact_dir / "test_metrics.json")
    np.savez_compressed(artifact_dir / "test_predictions.npz", **test_predictions)
    save_confusion_matrix(
        test_result.confusion_matrix.numpy(), bundle.classes, artifact_dir / "confusion_matrix.png"
    )

    profile = None
    if bool(config.get("profiling", {}).get("enabled", True)):
        profile = profile_model(model, test_loader, device, config.get("profiling", {}))
        write_json(profile, artifact_dir / "profile.json")

    summary = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "selection_metric": select_metric,
        "test": test_result.to_dict(),
        "trainable_parameters": parameter_count,
        "parameter_breakdown": parameter_breakdown,
        "profile": profile,
        "checkpoint": str((checkpoint_dir / "best.pt").resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "git_commit": _git_commit(),
    }
    write_json(summary, artifact_dir / "summary.json")
    logger.info(
        "Test | accuracy %.4f | macro-F1 %.4f | best epoch %d",
        test_result.accuracy,
        test_result.macro_f1,
        best_epoch,
    )
    logger.info("Artifacts: %s", artifact_dir)
    logger.info("Checkpoint: %s", checkpoint_dir / "best.pt")
    return summary


def audit_experiment(config: dict[str, Any], checkpoint_path: str | Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = checkpoint_path.parent.name
    artifact_dir = ensure_dir(Path(config["experiment"]["artifact_root"]) / run_id / "audit")
    logger = configure_logging(artifact_dir / "audit.log")
    bundle = build_dataset_bundle(config["dataset"], seed)
    num_classes = int(checkpoint.get("num_classes", len(bundle.classes)))
    model = build_model(config["model"], num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    criterion = make_criterion(config["training"])

    rows = []
    all_results = {}
    all_predictions = {}
    resolved_perturbation_methods = set()
    for item in config.get("audit", {}).get("perturbations", [{"name": "original"}]):
        spec = PerturbationSpec.from_dict(item)
        dataset = PerturbedDataset(bundle.test, spec)
        loader = build_loader(dataset, config["dataset"], shuffle=False)
        result, predictions = evaluate(model, loader, criterion, device, num_classes)
        label = spec.name if spec.target_mode == "keep" else f"{spec.name}__{spec.target_mode}"
        if label in all_results:
            raise ValueError(f"Duplicate audit condition label: {label}")
        resolved_perturbation_methods.add(dataset.resolved_method)
        resolved_item = dict(item)
        resolved_item["resolved_method"] = dataset.resolved_method
        metrics = result.to_dict()
        metrics["perturbation"] = resolved_item
        all_results[label] = metrics
        all_predictions[label] = predictions
        rows.append(
            {
                "condition": label,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "loss": result.loss,
                "resolved_method": dataset.resolved_method,
            }
        )
        if bool(config.get("audit", {}).get("save_predictions", True)):
            np.savez_compressed(artifact_dir / f"predictions__{label}.npz", **predictions)
        logger.info("%s | accuracy %.4f | macro-F1 %.4f", label, result.accuracy, result.macro_f1)

    original_metrics = all_results.get("original")
    original_predictions = all_predictions.get("original")
    paired_analysis = {}
    paired_csv_fields = (
        "prediction_changed_count",
        "prediction_changed_rate",
        "target_changed_count",
        "target_changed_rate",
        "correct_to_incorrect_count",
        "correct_to_incorrect_rate",
        "correct_to_incorrect_rate_given_original_correct",
        "incorrect_to_correct_count",
        "incorrect_to_correct_rate",
        "incorrect_to_correct_rate_given_original_incorrect",
    )
    for row in rows:
        label = row["condition"]
        if original_metrics is None:
            row["accuracy_drop_from_original"] = None
            row["macro_f1_drop_from_original"] = None
        else:
            row["accuracy_drop_from_original"] = original_metrics["accuracy"] - row["accuracy"]
            row["macro_f1_drop_from_original"] = (
                original_metrics["macro_f1"] - row["macro_f1"]
            )
            all_results[label]["degradation_from_original"] = {
                "accuracy_drop": row["accuracy_drop_from_original"],
                "macro_f1_drop": row["macro_f1_drop_from_original"],
            }

        analysis = None
        if original_predictions is not None:
            analysis = paired_prediction_analysis(
                original_predictions, all_predictions[label], num_classes
            )
            paired_analysis[label] = analysis
        for field in paired_csv_fields:
            row[field] = None if analysis is None else analysis[field]

    prefix_rows = []
    original_loader = build_loader(bundle.test, config["dataset"], shuffle=False)
    fractions = [float(value) for value in config.get("audit", {}).get("prefix_fractions", [1.0])]
    for fraction in fractions:
        result, _ = evaluate(
            model, original_loader, criterion, device, num_classes, prefix_fraction=fraction
        )
        prefix_rows.append(
            {"fraction": fraction, "accuracy": result.accuracy, "macro_f1": result.macro_f1}
        )
    if len(prefix_rows) >= 2:
        prefix_fraction_values = [row["fraction"] for row in prefix_rows]
        prefix_accuracy_values = [row["accuracy"] for row in prefix_rows]
        raw_area = prefix_auc(prefix_fraction_values, prefix_accuracy_values)
        normalized_area = normalized_prefix_auc(
            prefix_fraction_values, prefix_accuracy_values
        )
    else:
        raw_area = None
        normalized_area = None

    write_csv(rows, artifact_dir / "perturbation_summary.csv")
    write_csv(prefix_rows, artifact_dir / "prefix_curve.csv")
    save_prefix_curve(prefix_rows, artifact_dir / "prefix_curve.png")

    limitations = ["perturbation sensitivity alone does not prove causal temporal understanding"]
    if "paired_reversed_action_sample" in resolved_perturbation_methods:
        limitations.append(
            "reverse_actions uses the paired reversed-class sample with the same source filename; "
            "it is a valid reversed action chain, not a frame-exact rearrangement of the original tensor"
        )
    if "equal_temporal_chunks" in resolved_perturbation_methods:
        limitations.append(
            "reverse_segments uses equal temporal chunks because true action boundaries are unavailable"
        )

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "conditions": all_results,
        "paired_analysis": paired_analysis,
        "prefix_curve": prefix_rows,
        "prefix_accuracy_auc": raw_area,
        "prefix_accuracy_auc_raw": raw_area,
        "prefix_accuracy_auc_normalized": normalized_area,
        "prefix_auc_interval": [min(fractions), max(fractions)] if fractions else None,
        "limitations": limitations,
    }
    write_json(summary, artifact_dir / "audit_summary.json")
    if raw_area is not None:
        logger.info("Prefix accuracy AUC raw: %.4f", raw_area)
    if normalized_area is not None:
        logger.info("Prefix accuracy AUC normalized: %.4f", normalized_area)
    logger.info("Audit artifacts: %s", artifact_dir)
    return summary
