from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from etsr.config import save_config
from etsr.data.factory import build_loader
from etsr.evaluation.reports import save_confusion_matrix
from etsr.models.factory import build_model
from etsr.phase2.causal import (
    pairwise_margin,
    patch_with_counterfactual_activations,
    segment_aligned_time_indices,
)
from etsr.phase2.dataset import (
    Phase2DVSGCDataset,
    build_phase2_bundle,
    prepare_phase2_dataset,
)
from etsr.phase2.input_audit import collect_input_audit
from etsr.phase2.metrics import (
    aggregate_seed_scalars,
    aggregate_tidy_seed_rows,
    classification_metrics,
    factorized_content_order_metrics,
    grouped_bootstrap_interval,
    inverse_temporal_consistency,
    normalized_trapezoid_auc,
    prefix_trajectory_metrics,
    reverse_class_map,
    trapezoid_auc,
)
from etsr.phase2.probes import (
    fit_linear_probe,
    previous_primitive_examples,
    sample_probe_targets,
    timestep_primitive_targets,
)
from etsr.phase2.tracing import TraceCollection, collect_traces, save_trace_npz
from etsr.phase2.transformations import redistribute_segment_durations
from etsr.reproducibility import seed_everything
from etsr.training.checkpointing import save_checkpoint
from etsr.training.engine import (
    evaluate,
    make_criterion,
    make_optimizer,
    make_scheduler,
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


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_phase2_config(config: dict[str, Any]) -> None:
    if config.get("phase2", {}).get("protocol_version") != "phase2_tdup_v1":
        raise ValueError("Phase 2 requires phase2.protocol_version=phase2_tdup_v1.")
    seeds = [int(value) for value in config["experiment"].get("model_seeds", [])]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("Phase 2 requires at least three distinct model seeds.")
    if config["dataset"].get("name") != "dvsgc_phase2":
        raise ValueError("Phase 2 requires dataset.name=dvsgc_phase2.")
    if config["dataset"].get("official_split") != "train":
        raise ValueError("Phase 2 may only use the official training partition.")
    primitive_ids = [str(value) for value in config["dataset"].get("primitive_ids", [])]
    if len(primitive_ids) != 3 or len(set(primitive_ids)) != 3:
        raise ValueError("phase2_tdup_v1 requires exactly three distinct primitives.")
    if bool(config["dataset"].get("allow_consecutive_repetition", False)):
        raise ValueError("phase2_tdup_v1 requires consecutive repetitions to be disabled.")
    if not bool(config.get("split", {}).get("forbid_official_test", False)):
        raise ValueError("Phase 2 requires an explicit official-test embargo.")

    time_steps = int(config["dataset"]["frames_number"])
    prefix_config = config["phase2"]["prefix"]
    auc_start = int(prefix_config["auc_start_timestep"])
    tail_start = int(prefix_config["tail_start"])
    if not 1 <= auc_start < time_steps:
        raise ValueError("phase2.prefix.auc_start_timestep must be in [1, T).")
    if not auc_start <= tail_start < time_steps:
        raise ValueError("phase2.prefix.tail_start must be in [auc_start_timestep, T).")
    prefix_timesteps = [int(value) for value in config["phase2"]["probes"]["prefix_timesteps"]]
    if any(value < 1 or value > time_steps for value in prefix_timesteps):
        raise ValueError("Probe prefix timesteps must be in [1, T].")
    ratios = config["phase2"]["transformations"].get("duration_ratios", [])
    if not ratios or any(
        len(ratio) != 2 or any(float(value) <= 0 for value in ratio) for ratio in ratios
    ):
        raise ValueError("Phase 2 requires positive order-2 duration ratios.")
    if not config["phase2"]["probes"].get("regularization_grid"):
        raise ValueError("The probe regularization grid must not be empty.")
    allowed_regions = {"first_action", "second_action"}
    if set(config["phase2"]["causal"]["regions"]) - allowed_regions:
        raise ValueError("Unsupported Phase 2 causal region.")


def phase2_prepare(config: dict[str, Any]) -> dict[str, Any]:
    _validate_phase2_config(config)
    manifest = prepare_phase2_dataset(config)
    return {
        "dataset_root": str(Path(config["dataset"]["root"]).resolve()),
        "dataset_manifest": str(
            (Path(config["dataset"]["root"]) / "dataset_manifest.json").resolve()
        ),
        "samples": len(manifest["samples"]),
        "source_groups": len(manifest["source_filenames"]),
        "official_test_used": False,
    }


def phase2_train(config: dict[str, Any], seed: int) -> dict[str, Any]:
    _validate_phase2_config(config)
    configured_seeds = [int(value) for value in config["experiment"]["model_seeds"]]
    if seed not in configured_seeds:
        raise ValueError(f"Seed {seed} is not declared in experiment.model_seeds.")
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"{config['experiment']['name']}__{_timestamp()}__seed{seed}"
    artifact_dir = ensure_dir(Path(config["experiment"]["artifact_root"]) / run_id)
    checkpoint_dir = ensure_dir(Path(config["experiment"]["checkpoint_root"]) / run_id)
    logger = configure_logging(artifact_dir / "run.log")
    logger.info("Phase 2 run: %s", run_id)
    logger.info("Official test embargo: active")

    bundle = build_phase2_bundle(config["dataset"])
    num_classes = len(bundle.classes)
    model = build_model(config["model"], num_classes).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    train_loader = build_loader(bundle.train_core, config["dataset"], shuffle=True)
    validation_loader = build_loader(bundle.checkpoint_validation, config["dataset"], shuffle=False)
    optimizer = make_optimizer(model, config["training"])
    scheduler = make_scheduler(optimizer, int(config["training"]["epochs"]))
    criterion = make_criterion(config["training"])
    amp_enabled = bool(config["training"].get("amp", False) and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    dataset_root = Path(config["dataset"]["root"])
    split_manifest_path = dataset_root / "split_manifest.json"
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    resolved_config = dict(config)
    resolved_config["experiment"] = dict(config["experiment"])
    resolved_config["experiment"]["seed"] = seed
    resolved_config["model"] = dict(config["model"])
    resolved_config["model"]["num_classes"] = num_classes
    resolved_config["runtime"] = {
        "run_id": run_id,
        "device": str(device),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "trainable_parameters": parameter_count,
        "classes": bundle.classes,
        "official_test_used": False,
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "split_manifest_sha256": _sha256(split_manifest_path),
    }
    save_config(resolved_config, artifact_dir / "config_resolved.yaml")

    select_metric = str(config["training"].get("select_metric", "macro_f1"))
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
        validation_result, _ = evaluate(model, validation_loader, criterion, device, num_classes)
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
        score = getattr(validation_result, select_metric)
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            score,
            resolved_config,
            num_classes,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                score,
                resolved_config,
                num_classes,
            )
        logger.info(
            "Epoch %03d | train %.4f | checkpoint-val %.4f",
            epoch,
            train_metrics["accuracy"],
            validation_result.macro_f1,
        )

    restore_best_model(checkpoint_dir / "best.pt", model, device, logger)
    validation_result, _ = evaluate(model, validation_loader, criterion, device, num_classes)
    write_json(validation_result.to_dict(), artifact_dir / "checkpoint_validation_metrics.json")
    save_confusion_matrix(
        validation_result.confusion_matrix.numpy(),
        bundle.classes,
        artifact_dir / "checkpoint_validation_confusion_matrix.png",
    )
    summary = {
        "run_id": run_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_checkpoint_validation_score": best_score,
        "selection_metric": select_metric,
        "checkpoint_validation": validation_result.to_dict(),
        "checkpoint": str((checkpoint_dir / "best.pt").resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "official_test_used": False,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }
    write_json(summary, artifact_dir / "summary.json")
    return summary


def _load_phase2_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
    expected_seed: int,
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("num_classes", -1)) != num_classes:
        raise ValueError(f"Checkpoint class count mismatch: {checkpoint_path}")
    checkpoint_config = checkpoint.get("config", {})
    runtime = checkpoint_config.get("runtime", {})
    if runtime.get("official_test_used") is not False:
        raise RuntimeError(
            f"Checkpoint does not certify the Phase 2 test embargo: {checkpoint_path}"
        )
    if int(checkpoint_config.get("experiment", {}).get("seed", -1)) != expected_seed:
        raise RuntimeError(
            f"Checkpoint seed does not match seed {expected_seed}: {checkpoint_path}"
        )
    if runtime.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise RuntimeError(
            f"Checkpoint was trained on a different dataset manifest: {checkpoint_path}"
        )
    if runtime.get("split_manifest_sha256") != split_manifest_sha256:
        raise RuntimeError(
            f"Checkpoint was trained on a different split manifest: {checkpoint_path}"
        )
    checkpoint_model_config = dict(checkpoint_config.get("model", {}))
    checkpoint_model_config.pop("num_classes", None)
    if checkpoint_model_config != config["model"]:
        raise RuntimeError(
            f"Checkpoint architecture config differs from the audit config: {checkpoint_path}"
        )
    model = build_model(config["model"], num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _trace_metrics(
    trace: TraceCollection,
    classes: list[str],
    auc_start_timestep: int,
    tail_start: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    final_predictions = trace.predictions
    classification = classification_metrics(trace.targets, final_predictions, len(classes))
    factorized = factorized_content_order_metrics(trace.targets, final_predictions, classes)
    prefix_rows, transition_rows, arrays = prefix_trajectory_metrics(
        trace.cumulative_logits, trace.targets, classes, tail_start
    )
    auc_rows = [row for row in prefix_rows if row["timestep"] >= auc_start_timestep]
    auc_fractions = np.asarray([row["fraction"] for row in auc_rows])
    auc_accuracies = np.asarray([row["accuracy"] for row in auc_rows])
    prefix_auc = normalized_trapezoid_auc(auc_fractions, auc_accuracies)
    prefix_auc_raw = trapezoid_auc(auc_fractions, auc_accuracies)
    summary = {
        "classification": classification,
        "content_order": factorized,
        "prefix_accuracy_auc_raw": prefix_auc_raw,
        "prefix_accuracy_auc_normalized": prefix_auc,
        "prefix_auc_interval": [float(auc_fractions.min()), float(auc_fractions.max())],
    }
    return summary, prefix_rows, transition_rows, arrays


@torch.no_grad()
def _duration_audit(
    model: torch.nn.Module,
    dataset,
    ratios: list[list[float]],
    device: torch.device,
    classes: list[str],
    original_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for ratio in ratios:
        predictions = []
        targets = []
        count_errors = []
        for index in range(len(dataset)):
            frames, target, _stable_index = dataset[index]
            metadata = dataset.dataset.metadata_for_index(index)
            transformed, _target_lengths = redistribute_segment_durations(
                frames,
                [int(value) for value in metadata["segment_lengths"]],
                [float(value) for value in ratio],
            )
            count_errors.append(
                float((transformed.sum(dim=0) - frames.sum(dim=0)).abs().max().item())
            )
            logits = model(transformed.unsqueeze(0).to(device))
            predictions.append(int(logits.argmax(dim=1).item()))
            targets.append(int(target))
        predictions_array = np.asarray(predictions, dtype=np.int64)
        targets_array = np.asarray(targets, dtype=np.int64)
        metrics = classification_metrics(targets_array, predictions_array, len(classes))
        factorized = factorized_content_order_metrics(targets_array, predictions_array, classes)
        rows.append(
            {
                "transformation": "duration_redistribution",
                "ratio": ":".join(str(value) for value in ratio),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "content_accuracy": factorized["content_accuracy"],
                "conditional_order_accuracy": factorized["conditional_order_accuracy"],
                "prediction_consistency": float((predictions_array == original_predictions).mean()),
                "maximum_count_preservation_error": max(count_errors),
            }
        )
    return rows


def _probe_row(name: str, layer: str, timestep: int | None, result) -> dict[str, Any]:
    return {
        "probe": name,
        "layer": layer,
        "timestep": timestep,
        "lag": None,
        "regularization": result.regularization,
        "validation_macro_f1": result.validation_macro_f1,
        "audit_accuracy": result.audit_metrics["accuracy"],
        "audit_macro_f1": result.audit_metrics["macro_f1"],
        "shuffled_label_accuracy": result.shuffled_label_accuracy,
    }


def _run_probes(
    config: dict[str, Any],
    seed: int,
    train_trace: TraceCollection,
    validation_trace: TraceCollection,
    audit_trace: TraceCollection,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    probe_config = config["phase2"]["probes"]
    primitive_ids = [str(value) for value in config["dataset"]["primitive_ids"]]
    regularization = [float(value) for value in probe_config["regularization_grid"]]
    common_kwargs = {
        "regularization_grid": regularization,
        "epochs": int(probe_config.get("epochs", 80)),
        "learning_rate": float(probe_config.get("learning_rate", 0.05)),
        "seed": seed,
        "shuffled_label_control": bool(probe_config.get("shuffled_label_control", True)),
    }
    sample_targets = {
        "train": sample_probe_targets(train_trace.metadata, primitive_ids),
        "validation": sample_probe_targets(validation_trace.metadata, primitive_ids),
        "audit": sample_probe_targets(audit_trace.metadata, primitive_ids),
    }
    rows = []
    summary: dict[str, float] = {}
    for layer in probe_config["layers"]:
        for timestep in probe_config["prefix_timesteps"]:
            index = int(timestep) - 1
            if index >= audit_trace.features[layer].shape[1]:
                continue
            for target_name in ("content", "order"):
                result = fit_linear_probe(
                    train_trace.features[layer][:, index],
                    sample_targets["train"][target_name],
                    validation_trace.features[layer][:, index],
                    sample_targets["validation"][target_name],
                    audit_trace.features[layer][:, index],
                    sample_targets["audit"][target_name],
                    **common_kwargs,
                )
                rows.append(_probe_row(target_name, layer, int(timestep), result))

        train_current = timestep_primitive_targets(
            train_trace.metadata, primitive_ids, train_trace.features[layer].shape[1]
        )
        validation_current = timestep_primitive_targets(
            validation_trace.metadata, primitive_ids, validation_trace.features[layer].shape[1]
        )
        audit_current = timestep_primitive_targets(
            audit_trace.metadata, primitive_ids, audit_trace.features[layer].shape[1]
        )
        current_result = fit_linear_probe(
            train_trace.features[layer].reshape(-1, train_trace.features[layer].shape[-1]),
            train_current.reshape(-1),
            validation_trace.features[layer].reshape(
                -1, validation_trace.features[layer].shape[-1]
            ),
            validation_current.reshape(-1),
            audit_trace.features[layer].reshape(-1, audit_trace.features[layer].shape[-1]),
            audit_current.reshape(-1),
            **common_kwargs,
        )
        rows.append(_probe_row("current_primitive", layer, None, current_result))

        train_previous = previous_primitive_examples(
            train_trace.features[layer], train_trace.metadata, primitive_ids
        )
        validation_previous = previous_primitive_examples(
            validation_trace.features[layer], validation_trace.metadata, primitive_ids
        )
        audit_previous = previous_primitive_examples(
            audit_trace.features[layer], audit_trace.metadata, primitive_ids
        )
        previous_result = fit_linear_probe(
            train_previous[0],
            train_previous[1],
            validation_previous[0],
            validation_previous[1],
            audit_previous[0],
            audit_previous[1],
            **common_kwargs,
        )
        rows.append(_probe_row("previous_primitive", layer, None, previous_result))
        summary[f"previous_primitive_accuracy__{layer}"] = previous_result.audit_metrics["accuracy"]
        for lag in sorted(set(audit_previous[2].tolist())):
            mask = audit_previous[2] == lag
            metrics = classification_metrics(
                audit_previous[1][mask],
                previous_result.predictions[mask],
                len(primitive_ids),
            )
            rows.append(
                {
                    "probe": "previous_primitive_by_lag",
                    "layer": layer,
                    "timestep": None,
                    "lag": int(lag),
                    "audit_accuracy": metrics["accuracy"],
                    "audit_macro_f1": metrics["macro_f1"],
                }
            )
    return rows, summary


def _run_shortcut_baselines(
    config: dict[str, Any],
    seed: int,
    train_loader,
    validation_loader,
    audit_loader,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _train_rows, train_temporal, train_invariant, train_targets = collect_input_audit(
        train_loader, train_loader.dataset
    )
    _validation_rows, validation_temporal, validation_invariant, validation_targets = (
        collect_input_audit(validation_loader, validation_loader.dataset)
    )
    audit_rows, audit_temporal, audit_invariant, audit_targets = collect_input_audit(
        audit_loader, audit_loader.dataset
    )
    probe_config = config["phase2"]["probes"]
    kwargs = {
        "regularization_grid": [float(value) for value in probe_config["regularization_grid"]],
        "epochs": int(probe_config.get("epochs", 80)),
        "learning_rate": float(probe_config.get("learning_rate", 0.05)),
        "seed": seed + 500_000,
        "shuffled_label_control": bool(probe_config.get("shuffled_label_control", True)),
    }
    result_rows = []
    for name, train_features, validation_features, audit_features in (
        (
            "temporal_input_statistics",
            train_temporal,
            validation_temporal,
            audit_temporal,
        ),
        (
            "order_invariant_input_statistics",
            train_invariant,
            validation_invariant,
            audit_invariant,
        ),
    ):
        result = fit_linear_probe(
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            audit_features,
            audit_targets,
            **kwargs,
        )
        row = _probe_row(name, "input", None, result)
        factorized = factorized_content_order_metrics(
            audit_targets,
            result.predictions,
            list(audit_loader.dataset.classes),
        )
        row["content_accuracy"] = factorized["content_accuracy"]
        row["conditional_order_accuracy"] = factorized["conditional_order_accuracy"]
        result_rows.append(row)
    return result_rows, audit_rows


@torch.no_grad()
def _run_causal_audit(
    config: dict[str, Any],
    model: torch.nn.Module,
    dataset,
    classes: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    causal_config = config["phase2"]["causal"]
    if not bool(causal_config.get("enabled", True)):
        return []
    raw_dataset: Phase2DVSGCDataset = dataset.dataset
    per_class = int(causal_config.get("samples_per_class", 8))
    selected = []
    counts = {index: 0 for index in range(len(classes))}
    for index, item in enumerate(raw_dataset.metadata):
        target = int(item["target"])
        if counts[target] < per_class:
            selected.append(index)
            counts[target] += 1
    if any(value == 0 for value in counts.values()):
        raise RuntimeError("The development-audit split is missing one or more classes.")
    frames = torch.stack([dataset[index][0] for index in selected]).to(device)
    donor_indices = [raw_dataset.reverse_indices[index] for index in selected]
    donor_frames = torch.stack([dataset[index][0] for index in donor_indices]).to(device)
    targets = torch.as_tensor(
        [raw_dataset.metadata[index]["target"] for index in selected],
        dtype=torch.long,
        device=device,
    )
    reverse_mapping = reverse_class_map(classes)
    inverse_targets = torch.as_tensor(
        [reverse_mapping[int(target)] for target in targets.cpu().tolist()],
        dtype=torch.long,
        device=device,
    )
    baseline_logits = model(frames)
    baseline_margin = pairwise_margin(baseline_logits, targets, inverse_targets)
    recipient_lengths = torch.as_tensor(
        [raw_dataset.metadata[index]["segment_lengths"] for index in selected],
        dtype=torch.long,
        device=device,
    )
    donor_lengths = torch.as_tensor(
        [raw_dataset.metadata[index]["segment_lengths"] for index in donor_indices],
        dtype=torch.long,
        device=device,
    )
    donor_time_indices = segment_aligned_time_indices(recipient_lengths, donor_lengths)
    rows = []
    for layer in causal_config["layers"]:
        for region in causal_config["regions"]:
            mask = torch.zeros(frames.shape[:2], dtype=torch.bool, device=device)
            for batch_index, sample_index in enumerate(selected):
                transition = int(raw_dataset.metadata[sample_index]["transition_indices"][0])
                if region == "first_action":
                    mask[batch_index, :transition] = True
                elif region == "second_action":
                    mask[batch_index, transition:] = True
                else:
                    raise ValueError(f"Unsupported causal region: {region}")
            patched_logits = patch_with_counterfactual_activations(
                model,
                layer,
                frames,
                donor_frames,
                mask,
                donor_time_indices=donor_time_indices,
            )
            patched_margin = pairwise_margin(patched_logits, targets, inverse_targets)
            effect = patched_margin - baseline_margin
            common = {
                "layer": layer,
                "region": region,
                "intervention": ("matched_reverse_pair_segment_aligned_activation_patch"),
            }
            rows.append(
                common
                | {
                    "scope": "all_classes",
                    "class_name": None,
                    "samples": len(selected),
                    "mean_pairwise_margin_effect": float(effect.mean().item()),
                    "std_pairwise_margin_effect": float(effect.std(unbiased=True).item()),
                    "prediction_changed_rate": float(
                        (patched_logits.argmax(dim=1) != baseline_logits.argmax(dim=1))
                        .float()
                        .mean()
                        .item()
                    ),
                    "prediction_changed_to_inverse_rate": float(
                        (patched_logits.argmax(dim=1) == inverse_targets).float().mean().item()
                    ),
                }
            )
            for class_index, class_name in enumerate(classes):
                class_mask = targets == class_index
                class_effect = effect[class_mask]
                rows.append(
                    common
                    | {
                        "scope": "class",
                        "class_name": class_name,
                        "samples": int(class_mask.sum().item()),
                        "mean_pairwise_margin_effect": float(class_effect.mean().item()),
                        "std_pairwise_margin_effect": (
                            float(class_effect.std(unbiased=True).item())
                            if class_effect.numel() > 1
                            else 0.0
                        ),
                        "prediction_changed_rate": float(
                            (
                                patched_logits.argmax(dim=1)[class_mask]
                                != baseline_logits.argmax(dim=1)[class_mask]
                            )
                            .float()
                            .mean()
                            .item()
                        ),
                        "prediction_changed_to_inverse_rate": float(
                            (
                                patched_logits.argmax(dim=1)[class_mask]
                                == inverse_targets[class_mask]
                            )
                            .float()
                            .mean()
                            .item()
                        ),
                    }
                )
    return rows


def _verify_checkpoint_manifest(config: dict[str, Any], checkpoints: dict[int, Path]) -> None:
    expected = {int(value) for value in config["experiment"]["model_seeds"]}
    if set(checkpoints) != expected:
        raise ValueError(
            f"Phase 2 audit requires checkpoints for exactly {sorted(expected)}; "
            f"received {sorted(checkpoints)}."
        )
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 2 checkpoints: {missing}")


def phase2_audit(config: dict[str, Any], checkpoints: dict[int, str | Path]) -> dict[str, Any]:
    _validate_phase2_config(config)
    checkpoint_paths = {int(seed): Path(path) for seed, path in checkpoints.items()}
    _verify_checkpoint_manifest(config, checkpoint_paths)
    seed_everything(
        int(config["experiment"]["seed"]),
        bool(config["experiment"].get("deterministic", True)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase2_id = f"{config['experiment']['name']}__audit__{_timestamp()}"
    artifact_dir = ensure_dir(Path(config["experiment"]["artifact_root"]) / phase2_id)
    logger = configure_logging(artifact_dir / "phase2_audit.log")
    bundle = build_phase2_bundle(config["dataset"])
    classes = bundle.classes
    num_classes = len(classes)
    dataset_root = Path(config["dataset"]["root"])
    dataset_manifest_sha256 = _sha256(dataset_root / "dataset_manifest.json")
    split_manifest_sha256 = _sha256(dataset_root / "split_manifest.json")
    shutil.copy2(dataset_root / "dataset_manifest.json", artifact_dir / "dataset_manifest.json")
    shutil.copy2(dataset_root / "split_manifest.json", artifact_dir / "split_manifest.json")
    save_config(config, artifact_dir / "protocol_manifest.yaml")
    write_json(
        {
            "checkpoints": {
                str(seed): {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
                for seed, path in checkpoint_paths.items()
            },
            "official_test_used": False,
            "protocol_version": config["phase2"]["protocol_version"],
        },
        artifact_dir / "checkpoints_manifest.json",
    )

    layers = [str(value) for value in config["phase2"]["tracing"]["layers"]]
    auc_start_timestep = int(config["phase2"]["prefix"]["auc_start_timestep"])
    tail_start = int(config["phase2"]["prefix"]["tail_start"])
    seed_summaries = []
    tidy_rows: dict[str, list[dict[str, Any]]] = {
        "content_pair": [],
        "inverse_pair": [],
        "prefix": [],
        "late_update": [],
        "transformation": [],
        "probe": [],
        "shortcut": [],
        "causal": [],
    }
    for seed, checkpoint_path in sorted(checkpoint_paths.items()):
        logger.info("Auditing seed %d: %s", seed, checkpoint_path)
        seed_dir = ensure_dir(artifact_dir / f"seed_{seed}")
        model = _load_phase2_model(
            config,
            checkpoint_path,
            num_classes,
            device,
            expected_seed=seed,
            dataset_manifest_sha256=dataset_manifest_sha256,
            split_manifest_sha256=split_manifest_sha256,
        )
        train_loader = build_loader(bundle.train_core, config["dataset"], shuffle=False)
        validation_loader = build_loader(
            bundle.checkpoint_validation, config["dataset"], shuffle=False
        )
        audit_loader = build_loader(bundle.development_audit, config["dataset"], shuffle=False)
        audit_trace = collect_traces(model, audit_loader, device, layers)
        trace_summary, prefix_rows, transition_rows, trajectory_arrays = _trace_metrics(
            audit_trace, classes, auc_start_timestep, tail_start
        )
        predictions = audit_trace.predictions
        raw_audit: Phase2DVSGCDataset = bundle.development_audit.dataset
        reverse_predictions = np.asarray(
            [predictions[raw_audit.reverse_indices[index]] for index in range(len(raw_audit))]
        )
        inverse_metrics = inverse_temporal_consistency(
            predictions, reverse_predictions, audit_trace.targets, classes
        )

        source_groups = np.asarray([item["source_filename"] for item in audit_trace.metadata])
        correct = trajectory_arrays["predictions"] == audit_trace.targets[:, None]
        tail_before = tail_start - 1
        final_index = correct.shape[1] - 1
        late_utility_values = correct[:, final_index].astype(np.float64) - correct[
            :, tail_before
        ].astype(np.float64)
        bootstrap_config = config["phase2"]["bootstrap"]
        nlu_bootstrap = grouped_bootstrap_interval(
            source_groups,
            lambda indices, values=late_utility_values: float(values[indices].mean()),
            int(bootstrap_config["samples"]),
            float(bootstrap_config["confidence"]),
            seed + 700_000,
        )

        duration_rows = _duration_audit(
            model,
            bundle.development_audit,
            config["phase2"]["transformations"].get("duration_ratios", []),
            device,
            classes,
            predictions,
        )
        train_trace = collect_traces(model, train_loader, device, layers)
        validation_trace = collect_traces(model, validation_loader, device, layers)
        probe_rows, probe_summary = _run_probes(
            config, seed, train_trace, validation_trace, audit_trace
        )
        shortcut_rows, input_rows = _run_shortcut_baselines(
            config, seed, train_loader, validation_loader, audit_loader
        )
        causal_rows = _run_causal_audit(config, model, bundle.development_audit, classes, device)

        content_pair_rows = trace_summary["content_order"]["per_content_pair"]
        inverse_pair_rows = inverse_metrics["per_content_pair"]

        for name, rows in (
            ("content_pair", content_pair_rows),
            ("inverse_pair", inverse_pair_rows),
            ("prefix", prefix_rows),
            ("late_update", transition_rows),
            ("transformation", duration_rows),
            ("probe", probe_rows),
            ("shortcut", shortcut_rows),
            ("causal", causal_rows),
        ):
            tidy_rows[name].extend([dict(row) | {"seed": seed} for row in rows])

        write_csv(prefix_rows, seed_dir / "prefix_metrics.csv")
        write_csv(transition_rows, seed_dir / "late_update_metrics.csv")
        write_csv(duration_rows, seed_dir / "transformation_metrics.csv")
        write_csv(probe_rows, seed_dir / "probe_metrics.csv")
        write_csv(shortcut_rows, seed_dir / "input_shortcut_baselines.csv")
        write_csv(input_rows, seed_dir / "input_temporal_statistics.csv")
        write_csv(causal_rows, seed_dir / "causal_patching.csv")
        save_trace_npz(audit_trace, str(seed_dir / "audit_compact_traces.npz"))

        tail_row = next(
            row
            for row in transition_rows
            if row["from_timestep"] == tail_start
            and row["to_timestep"] == audit_trace.cumulative_logits.shape[1]
        )
        duration_consistency = (
            float(np.mean([row["prediction_consistency"] for row in duration_rows]))
            if duration_rows
            else 1.0
        )
        seed_summary = {
            "seed": seed,
            "checkpoint": str(checkpoint_path.resolve()),
            "official_test_used": False,
            "accuracy": trace_summary["classification"]["accuracy"],
            "macro_f1": trace_summary["classification"]["macro_f1"],
            "content_accuracy": trace_summary["content_order"]["content_accuracy"],
            "conditional_order_accuracy": trace_summary["content_order"][
                "conditional_order_accuracy"
            ],
            "inverse_temporal_consistency": inverse_metrics["inverse_temporal_consistency"],
            "prefix_accuracy_auc_raw": trace_summary["prefix_accuracy_auc_raw"],
            "prefix_accuracy_auc_normalized": trace_summary["prefix_accuracy_auc_normalized"],
            "net_late_utility": tail_row["net_late_utility"],
            "duration_prediction_consistency": duration_consistency,
            "net_late_utility_grouped_bootstrap": nlu_bootstrap,
            "classification": trace_summary["classification"],
            "content_order": trace_summary["content_order"],
            "inverse_consistency": inverse_metrics,
            "probe_availability": probe_summary,
            "input_shortcut_baselines": shortcut_rows,
            "causal_patching": causal_rows,
        }
        write_json(seed_summary, seed_dir / "seed_summary.json")
        seed_summaries.append(seed_summary)

    aggregate_specs = {
        "content_pair": {
            "keys": ["content"],
            "values": ["content_accuracy", "conditional_order_accuracy"],
        },
        "inverse_pair": {
            "keys": ["content"],
            "values": ["inverse_temporal_consistency"],
        },
        "prefix": {
            "keys": ["timestep", "fraction"],
            "values": [
                "accuracy",
                "macro_f1",
                "content_accuracy",
                "conditional_order_accuracy",
                "mean_correct_margin",
            ],
        },
        "late_update": {
            "keys": ["from_timestep", "to_timestep"],
            "values": [
                "late_harm_rate",
                "late_rescue_rate",
                "net_late_utility",
                "mean_margin_change",
            ],
        },
        "transformation": {
            "keys": ["transformation", "ratio"],
            "values": [
                "accuracy",
                "macro_f1",
                "content_accuracy",
                "conditional_order_accuracy",
                "prediction_consistency",
                "maximum_count_preservation_error",
            ],
        },
        "probe": {
            "keys": ["probe", "layer", "timestep", "lag"],
            "values": [
                "validation_macro_f1",
                "audit_accuracy",
                "audit_macro_f1",
                "shuffled_label_accuracy",
            ],
        },
        "shortcut": {
            "keys": ["probe", "layer", "timestep", "lag"],
            "values": [
                "audit_accuracy",
                "audit_macro_f1",
                "content_accuracy",
                "conditional_order_accuracy",
                "shuffled_label_accuracy",
            ],
        },
        "causal": {
            "keys": ["layer", "region", "scope", "class_name", "intervention"],
            "values": [
                "mean_pairwise_margin_effect",
                "prediction_changed_rate",
                "prediction_changed_to_inverse_rate",
            ],
        },
    }
    aggregate_artifacts = {}
    for name, specification in aggregate_specs.items():
        aggregate_rows = aggregate_tidy_seed_rows(
            tidy_rows[name], specification["keys"], specification["values"]
        )
        path = artifact_dir / f"aggregate_{name}.csv"
        write_csv(aggregate_rows, path)
        aggregate_artifacts[name] = path.name

    scalar_fields = [
        "accuracy",
        "macro_f1",
        "content_accuracy",
        "conditional_order_accuracy",
        "inverse_temporal_consistency",
        "prefix_accuracy_auc_raw",
        "prefix_accuracy_auc_normalized",
        "net_late_utility",
        "duration_prediction_consistency",
    ]
    aggregate = aggregate_seed_scalars(seed_summaries, scalar_fields)
    profile = {
        "phase2_id": phase2_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "official_test_used": False,
        "protocol_version": config["phase2"]["protocol_version"],
        "construct": "Temporal Dynamics Utilization Profile",
        "primary_components": {
            "content_order_factorization": [
                "content_accuracy",
                "conditional_order_accuracy",
            ],
            "semantic_temporal_consistency": [
                "inverse_temporal_consistency",
                "duration_prediction_consistency",
            ],
            "evidence_update_utility": [
                "prefix_accuracy_auc_raw",
                "prefix_accuracy_auc_normalized",
                "net_late_utility",
            ],
            "availability_and_causal_use": [
                "probe_metrics.csv",
                "causal_patching.csv",
            ],
            "shortcut_controls": ["input_shortcut_baselines.csv"],
        },
        "aggregation": aggregate,
        "aggregate_artifacts": aggregate_artifacts,
        "per_seed": seed_summaries,
        "interpretation_rule": (
            "The profile is primary; no single composite scalar is used. "
            "Probe accessibility is not interpreted as causal use."
        ),
    }
    write_json(profile, artifact_dir / "temporal_utilization_profile.json")
    write_json(profile, artifact_dir / "phase2_summary.json")
    logger.info("Phase 2 artifacts: %s", artifact_dir)
    return profile


def parse_checkpoint_arguments(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must use SEED=PATH: {value}")
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"Duplicate checkpoint seed: {seed}")
        result[seed] = Path(path_text)
    return result
