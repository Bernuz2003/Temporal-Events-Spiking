"""Cheap global-statistic controls for the physical-time E0 representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from etsr.data.events import EventSample
from etsr.dvslip.dataset import DvsLipDataset, DvsLipExpectations, load_dvslip_index
from etsr.evaluation.metrics import classification_metrics
from etsr.reproducibility import git_commit, git_is_dirty
from etsr.utils.io import ensure_dir, write_json


def sample_shortcut_statistics(sample: EventSample, bin_width_us: int) -> dict[str, float | int]:
    """Return raw metadata and the aggregate statistics directly observable in E0."""

    if type(bin_width_us) is not int or bin_width_us <= 0:
        raise ValueError("bin_width_us must be a positive integer.")
    event_count = len(sample.t_us)
    if event_count == 0:
        raise ValueError(f"Shortcut statistics require a non-empty sample: {sample.sample_id}")
    return {
        "duration_us": int(sample.duration_us),
        "event_count": int(event_count),
        "on_fraction": float(np.count_nonzero(sample.polarity == 1) / event_count),
        "active_time_bins": int(
            np.unique(np.asarray(sample.t_us, dtype=np.int64) // bin_width_us).size
        ),
    }


def temporal_count_features(
    sample: EventSample,
    *,
    bin_width_us: int,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned and permutation-invariant views of the same per-bin polarity counts."""

    if type(bin_width_us) is not int or bin_width_us <= 0:
        raise ValueError("bin_width_us must be a positive integer.")
    if type(time_steps) is not int or time_steps <= 0:
        raise ValueError("time_steps must be a positive integer.")

    time_bins = np.asarray(sample.t_us, dtype=np.int64) // bin_width_us
    if time_bins.size == 0 or time_bins[0] < 0 or time_bins[-1] >= time_steps:
        raise ValueError(f"Sample {sample.sample_id} does not fit the requested temporal window.")
    polarities = np.asarray(sample.polarity, dtype=np.int64)
    linear_indices = time_bins * 2 + polarities
    counts = np.bincount(linear_indices, minlength=time_steps * 2).reshape(time_steps, 2)

    # Sorting complete OFF/ON rows preserves the exact multiset of temporal bins while removing
    # their positions. It is therefore a controlled order-invariant counterpart, not a weaker
    # aggregate with different information content.
    invariant_order = np.lexsort((counts[:, 1], counts[:, 0]))
    return counts.reshape(-1), counts[invariant_order].reshape(-1)


def eta_squared(values: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of scalar variance explained by the class partition."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if values.size == 0 or values.shape != targets.shape:
        raise ValueError("eta_squared requires non-empty aligned vectors.")
    centered = values - values.mean()
    total = float(np.square(centered).sum())
    if total == 0.0:
        return 0.0
    between = 0.0
    for target in np.unique(targets):
        group = values[targets == target]
        between += group.size * float(group.mean() - values.mean()) ** 2
    return between / total


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Pearson correlation requires aligned vectors with at least two values.")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _collect_features(
    dataset: DvsLipDataset,
    bin_width_us: int,
    *,
    time_steps: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    statistics = []
    targets = []
    aligned_temporal = []
    invariant_temporal = []
    for index in range(len(dataset)):
        sample = dataset[index]
        statistics.append(sample_shortcut_statistics(sample, bin_width_us))
        targets.append(sample.target)
        if time_steps is not None:
            aligned, invariant = temporal_count_features(
                sample,
                bin_width_us=bin_width_us,
                time_steps=time_steps,
            )
            aligned_temporal.append(aligned)
            invariant_temporal.append(invariant)
    columns = {
        name: np.asarray([row[name] for row in statistics], dtype=np.float64)
        for name in ("duration_us", "event_count", "on_fraction", "active_time_bins")
    }
    temporal = None
    if time_steps is not None:
        temporal = (
            np.asarray(aligned_temporal, dtype=np.float64),
            np.asarray(invariant_temporal, dtype=np.float64),
        )
    return columns, np.asarray(targets, dtype=np.int64), temporal


def _feature_matrix(columns: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([columns[name] for name in names])


def _fit_logistic_control(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    num_classes: int,
    *,
    l2_penalty: float,
    max_iterations: int,
) -> dict[str, Any]:
    """Fit one deterministic convex multinomial control with no validation tuning."""

    if l2_penalty < 0.0 or max_iterations <= 0:
        raise ValueError("Shortcut l2_penalty must be non-negative and iterations positive.")
    mean = train_features.mean(axis=0, keepdims=True)
    scale = train_features.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    train_x = torch.as_tensor((train_features - mean) / scale, dtype=torch.float64)
    validation_x = torch.as_tensor((validation_features - mean) / scale, dtype=torch.float64)
    train_y = torch.as_tensor(train_targets, dtype=torch.long)

    classifier = nn.Linear(train_x.shape[1], num_classes, dtype=torch.float64)
    nn.init.zeros_(classifier.weight)
    nn.init.zeros_(classifier.bias)
    optimizer = torch.optim.LBFGS(
        classifier.parameters(),
        lr=1.0,
        max_iter=max_iterations,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        loss = loss + 0.5 * l2_penalty * classifier.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        train_logits = classifier(train_x)
        final_loss = float(
            (
                nn.functional.cross_entropy(train_logits, train_y)
                + 0.5 * l2_penalty * classifier.weight.square().sum()
            ).item()
        )
        train_predictions = train_logits.argmax(dim=1).numpy()
        validation_predictions = classifier(validation_x).argmax(dim=1).numpy()
    state = optimizer.state[next(iter(classifier.parameters()))]
    return {
        "train": classification_metrics(train_targets, train_predictions, num_classes),
        "validation": classification_metrics(
            validation_targets, validation_predictions, num_classes
        ),
        "optimization": {
            "final_loss": final_loss,
            "function_evaluations": int(state.get("func_evals", 0)),
            "iterations": int(state.get("n_iter", 0)),
        },
    }


def run_dvslip_shortcut_control(
    train_root: str | Path,
    split_manifest: str | Path,
    output_path: str | Path,
    *,
    bin_width_us: int,
    expectations: DvsLipExpectations | None = None,
    l2_penalty: float = 1e-4,
    max_iterations: int = 100,
    time_steps: int | None = None,
) -> dict[str, Any]:
    """Fit fixed global controls and optional aligned/order-invariant temporal controls."""

    dataset_index = load_dvslip_index(train_root, split_manifest, expectations=expectations)
    train = DvsLipDataset(dataset_index, "train")
    validation = DvsLipDataset(dataset_index, "validation")

    train_columns, train_targets, train_temporal = _collect_features(
        train,
        bin_width_us,
        time_steps=time_steps,
    )
    validation_columns, validation_targets, validation_temporal = _collect_features(
        validation,
        bin_width_us,
        time_steps=time_steps,
    )
    all_targets = np.concatenate([train_targets, validation_targets])
    all_columns = {
        name: np.concatenate([train_columns[name], validation_columns[name]])
        for name in train_columns
    }
    feature_sets = {
        "raw_duration_metadata": ("duration_us", "event_count", "on_fraction"),
        "e0_global_statistics": ("active_time_bins", "event_count", "on_fraction"),
    }
    controls = {
        name: {
            "features": list(feature_names),
            **_fit_logistic_control(
                _feature_matrix(train_columns, feature_names),
                train_targets,
                _feature_matrix(validation_columns, feature_names),
                validation_targets,
                len(train.classes),
                l2_penalty=l2_penalty,
                max_iterations=max_iterations,
            ),
        }
        for name, feature_names in feature_sets.items()
    }
    if time_steps is not None:
        assert train_temporal is not None and validation_temporal is not None
        temporal_feature_sets = {
            "time_aligned_polarity_counts": (train_temporal[0], validation_temporal[0]),
            "order_invariant_polarity_counts": (train_temporal[1], validation_temporal[1]),
        }
        for name, (train_features, validation_features) in temporal_feature_sets.items():
            controls[name] = {
                "features": ["off_count", "on_count"],
                "feature_count": int(train_features.shape[1]),
                **_fit_logistic_control(
                    train_features,
                    train_targets,
                    validation_features,
                    validation_targets,
                    len(train.classes),
                    l2_penalty=l2_penalty,
                    max_iterations=max_iterations,
                ),
            }
    report = {
        "schema_version": 1,
        "control_id": (
            "dvslip_temporal_shortcuts_v1"
            if time_steps is not None
            else "dvslip_global_shortcuts_v1"
        ),
        "official_source_split": "train",
        "official_test_used": False,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "dataset_index_sha256": train.dataset_index_sha256,
        "split_manifest_sha256": train.split_manifest_sha256,
        "sample_counts": {"train": len(train), "validation": len(validation)},
        "class_count": len(train.classes),
        "bin_width_us": bin_width_us,
        "time_steps": time_steps,
        "eta_squared": {
            name: eta_squared(values, all_targets) for name, values in all_columns.items()
        },
        "duration_active_bins_pearson": _pearson(
            all_columns["duration_us"], all_columns["active_time_bins"]
        ),
        "classifier": {
            "type": "multinomial_logistic_regression",
            "optimizer": "full_batch_lbfgs_strong_wolfe",
            "initialization": "zeros",
            "standardization": "train_mean_and_std",
            "l2_penalty": l2_penalty,
            "maximum_iterations": max_iterations,
            "validation_hyperparameter_selection": False,
        },
        "controls": controls,
        "interpretation": (
            "Non-trivial accuracy is evidence of a shortcut floor, not evidence that the full "
            "model uses the same shortcut. A gain from time-aligned over order-invariant counts "
            "shows that bin position is informative to this linear control; P2-02 must test whether "
            "the neural model uses cross-time dependencies."
        ),
    }
    ensure_dir(Path(output_path).parent)
    write_json(report, output_path)
    return report


def align_prediction_shortcuts(
    dataset: Any,
    predictions: dict[str, np.ndarray],
    *,
    bin_width_us: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce the single per-sample D016 artifact after final model selection."""

    raw_dataset = dataset.raw_dataset
    arrays = {
        name: np.asarray(predictions[name]).reshape(-1)
        for name in ("indices", "targets", "predictions", "margins")
    }
    if len({values.size for values in arrays.values()}) != 1:
        raise ValueError("Prediction arrays must be aligned.")

    rows = []
    for index, target, prediction, margin in zip(
        arrays["indices"],
        arrays["targets"],
        arrays["predictions"],
        arrays["margins"],
        strict=True,
    ):
        sample = raw_dataset[int(index)]
        if sample.target != int(target):
            raise ValueError("Prediction targets do not match the DVS-Lip dataset.")
        rows.append(
            {
                "sample_id": sample.sample_id,
                "target": int(target),
                "prediction": int(prediction),
                "correct": int(prediction == target),
                "margin": float(margin),
                **sample_shortcut_statistics(sample, bin_width_us),
            }
        )

    correct = np.asarray([row["correct"] for row in rows], dtype=np.float64)
    margins = np.asarray([row["margin"] for row in rows], dtype=np.float64)
    features = ("duration_us", "active_time_bins", "event_count", "on_fraction")
    return rows, {
        "samples": len(rows),
        "pearson_with_correctness": {
            name: _pearson(np.asarray([row[name] for row in rows]), correct) for name in features
        },
        "pearson_with_margin": {
            name: _pearson(np.asarray([row[name] for row in rows]), margins) for name in features
        },
    }
