from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from etsr.evaluation.metrics import classification_metrics


@dataclass
class ProbeResult:
    regularization: float
    validation_macro_f1: float
    audit_metrics: dict[str, Any]
    predictions: np.ndarray
    shuffled_label_accuracy: float | None


def _standardize(
    train: np.ndarray, validation: np.ndarray, audit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (validation - mean) / scale, (audit - mean) / scale


def _train_linear_classifier(
    features: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    regularization: float,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> nn.Linear:
    torch.manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.long)
    classifier = nn.Linear(x.shape[1], num_classes)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(x)
        penalty = classifier.weight.square().mean()
        loss = nn.functional.cross_entropy(logits, y) + regularization * penalty
        loss.backward()
        optimizer.step()
    return classifier


def fit_linear_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    audit_features: np.ndarray,
    audit_targets: np.ndarray,
    regularization_grid: list[float],
    epochs: int,
    learning_rate: float,
    seed: int,
    shuffled_label_control: bool,
) -> ProbeResult:
    arrays = [
        np.asarray(value)
        for value in (
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            audit_features,
            audit_targets,
        )
    ]
    (
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        audit_features,
        audit_targets,
    ) = arrays
    if train_features.ndim != 2 or validation_features.ndim != 2 or audit_features.ndim != 2:
        raise ValueError("Probe features must be matrices.")
    if any(value.size == 0 for value in arrays):
        raise ValueError("Probe arrays must not be empty.")
    if not (
        train_features.shape[0] == train_targets.shape[0]
        and validation_features.shape[0] == validation_targets.shape[0]
        and audit_features.shape[0] == audit_targets.shape[0]
    ):
        raise ValueError("Probe feature and target rows must be aligned.")
    if not (train_features.shape[1] == validation_features.shape[1] == audit_features.shape[1]):
        raise ValueError("Probe feature dimensions must agree across splits.")
    train_features, validation_features, audit_features = _standardize(
        train_features, validation_features, audit_features
    )
    num_classes = int(train_targets.max() + 1)
    if num_classes < 2:
        raise ValueError("A probe requires at least two target classes.")
    if any(
        np.any((targets < 0) | (targets >= num_classes))
        for targets in (train_targets, validation_targets, audit_targets)
    ):
        raise ValueError("Validation/audit probe labels are outside the training label space.")

    best_model = None
    best_regularization = None
    best_score = float("-inf")
    for regularization in regularization_grid:
        classifier = _train_linear_classifier(
            train_features,
            train_targets,
            num_classes,
            float(regularization),
            epochs,
            learning_rate,
            seed,
        )
        with torch.no_grad():
            predictions = (
                classifier(torch.as_tensor(validation_features, dtype=torch.float32))
                .argmax(dim=1)
                .numpy()
            )
        score = classification_metrics(validation_targets, predictions, num_classes)["macro_f1"]
        if score > best_score:
            best_score = score
            best_model = classifier
            best_regularization = float(regularization)

    if best_model is None or best_regularization is None:
        raise RuntimeError("Probe regularization grid is empty.")
    with torch.no_grad():
        audit_predictions = (
            best_model(torch.as_tensor(audit_features, dtype=torch.float32)).argmax(dim=1).numpy()
        )
    audit_metrics = classification_metrics(audit_targets, audit_predictions, num_classes)

    shuffled_accuracy = None
    if shuffled_label_control:
        rng = np.random.default_rng(seed + 100_000)
        shuffled_targets = rng.permutation(train_targets)
        shuffled_model = _train_linear_classifier(
            train_features,
            shuffled_targets,
            num_classes,
            best_regularization,
            epochs,
            learning_rate,
            seed + 200_000,
        )
        with torch.no_grad():
            shuffled_predictions = (
                shuffled_model(torch.as_tensor(audit_features, dtype=torch.float32))
                .argmax(dim=1)
                .numpy()
            )
        shuffled_accuracy = float((shuffled_predictions == audit_targets).mean())

    return ProbeResult(
        regularization=best_regularization,
        validation_macro_f1=float(best_score),
        audit_metrics=audit_metrics,
        predictions=audit_predictions,
        shuffled_label_accuracy=shuffled_accuracy,
    )


def sample_probe_targets(
    metadata: list[dict[str, Any]], primitive_ids: list[str]
) -> dict[str, np.ndarray]:
    primitive_to_idx = {name: index for index, name in enumerate(primitive_ids)}
    content_values = sorted({tuple(sorted(item["primitive_sequence"])) for item in metadata})
    content_to_idx = {value: index for index, value in enumerate(content_values)}
    return {
        "content": np.asarray(
            [content_to_idx[tuple(sorted(item["primitive_sequence"]))] for item in metadata],
            dtype=np.int64,
        ),
        "order": np.asarray(
            [
                int(tuple(item["primitive_sequence"]) != tuple(sorted(item["primitive_sequence"])))
                for item in metadata
            ],
            dtype=np.int64,
        ),
        "first_primitive": np.asarray(
            [primitive_to_idx[item["primitive_sequence"][0]] for item in metadata],
            dtype=np.int64,
        ),
    }


def timestep_primitive_targets(
    metadata: list[dict[str, Any]], primitive_ids: list[str], time_steps: int
) -> np.ndarray:
    primitive_to_idx = {name: index for index, name in enumerate(primitive_ids)}
    targets = np.empty((len(metadata), time_steps), dtype=np.int64)
    for sample_index, item in enumerate(metadata):
        transition = int(item["transition_indices"][0])
        sequence = item["primitive_sequence"]
        for timestep in range(time_steps):
            primitive = sequence[0] if timestep < transition else sequence[1]
            targets[sample_index, timestep] = primitive_to_idx[primitive]
    return targets


def previous_primitive_examples(
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    primitive_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primitive_to_idx = {name: index for index, name in enumerate(primitive_ids)}
    examples = []
    targets = []
    lags = []
    for sample_index, item in enumerate(metadata):
        transition = int(item["transition_indices"][0])
        previous_target = primitive_to_idx[item["primitive_sequence"][0]]
        for timestep in range(transition, features.shape[1]):
            examples.append(features[sample_index, timestep])
            targets.append(previous_target)
            lags.append(timestep - transition + 1)
    return np.asarray(examples), np.asarray(targets, dtype=np.int64), np.asarray(lags)
