from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ClassificationResult:
    accuracy: float
    macro_f1: float
    loss: float
    samples: int
    confusion_matrix: torch.Tensor

    def to_dict(self) -> dict[str, float | int | list[list[int]]]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "loss": self.loss,
            "samples": self.samples,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


class ClassificationAccumulator:
    """Accumulate only the aggregates needed by validation."""

    def __init__(self, num_classes: int, collect_predictions: bool = False) -> None:
        self.num_classes = num_classes
        self.collect_predictions = collect_predictions
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.loss_sum = 0.0
        self.samples = 0
        self.indices: list[int] = []
        self.targets: list[int] = []
        self.predictions: list[int] = []
        self.margins: list[float] = []

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        predictions = logits.argmax(dim=1)
        encoded = targets.detach().cpu() * self.num_classes + predictions.detach().cpu()
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)
        batch_size = int(targets.numel())
        self.loss_sum += float(loss.detach().item()) * batch_size
        self.samples += batch_size
        if self.collect_predictions:
            detached = logits.detach()
            correct = detached.gather(1, targets[:, None]).squeeze(1)
            competitors = detached.clone()
            competitors.scatter_(1, targets[:, None], float("-inf"))
            margins = correct - competitors.max(dim=1).values
            self.indices.extend(indices.tolist())
            self.targets.extend(targets.detach().cpu().tolist())
            self.predictions.extend(predictions.detach().cpu().tolist())
            self.margins.extend(margins.cpu().tolist())

    def compute(self) -> ClassificationResult:
        true_positive = self.confusion.diag().float()
        predicted = self.confusion.sum(dim=0).float()
        actual = self.confusion.sum(dim=1).float()
        precision = true_positive / predicted.clamp_min(1.0)
        recall = true_positive / actual.clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        return ClassificationResult(
            accuracy=float(true_positive.sum().item() / max(1, self.samples)),
            macro_f1=float(f1.mean().item()),
            loss=self.loss_sum / max(1, self.samples),
            samples=self.samples,
            confusion_matrix=self.confusion.clone(),
        )

    def prediction_arrays(self) -> dict[str, np.ndarray]:
        if not self.collect_predictions:
            raise RuntimeError("Prediction collection was not enabled.")
        return {
            "indices": np.asarray(self.indices, dtype=np.int64),
            "targets": np.asarray(self.targets, dtype=np.int64),
            "predictions": np.asarray(self.predictions, dtype=np.int64),
            "margins": np.asarray(self.margins, dtype=np.float64),
        }


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have the same shape.")
    for name, values in (("targets", targets), ("predictions", predictions)):
        if np.any((values < 0) | (values >= num_classes)):
            raise ValueError(f"{name} are outside the class range.")

    encoded = targets * num_classes + predictions
    matrix = np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    samples = int(matrix.sum())
    return {
        "samples": samples,
        "accuracy": float(true_positive.sum() / max(samples, 1)),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
        "recall_per_class": recall.tolist(),
    }


def grouped_accuracies(
    confusion_matrix: torch.Tensor,
    classes: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float | int]]:
    """Compute accuracy on named subsets of true classes."""

    if confusion_matrix.ndim != 2 or confusion_matrix.shape != (len(classes), len(classes)):
        raise ValueError("Confusion matrix shape must match the class vocabulary.")
    if len(classes) != len(set(classes)):
        raise ValueError("Class vocabulary must contain unique names.")

    class_indices = {name: index for index, name in enumerate(classes)}
    results: dict[str, dict[str, float | int]] = {}
    for group_name, group_classes in groups.items():
        if not group_name or not group_classes:
            raise ValueError("Metric groups require a name and at least one class.")
        if len(group_classes) != len(set(group_classes)):
            raise ValueError(f"Metric group {group_name!r} contains duplicate classes.")
        unknown = sorted(set(group_classes) - set(class_indices))
        if unknown:
            raise ValueError(f"Metric group {group_name!r} contains unknown classes: {unknown}")
        indices = [class_indices[name] for name in group_classes]
        samples = int(confusion_matrix[indices, :].sum().item())
        correct = int(confusion_matrix[indices, indices].sum().item())
        results[group_name] = {
            "accuracy": correct / max(1, samples),
            "samples": samples,
            "class_count": len(indices),
        }
    return results


def paired_confusions(
    confusion_matrix: torch.Tensor,
    classes: Sequence[str],
    pairs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Measure direct substitutions within declared confusable class pairs."""

    if confusion_matrix.ndim != 2 or confusion_matrix.shape != (len(classes), len(classes)):
        raise ValueError("Confusion matrix shape must match the class vocabulary.")
    class_indices = {name: index for index, name in enumerate(classes)}
    flattened = [name for pair in pairs for name in pair]
    if any(len(pair) != 2 for pair in pairs) or len(flattened) != len(set(flattened)):
        raise ValueError("Confusable pairs must be disjoint two-class groups.")
    unknown = sorted(set(flattened) - set(class_indices))
    if unknown:
        raise ValueError(f"Confusable pairs contain unknown classes: {unknown}")

    pair_rows: list[dict[str, Any]] = []
    paired_samples = 0
    paired_class_errors = 0
    within_pair_errors = 0
    for first, second in pairs:
        first_index = class_indices[first]
        second_index = class_indices[second]
        first_samples = int(confusion_matrix[first_index, :].sum().item())
        second_samples = int(confusion_matrix[second_index, :].sum().item())
        first_as_second = int(confusion_matrix[first_index, second_index].item())
        second_as_first = int(confusion_matrix[second_index, first_index].item())
        pair_samples = first_samples + second_samples
        pair_errors = first_as_second + second_as_first
        pair_rows.append(
            {
                "classes": [first, second],
                "first_as_second": first_as_second,
                "second_as_first": second_as_first,
                "within_pair_errors": pair_errors,
                "samples": pair_samples,
                "within_pair_error_rate": pair_errors / max(1, pair_samples),
            }
        )
        paired_samples += pair_samples
        paired_class_errors += pair_samples - int(
            confusion_matrix[first_index, first_index].item()
            + confusion_matrix[second_index, second_index].item()
        )
        within_pair_errors += pair_errors

    pair_rows.sort(key=lambda row: int(row["within_pair_errors"]), reverse=True)
    return {
        "pair_count": len(pair_rows),
        "paired_class_samples": paired_samples,
        "paired_class_errors": paired_class_errors,
        "within_pair_errors": within_pair_errors,
        "within_pair_errors_per_paired_sample": within_pair_errors / max(1, paired_samples),
        "within_pair_share_of_paired_class_errors": within_pair_errors
        / max(1, paired_class_errors),
        "pairs": pair_rows,
    }


def trapezoidal_auc(points: Sequence[float], values: Sequence[float]) -> float:
    """Integrate bounded values over a strictly increasing positive interval."""

    if len(points) != len(values) or len(points) < 2:
        raise ValueError("AUC requires aligned sequences with at least two points.")
    x = [float(value) for value in points]
    y = [float(value) for value in values]
    if not all(np.isfinite(x + y)):
        raise ValueError("AUC points and values must be finite.")
    if any(value <= 0.0 for value in x):
        raise ValueError("AUC points must be positive.")
    if any(right <= left for left, right in zip(x, x[1:], strict=False)):
        raise ValueError("AUC points must be strictly increasing.")
    if any(not 0.0 <= value <= 1.0 for value in y):
        raise ValueError("AUC values must be in [0, 1].")
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for left_x, right_x, left_y, right_y in zip(x, x[1:], y, y[1:], strict=False)
    )


def interval_normalized_auc(points: Sequence[float], values: Sequence[float]) -> float:
    """Normalize trapezoidal AUC by the width of the measured interval."""

    area = trapezoidal_auc(points, values)
    interval = float(points[-1]) - float(points[0])
    return area / interval
