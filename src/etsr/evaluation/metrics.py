from __future__ import annotations

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
