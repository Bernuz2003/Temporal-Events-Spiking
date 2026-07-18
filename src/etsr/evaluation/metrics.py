from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
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
            "confusion_matrix": self.confusion_matrix.cpu().tolist(),
        }


class ClassificationAccumulator:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.loss_sum = 0.0
        self.samples = 0
        self.indices: list[int] = []
        self.targets: list[int] = []
        self.predictions: list[int] = []

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
        self.indices.extend(indices.detach().cpu().tolist())
        self.targets.extend(targets.detach().cpu().tolist())
        self.predictions.extend(predictions.detach().cpu().tolist())

    def compute(self) -> ClassificationResult:
        true_positive = self.confusion.diag().float()
        predicted = self.confusion.sum(dim=0).float()
        actual = self.confusion.sum(dim=1).float()
        precision = true_positive / predicted.clamp_min(1.0)
        recall = true_positive / actual.clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        accuracy = float(true_positive.sum().item() / max(1, self.samples))
        return ClassificationResult(
            accuracy=accuracy,
            macro_f1=float(f1.mean().item()),
            loss=self.loss_sum / max(1, self.samples),
            samples=self.samples,
            confusion_matrix=self.confusion.clone(),
        )


def prefix_auc(fractions: list[float], accuracies: list[float]) -> float:
    if len(fractions) != len(accuracies) or len(fractions) < 2:
        raise ValueError("prefix_auc requires aligned lists with at least two points")
    x = torch.tensor(fractions, dtype=torch.float64)
    y = torch.tensor(accuracies, dtype=torch.float64)
    order = torch.argsort(x)
    return float(torch.trapz(y[order], x[order]).item())


def normalized_prefix_auc(fractions: list[float], accuracies: list[float]) -> float:
    """Normalize prefix AUC by the width of the observed fraction interval."""
    raw_area = prefix_auc(fractions, accuracies)
    interval = max(fractions) - min(fractions)
    if interval <= 0:
        raise ValueError("normalized_prefix_auc requires at least two distinct fractions")
    return raw_area / interval


def paired_prediction_analysis(
    original: dict[str, np.ndarray],
    condition: dict[str, np.ndarray],
    num_classes: int,
) -> dict[str, int | float | None | list[list[int]]]:
    """Compare sample-aligned predictions from the original and perturbed conditions."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    required = ("indices", "targets", "predictions")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, values in (("original", original), ("condition", condition)):
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"{name} predictions are missing: {', '.join(missing)}")
        arrays[name] = {
            key: np.asarray(values[key], dtype=np.int64).reshape(-1) for key in required
        }
        lengths = {array.size for array in arrays[name].values()}
        if len(lengths) != 1:
            raise ValueError(f"{name} prediction arrays must have equal length")
        if np.unique(arrays[name]["indices"]).size != arrays[name]["indices"].size:
            raise ValueError(f"{name} sample indices must be unique")

    original_order = np.argsort(arrays["original"]["indices"])
    condition_order = np.argsort(arrays["condition"]["indices"])
    original_indices = arrays["original"]["indices"][original_order]
    condition_indices = arrays["condition"]["indices"][condition_order]
    if not np.array_equal(original_indices, condition_indices):
        raise ValueError("Original and condition predictions must contain the same sample indices")

    original_targets = arrays["original"]["targets"][original_order]
    original_predictions = arrays["original"]["predictions"][original_order]
    condition_targets = arrays["condition"]["targets"][condition_order]
    condition_predictions = arrays["condition"]["predictions"][condition_order]
    for name, values in (
        ("original targets", original_targets),
        ("original predictions", original_predictions),
        ("condition targets", condition_targets),
        ("condition predictions", condition_predictions),
    ):
        if np.any((values < 0) | (values >= num_classes)):
            raise ValueError(f"{name} must be in [0, num_classes)")

    samples = int(original_indices.size)
    original_correct = original_predictions == original_targets
    condition_correct = condition_predictions == condition_targets
    correct_to_incorrect = original_correct & ~condition_correct
    incorrect_to_correct = ~original_correct & condition_correct
    prediction_changed = original_predictions != condition_predictions
    target_changed = original_targets != condition_targets

    original_correct_count = int(original_correct.sum())
    original_incorrect_count = samples - original_correct_count
    correct_to_incorrect_count = int(correct_to_incorrect.sum())
    incorrect_to_correct_count = int(incorrect_to_correct.sum())

    def rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    encoded_transitions = original_predictions * num_classes + condition_predictions
    transition_matrix = np.bincount(
        encoded_transitions, minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    return {
        "samples": samples,
        "prediction_changed_count": int(prediction_changed.sum()),
        "prediction_changed_rate": rate(int(prediction_changed.sum()), samples),
        "target_changed_count": int(target_changed.sum()),
        "target_changed_rate": rate(int(target_changed.sum()), samples),
        "original_correct_count": original_correct_count,
        "condition_correct_count": int(condition_correct.sum()),
        "correct_to_incorrect_count": correct_to_incorrect_count,
        "correct_to_incorrect_rate": rate(correct_to_incorrect_count, samples),
        "correct_to_incorrect_rate_given_original_correct": rate(
            correct_to_incorrect_count, original_correct_count
        ),
        "incorrect_to_correct_count": incorrect_to_correct_count,
        "incorrect_to_correct_rate": rate(incorrect_to_correct_count, samples),
        "incorrect_to_correct_rate_given_original_incorrect": rate(
            incorrect_to_correct_count, original_incorrect_count
        ),
        "both_correct_count": int((original_correct & condition_correct).sum()),
        "both_incorrect_count": int((~original_correct & ~condition_correct).sum()),
        "prediction_transition_matrix": transition_matrix.tolist(),
    }
