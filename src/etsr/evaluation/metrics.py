from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have the same shape.")
    for name, values in (("targets", targets), ("predictions", predictions)):
        if np.any((values < 0) | (values >= num_classes)):
            raise ValueError(f"{name} are outside the class range.")
    encoded = targets * num_classes + predictions
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> dict[str, Any]:
    matrix = confusion_matrix(targets, predictions, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0
    )
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


def _class_tokens(class_name: str) -> tuple[str, ...]:
    if "->" in class_name:
        return tuple(class_name.split("->"))
    return tuple(class_name)


def _content_signatures(classes: list[str]) -> list[tuple[str, ...]]:
    return [tuple(sorted(_class_tokens(class_name))) for class_name in classes]


def factorized_content_order_metrics(
    targets: np.ndarray, predictions: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have the same shape.")
    for name, values in (("targets", targets), ("predictions", predictions)):
        if np.any((values < 0) | (values >= len(classes))):
            raise ValueError(f"{name} are outside the class range.")

    signatures = _content_signatures(classes)
    # Content ignores order; exact correctness then isolates order once content is right.
    content_correct = np.asarray(
        [
            signatures[int(target)] == signatures[int(prediction)]
            for target, prediction in zip(targets, predictions, strict=True)
        ],
        dtype=bool,
    )
    exact_correct = targets == predictions
    content_count = int(content_correct.sum())

    pair_rows = []
    for signature in sorted(set(signatures)):
        class_indices = [index for index, value in enumerate(signatures) if value == signature]
        mask = np.isin(targets, class_indices)
        if not mask.any():
            continue
        pair_content = int(content_correct[mask].sum())
        pair_exact = int(exact_correct[mask].sum())
        pair_rows.append(
            {
                "content": "|".join(signature),
                "classes": [classes[index] for index in class_indices],
                "samples": int(mask.sum()),
                "content_accuracy": float(content_correct[mask].mean()),
                "conditional_order_accuracy": (
                    pair_exact / pair_content if pair_content else None
                ),
            }
        )
    return {
        "samples": int(targets.size),
        "content_correct_count": content_count,
        "content_accuracy": content_count / max(int(targets.size), 1),
        "order_correct_given_content_count": int(exact_correct.sum()),
        "conditional_order_accuracy": (
            float(exact_correct.sum() / content_count) if content_count else None
        ),
        "per_content_pair": pair_rows,
    }


def reverse_class_map(classes: list[str]) -> dict[int, int]:
    class_to_idx = {name: index for index, name in enumerate(classes)}
    result = {}
    for index, name in enumerate(classes):
        tokens = _class_tokens(name)
        reverse_name = (
            "->".join(reversed(tokens)) if "->" in name else "".join(reversed(tokens))
        )
        if reverse_name not in class_to_idx:
            raise ValueError(f"Missing reversed class for {name}")
        result[index] = class_to_idx[reverse_name]
    return result


def inverse_temporal_consistency(
    predictions: np.ndarray,
    reverse_predictions: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    reverse_predictions = np.asarray(reverse_predictions, dtype=np.int64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if not (predictions.shape == reverse_predictions.shape == targets.shape):
        raise ValueError("Inverse consistency arrays must be aligned.")
    mapping = reverse_class_map(classes)
    expected = np.asarray([mapping[int(value)] for value in predictions], dtype=np.int64)
    consistent = reverse_predictions == expected

    pair_rows = []
    signatures = _content_signatures(classes)
    for signature in sorted(set(signatures)):
        indices = [index for index, value in enumerate(signatures) if value == signature]
        mask = np.isin(targets, indices)
        if mask.any():
            pair_rows.append(
                {
                    "content": "|".join(signature),
                    "classes": [classes[index] for index in indices],
                    "samples": int(mask.sum()),
                    "inverse_temporal_consistency": float(consistent[mask].mean()),
                }
            )
    return {
        "samples": int(consistent.size),
        "consistent_count": int(consistent.sum()),
        "inverse_temporal_consistency": float(consistent.mean()),
        "per_content_pair": pair_rows,
    }


def correct_class_margin(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if logits.ndim != 3 or logits.shape[0] != targets.size:
        raise ValueError("Expected logits [N, T, C] aligned with targets [N].")
    correct = np.take_along_axis(logits, targets[:, None, None], axis=2).squeeze(2)
    competitors = logits.copy()
    rows = np.arange(targets.size)[:, None]
    times = np.arange(logits.shape[1])[None, :]
    competitors[rows, times, targets[:, None]] = -np.inf
    return correct - competitors.max(axis=2)


def prefix_trajectory_metrics(
    cumulative_logits: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    tail_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    logits = np.asarray(cumulative_logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if logits.ndim != 3 or logits.shape[0] != targets.size:
        raise ValueError("Expected cumulative logits [N, T, C].")
    predictions = logits.argmax(axis=2)
    margins = correct_class_margin(logits, targets)

    prefix_rows = []
    for timestep in range(logits.shape[1]):
        metrics = classification_metrics(targets, predictions[:, timestep], len(classes))
        factorized = factorized_content_order_metrics(
            targets, predictions[:, timestep], classes
        )
        prefix_rows.append(
            {
                "timestep": timestep + 1,
                "fraction": (timestep + 1) / logits.shape[1],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "content_accuracy": factorized["content_accuracy"],
                "conditional_order_accuracy": factorized["conditional_order_accuracy"],
                "mean_correct_margin": float(margins[:, timestep].mean()),
            }
        )

    transitions = [(index, index + 1) for index in range(logits.shape[1] - 1)]
    tail_transition = (tail_start - 1, logits.shape[1] - 1)
    if 0 < tail_start < logits.shape[1] and tail_transition not in transitions:
        transitions.append(tail_transition)

    transition_rows = []
    for before, after in transitions:
        correct_before = predictions[:, before] == targets
        correct_after = predictions[:, after] == targets
        harm = correct_before & ~correct_after
        rescue = ~correct_before & correct_after
        transition_rows.append(
            {
                "from_timestep": before + 1,
                "to_timestep": after + 1,
                "late_harm_count": int(harm.sum()),
                "late_harm_rate": float(harm.mean()),
                "late_rescue_count": int(rescue.sum()),
                "late_rescue_rate": float(rescue.mean()),
                "net_late_utility": float(rescue.mean() - harm.mean()),
                "mean_margin_change": float((margins[:, after] - margins[:, before]).mean()),
            }
        )
    return prefix_rows, transition_rows, {"predictions": predictions, "margins": margins}


def grouped_bootstrap_interval(
    group_ids: np.ndarray,
    metric: Callable[[np.ndarray], float],
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    group_ids = np.asarray(group_ids)
    unique_groups = np.unique(group_ids)
    if unique_groups.size < 2:
        raise ValueError("Grouped bootstrap requires at least two groups.")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration.")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        # Repeated groups must repeat all their rows, preserving the clustered sampling unit.
        indices = np.concatenate([np.flatnonzero(group_ids == group) for group in selected])
        estimates.append(float(metric(indices)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(estimates)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": samples,
    }


def aggregate_seed_scalars(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"seeds": [int(row["seed"]) for row in rows]}
    for field in fields:
        raw_values = [row[field] for row in rows]
        if any(value is None for value in raw_values):
            result[field] = {
                "values": raw_values,
                "mean": None,
                "std": None,
                "same_sign": None,
            }
            continue
        values = np.asarray(raw_values, dtype=np.float64)
        result[field] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "same_sign": bool(np.all(values >= 0) or np.all(values <= 0)),
        }
    return result


def aggregate_tidy_seed_rows(
    rows: list[dict[str, Any]], key_fields: list[str], value_fields: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in key_fields), []).append(row)

    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate = dict(zip(key_fields, key, strict=True))
        aggregate["seeds"] = "|".join(str(int(row["seed"])) for row in group)
        aggregate["seed_count"] = len(group)
        for field in value_fields:
            values = np.asarray(
                [float(row[field]) for row in group if row.get(field) is not None],
                dtype=np.float64,
            )
            aggregate[f"{field}_mean"] = float(values.mean()) if values.size else None
            aggregate[f"{field}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else None
            )
            aggregate[f"{field}_n"] = int(values.size)
            aggregate[f"{field}_same_sign"] = (
                bool(np.all(values >= 0) or np.all(values <= 0)) if values.size else None
            )
        output.append(aggregate)
    return output
