from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have the same shape.")
    if np.any((targets < 0) | (targets >= num_classes)):
        raise ValueError("Targets are outside the class range.")
    if np.any((predictions < 0) | (predictions >= num_classes)):
        raise ValueError("Predictions are outside the class range.")
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


def _class_tokens(class_name: str) -> tuple[str, ...]:
    """Decode a class label without tying metrics to single-character primitives."""
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
    if np.any((targets < 0) | (targets >= len(classes))):
        raise ValueError("Targets are outside the class range.")
    if np.any((predictions < 0) | (predictions >= len(classes))):
        raise ValueError("Predictions are outside the class range.")
    signatures = _content_signatures(classes)
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
        pair_samples = int(mask.sum())
        if pair_samples == 0:
            continue
        pair_content = int(content_correct[mask].sum())
        pair_exact = int(exact_correct[mask].sum())
        pair_rows.append(
            {
                "content": "|".join(signature),
                "classes": [classes[index] for index in class_indices],
                "samples": pair_samples,
                "content_accuracy": pair_content / pair_samples,
                "conditional_order_accuracy": (pair_exact / pair_content if pair_content else None),
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
        reverse_name = "->".join(reversed(tokens)) if "->" in name else "".join(reversed(tokens))
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
    rows = []
    signatures = _content_signatures(classes)
    for signature in sorted(set(signatures)):
        indices = [index for index, value in enumerate(signatures) if value == signature]
        mask = np.isin(targets, indices)
        if mask.any():
            rows.append(
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
        "per_content_pair": rows,
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
        factorized = factorized_content_order_metrics(targets, predictions[:, timestep], classes)
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

    transition_rows = []
    pairs = [(index, index + 1) for index in range(logits.shape[1] - 1)]
    if 0 < tail_start < logits.shape[1] and (tail_start - 1, logits.shape[1] - 1) not in pairs:
        pairs.append((tail_start - 1, logits.shape[1] - 1))
    for before, after in pairs:
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
    arrays = {"predictions": predictions, "margins": margins}
    return prefix_rows, transition_rows, arrays


def trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        raise ValueError("AUC requires aligned arrays with at least two points.")
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def normalized_trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        raise ValueError("AUC requires aligned arrays with at least two points.")
    order = np.argsort(x)
    width = float(x[order][-1] - x[order][0])
    if width <= 0:
        raise ValueError("AUC interval must have positive width.")
    return trapezoid_auc(x[order], y[order]) / width


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
        values = np.asarray([float(value) for value in raw_values], dtype=np.float64)
        result[field] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "same_sign": bool(np.all(values >= 0) or np.all(values <= 0)),
        }
    return result


def aggregate_tidy_seed_rows(
    rows: list[dict[str, Any]],
    key_fields: list[str],
    value_fields: list[str],
) -> list[dict[str, Any]]:
    """Aggregate aligned tidy rows without silently mixing different conditions."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        groups.setdefault(key, []).append(row)

    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate = dict(zip(key_fields, key, strict=True))
        aggregate["seeds"] = "|".join(str(int(row["seed"])) for row in group)
        aggregate["seed_count"] = len(group)
        for field in value_fields:
            values = [row.get(field) for row in group]
            valid = np.asarray(
                [float(value) for value in values if value is not None], dtype=np.float64
            )
            aggregate[f"{field}_mean"] = float(valid.mean()) if valid.size else None
            aggregate[f"{field}_std"] = (
                float(valid.std(ddof=1)) if valid.size > 1 else 0.0 if valid.size else None
            )
            aggregate[f"{field}_n"] = int(valid.size)
            aggregate[f"{field}_same_sign"] = (
                bool(np.all(valid >= 0) or np.all(valid <= 0)) if valid.size else None
            )
        output.append(aggregate)
    return output
