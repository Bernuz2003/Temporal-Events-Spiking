import numpy as np
import torch

from etsr.evaluation.metrics import (
    ClassificationAccumulator,
    normalized_prefix_auc,
    paired_prediction_analysis,
    prefix_auc,
)


def test_classification_metrics():
    accumulator = ClassificationAccumulator(num_classes=2)
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0], [2.0, 3.0]])
    targets = torch.tensor([0, 1, 1, 1])
    accumulator.update(logits, targets, torch.tensor(0.5), torch.arange(4))
    result = accumulator.compute()
    assert result.accuracy == 0.75
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.confusion_matrix.tolist() == [[1, 0], [1, 2]]


def test_prefix_auc():
    area = prefix_auc([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    assert abs(area - 0.5) < 1e-9


def test_normalized_prefix_auc():
    raw_area = prefix_auc([0.25, 0.5, 0.75, 1.0], [1.0, 1.0, 1.0, 1.0])
    normalized_area = normalized_prefix_auc(
        [0.25, 0.5, 0.75, 1.0], [1.0, 1.0, 1.0, 1.0]
    )
    assert raw_area == 0.75
    assert normalized_area == 1.0


def test_paired_prediction_analysis_aligns_samples_by_index():
    original = {
        "indices": np.array([20, 10, 30, 40]),
        "targets": np.array([1, 0, 1, 0]),
        "predictions": np.array([1, 1, 0, 0]),
    }
    condition = {
        "indices": np.array([30, 20, 40, 10]),
        "targets": np.array([1, 1, 0, 0]),
        "predictions": np.array([1, 0, 0, 1]),
    }

    result = paired_prediction_analysis(original, condition, num_classes=2)

    assert result["samples"] == 4
    assert result["prediction_changed_count"] == 2
    assert result["prediction_changed_rate"] == 0.5
    assert result["target_changed_count"] == 0
    assert result["correct_to_incorrect_count"] == 1
    assert result["correct_to_incorrect_rate_given_original_correct"] == 0.5
    assert result["incorrect_to_correct_count"] == 1
    assert result["incorrect_to_correct_rate_given_original_incorrect"] == 0.5
    assert result["prediction_transition_matrix"] == [[1, 1], [1, 1]]


def test_paired_prediction_analysis_marks_undefined_conditional_rate():
    predictions = {
        "indices": np.array([0, 1]),
        "targets": np.array([0, 1]),
        "predictions": np.array([1, 0]),
    }

    result = paired_prediction_analysis(predictions, predictions, num_classes=2)

    assert result["correct_to_incorrect_rate_given_original_correct"] is None
