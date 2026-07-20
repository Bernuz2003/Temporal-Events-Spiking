import numpy as np
import torch

from etsr.evaluation.metrics import (
    ClassificationAccumulator,
    aggregate_tidy_seed_rows,
    factorized_content_order_metrics,
    inverse_temporal_consistency,
    normalized_prefix_auc,
    paired_prediction_analysis,
    prefix_auc,
    prefix_trajectory_metrics,
    reverse_class_map,
)

CLASSES = ["13", "18", "31", "38", "81", "83"]


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


def test_factorized_metrics_separate_content_from_order():
    targets = np.asarray([0, 2, 1, 4])
    predictions = np.asarray([2, 2, 4, 1])

    result = factorized_content_order_metrics(targets, predictions, CLASSES)

    assert result["content_accuracy"] == 1.0
    assert result["conditional_order_accuracy"] == 0.25


def test_reverse_map_supports_explicit_multicharacter_tokens():
    classes = ["walk->clap", "clap->walk"]
    assert reverse_class_map(classes) == {0: 1, 1: 0}


def test_inverse_consistency_reports_unique_pair_correctness():
    predictions = np.asarray([0, 1, 2])
    reverse_predictions = np.asarray([2, 4, 0])
    targets = np.asarray([0, 1, 3])

    result = inverse_temporal_consistency(predictions, reverse_predictions, targets, CLASSES)

    assert result["inverse_temporal_consistency"] == 1.0
    assert result["pairs"] == 3
    assert result["canonical_accuracy"] == 2 / 3
    assert result["reverse_target_accuracy"] == 2 / 3
    assert result["joint_pair_accuracy"] == 2 / 3


def test_prefix_metrics_measure_late_harm_and_rescue():
    targets = np.asarray([0, 1])
    logits = np.asarray(
        [
            [[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]],
            [[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]],
        ]
    )

    _prefix, transitions, _arrays = prefix_trajectory_metrics(
        logits, targets, ["13", "31"], tail_start=2
    )
    tail = next(row for row in transitions if row["from_timestep"] == 2)

    assert tail["late_harm_rate"] == 0.5
    assert tail["late_rescue_rate"] == 0.5
    assert tail["net_late_utility"] == 0.0


def test_tidy_aggregation_never_mixes_conditions():
    rows = [
        {"seed": 42, "layer": "stage1", "region": "first", "effect": -2.0},
        {"seed": 123, "layer": "stage1", "region": "first", "effect": -1.0},
        {"seed": 42, "layer": "stage1", "region": "second", "effect": 3.0},
        {"seed": 123, "layer": "stage1", "region": "second", "effect": 1.0},
    ]

    result = aggregate_tidy_seed_rows(rows, ["layer", "region"], ["effect"])

    assert len(result) == 2
    first = next(row for row in result if row["region"] == "first")
    assert first["effect_mean"] == -1.5
    assert first["effect_same_sign"] is True
