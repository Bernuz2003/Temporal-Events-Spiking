import numpy as np
import pytest
import torch

from etsr.evaluation.metrics import (
    ClassificationAccumulator,
    classification_metrics,
    grouped_accuracies,
    interval_normalized_auc,
    paired_confusions,
    trapezoidal_auc,
)


def test_classification_accumulator_keeps_only_aggregate_metrics():
    accumulator = ClassificationAccumulator(num_classes=2)
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0], [2.0, 3.0]])
    targets = torch.tensor([0, 1, 1, 1])

    accumulator.update(logits, targets, torch.tensor(0.5), torch.arange(4))
    result = accumulator.compute()

    assert result.accuracy == 0.75
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.confusion_matrix.tolist() == [[1, 0], [1, 2]]


def test_numpy_metrics_match_the_training_aggregates():
    result = classification_metrics(
        np.array([0, 1, 1, 1]),
        np.array([0, 1, 0, 1]),
        num_classes=2,
    )

    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == [[1, 0], [1, 2]]


def test_grouped_accuracies_select_rows_by_true_class():
    matrix = torch.tensor([[4, 1, 0], [0, 3, 1], [1, 0, 1]])

    result = grouped_accuracies(
        matrix,
        ["zero", "one", "two"],
        {"first_two": ["zero", "one"], "last": ["two"]},
    )

    assert result["first_two"] == {
        "accuracy": pytest.approx(7 / 9),
        "samples": 9,
        "class_count": 2,
    }
    assert result["last"] == {
        "accuracy": 0.5,
        "samples": 2,
        "class_count": 1,
    }


def test_auc_reports_raw_and_interval_normalized_values_on_either_time_axis():
    points = [250_000, 500_000, 1_000_000]
    accuracies = [0.2, 0.4, 0.8]

    assert trapezoidal_auc(points, accuracies) == pytest.approx(412_500)
    assert interval_normalized_auc(points, accuracies) == pytest.approx(0.55)

    with pytest.raises(ValueError, match="strictly increasing"):
        trapezoidal_auc([0.5, 0.5], [0.2, 0.3])


def test_paired_confusions_distinguish_direct_pair_errors_from_all_group_errors():
    matrix = torch.tensor(
        [
            [3, 2, 0, 0],
            [1, 2, 1, 0],
            [0, 1, 2, 1],
            [0, 0, 2, 3],
        ]
    )

    result = paired_confusions(
        matrix,
        ["a", "b", "c", "d"],
        [["a", "b"], ["c", "d"]],
    )

    assert result["paired_class_samples"] == 18
    assert result["paired_class_errors"] == 8
    assert result["within_pair_errors"] == 6
    assert result["within_pair_errors_per_paired_sample"] == pytest.approx(1 / 3)
    assert result["within_pair_share_of_paired_class_errors"] == pytest.approx(0.75)
