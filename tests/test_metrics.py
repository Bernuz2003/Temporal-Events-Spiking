import numpy as np
import torch

from etsr.evaluation.metrics import ClassificationAccumulator, classification_metrics


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
