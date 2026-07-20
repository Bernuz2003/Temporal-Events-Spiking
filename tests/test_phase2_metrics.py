import numpy as np

from etsr.phase2.metrics import (
    aggregate_tidy_seed_rows,
    factorized_content_order_metrics,
    inverse_temporal_consistency,
    normalized_trapezoid_auc,
    prefix_trajectory_metrics,
    reverse_class_map,
    trapezoid_auc,
)

CLASSES = ["13", "18", "31", "38", "81", "83"]


def test_factorized_metrics_separate_content_from_order():
    targets = np.asarray([0, 2, 1, 4])
    predictions = np.asarray([2, 2, 4, 1])

    result = factorized_content_order_metrics(targets, predictions, CLASSES)

    assert result["content_accuracy"] == 1.0
    assert result["conditional_order_accuracy"] == 0.25


def test_reverse_map_supports_explicit_multicharacter_tokens():
    classes = ["walk->clap", "clap->walk"]
    assert reverse_class_map(classes) == {0: 1, 1: 0}


def test_inverse_consistency_is_a_paired_prediction_property():
    predictions = np.asarray([0, 1, 2, 5])
    reverse_predictions = np.asarray([2, 4, 0, 3])
    targets = np.asarray([0, 1, 2, 5])

    result = inverse_temporal_consistency(predictions, reverse_predictions, targets, CLASSES)

    assert result["inverse_temporal_consistency"] == 1.0


def test_prefix_auc_reports_raw_and_interval_normalized_values():
    x = np.asarray([0.25, 0.5, 0.75, 1.0])
    y = np.ones(4)

    assert trapezoid_auc(x, y) == 0.75
    assert normalized_trapezoid_auc(x, y) == 1.0


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
    tail = next(row for row in transitions if row["from_timestep"] == 2 and row["to_timestep"] == 3)

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
