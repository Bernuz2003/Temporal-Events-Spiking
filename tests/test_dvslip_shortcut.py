import numpy as np
import pytest

from etsr.data.events import EventSample
from etsr.dvslip.shortcut import (
    _fit_logistic_control,
    align_prediction_shortcuts,
    eta_squared,
    sample_shortcut_statistics,
)


def test_shortcut_statistics_separate_raw_duration_from_e0_observable_bins():
    sample = EventSample(
        x=np.array([0, 1, 2, 3]),
        y=np.array([0, 1, 2, 3]),
        t_us=np.array([0, 49_999, 50_000, 199_999]),
        polarity=np.array([0, 1, 1, 0]),
        target=0,
        sample_id="word/0.npy",
        speaker_id=None,
        duration_us=199_999,
        metadata={},
    )

    statistics = sample_shortcut_statistics(sample, 50_000)

    assert statistics == {
        "duration_us": 199_999,
        "event_count": 4,
        "on_fraction": 0.5,
        "active_time_bins": 3,
    }


def test_eta_squared_reports_class_explained_scalar_variance():
    assert eta_squared(np.array([0.0, 0.0, 10.0, 10.0]), np.array([0, 0, 1, 1])) == 1.0
    assert eta_squared(np.ones(4), np.array([0, 0, 1, 1])) == 0.0


def test_fixed_logistic_control_learns_a_separable_global_shortcut():
    train_features = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    train_targets = np.array([0, 0, 0, 1, 1, 1])
    validation_features = np.array([[-1.5], [1.5]])
    validation_targets = np.array([0, 1])

    result = _fit_logistic_control(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        2,
        l2_penalty=1e-4,
        max_iterations=40,
    )

    assert result["validation"]["accuracy"] == 1.0
    assert result["optimization"]["iterations"] <= 40


def test_final_prediction_artifact_uses_stable_dataset_indices():
    samples = [
        EventSample(
            x=np.array([0, 1]),
            y=np.array([0, 1]),
            t_us=np.array([0, duration]),
            polarity=np.array([0, 1]),
            target=index,
            sample_id=f"word/{index}.npy",
            speaker_id=None,
            duration_us=duration,
            metadata={},
        )
        for index, duration in enumerate((49_999, 149_999))
    ]

    class Dataset:
        raw_dataset = samples

    rows, summary = align_prediction_shortcuts(
        Dataset(),
        {
            "indices": np.array([1, 0]),
            "targets": np.array([1, 0]),
            "predictions": np.array([0, 0]),
            "margins": np.array([0.8, 0.2]),
        },
        bin_width_us=50_000,
    )

    assert [row["sample_id"] for row in rows] == ["word/1.npy", "word/0.npy"]
    assert summary["pearson_with_correctness"]["active_time_bins"] == pytest.approx(-1.0)
