"""Exhaustive validation and profile of prepared official-train DVS-Gesture."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from etsr.dvsgesture.dataset import (
    DVS_GESTURE_CLASSES,
    _discover_samples,
    _index_sha256,
    load_prepared_sample,
)
from etsr.utils.io import write_json


def _summary(values: list[int]) -> dict[str, int]:
    array = np.asarray(values, dtype=np.uint64)
    return {
        "minimum": int(array.min()),
        "p05": int(np.percentile(array, 5)),
        "median": int(np.median(array)),
        "p95": int(np.percentile(array, 95)),
        "maximum": int(array.max()),
    }


def run_dvsgesture_profile(
    train_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(train_root)
    samples, subjects = _discover_samples(root)
    class_counts: Counter[int] = Counter()
    subject_counts: Counter[int] = Counter()
    durations: list[int] = []
    event_counts: list[int] = []
    maximum_relative_timestamp = 0

    for path in samples:
        sample_id = path.relative_to(root).as_posix()
        target = int(Path(sample_id).parts[0])
        subject = subjects[sample_id]
        events, metadata = load_prepared_sample(
            path,
            expected_target=target,
            expected_subject=subject,
        )
        duration = int(metadata["segment_end_us"]) - int(metadata["segment_start_us"])
        class_counts[target] += 1
        subject_counts[subject] += 1
        durations.append(duration)
        event_counts.append(len(events["t"]))
        maximum_relative_timestamp = max(maximum_relative_timestamp, int(events["t"][-1]))

    report = {
        "schema_version": 1,
        "dataset": {
            "name": "dvsgesture",
            "source_split": "train",
            "sample_count": len(samples),
            "class_count": len(DVS_GESTURE_CLASSES),
            "classes": list(DVS_GESTURE_CLASSES),
            "subject_count": len(subject_counts),
            "sensor_size": [128, 128],
            "dataset_index_sha256": _index_sha256(root, samples),
        },
        "class_counts": {str(target): class_counts[target] for target in range(11)},
        "subject_counts": {str(subject): subject_counts[subject] for subject in sorted(subject_counts)},
        "duration_us": _summary(durations),
        "event_count": _summary(event_counts),
        "maximum_relative_timestamp_us": maximum_relative_timestamp,
        "validation": {"all_samples_valid": True},
        "official_test_used": False,
    }
    write_json(report, output_path)
    return report
