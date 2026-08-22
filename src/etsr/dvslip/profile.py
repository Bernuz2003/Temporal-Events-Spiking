"""Decision-oriented statistics for the official-train DVS-Lip archive."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from etsr.dvslip.dataset import load_event_array
from etsr.utils.io import ensure_dir, write_json

if TYPE_CHECKING:
    from etsr.dvslip.preflight import DvsLipExpectations


def _distribution(values: list[int] | list[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    p05, p25, median, p75, p95 = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p05": float(p05),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "p95": float(p95),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


@dataclass
class _ProfileAccumulator:
    durations_us: list[int] = field(default_factory=list)
    events_per_sample: list[int] = field(default_factory=list)
    event_rates_hz: list[float] = field(default_factory=list)
    on_fractions: list[float] = field(default_factory=list)
    spatial_occupancies: list[float] = field(default_factory=list)
    on_events: int = 0
    off_events: int = 0
    zero_duration_samples: int = 0

    def add(self, events: np.ndarray, *, height: int, width: int) -> None:
        event_count = int(events.shape[0])
        duration_us = int(events["t"][-1]) - int(events["t"][0])
        on_count = int(np.count_nonzero(events["p"] == 1))
        off_count = event_count - on_count
        pixel_ids = events["y"].astype(np.int64) * width + events["x"].astype(np.int64)

        self.durations_us.append(duration_us)
        self.events_per_sample.append(event_count)
        self.on_fractions.append(on_count / event_count)
        self.spatial_occupancies.append(int(np.unique(pixel_ids).size) / float(height * width))
        self.on_events += on_count
        self.off_events += off_count
        if duration_us > 0:
            self.event_rates_hz.append(event_count * 1_000_000.0 / duration_us)
        else:
            self.zero_duration_samples += 1

    def summary(self) -> dict[str, Any]:
        total_events = self.on_events + self.off_events
        return {
            "sample_count": len(self.events_per_sample),
            "event_count": total_events,
            "duration_us": _distribution(self.durations_us),
            "events_per_sample": _distribution(self.events_per_sample),
            "event_rate_hz": {
                **_distribution(self.event_rates_hz),
                "undefined_zero_duration_samples": self.zero_duration_samples,
            },
            "polarity": {
                "on_events": self.on_events,
                "off_events": self.off_events,
                "on_fraction": self.on_events / total_events if total_events else None,
                "off_fraction": self.off_events / total_events if total_events else None,
                "per_sample_on_fraction": _distribution(self.on_fractions),
            },
            "spatial_occupancy_fraction": _distribution(self.spatial_occupancies),
        }


def run_dvslip_profile(
    train_root: str | Path,
    split_manifest: str | Path,
    output_path: str | Path,
    *,
    expectations: DvsLipExpectations | None = None,
) -> dict[str, Any]:
    """Validate every official-train sample and write one profile for scale selection."""

    from etsr.dvslip.preflight import DvsLipExpectations, discover_training_samples
    from etsr.dvslip.split import load_development_split_manifest

    expected = expectations or DvsLipExpectations()
    root, samples, class_counts = discover_training_samples(train_root, expected)
    relative_paths = [sample.relative_to(root).as_posix() for sample in samples]
    assignments, split_summary = load_development_split_manifest(split_manifest, relative_paths)

    overall = _ProfileAccumulator()
    by_class: dict[str, _ProfileAccumulator] = defaultdict(_ProfileAccumulator)
    by_split: dict[str, _ProfileAccumulator] = defaultdict(_ProfileAccumulator)
    for path, sample_id in zip(samples, relative_paths, strict=True):
        events = load_event_array(path, height=expected.height, width=expected.width)
        class_name = Path(sample_id).parts[0]
        development_split = assignments[sample_id]
        overall.add(events, height=expected.height, width=expected.width)
        by_class[class_name].add(events, height=expected.height, width=expected.width)
        by_split[development_split].add(events, height=expected.height, width=expected.width)

    report = {
        "schema_version": 1,
        "official_source_split": "train",
        "official_test_used": False,
        "sensor_size": [expected.height, expected.width],
        "dataset": {
            "sample_count": len(samples),
            "class_count": len(class_counts),
            "class_counts": class_counts,
        },
        "split_manifest": split_summary,
        "validation": {
            "all_samples_valid": True,
            "validated_sample_count": len(samples),
            "missing_sample_count": 0,
            "corrupt_sample_count": 0,
        },
        "speaker_statistics": {
            "available": False,
            "reason": "authoritative_sample_to_speaker_mapping_unavailable",
        },
        "overall": overall.summary(),
        "development_splits": {
            split: by_split[split].summary() for split in ("train", "validation")
        },
        "classes": {
            class_name: by_class[class_name].summary() for class_name in sorted(class_counts)
        },
    }
    ensure_dir(Path(output_path).parent)
    write_json(report, output_path)
    return report
