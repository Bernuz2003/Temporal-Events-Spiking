"""Canonical train-only raw-event access for DVS-Lip."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from etsr.dvslip.preflight import DvsLipExpectations

REQUIRED_EVENT_FIELDS = ("t", "x", "y", "p")
DevelopmentSplit = Literal["train", "validation"]


def dataset_index_sha256(train_root: str | Path, samples: list[Path]) -> str:
    """Hash sample identities and sizes without reading the official-test sibling."""

    root = Path(train_root)
    digest = hashlib.sha256()
    for sample in samples:
        relative_path = sample.relative_to(root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(sample.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class EventSample:
    """One DVS-Lip sample with its original event values and stable identity."""

    x: np.ndarray
    y: np.ndarray
    t_us: np.ndarray
    polarity: np.ndarray
    target: int
    sample_id: str
    speaker_id: str | int | None
    duration_us: int
    metadata: dict[str, Any]


def load_event_array(
    path: str | Path,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Load and validate one released structured array without changing its values."""

    sample_path = Path(path)
    try:
        events = np.load(sample_path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read DVS-Lip sample {sample_path}: {exc}") from exc

    if not isinstance(events, np.ndarray):
        raise ValueError(f"DVS-Lip sample is not a NumPy array: {sample_path}")
    field_order = events.dtype.names
    if field_order is None:
        raise ValueError(
            f"Sample {sample_path} is not a structured array. Do not infer dense column semantics; "
            "record an explicit conversion decision first."
        )
    if events.ndim != 1 or not events.shape[0]:
        raise ValueError(
            f"Expected a non-empty one-dimensional structured event array at {sample_path}; "
            f"found shape {events.shape}."
        )
    if set(field_order) != set(REQUIRED_EVENT_FIELDS) or len(field_order) != 4:
        raise ValueError(
            f"Sample {sample_path} fields are {field_order}; expected exactly "
            f"{REQUIRED_EVENT_FIELDS}."
        )

    for field in REQUIRED_EVENT_FIELDS:
        dtype = events.dtype.fields[field][0]
        if dtype.shape or not np.issubdtype(dtype, np.integer):
            raise ValueError(
                f"Sample {sample_path} field {field!r} must be a scalar integer, found {dtype}."
            )

    timestamps = events["t"]
    x_coords = events["x"]
    y_coords = events["y"]
    polarities = events["p"]
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError(f"Sample timestamps are not monotonic at {sample_path}.")
    if int(timestamps.min()) < 0:
        raise ValueError(f"Sample contains negative timestamps at {sample_path}.")
    if int(x_coords.min()) < 0 or int(x_coords.max()) >= width:
        raise ValueError(f"Sample x coordinates exceed [0,{width}) at {sample_path}.")
    if int(y_coords.min()) < 0 or int(y_coords.max()) >= height:
        raise ValueError(f"Sample y coordinates exceed [0,{height}) at {sample_path}.")
    polarity_values = {int(value) for value in np.unique(polarities)}
    if not polarity_values.issubset({0, 1}):
        raise ValueError(
            f"Sample polarity values must be a subset of [0, 1] at {sample_path}; "
            f"found {sorted(polarity_values)}."
        )
    return events


def summarize_event_array(events: np.ndarray) -> dict[str, Any]:
    """Return the serializable schema/range evidence used by the preflight report."""

    field_order = events.dtype.names
    if field_order is None:  # Defensive: load_event_array already rejects this case.
        raise ValueError("Cannot summarize an unstructured DVS-Lip event array.")
    values = {field: np.asarray(events[field]) for field in REQUIRED_EVENT_FIELDS}
    return {
        "event_count": int(events.shape[0]),
        "structured_dtype": str(events.dtype),
        "field_order": list(field_order),
        "field_dtypes": {
            field: events.dtype.fields[field][0].str for field in REQUIRED_EVENT_FIELDS
        },
        "timestamp_min": int(values["t"].min()),
        "timestamp_max": int(values["t"].max()),
        "timestamps_monotonic": True,
        "x_range": [int(values["x"].min()), int(values["x"].max())],
        "y_range": [int(values["y"].min()), int(values["y"].max())],
        "polarity_values": sorted(int(value) for value in np.unique(values["p"])),
    }


class DvsLipDataset:
    """A deterministic view of one development split from official ``train/`` only."""

    def __init__(
        self,
        train_root: str | Path,
        split_manifest: str | Path,
        split: DevelopmentSplit,
        *,
        expectations: DvsLipExpectations | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("DVS-Lip development split must be 'train' or 'validation'.")

        # Local imports keep the raw contract independent of preflight orchestration and Torch.
        from etsr.dvslip.preflight import DvsLipExpectations, discover_training_samples
        from etsr.dvslip.split import load_development_split_manifest

        expected = expectations or DvsLipExpectations()
        root, discovered, class_counts = discover_training_samples(train_root, expected)
        relative_paths = [path.relative_to(root).as_posix() for path in discovered]
        assignments, split_summary = load_development_split_manifest(split_manifest, relative_paths)

        self.root = root
        self.split: DevelopmentSplit = split
        self.height = int(expected.height)
        self.width = int(expected.width)
        self.classes = sorted(class_counts)
        self.class_to_idx = {class_name: target for target, class_name in enumerate(self.classes)}
        self.dataset_index_sha256 = dataset_index_sha256(root, discovered)
        self.sample_ids = sorted(
            sample_id for sample_id, assignment in assignments.items() if assignment == split
        )
        self.split_manifest_sha256 = str(split_summary["sha256"])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> EventSample:
        sample_id = self.sample_ids[index]
        path = self.root / sample_id
        events = load_event_array(path, height=self.height, width=self.width)
        timestamps = np.asarray(events["t"]).copy()
        class_name = Path(sample_id).parts[0]

        return EventSample(
            x=np.asarray(events["x"]).copy(),
            y=np.asarray(events["y"]).copy(),
            t_us=timestamps,
            polarity=np.asarray(events["p"]).copy(),
            target=self.class_to_idx[class_name],
            sample_id=sample_id,
            speaker_id=None,
            duration_us=int(timestamps[-1]) - int(timestamps[0]),
            metadata={
                "class_name": class_name,
                "development_split": self.split,
                "official_source_split": "train",
                "official_test_used": False,
                "speaker_identity_available": False,
                "time_unit": "microsecond",
                "timestamp_normalized": False,
                "polarity_convention": {"OFF": 0, "ON": 1},
                "sensor_size": [self.height, self.width],
                "structured_dtype": str(events.dtype),
                "split_manifest_sha256": self.split_manifest_sha256,
            },
        )
