"""Canonical train-only raw-event access for DVS-Lip."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from etsr.data.events import EventSample

REQUIRED_EVENT_FIELDS = ("t", "x", "y", "p")
DevelopmentSplit = Literal["train", "validation"]


@dataclass(frozen=True)
class DvsLipExpectations:
    """Published official-train invariants."""

    class_count: int = 100
    sample_count: int = 14_896
    height: int = 128
    width: int = 128


def discover_training_samples(
    train_root: str | Path,
    expectations: DvsLipExpectations,
) -> tuple[Path, list[Path], dict[str, int]]:
    """Enumerate official train without resolving or traversing a test sibling."""

    root = Path(train_root)
    if root.name != "train":
        raise ValueError(
            "DVS-Lip accepts only a path whose final component is 'train'; "
            "the official test partition must not be passed."
        )
    if root.is_symlink():
        raise ValueError("The training root must not be a symlink.")
    if not root.is_dir():
        raise FileNotFoundError(f"DVS-Lip training root does not exist: {root}")

    root_entries = sorted(
        (entry for entry in root.iterdir() if not entry.name.startswith(".")),
        key=lambda entry: entry.name,
    )
    class_dirs = [entry for entry in root_entries if not entry.is_symlink() and entry.is_dir()]
    unexpected = [entry.name for entry in root_entries if entry not in class_dirs]
    if unexpected:
        raise ValueError(f"Unexpected entries directly under the training root: {unexpected}")
    if len(class_dirs) != expectations.class_count:
        raise ValueError(
            f"Expected {expectations.class_count} class directories, found {len(class_dirs)}."
        )

    samples: list[Path] = []
    class_counts: dict[str, int] = {}
    for class_dir in class_dirs:
        entries = sorted(
            (entry for entry in class_dir.iterdir() if not entry.name.startswith(".")),
            key=lambda entry: entry.name,
        )
        invalid = [
            entry.name
            for entry in entries
            if entry.is_symlink()
            or not entry.is_file()
            or entry.suffix != ".npy"
            or not entry.stem.isdecimal()
        ]
        if invalid:
            raise ValueError(
                f"Invalid entries in class {class_dir.name!r}; "
                f"expected integer .npy files: {invalid}"
            )
        if not entries:
            raise ValueError(f"DVS-Lip class {class_dir.name!r} is empty.")
        class_counts[class_dir.name] = len(entries)
        samples.extend(entries)

    if len(samples) != expectations.sample_count:
        raise ValueError(
            f"Expected {expectations.sample_count} official-train samples, found {len(samples)}."
        )
    return root, samples, class_counts


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
class DvsLipIndex:
    """One validated index shared by all development-split views."""

    root: Path
    sample_ids: tuple[str, ...]
    assignments: dict[str, str]
    classes: tuple[str, ...]
    height: int
    width: int
    dataset_index_sha256: str
    split_manifest_sha256: str


def load_dvslip_index(
    train_root: str | Path,
    split_manifest: str | Path,
    *,
    expectations: DvsLipExpectations | None = None,
) -> DvsLipIndex:
    """Discover and validate the development archive once per workflow."""

    from etsr.dvslip.split import load_development_split_manifest

    expected = expectations or DvsLipExpectations()
    root, discovered, class_counts = discover_training_samples(train_root, expected)
    sample_ids = tuple(path.relative_to(root).as_posix() for path in discovered)
    assignments, split_summary = load_development_split_manifest(split_manifest, list(sample_ids))
    return DvsLipIndex(
        root=root,
        sample_ids=sample_ids,
        assignments=assignments,
        classes=tuple(sorted(class_counts)),
        height=int(expected.height),
        width=int(expected.width),
        dataset_index_sha256=dataset_index_sha256(root, discovered),
        split_manifest_sha256=str(split_summary["sha256"]),
    )


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
        dataset_index: DvsLipIndex,
        split: DevelopmentSplit,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("DVS-Lip development split must be 'train' or 'validation'.")

        self.root = dataset_index.root
        self.split: DevelopmentSplit = split
        self.height = dataset_index.height
        self.width = dataset_index.width
        self.classes = list(dataset_index.classes)
        self.class_to_idx = {class_name: target for target, class_name in enumerate(self.classes)}
        self.dataset_index_sha256 = dataset_index.dataset_index_sha256
        self.sample_ids = sorted(
            sample_id
            for sample_id in dataset_index.sample_ids
            if dataset_index.assignments[sample_id] == split
        )
        self.targets = tuple(
            self.class_to_idx[Path(sample_id).parts[0]] for sample_id in self.sample_ids
        )
        self.split_manifest_sha256 = dataset_index.split_manifest_sha256
        self.runtime_metadata = {
            "dataset_index_sha256": self.dataset_index_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
        }

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> EventSample:
        sample_id = self.sample_ids[index]
        path = self.root / sample_id
        events = load_event_array(path, height=self.height, width=self.width)
        timestamps = events["t"]
        class_name = Path(sample_id).parts[0]

        return EventSample(
            x=events["x"],
            y=events["y"],
            t_us=timestamps,
            polarity=events["p"],
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
