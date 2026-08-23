"""Raw-event access to prepared official-train DVS-Gesture samples."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from etsr.data.events import EventSample

DVS_GESTURE_CLASSES = (
    "hand_clapping",
    "right_hand_wave",
    "left_hand_wave",
    "right_arm_clockwise",
    "right_arm_counter_clockwise",
    "left_arm_clockwise",
    "left_arm_counter_clockwise",
    "arm_roll",
    "air_drums",
    "air_guitar",
    "other_gestures",
)
SAMPLE_PATTERN = re.compile(r"^user(?P<subject>\d{2})_[A-Za-z0-9_]+__\d+\.npz$")
DevelopmentSplit = Literal["train", "validation"]


@dataclass(frozen=True)
class DvsGestureIndex:
    root: Path
    sample_ids: tuple[str, ...]
    subjects: dict[str, int]
    validation_subjects: tuple[int, ...]
    dataset_index_sha256: str
    development_split_sha256: str
    height: int = 128
    width: int = 128
    classes: tuple[str, ...] = DVS_GESTURE_CLASSES


def _discover_samples(root: Path) -> tuple[list[Path], dict[str, int]]:
    if root.name != "train":
        raise ValueError(
            "DVS-Gesture development root must end in 'train'; official test is not a "
            "development input."
        )
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Prepared DVS-Gesture train root does not exist: {root}")
    entries = [entry for entry in root.iterdir() if not entry.name.startswith(".")]
    if {entry.name for entry in entries} != {str(target) for target in range(11)}:
        raise ValueError("Prepared DVS-Gesture train must contain exactly class directories 0..10.")
    entries.sort(key=lambda entry: int(entry.name))

    samples: list[Path] = []
    subjects: dict[str, int] = {}
    for class_dir in entries:
        if class_dir.is_symlink() or not class_dir.is_dir():
            raise ValueError(f"Invalid DVS-Gesture class directory: {class_dir}")
        class_samples = sorted(class_dir.iterdir(), key=lambda path: path.name)
        if not class_samples:
            raise ValueError(f"Prepared DVS-Gesture class {class_dir.name} is empty.")
        for path in class_samples:
            match = SAMPLE_PATTERN.fullmatch(path.name)
            if path.is_symlink() or not path.is_file() or match is None:
                raise ValueError(f"Invalid prepared DVS-Gesture sample: {path}")
            sample_id = path.relative_to(root).as_posix()
            subjects[sample_id] = int(match.group("subject"))
            samples.append(path)
    return samples, subjects


def _index_sha256(root: Path, samples: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in samples:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_dvsgesture_index(
    train_root: str | Path,
    validation_subjects: list[int] | tuple[int, ...],
) -> DvsGestureIndex:
    """Build and hash one subject-disjoint development split from official train."""

    root = Path(train_root)
    samples, subjects = _discover_samples(root)
    observed_subjects = set(subjects.values())
    selected = tuple(sorted(set(int(subject) for subject in validation_subjects)))
    if not selected:
        raise ValueError("DVS-Gesture validation_subjects must not be empty.")
    if not set(selected).issubset(observed_subjects):
        raise ValueError(
            f"DVS-Gesture validation subjects are absent from official train: "
            f"{sorted(set(selected) - observed_subjects)}"
        )
    if set(selected) == observed_subjects:
        raise ValueError("DVS-Gesture validation_subjects cannot consume every train subject.")

    sample_ids = tuple(path.relative_to(root).as_posix() for path in samples)
    split_digest = hashlib.sha256()
    for sample_id in sample_ids:
        assignment = "validation" if subjects[sample_id] in selected else "train"
        split_digest.update(f"{sample_id}\0{assignment}\n".encode())
    return DvsGestureIndex(
        root=root,
        sample_ids=sample_ids,
        subjects=subjects,
        validation_subjects=selected,
        dataset_index_sha256=_index_sha256(root, samples),
        development_split_sha256=split_digest.hexdigest(),
    )


def _scalar(archive: Any, field: str, path: Path) -> Any:
    value = archive[field]
    if value.shape != ():
        raise ValueError(f"Prepared DVS-Gesture field {field!r} must be scalar: {path}")
    return value.item()


def load_prepared_sample(
    path: str | Path,
    *,
    expected_target: int,
    expected_subject: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and fail-loud validate one prepared gesture segment."""

    sample_path = Path(path)
    required = {
        "t",
        "x",
        "y",
        "p",
        "target",
        "subject",
        "segment_start_us",
        "segment_end_us",
        "source_recording",
    }
    try:
        with np.load(sample_path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError(
                    f"Prepared DVS-Gesture fields are {sorted(archive.files)}, "
                    f"expected {sorted(required)}: {sample_path}"
                )
            events = {field: np.asarray(archive[field]) for field in ("t", "x", "y", "p")}
            metadata = {
                field: _scalar(archive, field, sample_path)
                for field in required - {"t", "x", "y", "p"}
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read prepared DVS-Gesture sample {sample_path}: {exc}") from exc

    lengths = {len(values) for values in events.values() if values.ndim == 1}
    if any(values.ndim != 1 for values in events.values()) or lengths != {len(events["t"])}:
        raise ValueError(f"DVS-Gesture event arrays must be aligned and one-dimensional: {sample_path}")
    if not len(events["t"]):
        raise ValueError(f"DVS-Gesture sample contains no events: {sample_path}")
    for field, values in events.items():
        if field != "p" and not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"DVS-Gesture event field {field!r} must be integer: {sample_path}")
    if np.any(events["t"][1:] < events["t"][:-1]):
        raise ValueError(f"DVS-Gesture timestamps are not monotonic: {sample_path}")
    if int(events["x"].min()) < 0 or int(events["x"].max()) >= 128:
        raise ValueError(f"DVS-Gesture x coordinates exceed [0,128): {sample_path}")
    if int(events["y"].min()) < 0 or int(events["y"].max()) >= 128:
        raise ValueError(f"DVS-Gesture y coordinates exceed [0,128): {sample_path}")
    polarity_values = {int(value) for value in np.unique(events["p"])}
    if not polarity_values.issubset({0, 1}):
        raise ValueError(f"DVS-Gesture polarity must be 0/1: {sample_path}")
    if int(metadata["target"]) != expected_target or int(metadata["subject"]) != expected_subject:
        raise ValueError(f"DVS-Gesture path and embedded target/subject disagree: {sample_path}")
    start = int(metadata["segment_start_us"])
    end = int(metadata["segment_end_us"])
    if end <= start or int(events["t"].min()) < 0 or int(events["t"].max()) >= end - start:
        raise ValueError(f"DVS-Gesture timestamps exceed their annotated segment: {sample_path}")
    return events, metadata


class DvsGestureDataset:
    """Subject-disjoint development view of prepared official-train gestures."""

    def __init__(self, dataset_index: DvsGestureIndex, split: DevelopmentSplit) -> None:
        if split not in ("train", "validation"):
            raise ValueError("DVS-Gesture development split must be 'train' or 'validation'.")
        self.root = dataset_index.root
        self.split: DevelopmentSplit = split
        self.height = dataset_index.height
        self.width = dataset_index.width
        self.classes = list(dataset_index.classes)
        self.class_to_idx = {name: target for target, name in enumerate(self.classes)}
        validation_subjects = set(dataset_index.validation_subjects)
        self.sample_ids = [
            sample_id
            for sample_id in dataset_index.sample_ids
            if (dataset_index.subjects[sample_id] in validation_subjects) == (split == "validation")
        ]
        self.targets = tuple(int(Path(sample_id).parts[0]) for sample_id in self.sample_ids)
        self.subjects = tuple(dataset_index.subjects[sample_id] for sample_id in self.sample_ids)
        self.dataset_index_sha256 = dataset_index.dataset_index_sha256
        self.development_split_sha256 = dataset_index.development_split_sha256
        self.runtime_metadata = {
            "dataset_index_sha256": self.dataset_index_sha256,
            "development_split_sha256": self.development_split_sha256,
            "validation_subjects": list(dataset_index.validation_subjects),
            "speaker_disjoint": True,
        }

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> EventSample:
        sample_id = self.sample_ids[index]
        target = self.targets[index]
        subject = self.subjects[index]
        events, source = load_prepared_sample(
            self.root / sample_id,
            expected_target=target,
            expected_subject=subject,
        )
        duration_us = int(source["segment_end_us"]) - int(source["segment_start_us"])
        return EventSample(
            x=events["x"],
            y=events["y"],
            t_us=events["t"],
            polarity=events["p"],
            target=target,
            sample_id=sample_id,
            speaker_id=subject,
            duration_us=duration_us,
            metadata={
                "class_name": self.classes[target],
                "development_split": self.split,
                "official_source_split": "train",
                "official_test_used": False,
                "speaker_identity_available": True,
                "speaker_disjoint": True,
                "source_recording": str(source["source_recording"]),
                "annotated_segment_us": [
                    int(source["segment_start_us"]),
                    int(source["segment_end_us"]),
                ],
                "time_unit": "microsecond",
                "timestamp_origin": "annotated_segment_start",
                "timestamp_normalized": False,
                "polarity_convention": {"OFF": 0, "ON": 1},
                "sensor_size": [self.height, self.width],
            },
        )
