"""Train-only preparation of the official DVS-Gesture AEDAT recordings."""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from etsr.dvsgesture.aedat import load_aedat_v3
from etsr.utils.io import write_json

RECORDING_PATTERN = re.compile(r"^user(?P<subject>\d{2})_[A-Za-z0-9_]+\.aedat$")


def _official_train_recordings(source_root: Path) -> list[str]:
    split_path = source_root / "trials_to_train.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Official DVS-Gesture train list missing: {split_path}")

    names = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names or len(names) != len(set(names)):
        raise ValueError("The official DVS-Gesture train list is empty or contains duplicates.")
    invalid = [name for name in names if RECORDING_PATTERN.fullmatch(name) is None]
    if invalid:
        raise ValueError(f"Invalid recording names in trials_to_train.txt: {invalid}")
    return names


def _labels(path: Path) -> np.ndarray:
    try:
        labels = np.loadtxt(path, dtype=np.uint64, delimiter=",", skiprows=1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read DVS-Gesture labels {path}: {exc}") from exc
    labels = np.atleast_2d(labels)
    if labels.ndim != 2 or labels.shape[1] != 3 or not labels.shape[0]:
        raise ValueError(f"Expected class,startTime_usec,endTime_usec rows in {path}")
    if np.any(labels[:, 0] < 1) or np.any(labels[:, 0] > 11):
        raise ValueError(f"DVS-Gesture labels must be in [1, 11]: {path}")
    if np.any(labels[:, 2] <= labels[:, 1]):
        raise ValueError(f"DVS-Gesture label intervals must have positive duration: {path}")
    return labels


def _source_index_sha256(source_root: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        for path in (source_root / name, source_root / f"{Path(name).stem}_labels.csv"):
            if not path.is_file():
                raise FileNotFoundError(f"Official DVS-Gesture source file missing: {path}")
            digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def prepare_dvsgesture_train(
    source_root: str | Path,
    output_root: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Segment only official-train recordings using their released label intervals."""

    source = Path(source_root)
    output = Path(output_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Extracted DVS-Gesture source directory missing: {source}")
    if output.exists():
        raise FileExistsError(
            f"Prepared DVS-Gesture output already exists: {output}. "
            "Keep it or remove it explicitly before rebuilding."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    recording_names = _official_train_recordings(source)
    source_hash = _source_index_sha256(source, recording_names)
    sample_count = 0
    class_counts: Counter[int] = Counter()
    subject_counts: Counter[int] = Counter()
    durations: list[int] = []
    event_counts: list[int] = []

    with tempfile.TemporaryDirectory(prefix="dvsgesture-train-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        for target in range(11):
            (temporary_root / str(target)).mkdir()

        for recording_name in recording_names:
            match = RECORDING_PATTERN.fullmatch(recording_name)
            if match is None:  # Guarded by _official_train_recordings.
                raise ValueError(f"Invalid DVS-Gesture recording name: {recording_name}")
            subject = int(match.group("subject"))
            if not 1 <= subject <= 23:
                raise ValueError(
                    f"Official-train recording has out-of-protocol subject {subject}: "
                    f"{recording_name}"
                )
            events = load_aedat_v3(source / recording_name)
            if int(events["x"].max()) >= 128 or int(events["y"].max()) >= 128:
                raise ValueError(f"DVS-Gesture coordinates exceed the DVS128 sensor: {recording_name}")
            label_rows = _labels(source / f"{Path(recording_name).stem}_labels.csv")
            recording_stem = Path(recording_name).stem

            for row_index, (released_label, start_us, end_us) in enumerate(label_rows):
                target = int(released_label) - 1
                start = int(start_us)
                end = int(end_us)
                mask = (events["t"] >= start) & (events["t"] < end)
                if not bool(mask.any()):
                    raise ValueError(
                        f"Annotated segment {row_index} contains no events: {recording_name}"
                    )
                segment_timestamps = events["t"][mask]
                if np.any(segment_timestamps[1:] < segment_timestamps[:-1]):
                    raise ValueError(
                        f"Annotated segment {row_index} has non-monotonic timestamps: "
                        f"{recording_name}"
                    )
                relative_timestamps = segment_timestamps - start
                sample_path = temporary_root / str(target) / f"{recording_stem}__{row_index:02d}.npz"
                np.savez_compressed(
                    sample_path,
                    t=relative_timestamps,
                    x=events["x"][mask],
                    y=events["y"][mask],
                    p=events["p"][mask],
                    target=np.asarray(target, dtype=np.uint8),
                    subject=np.asarray(subject, dtype=np.uint8),
                    segment_start_us=np.asarray(start, dtype=np.uint64),
                    segment_end_us=np.asarray(end, dtype=np.uint64),
                    source_recording=np.asarray(recording_name),
                )
                sample_count += 1
                class_counts[target] += 1
                subject_counts[subject] += 1
                durations.append(end - start)
                event_counts.append(int(mask.sum()))

        if set(class_counts) != set(range(11)):
            raise ValueError(f"Prepared DVS-Gesture train is missing classes: {sorted(set(range(11)) - set(class_counts))}")
        temporary_root.replace(output)

    duration_array = np.asarray(durations, dtype=np.uint64)
    event_count_array = np.asarray(event_counts, dtype=np.uint64)
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "dvsgesture",
        "source_split": "train",
        "official_test_used": False,
        "source_root": str(source.resolve()),
        "output_root": str(output.resolve()),
        "source_index_sha256": source_hash,
        "recording_count": len(recording_names),
        "sample_count": sample_count,
        "class_counts": {str(target): class_counts[target] for target in range(11)},
        "subject_counts": {str(subject): subject_counts[subject] for subject in sorted(subject_counts)},
        "duration_us": _summary(duration_array),
        "event_count": _summary(event_count_array),
        "timestamp_origin": "annotated_segment_start",
    }
    if report_path is not None:
        write_json(report, report_path)
    return report


def _summary(values: np.ndarray) -> dict[str, int]:
    return {
        "minimum": int(values.min()),
        "p05": int(np.percentile(values, 5)),
        "median": int(np.median(values)),
        "p95": int(np.percentile(values, 95)),
        "maximum": int(values.max()),
    }
