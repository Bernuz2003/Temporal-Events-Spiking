"""Strict, train-only preflight checks for a prospective DVS-Lip archive.

This module does not implement a training dataset. Its purpose is to turn the external evidence
needed by the raw loader into a machine-readable gate without opening the official test partition.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from etsr.reproducibility import sha256_file
from etsr.utils.io import ensure_dir, write_json

REQUIRED_EVENT_FIELDS = ("t", "x", "y", "p")
DEVELOPMENT_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class DvsLipExpectations:
    """Published official-train invariants used by the real preflight command."""

    class_count: int = 100
    sample_count: int = 14_896
    speaker_count: int = 30
    train_speaker_count: int = 24
    validation_speaker_count: int = 6
    height: int = 128
    width: int = 128


def _validate_relative_sample_path(value: str) -> str:
    candidate = PurePosixPath(value.strip())
    if not value.strip() or candidate.is_absolute():
        raise ValueError(f"Sample path must be non-empty and relative: {value!r}")
    if any(part in ("", ".", "..") for part in candidate.parts):
        raise ValueError(f"Sample path contains an unsafe component: {value!r}")
    if len(candidate.parts) != 2 or candidate.suffix != ".npy":
        raise ValueError(
            "Speaker-manifest paths must use the exact '<class>/<integer>.npy' train layout: "
            f"{value!r}"
        )
    return candidate.as_posix()


def discover_training_samples(
    train_root: str | Path,
    expectations: DvsLipExpectations,
) -> tuple[Path, list[Path], dict[str, int]]:
    """Enumerate the official training tree without resolving or traversing a test sibling."""

    root = Path(train_root)
    if root.name != "train":
        raise ValueError(
            "DVS-Lip preflight accepts only a path whose final component is 'train'; "
            "the official test partition must not be passed to this command."
        )
    if root.is_symlink():
        raise ValueError("The training root must not be a symlink.")
    if not root.is_dir():
        raise FileNotFoundError(f"DVS-Lip training root does not exist: {root}")

    visible_entries = sorted(
        (entry for entry in root.iterdir() if not entry.name.startswith(".")),
        key=lambda entry: entry.name,
    )
    class_dirs = [entry for entry in visible_entries if not entry.is_symlink() and entry.is_dir()]
    unexpected_root_entries = [
        entry.name for entry in visible_entries if entry not in class_dirs
    ]
    if unexpected_root_entries:
        raise ValueError(
            f"Unexpected entries directly under the training root: {unexpected_root_entries}"
        )
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
                f"Invalid entries in class {class_dir.name!r}; expected integer .npy files: {invalid}"
            )
        class_samples = [entry for entry in entries if entry.is_file()]
        if not class_samples:
            raise ValueError(f"DVS-Lip class {class_dir.name!r} is empty.")
        class_counts[class_dir.name] = len(class_samples)
        samples.extend(class_samples)

    if len(samples) != expectations.sample_count:
        raise ValueError(
            f"Expected {expectations.sample_count} official-train samples, found {len(samples)}."
        )
    return root, samples, class_counts


def inspect_event_sample(
    path: str | Path,
    expectations: DvsLipExpectations,
) -> dict[str, Any]:
    """Validate and summarize one structured raw-event sample without changing its time base."""

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

    field_dtypes: dict[str, str] = {}
    values: dict[str, np.ndarray] = {}
    for field in REQUIRED_EVENT_FIELDS:
        dtype = events.dtype.fields[field][0]
        if dtype.shape or not np.issubdtype(dtype, np.integer):
            raise ValueError(
                f"Sample {sample_path} field {field!r} must be a scalar integer, found {dtype}."
            )
        field_dtypes[field] = dtype.str
        values[field] = np.asarray(events[field])

    timestamps = values["t"]
    x_coords = values["x"]
    y_coords = values["y"]
    polarities = values["p"]
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError(f"Sample timestamps are not monotonic at {sample_path}.")
    if int(timestamps.min()) < 0:
        raise ValueError(f"Sample contains negative timestamps at {sample_path}.")
    if int(x_coords.min()) < 0 or int(x_coords.max()) >= expectations.width:
        raise ValueError(f"Sample x coordinates exceed [0,{expectations.width}) at {sample_path}.")
    if int(y_coords.min()) < 0 or int(y_coords.max()) >= expectations.height:
        raise ValueError(f"Sample y coordinates exceed [0,{expectations.height}) at {sample_path}.")
    polarity_values = sorted(int(value) for value in np.unique(polarities))
    if not set(polarity_values).issubset({0, 1}):
        raise ValueError(
            f"Sample polarity values must be a subset of [0, 1] at {sample_path}; "
            f"found {polarity_values}."
        )

    return {
        "relative_path": sample_path.name,
        "event_count": int(events.shape[0]),
        "structured_dtype": str(events.dtype),
        "field_order": list(field_order),
        "field_dtypes": field_dtypes,
        "timestamp_min": int(timestamps.min()),
        "timestamp_max": int(timestamps.max()),
        "timestamps_monotonic": True,
        "x_range": [int(x_coords.min()), int(x_coords.max())],
        "y_range": [int(y_coords.min()), int(y_coords.max())],
        "polarity_values": polarity_values,
    }


def _representative_samples(
    train_root: Path,
    samples: list[Path],
    samples_per_class: int,
) -> list[Path]:
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least one.")
    selected: list[Path] = []
    current_class: str | None = None
    selected_in_class = 0
    for sample in samples:
        class_name = sample.relative_to(train_root).parts[0]
        if class_name != current_class:
            current_class = class_name
            selected_in_class = 0
        if selected_in_class < samples_per_class:
            selected.append(sample)
            selected_in_class += 1
    return selected


def _validate_inspection_consistency(inspections: list[dict[str, Any]]) -> None:
    if not inspections:
        raise ValueError("No representative DVS-Lip samples were inspected.")
    reference = inspections[0]
    for inspection in inspections[1:]:
        for key in ("field_order", "field_dtypes"):
            if inspection[key] != reference[key]:
                raise ValueError(
                    f"Representative sample schema mismatch for {key}: "
                    f"{inspection[key]} != {reference[key]}."
                )


def load_speaker_manifest(
    path: str | Path,
    expected_samples: list[str],
    expectations: DvsLipExpectations,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load an explicit sample-to-speaker CSV and require exact train-set coverage."""

    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"relative_path", "speaker_id"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Speaker manifest must contain columns {sorted(required_columns)}."
            )
        assignments: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            relative_path = _validate_relative_sample_path(row["relative_path"])
            speaker_id = row["speaker_id"].strip()
            if not speaker_id:
                raise ValueError(f"Empty speaker_id at {manifest_path}:{line_number}.")
            if relative_path in assignments:
                raise ValueError(f"Duplicate speaker mapping for {relative_path!r}.")
            assignments[relative_path] = speaker_id

    expected = set(expected_samples)
    observed = set(assignments)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            "Speaker manifest does not exactly cover official-train samples: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)."
        )
    speaker_ids = sorted(set(assignments.values()))
    if len(speaker_ids) != expectations.speaker_count:
        raise ValueError(
            f"Expected {expectations.speaker_count} official-train speakers, "
            f"found {len(speaker_ids)}."
        )
    return assignments, {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "samples": len(assignments),
        "speaker_count": len(speaker_ids),
        "speaker_ids": speaker_ids,
    }


def load_split_manifest(
    path: str | Path,
    speaker_ids: list[str],
    expectations: DvsLipExpectations,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate a versioned 24/6 speaker assignment without inventing its selection rule."""

    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("DVS-Lip split manifest requires schema_version=1.")
    if payload.get("official_source_split") != "train":
        raise ValueError("DVS-Lip development splits may only derive from official train.")
    if payload.get("official_test_used") is not False:
        raise ValueError("DVS-Lip split manifest must declare official_test_used=false.")
    if type(payload.get("split_seed")) is not int:
        raise ValueError("DVS-Lip split manifest must record an integer split_seed.")
    if not isinstance(payload.get("strategy"), str) or not payload["strategy"].strip():
        raise ValueError("DVS-Lip split manifest must record a non-empty strategy.")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in assignments.items()
    ):
        raise ValueError("DVS-Lip split manifest assignments must map speaker IDs to split names.")
    invalid_splits = sorted(set(assignments.values()) - set(DEVELOPMENT_SPLITS))
    if invalid_splits:
        raise ValueError(f"Unknown DVS-Lip development split names: {invalid_splits}")
    expected_speakers = set(speaker_ids)
    assigned_speakers = set(assignments)
    if assigned_speakers != expected_speakers:
        raise ValueError(
            "Split manifest must assign every and only official-train speaker exactly once."
        )
    split_counts = {
        split: sum(value == split for value in assignments.values())
        for split in DEVELOPMENT_SPLITS
    }
    expected_counts = {
        "train": expectations.train_speaker_count,
        "validation": expectations.validation_speaker_count,
    }
    if split_counts != expected_counts:
        raise ValueError(
            f"Expected speaker split counts {expected_counts}, found {split_counts}."
        )
    return assignments, {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "split_seed": payload["split_seed"],
        "strategy": payload["strategy"],
        "speaker_counts": split_counts,
        "official_source_split": "train",
        "official_test_used": False,
    }


def load_class_groups_manifest(
    path: str | Path,
    observed_classes: list[str],
    expectations: DvsLipExpectations,
) -> dict[str, Any]:
    """Validate paper-semantic Acc1/Acc2 groups against the discovered class directories."""

    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("DVS-Lip class-groups manifest requires schema_version=1.")
    semantics = payload.get("paper_semantics")
    expected_semantics = {
        "Acc1": "visually_confusable_words",
        "Acc2": "common_words",
    }
    if semantics != expected_semantics:
        raise ValueError(
            f"DVS-Lip metric semantics must be {expected_semantics}, found {semantics}."
        )

    classes = payload.get("classes")
    confusable = payload.get("visually_confusable_words")
    common = payload.get("common_words")
    pairs = payload.get("visually_confusable_pairs")
    if not all(isinstance(group, list) for group in (classes, confusable, common, pairs)):
        raise ValueError("DVS-Lip class groups must be JSON lists.")
    if not all(isinstance(word, str) and word for word in classes + confusable + common):
        raise ValueError("DVS-Lip class names must be non-empty strings.")
    if len(classes) != expectations.class_count or len(set(classes)) != len(classes):
        raise ValueError(
            f"Expected {expectations.class_count} unique DVS-Lip classes in the metric manifest."
        )
    if set(classes) != set(observed_classes):
        raise ValueError(
            "Metric-manifest classes do not exactly match discovered training directories."
        )
    if len(confusable) != len(common) or len(confusable) + len(common) != len(classes):
        raise ValueError("DVS-Lip Acc1/Acc2 groups must form equal halves of the vocabulary.")
    if set(confusable) & set(common) or set(confusable) | set(common) != set(classes):
        raise ValueError("DVS-Lip Acc1/Acc2 groups must be disjoint and cover every class.")
    if len(pairs) * 2 != len(confusable) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(word, str) for word in pair)
        for pair in pairs
    ):
        raise ValueError("DVS-Lip visually-confusable groups must contain exact word pairs.")
    flattened_pairs = [word for pair in pairs for word in pair]
    if len(set(flattened_pairs)) != len(flattened_pairs) or set(flattened_pairs) != set(confusable):
        raise ValueError(
            "DVS-Lip visually-confusable pairs must cover each Acc1 class exactly once."
        )

    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "source_id": payload.get("source_id"),
        "source_commit": payload.get("source_commit"),
        "class_count": len(classes),
        "visually_confusable_word_count": len(confusable),
        "common_word_count": len(common),
        "visually_confusable_pair_count": len(pairs),
        "paper_semantics": semantics,
    }


def _dataset_index_digest(train_root: Path, samples: list[Path]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        relative_path = sample.relative_to(train_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(sample.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dataset_content_digest(train_root: Path, samples: list[Path]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        relative_path = sample.relative_to(train_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(sample).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_dvslip_preflight(
    train_root: str | Path,
    *,
    speaker_manifest: str | Path | None = None,
    split_manifest: str | Path | None = None,
    class_groups_manifest: str | Path | None = None,
    terms_path: str | Path | None = None,
    output_path: str | Path | None = None,
    hash_samples: bool = False,
    samples_per_class: int = 1,
    expectations: DvsLipExpectations | None = None,
) -> dict[str, Any]:
    """Run train-only validation and report unresolved protocol blockers explicitly."""

    expected = expectations or DvsLipExpectations()
    root, samples, class_counts = discover_training_samples(train_root, expected)
    relative_paths = [sample.relative_to(root).as_posix() for sample in samples]
    inspected_paths = _representative_samples(root, samples, samples_per_class)
    inspections = []
    for sample in inspected_paths:
        inspection = inspect_event_sample(sample, expected)
        inspection["relative_path"] = sample.relative_to(root).as_posix()
        inspections.append(inspection)
    _validate_inspection_consistency(inspections)

    blockers: list[str] = []
    speaker_assignments: dict[str, str] | None = None
    speaker_summary: dict[str, Any] | None = None
    if speaker_manifest is None:
        blockers.append("authoritative_sample_to_speaker_manifest_missing")
    else:
        speaker_assignments, speaker_summary = load_speaker_manifest(
            speaker_manifest, relative_paths, expected
        )

    split_summary: dict[str, Any] | None = None
    if split_manifest is None:
        blockers.append("versioned_24_6_speaker_split_missing")
    elif speaker_summary is None:
        raise ValueError("A split manifest cannot be validated without a speaker manifest.")
    else:
        split_assignments, split_summary = load_split_manifest(
            split_manifest, speaker_summary["speaker_ids"], expected
        )
        sample_split_counts = {split: 0 for split in DEVELOPMENT_SPLITS}
        assert speaker_assignments is not None
        for relative_path in relative_paths:
            speaker_id = speaker_assignments[relative_path]
            sample_split_counts[split_assignments[speaker_id]] += 1
        split_summary["sample_counts"] = sample_split_counts
        split_summary["speaker_overlap"] = []

    terms_summary: dict[str, Any] | None = None
    if terms_path is None:
        blockers.append("dataset_terms_missing")
    else:
        terms = Path(terms_path)
        if not terms.is_file() or terms.stat().st_size == 0:
            raise ValueError(f"Dataset terms must be a non-empty regular file: {terms}")
        terms_summary = {
            "path": str(terms.resolve()),
            "sha256": sha256_file(terms),
            "bytes": terms.stat().st_size,
            "legal_review_claimed": False,
        }

    class_groups_summary: dict[str, Any] | None = None
    if class_groups_manifest is None:
        blockers.append("semantic_acc1_acc2_manifest_missing")
    else:
        class_groups_summary = load_class_groups_manifest(
            class_groups_manifest, sorted(class_counts), expected
        )

    content_digest: str | None = None
    if hash_samples:
        content_digest = _dataset_content_digest(root, samples)
    else:
        blockers.append("full_sample_content_hash_not_computed")

    protocol_blockers = [*blockers, "official_test_physical_quarantine_not_verified"]
    report = {
        "schema_version": 1,
        "validation_status": "passed",
        "preflight_gate_status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "protocol_gate_status": "blocked",
        "protocol_blockers": protocol_blockers,
        "access_scope": "official_train_only",
        "official_source_split": "train",
        "official_test_used": False,
        "train_root": str(root.resolve()),
        "expectations": asdict(expected),
        "dataset_index": {
            "class_count": len(class_counts),
            "sample_count": len(samples),
            "class_counts": class_counts,
            "sha256": _dataset_index_digest(root, samples),
            "digest_definition": "sha256(relative_posix + NUL + file_size + LF)",
        },
        "dataset_content": {
            "sha256": content_digest,
            "files_hashed": len(samples) if hash_samples else 0,
            "complete": hash_samples,
            "digest_definition": "sha256(relative_posix + NUL + sha256(file) + LF)",
        },
        "sample_inspection": {
            "samples_per_class": samples_per_class,
            "samples_inspected": len(inspections),
            "records": inspections,
        },
        "speaker_manifest": speaker_summary,
        "split_manifest": split_summary,
        "dataset_terms": terms_summary,
        "class_groups_manifest": class_groups_summary,
        "claims": {
            "dataset_license_granted": False,
            "raw_loader_contract_frozen": False,
            "official_test_quarantine_verified": False,
        },
    }
    if output_path is not None:
        ensure_dir(Path(output_path).parent)
        write_json(report, output_path)
    return report
