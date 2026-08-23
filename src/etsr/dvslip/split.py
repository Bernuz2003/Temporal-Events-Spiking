"""One reproducible DVS-Lip development split, without unverified speaker claims."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from etsr.dvslip.dataset import DvsLipExpectations, discover_training_samples
from etsr.utils.io import ensure_dir, sha256_file, write_json

DEVELOPMENT_SPLITS = ("train", "validation")
GENERATOR_VERSION = "dvslip_sample_stratified_hash_v1"
SPLIT_SEED = 314159
VALIDATION_FRACTION = 0.2


def _require_relative_sample_path(value: str) -> str:
    candidate = PurePosixPath(value.strip())
    if (
        not value.strip()
        or candidate.is_absolute()
        or any(part in ("", ".", "..") for part in candidate.parts)
        or len(candidate.parts) != 2
        or candidate.suffix != ".npy"
        or not candidate.stem.isdecimal()
    ):
        raise ValueError(
            f"DVS-Lip sample paths must use the exact '<class>/<integer>.npy' layout: {value!r}"
        )
    return candidate.as_posix()


def _sample_rank(split_seed: int, relative_path: str) -> tuple[str, str]:
    material = f"{split_seed}\0{relative_path}".encode()
    return hashlib.sha256(material).hexdigest(), relative_path


def _sample_paths_digest(relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths):
        digest.update(relative_path.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_parameters(split_seed: int, validation_fraction: float) -> None:
    if type(split_seed) is not int:
        raise ValueError("DVS-Lip split_seed must be an integer.")
    if not isinstance(  # noqa: UP038 - removed by modern Ruff; tuple form is intentional.
        validation_fraction, (int, float)
    ) or isinstance(validation_fraction, bool):
        raise ValueError("DVS-Lip validation_fraction must be numeric.")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("DVS-Lip validation_fraction must be strictly between zero and one.")


def build_development_split_manifest(
    relative_paths: list[str],
    *,
    split_seed: int = SPLIT_SEED,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    """Assign each class by stable SHA-256 rank and return one self-contained manifest."""

    _validate_parameters(split_seed, validation_fraction)
    normalized = [_require_relative_sample_path(path) for path in relative_paths]
    if len(normalized) != len(set(normalized)):
        raise ValueError("DVS-Lip development split input contains duplicate sample paths.")

    by_class: dict[str, list[str]] = defaultdict(list)
    for relative_path in normalized:
        by_class[PurePosixPath(relative_path).parts[0]].append(relative_path)

    assignments: dict[str, str] = {}
    for class_name in sorted(by_class):
        class_paths = by_class[class_name]
        if len(class_paths) < 2:
            raise ValueError(
                f"Class {class_name!r} needs at least two samples for a train/validation split."
            )
        validation_count = int(len(class_paths) * float(validation_fraction) + 0.5)
        validation_count = max(1, min(len(class_paths) - 1, validation_count))
        ranked = sorted(class_paths, key=lambda path: _sample_rank(split_seed, path))
        validation_paths = set(ranked[:validation_count])
        for relative_path in class_paths:
            assignments[relative_path] = (
                "validation" if relative_path in validation_paths else "train"
            )

    sample_counts = {
        split: sum(assignment == split for assignment in assignments.values())
        for split in DEVELOPMENT_SPLITS
    }
    return {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "official_source_split": "train",
        "official_test_used": False,
        "split_unit": "sample",
        "class_stratified": True,
        "speaker_identity_available": False,
        "speaker_disjoint": False,
        "split_seed": split_seed,
        "validation_fraction": float(validation_fraction),
        "dataset_sample_paths_sha256": _sample_paths_digest(normalized),
        "sample_counts": sample_counts,
        "assignments": assignments,
    }


def prepare_dvslip_development_split(
    train_root: str | Path,
    output_path: str | Path,
    *,
    expectations: Any = None,
) -> dict[str, Any]:
    """Discover official-train filenames and write the sole local split manifest."""

    expected = expectations or DvsLipExpectations()
    root, samples, _ = discover_training_samples(train_root, expected)
    relative_paths = [sample.relative_to(root).as_posix() for sample in samples]
    manifest = build_development_split_manifest(relative_paths)
    ensure_dir(Path(output_path).parent)
    write_json(manifest, output_path)
    return manifest


def load_development_split_manifest(
    path: str | Path,
    expected_samples: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Recompute the self-contained manifest and reject any drift or stronger claim."""

    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    required_values = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "official_source_split": "train",
        "official_test_used": False,
        "split_unit": "sample",
        "class_stratified": True,
        "speaker_identity_available": False,
        "speaker_disjoint": False,
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in required_values.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Invalid DVS-Lip development split fields: {mismatches}")

    expected_manifest = build_development_split_manifest(
        expected_samples,
        split_seed=payload.get("split_seed"),
        validation_fraction=payload.get("validation_fraction"),
    )
    if payload != expected_manifest:
        raise ValueError(
            "DVS-Lip development split does not reproduce exactly from its recorded parameters."
        )

    assignments = expected_manifest["assignments"]
    return assignments, {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "generator_version": GENERATOR_VERSION,
        "split_seed": expected_manifest["split_seed"],
        "validation_fraction": expected_manifest["validation_fraction"],
        "sample_counts": expected_manifest["sample_counts"],
        "dataset_sample_paths_sha256": expected_manifest["dataset_sample_paths_sha256"],
        "split_unit": "sample",
        "class_stratified": True,
        "speaker_identity_available": False,
        "speaker_disjoint": False,
        "official_source_split": "train",
        "official_test_used": False,
    }
