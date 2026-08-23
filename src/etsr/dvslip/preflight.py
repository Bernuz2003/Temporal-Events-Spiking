"""Strict, train-only preflight checks for a prospective DVS-Lip archive.

This module does not implement a training dataset. Its purpose is to turn the external evidence
needed by the raw loader into a machine-readable gate without opening the official test partition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from etsr.dvslip.dataset import (
    DvsLipExpectations,
    dataset_index_sha256,
    discover_training_samples,
    load_event_array,
    summarize_event_array,
)
from etsr.dvslip.split import load_development_split_manifest
from etsr.utils.io import ensure_dir, sha256_file, write_json


def inspect_event_sample(
    path: str | Path,
    expectations: DvsLipExpectations,
) -> dict[str, Any]:
    """Validate and summarize one structured raw-event sample without changing its time base."""

    sample_path = Path(path)
    events = load_event_array(
        sample_path,
        height=expectations.height,
        width=expectations.width,
    )
    return {"relative_path": sample_path.name, **summarize_event_array(events)}


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
    split_manifest: str | Path | None = None,
    class_groups_manifest: str | Path | None = None,
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
    known_limitations = [
        "sample_to_speaker_mapping_unavailable",
        "development_validation_is_not_speaker_disjoint",
        "development_validation_is_not_an_unseen_speaker_estimate",
        "dataset_license_not_identified",
        "dataset_redistribution_not_authorized_by_available_evidence",
    ]

    split_summary: dict[str, Any] | None = None
    if split_manifest is None:
        blockers.append("deterministic_development_split_manifest_missing")
    else:
        _split_assignments, split_summary = load_development_split_manifest(
            split_manifest,
            relative_paths,
        )

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

    protocol_blockers = list(blockers)
    report = {
        "schema_version": 3,
        "validation_status": "passed",
        "preflight_gate_status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "protocol_gate_status": "ready" if not protocol_blockers else "blocked",
        "protocol_blockers": protocol_blockers,
        "known_limitations": known_limitations,
        "access_scope": "official_train_only",
        "official_source_split": "train",
        "official_test_used": False,
        "dataset_source": {
            "official_page": "https://sites.google.com/view/event-based-lipreading",
            "distribution": "google_drive_zip_linked_from_official_page",
            "missing_metadata_resolution": "D011",
        },
        "train_root": str(root.resolve()),
        "expectations": asdict(expected),
        "dataset_index": {
            "class_count": len(class_counts),
            "sample_count": len(samples),
            "class_counts": class_counts,
            "sha256": dataset_index_sha256(root, samples),
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
        "split_manifest": split_summary,
        "class_groups_manifest": class_groups_summary,
        "claims": {
            "dataset_license_granted": False,
            "dataset_redistribution_authorized": False,
            "speaker_identity_available": False,
            "development_validation_speaker_disjoint": False,
            "development_validation_unseen_speaker_estimate": False,
            "raw_loader_contract_frozen": True,
            "official_test_logical_embargo_enforced": True,
        },
    }
    if output_path is not None:
        ensure_dir(Path(output_path).parent)
        write_json(report, output_path)
    return report
