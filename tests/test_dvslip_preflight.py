import csv
import json
from pathlib import Path

import numpy as np
import pytest

from etsr.data.dvslip_preflight import (
    DvsLipExpectations,
    inspect_event_sample,
    load_class_groups_manifest,
    run_dvslip_preflight,
)


def _write_event_sample(path, *, polarity=(0, 1, 1)):
    path.parent.mkdir(parents=True, exist_ok=True)
    events = np.zeros(
        len(polarity),
        dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")],
    )
    events["t"] = np.arange(len(polarity), dtype=np.uint64) * 100
    events["x"] = np.arange(len(polarity), dtype=np.uint16)
    events["y"] = np.arange(len(polarity), dtype=np.uint16) + 1
    events["p"] = polarity
    np.save(path, events)


def _small_archive(tmp_path):
    train_root = tmp_path / "DVS-Lip" / "train"
    relative_paths = []
    for class_name in ("alpha", "beta", "gamma", "delta"):
        relative_path = f"{class_name}/0.npy"
        _write_event_sample(train_root / relative_path)
        relative_paths.append(relative_path)
    expectations = DvsLipExpectations(
        class_count=4,
        sample_count=4,
        speaker_count=2,
        train_speaker_count=1,
        validation_speaker_count=1,
        height=8,
        width=8,
    )
    return train_root, relative_paths, expectations


def _write_speaker_manifest(path, relative_paths):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "speaker_id"])
        writer.writeheader()
        for index, relative_path in enumerate(relative_paths):
            writer.writerow(
                {
                    "relative_path": relative_path,
                    "speaker_id": f"speaker_{index % 2}",
                }
            )


def test_preflight_validates_complete_train_only_protocol_inputs(tmp_path):
    train_root, relative_paths, expectations = _small_archive(tmp_path)
    speaker_manifest = tmp_path / "speakers.csv"
    _write_speaker_manifest(speaker_manifest, relative_paths)
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "official_source_split": "train",
                "official_test_used": False,
                "split_seed": 17,
                "strategy": "fixture assignment",
                "assignments": {
                    "speaker_0": "train",
                    "speaker_1": "validation",
                },
            }
        ),
        encoding="utf-8",
    )
    terms = tmp_path / "DATASET_TERMS.txt"
    terms.write_text("Fixture terms; not a real license.\n", encoding="utf-8")
    class_groups = tmp_path / "class_groups.json"
    class_groups.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "paper_semantics": {
                    "Acc1": "visually_confusable_words",
                    "Acc2": "common_words",
                },
                "classes": ["alpha", "beta", "gamma", "delta"],
                "visually_confusable_pairs": [["alpha", "beta"]],
                "visually_confusable_words": ["alpha", "beta"],
                "common_words": ["gamma", "delta"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "preflight.json"

    report = run_dvslip_preflight(
        train_root,
        speaker_manifest=speaker_manifest,
        split_manifest=split_manifest,
        class_groups_manifest=class_groups,
        terms_path=terms,
        output_path=output,
        hash_samples=True,
        expectations=expectations,
    )

    assert report["validation_status"] == "passed"
    assert report["preflight_gate_status"] == "ready"
    assert report["protocol_gate_status"] == "blocked"
    assert report["blockers"] == []
    assert report["protocol_blockers"] == [
        "official_test_physical_quarantine_not_verified",
    ]
    assert report["official_test_used"] is False
    assert report["access_scope"] == "official_train_only"
    assert report["dataset_content"]["complete"] is True
    assert report["dataset_content"]["files_hashed"] == 4
    assert report["sample_inspection"]["samples_inspected"] == 4
    assert report["split_manifest"]["speaker_overlap"] == []
    assert report["split_manifest"]["sample_counts"] == {
        "train": 2,
        "validation": 2,
    }
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["official_test_used"] is False


def test_preflight_reports_missing_external_evidence_without_inventing_it(tmp_path):
    train_root, _relative_paths, expectations = _small_archive(tmp_path)

    report = run_dvslip_preflight(train_root, expectations=expectations)

    assert report["validation_status"] == "passed"
    assert report["preflight_gate_status"] == "blocked"
    assert report["protocol_gate_status"] == "blocked"
    assert report["blockers"] == [
        "authoritative_sample_to_speaker_manifest_missing",
        "versioned_24_6_speaker_split_missing",
        "dataset_terms_missing",
        "semantic_acc1_acc2_manifest_missing",
        "full_sample_content_hash_not_computed",
    ]
    assert report["speaker_manifest"] is None
    assert report["dataset_terms"] is None
    assert report["claims"]["raw_loader_contract_frozen"] is False


def test_preflight_rejects_the_official_test_root_before_sample_access(tmp_path):
    test_root = tmp_path / "DVS-Lip" / "test"
    test_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="official test partition"):
        run_dvslip_preflight(
            test_root,
            expectations=DvsLipExpectations(class_count=0, sample_count=0),
        )


def test_event_inspection_rejects_unverified_dense_column_semantics(tmp_path):
    sample = tmp_path / "dense.npy"
    np.save(sample, np.zeros((4, 4), dtype=np.int64))

    with pytest.raises(ValueError, match="Do not infer dense column semantics"):
        inspect_event_sample(sample, DvsLipExpectations())


def test_preflight_rejects_incomplete_speaker_coverage(tmp_path):
    train_root, relative_paths, expectations = _small_archive(tmp_path)
    speaker_manifest = tmp_path / "speakers.csv"
    _write_speaker_manifest(speaker_manifest, relative_paths[:-1])

    with pytest.raises(ValueError, match="does not exactly cover"):
        run_dvslip_preflight(
            train_root,
            speaker_manifest=speaker_manifest,
            expectations=expectations,
        )


def test_versioned_class_groups_encode_paper_acc_semantics():
    path = Path("configs/dvslip_class_groups.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["classes"]) == 100
    assert len(set(payload["classes"])) == 100
    assert len(payload["visually_confusable_pairs"]) == 25
    assert len(payload["visually_confusable_words"]) == 50
    assert len(payload["common_words"]) == 50
    assert set(payload["visually_confusable_words"]).isdisjoint(payload["common_words"])
    assert payload["paper_semantics"] == {
        "Acc1": "visually_confusable_words",
        "Acc2": "common_words",
    }
    summary = load_class_groups_manifest(
        path,
        payload["classes"],
        DvsLipExpectations(),
    )
    assert summary["visually_confusable_pair_count"] == 25
