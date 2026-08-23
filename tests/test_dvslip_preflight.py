import json
from pathlib import Path

import numpy as np
import pytest

from etsr.dvslip.dataset import DvsLipExpectations
from etsr.dvslip.preflight import (
    inspect_event_sample,
    load_class_groups_manifest,
    run_dvslip_preflight,
)
from etsr.dvslip.split import (
    prepare_dvslip_development_split,
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
        for sample_index in range(5):
            relative_path = f"{class_name}/{sample_index}.npy"
            _write_event_sample(train_root / relative_path)
            relative_paths.append(relative_path)
    expectations = DvsLipExpectations(
        class_count=4,
        sample_count=20,
        height=8,
        width=8,
    )
    return train_root, relative_paths, expectations


def test_preflight_validates_complete_train_only_protocol_inputs(tmp_path):
    train_root, _relative_paths, expectations = _small_archive(tmp_path)
    split_manifest = tmp_path / "split.json"
    prepare_dvslip_development_split(
        train_root,
        split_manifest,
        expectations=expectations,
    )
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
        split_manifest=split_manifest,
        class_groups_manifest=class_groups,
        output_path=output,
        hash_samples=True,
        expectations=expectations,
    )

    assert report["validation_status"] == "passed"
    assert report["preflight_gate_status"] == "ready"
    assert report["protocol_gate_status"] == "ready"
    assert report["blockers"] == []
    assert report["protocol_blockers"] == []
    assert report["official_test_used"] is False
    assert report["access_scope"] == "official_train_only"
    assert report["dataset_content"]["complete"] is True
    assert report["dataset_content"]["files_hashed"] == 20
    assert report["sample_inspection"]["samples_inspected"] == 4
    assert report["split_manifest"]["speaker_disjoint"] is False
    assert report["split_manifest"]["sample_counts"] == {
        "train": 16,
        "validation": 4,
    }
    assert report["claims"]["development_validation_speaker_disjoint"] is False
    assert report["claims"]["raw_loader_contract_frozen"] is True
    assert report["claims"]["official_test_logical_embargo_enforced"] is True
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["official_test_used"] is False


def test_preflight_reports_missing_external_evidence_without_inventing_it(tmp_path):
    train_root, _relative_paths, expectations = _small_archive(tmp_path)

    report = run_dvslip_preflight(train_root, expectations=expectations)

    assert report["validation_status"] == "passed"
    assert report["preflight_gate_status"] == "blocked"
    assert report["protocol_gate_status"] == "blocked"
    assert report["blockers"] == [
        "deterministic_development_split_manifest_missing",
        "semantic_acc1_acc2_manifest_missing",
        "full_sample_content_hash_not_computed",
    ]
    assert "sample_to_speaker_mapping_unavailable" in report["known_limitations"]
    assert report["claims"]["raw_loader_contract_frozen"] is True


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


def test_preflight_rejects_a_split_assignment_that_does_not_reproduce(tmp_path):
    train_root, _relative_paths, expectations = _small_archive(tmp_path)
    split_manifest = tmp_path / "split.json"
    prepare_dvslip_development_split(
        train_root,
        split_manifest,
        expectations=expectations,
    )
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    first_path = sorted(payload["assignments"])[0]
    payload["assignments"][first_path] = (
        "validation" if payload["assignments"][first_path] == "train" else "train"
    )
    split_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not reproduce exactly"):
        run_dvslip_preflight(
            train_root,
            split_manifest=split_manifest,
            expectations=expectations,
        )


def test_sample_split_generation_is_deterministic_and_never_claims_speaker_disjointness(
    tmp_path,
):
    train_root, _relative_paths, expectations = _small_archive(tmp_path)
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    first = prepare_dvslip_development_split(train_root, first_output, expectations=expectations)
    second = prepare_dvslip_development_split(train_root, second_output, expectations=expectations)

    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first["speaker_identity_available"] is False
    assert first["speaker_disjoint"] is False
    assert first["official_test_used"] is False
    assert first["sample_counts"] == {"train": 16, "validation": 4}


def test_split_manifest_rejects_an_unverified_speaker_disjoint_claim(tmp_path):
    train_root, _relative_paths, expectations = _small_archive(tmp_path)
    path = tmp_path / "invalid_split.json"
    prepare_dvslip_development_split(
        train_root,
        path,
        expectations=expectations,
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["speaker_disjoint"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="speaker_disjoint"):
        run_dvslip_preflight(
            train_root,
            split_manifest=path,
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
