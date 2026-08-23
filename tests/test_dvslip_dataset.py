import json

import numpy as np
import pytest

from etsr.data.events import EventSample
from etsr.dvslip.dataset import (
    DvsLipDataset,
    DvsLipExpectations,
    load_dvslip_index,
)
from etsr.dvslip.split import prepare_dvslip_development_split


def _write_event_sample(path, timestamps=(0, 100, 400)):
    path.parent.mkdir(parents=True, exist_ok=True)
    events = np.zeros(
        len(timestamps),
        dtype=[("t", "<i4"), ("x", "i1"), ("y", "i1"), ("p", "i1")],
    )
    events["t"] = timestamps
    events["x"] = [1, 4, 7]
    events["y"] = [2, 5, 6]
    events["p"] = [0, 1, 0]
    np.save(path, events)


def _development_archive(tmp_path):
    root = tmp_path / "DVS-Lip" / "train"
    for class_name in ("alpha", "beta"):
        for index in range(5):
            _write_event_sample(root / class_name / f"{index}.npy")
    expectations = DvsLipExpectations(
        class_count=2,
        sample_count=10,
        height=8,
        width=8,
    )
    manifest = tmp_path / "split.json"
    prepare_dvslip_development_split(root, manifest, expectations=expectations)
    return root, manifest, expectations


def test_raw_dataset_preserves_events_and_uses_stable_train_only_identity(tmp_path):
    root, manifest, expectations = _development_archive(tmp_path)
    # A sibling may exist locally, but development loading must neither inspect nor require it.
    (root.parent / "test").mkdir()
    (root.parent / "test" / "ignored.txt").write_text("not dataset input", encoding="utf-8")

    dataset_index = load_dvslip_index(root, manifest, expectations=expectations)
    training = DvsLipDataset(dataset_index, "train")
    validation = DvsLipDataset(dataset_index, "validation")

    assert len(training) == 8
    assert len(validation) == 2
    assert training.dataset_index_sha256 == validation.dataset_index_sha256
    assert len(training.dataset_index_sha256) == 64
    assert set(training.sample_ids).isdisjoint(validation.sample_ids)
    assert set(training.sample_ids) | set(validation.sample_ids) == {
        f"{class_name}/{index}.npy" for class_name in ("alpha", "beta") for index in range(5)
    }

    selected_path = root / training.sample_ids[0]
    _write_event_sample(selected_path, timestamps=(100, 300, 900))
    sample = training[0]

    assert isinstance(sample, EventSample)
    assert sample.sample_id == training.sample_ids[0]
    assert sample.target == training.class_to_idx[sample.metadata["class_name"]]
    assert sample.speaker_id is None
    assert sample.duration_us == 800
    assert np.array_equal(sample.t_us, np.array([100, 300, 900], dtype=np.int32))
    assert np.array_equal(sample.x, np.array([1, 4, 7], dtype=np.int8))
    assert np.array_equal(sample.y, np.array([2, 5, 6], dtype=np.int8))
    assert np.array_equal(sample.polarity, np.array([0, 1, 0], dtype=np.int8))
    assert sample.metadata["timestamp_normalized"] is False
    assert sample.metadata["official_test_used"] is False
    assert sample.metadata["polarity_convention"] == {"OFF": 0, "ON": 1}
    json.dumps(sample.metadata)


def test_raw_dataset_validates_each_sample_when_accessed(tmp_path):
    root, manifest, expectations = _development_archive(tmp_path)
    dataset_index = load_dvslip_index(root, manifest, expectations=expectations)
    dataset = DvsLipDataset(dataset_index, "train")
    _write_event_sample(root / dataset.sample_ids[0], timestamps=(100, 50, 900))

    with pytest.raises(ValueError, match="timestamps are not monotonic"):
        dataset[0]


def test_raw_dataset_accepts_only_development_splits(tmp_path):
    root, manifest, expectations = _development_archive(tmp_path)

    with pytest.raises(ValueError, match="'train' or 'validation'"):
        DvsLipDataset(load_dvslip_index(root, manifest, expectations=expectations), "test")


def test_train_and_validation_share_one_discovery(tmp_path, monkeypatch):
    root, manifest, expectations = _development_archive(tmp_path)
    from etsr.dvslip import dataset as dataset_module

    original = dataset_module.discover_training_samples
    calls = 0

    def counted_discovery(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "discover_training_samples", counted_discovery)
    dataset_index = load_dvslip_index(root, manifest, expectations=expectations)
    DvsLipDataset(dataset_index, "train")
    DvsLipDataset(dataset_index, "validation")

    assert calls == 1
