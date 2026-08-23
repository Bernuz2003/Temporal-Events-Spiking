import json

import numpy as np
import pytest

from etsr.dvslip.dataset import DvsLipExpectations
from etsr.dvslip.profile import run_dvslip_profile
from etsr.dvslip.split import prepare_dvslip_development_split


def _write_event_sample(path, timestamps, polarities=(0, 1, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    events = np.zeros(
        len(timestamps),
        dtype=[("t", "<i4"), ("x", "i1"), ("y", "i1"), ("p", "i1")],
    )
    events["t"] = timestamps
    events["x"] = np.arange(len(timestamps), dtype=np.int8)
    events["y"] = np.arange(len(timestamps), dtype=np.int8)
    events["p"] = polarities
    np.save(path, events)


def _archive(tmp_path):
    root = tmp_path / "DVS-Lip" / "train"
    durations = {
        "alpha/0.npy": 200,
        "alpha/1.npy": 400,
        "beta/0.npy": 1_000,
        "beta/1.npy": 2_000,
    }
    for sample_id, duration in durations.items():
        _write_event_sample(root / sample_id, (0, duration // 2, duration))
    expectations = DvsLipExpectations(
        class_count=2,
        sample_count=4,
        height=4,
        width=4,
    )
    manifest = tmp_path / "split.json"
    prepare_dvslip_development_split(root, manifest, expectations=expectations)
    return root, manifest, expectations


def test_profile_summarizes_every_train_sample_and_development_split(tmp_path):
    root, manifest, expectations = _archive(tmp_path)
    output = tmp_path / "artifacts" / "profile.json"

    report = run_dvslip_profile(
        root,
        manifest,
        output,
        expectations=expectations,
    )

    assert report["official_test_used"] is False
    assert report["dataset"] == {
        "sample_count": 4,
        "class_count": 2,
        "class_counts": {"alpha": 2, "beta": 2},
    }
    assert report["validation"] == {
        "all_samples_valid": True,
        "validated_sample_count": 4,
        "missing_sample_count": 0,
        "corrupt_sample_count": 0,
    }
    assert report["speaker_statistics"]["available"] is False
    assert report["overall"]["duration_us"]["median"] == 700.0
    assert report["overall"]["events_per_sample"]["median"] == 3.0
    assert report["overall"]["polarity"]["on_events"] == 4
    assert report["overall"]["polarity"]["off_events"] == 8
    assert report["overall"]["spatial_occupancy_fraction"]["median"] == 3 / 16
    assert report["classes"]["alpha"]["sample_count"] == 2
    assert report["classes"]["beta"]["duration_us"]["maximum"] == 2_000.0
    assert report["development_splits"]["train"]["sample_count"] == 2
    assert report["development_splits"]["validation"]["sample_count"] == 2
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_profile_fails_without_writing_partial_output_on_corrupt_sample(tmp_path):
    root, manifest, expectations = _archive(tmp_path)
    _write_event_sample(root / "alpha" / "0.npy", (0, 100, 200), polarities=(0, 2, 0))
    output = tmp_path / "profile.json"

    with pytest.raises(ValueError, match="polarity values"):
        run_dvslip_profile(
            root,
            manifest,
            output,
            expectations=expectations,
        )

    assert not output.exists()
