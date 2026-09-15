import csv
import json
import struct
from pathlib import Path

import pytest

from etsr.config import ConfigError, load_config
from etsr.data.factory import build_dataset_bundle
from etsr.dvsgesture.dataset import DvsGestureDataset, load_dvsgesture_index
from etsr.dvsgesture.prepare import prepare_dvsgesture_train
from etsr.dvsgesture.profile import run_dvsgesture_profile


def test_canonical_dvsgesture_config_follows_the_profiled_protocol():
    config = load_config(Path(__file__).parents[1] / "configs" / "dvsgesture_e0.yaml")

    assert config["dataset"]["validation_subjects"] == [8, 11, 17, 19, 22]
    assert config["representation"]["window_us"] == 20_000_000
    assert config["representation"]["bin_width_us"] == 200_000
    assert config["dataset"]["batch_size"] * config["training"][
        "gradient_accumulation_steps"
    ] == 32
    assert config["evaluation"]["absolute_prefix_times_us"][-1] == 20_000_000
    assert config["evaluation"]["relative_prefix_fractions"] == [0.1, 0.25, 0.5, 0.75, 1.0]


def test_dvsgesture_frozen_transfer_changes_only_the_model_topology():
    root = Path(__file__).parents[1]
    baseline = load_config(root / "configs" / "dvsgesture_e0.yaml")
    transfer = load_config(root / "configs" / "dvsgesture_f_tcap_stage1_dwc3_d8.yaml")
    for section in ("dataset", "representation", "augmentation", "evaluation", "training"):
        assert transfer[section] == baseline[section]
    assert transfer["model"] == {
        **baseline["model"],
        "frontend": "pyramidal",
        "temporal_channel_mixer": True,
        "temporal_channel_mixer_delays": [1, 2, 4, 8],
        "stage1_mixer": "depthwise_conv",
        "stage1_depthwise_kernel_size": 3,
    }


def _write_recording(source_root, subject, *, invert_annotated_segment=False):
    stem = f"user{subject:02d}_lab"
    rows = []
    events = []
    for target in range(11):
        start = target * 1_000
        end = start + 100
        rows.append((target + 1, start, end))
        deltas = (50, 10, 90) if invert_annotated_segment and target == 0 else (10, 50, 90)
        for delta, polarity in zip(deltas, (0, 1, 0), strict=True):
            x = target + 1
            y = target + 2
            address = (x << 17) | (y << 2) | (polarity << 1) | 1
            events.append((address, start + delta))
        if target == 4:
            # The official user08_led file has one reversal in this unannotated inter-gesture gap.
            events.extend(((1, 4_500), (1, 4_400)))

    payload = b"".join(struct.pack("<II", address, timestamp) for address, timestamp in events)
    packet_header = struct.pack(
        "<HHIIIIII",
        1,
        0,
        8,
        4,
        0,
        len(events),
        len(events),
        len(events),
    )
    (source_root / f"{stem}.aedat").write_bytes(
        b"#!AER-DAT3.1\r\n#!END-HEADER\r\n" + packet_header + payload
    )
    with (source_root / f"{stem}_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "startTime_usec", "endTime_usec"))
        writer.writerows(rows)
    return f"{stem}.aedat"


def _prepare_fixture(tmp_path):
    source_root = tmp_path / "DVS-Gesture" / "DvsGesture"
    source_root.mkdir(parents=True)
    names = [_write_recording(source_root, subject) for subject in (1, 2)]
    (source_root / "trials_to_train.txt").write_text("\n".join(names), encoding="utf-8")
    # Preparation is deliberately train-only and must not inspect this list.
    (source_root / "trials_to_test.txt").write_text("missing_test.aedat\n", encoding="utf-8")
    train_root = tmp_path / "DVS-Gesture" / "events" / "train"
    report_path = tmp_path / "preparation.json"
    report = prepare_dvsgesture_train(source_root, train_root, report_path)
    return train_root, report, report_path


def _write_config(path, train_root, flip_probability=0.0):
    path.write_text(
        f"""
experiment:
  name: dvsgesture_fixture
  seed: 7
dataset:
  name: dvsgesture
  root: {train_root}
  validation_subjects: [2]
  batch_size: 2
  num_workers: 0
  pin_memory: false
representation:
  name: count_frames_e0
  window_us: 100
  bin_width_us: 10
  count_cap: 255
augmentation:
  horizontal_flip_probability: {flip_probability}
model:
  name: mini_qkformer
  in_channels: 2
  surrogate_alpha: 4.0
training:
  recipe_id: dvsgesture_fixture_e0
  epochs: 1
  optimizer: adamw
  learning_rate: 0.001
  min_learning_rate: 0.000001
  scheduler: cosine
  warmup_epochs: 0
  warmup_start_factor: 0.01
  weight_decay: 0.0005
  label_smoothing: 0.1
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  amp: false
  select_metric: macro_f1
""".strip(),
        encoding="utf-8",
    )


def test_preparation_preserves_physical_events_and_ignores_official_test(tmp_path):
    train_root, report, report_path = _prepare_fixture(tmp_path)

    assert report["sample_count"] == 22
    assert report["recording_count"] == 2
    assert report["official_test_used"] is False
    assert report["duration_us"]["minimum"] == 100
    assert json.loads(report_path.read_text(encoding="utf-8"))["source_split"] == "train"

    dataset_index = load_dvsgesture_index(train_root, validation_subjects=[2])
    training = DvsGestureDataset(dataset_index, "train")
    validation = DvsGestureDataset(dataset_index, "validation")
    assert len(training) == 11
    assert len(validation) == 11
    assert set(training.subjects) == {1}
    assert set(validation.subjects) == {2}
    assert set(training.sample_ids).isdisjoint(validation.sample_ids)
    assert training.development_split_sha256 == validation.development_split_sha256

    sample = validation[0]
    assert sample.speaker_id == 2
    assert sample.duration_us == 100
    assert sample.t_us.tolist() == [10, 50, 90]
    assert sample.metadata["timestamp_origin"] == "annotated_segment_start"
    assert sample.metadata["timestamp_normalized"] is False
    assert sample.metadata["official_test_used"] is False


def test_profile_and_factory_validate_the_shared_dvsgesture_path(tmp_path):
    train_root, _, _ = _prepare_fixture(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile = run_dvsgesture_profile(train_root, profile_path)
    assert profile["dataset"]["sample_count"] == 22
    assert profile["dataset"]["class_count"] == 11
    assert profile["dataset"]["subject_count"] == 2
    assert profile["validation"]["all_samples_valid"] is True

    config_path = tmp_path / "dvsgesture.yaml"
    _write_config(config_path, train_root)
    bundle = build_dataset_bundle(load_config(config_path))
    frames, target, index = bundle.validation[0]
    assert len(bundle.classes) == 11
    assert frames.shape == (10, 2, 128, 128)
    assert int(frames.sum()) == 3
    assert target == 0
    assert index == 0
    assert bundle.train.runtime_metadata["speaker_disjoint"] is True


def test_dvsgesture_rejects_label_changing_horizontal_flip(tmp_path):
    train_root, _, _ = _prepare_fixture(tmp_path)
    config_path = tmp_path / "dvsgesture.yaml"
    _write_config(config_path, train_root, flip_probability=0.5)

    with pytest.raises(ConfigError, match="changes left/right gesture labels"):
        load_config(config_path)


def test_preparation_refuses_to_overwrite_existing_derived_data(tmp_path):
    train_root, _, _ = _prepare_fixture(tmp_path)
    source_root = tmp_path / "DVS-Gesture" / "DvsGesture"

    with pytest.raises(FileExistsError, match="remove it explicitly"):
        prepare_dvsgesture_train(source_root, train_root)


def test_preparation_rejects_nonmonotonic_events_inside_an_annotation(tmp_path):
    source_root = tmp_path / "DvsGesture"
    source_root.mkdir()
    name = _write_recording(source_root, 1, invert_annotated_segment=True)
    (source_root / "trials_to_train.txt").write_text(name, encoding="utf-8")

    with pytest.raises(ValueError, match="Annotated segment 0 has non-monotonic timestamps"):
        prepare_dvsgesture_train(source_root, tmp_path / "events" / "train")
