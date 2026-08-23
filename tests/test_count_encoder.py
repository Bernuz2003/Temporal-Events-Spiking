from dataclasses import replace

import numpy as np
import pytest
import torch

from etsr.config import ConfigError, load_config
from etsr.data.common import build_loader
from etsr.dvslip.dataset import EventSample
from etsr.dvslip.encoded import EncodedDvsLipDataset
from etsr.encoders.count import CountFrameEncoder


def _sample() -> EventSample:
    return EventSample(
        x=np.array([1, 1, 1, 2], dtype=np.int8),
        y=np.array([2, 2, 2, 3], dtype=np.int8),
        t_us=np.array([0, 49_999, 50_000, 1_999_999], dtype=np.int32),
        polarity=np.array([0, 1, 1, 0], dtype=np.int8),
        target=3,
        sample_id="word/7.npy",
        speaker_id=None,
        duration_us=1_999_999,
        metadata={},
    )


def _encoder() -> CountFrameEncoder:
    return CountFrameEncoder(
        height=4,
        width=4,
        window_us=2_000_000,
        bin_width_us=50_000,
        count_cap=255,
    )


def test_e0_count_encoder_preserves_counts_and_physical_time_contract():
    encoded = _encoder()(_sample())

    assert encoded.tensor.dtype == torch.uint8
    assert encoded.tensor.shape == (40, 2, 4, 4)
    assert int(encoded.tensor.sum()) == 4
    assert encoded.tensor[0, 0, 2, 1] == 1
    assert encoded.tensor[0, 1, 2, 1] == 1
    assert encoded.tensor[1, 1, 2, 1] == 1
    assert encoded.tensor[39, 0, 3, 2] == 1
    assert encoded.time_axis == 0
    assert encoded.time_bin_edges_us == tuple(range(0, 2_050_000, 50_000))
    assert encoded.representation_name == "count_frames_e0"
    assert encoded.representation_parameters["logical_count_bits"] == 8
    assert encoded.representation_parameters["overflow_policy"] == "error"
    assert encoded.state_profile["persistent_state_bits"] == 0
    assert encoded.metadata["timestamp_normalized"] is False
    assert encoded.metadata["encoded_event_count"] == encoded.metadata["source_event_count"]


def test_e0_encoder_fails_instead_of_clipping_time_or_counts():
    outside_window = replace(
        _sample(),
        t_us=np.array([0, 49_999, 50_000, 2_000_000], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="must fit"):
        _encoder()(outside_window)

    event_count = 256
    overflow = EventSample(
        x=np.zeros(event_count, dtype=np.int8),
        y=np.zeros(event_count, dtype=np.int8),
        t_us=np.zeros(event_count, dtype=np.int32),
        polarity=np.zeros(event_count, dtype=np.int8),
        target=0,
        sample_id="word/overflow.npy",
        speaker_id=None,
        duration_us=0,
        metadata={},
    )
    with pytest.raises(ValueError, match="exceeding the explicit E0 uint8 cap"):
        _encoder()(overflow)


def test_encoded_dataset_adapts_to_the_shared_training_batch_contract():
    class RawFixture:
        height = 4
        width = 4
        classes = ["word"]
        class_to_idx = {"word": 0}
        sample_ids = ["word/7.npy", "word/8.npy"]
        targets = (0, 0)
        dataset_index_sha256 = "dataset-index-fixture"
        split_manifest_sha256 = "fixture-hash"

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return replace(_sample(), target=0, sample_id=self.sample_ids[index])

    dataset = EncodedDvsLipDataset(RawFixture(), _encoder())
    loader = build_loader(dataset, {"batch_size": 2, "num_workers": 0}, shuffle=False)
    frames, targets, indices = next(iter(loader))

    assert frames.shape == (2, 40, 2, 4, 4)
    assert frames.dtype == torch.float32
    assert targets.tolist() == [0, 0]
    assert indices.tolist() == [0, 1]
    assert dataset.sample_ids == ["word/7.npy", "word/8.npy"]
    assert dataset.representation_metadata["name"] == "count_frames_e0"


def test_encoded_dataset_applies_training_only_horizontal_flip():
    class RawFixture:
        height = 4
        width = 4
        classes = ["word"]
        class_to_idx = {"word": 0}
        sample_ids = ["word/7.npy"]
        targets = (0,)
        dataset_index_sha256 = "dataset-index-fixture"
        split_manifest_sha256 = "fixture-hash"

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return replace(_sample(), target=0, sample_id=self.sample_ids[index])

    plain_frames, _, _ = EncodedDvsLipDataset(RawFixture(), _encoder())[0]
    flipped_frames, _, _ = EncodedDvsLipDataset(
        RawFixture(), _encoder(), horizontal_flip_probability=1.0
    )[0]

    assert torch.equal(flipped_frames, torch.flip(plain_frames, dims=(-1,)))
    assert int(flipped_frames.sum()) == int(plain_frames.sum())


def test_dvslip_config_requires_explicit_representation_and_train_root(tmp_path):
    config_path = tmp_path / "dvslip.yaml"
    config_path.write_text(
        """
experiment:
  name: fixture
  seed: 7
dataset:
  name: dvslip
  root: data/DVS-Lip/train
  split_manifest: data/dvslip_development_split.json
  batch_size: 4
representation:
  name: count_frames_e0
  window_us: 2000000
  bin_width_us: 50000
  count_cap: 255
model:
  name: mini_qkformer
  in_channels: 2
training:
  recipe_id: fixture_e0
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
augmentation:
  horizontal_flip_probability: 0.0
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config["representation"]["window_us"] == 2_000_000

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "root: data/DVS-Lip/train", "root: data/DVS-Lip/test"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must end in 'train'"):
        load_config(config_path)
