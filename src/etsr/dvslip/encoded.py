"""Explicit adapter from raw DVS-Lip events to the shared dense training contract."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from etsr.data.common import DatasetBundle
from etsr.dvslip.dataset import DvsLipDataset, load_dvslip_index
from etsr.encoders.count import CountFrameEncoder


class EncodedDvsLipDataset(Dataset):
    """Encode raw samples on access while retaining stable index-to-sample-ID mapping."""

    def __init__(
        self,
        raw_dataset: DvsLipDataset,
        encoder: CountFrameEncoder,
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        if (raw_dataset.height, raw_dataset.width) != (encoder.height, encoder.width):
            raise ValueError("DVS-Lip dataset and encoder sensor sizes differ.")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")
        self.raw_dataset = raw_dataset
        self.encoder = encoder
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.classes = raw_dataset.classes
        self.class_to_idx = raw_dataset.class_to_idx
        self.sample_ids = raw_dataset.sample_ids
        self.targets = raw_dataset.targets
        self.dataset_index_sha256 = raw_dataset.dataset_index_sha256
        self.split_manifest_sha256 = raw_dataset.split_manifest_sha256
        self.representation_metadata = {
            "name": encoder.name,
            "parameters": encoder.parameters,
            "state_profile": encoder.state_profile,
        }

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        encoded = self.encoder(self.raw_dataset[index])
        frames = encoded.tensor
        if self.horizontal_flip_probability and bool(
            torch.rand(()) < self.horizontal_flip_probability
        ):
            frames = torch.flip(frames, dims=(-1,))
        return frames.to(torch.float32), encoded.target, int(index)


def build_dvslip_bundle(
    dataset_config: dict[str, Any],
    representation_config: dict[str, Any],
    augmentation_config: dict[str, Any],
) -> DatasetBundle:
    """Build development train/validation datasets with no official-test holdout."""

    if representation_config.get("name") != "count_frames_e0":
        raise ValueError(f"Unsupported DVS-Lip representation: {representation_config.get('name')}")
    root = dataset_config["root"]
    split_manifest = dataset_config["split_manifest"]
    dataset_index = load_dvslip_index(root, split_manifest)
    raw_train = DvsLipDataset(dataset_index, "train")
    raw_validation = DvsLipDataset(dataset_index, "validation")
    encoder = CountFrameEncoder(
        height=raw_train.height,
        width=raw_train.width,
        window_us=int(representation_config["window_us"]),
        bin_width_us=int(representation_config["bin_width_us"]),
        count_cap=int(representation_config["count_cap"]),
    )
    train = EncodedDvsLipDataset(
        raw_train,
        encoder,
        horizontal_flip_probability=float(augmentation_config["horizontal_flip_probability"]),
    )
    validation = EncodedDvsLipDataset(raw_validation, encoder)
    return DatasetBundle(
        train=train,
        validation=validation,
        holdout=None,
        classes=raw_train.classes,
    )
