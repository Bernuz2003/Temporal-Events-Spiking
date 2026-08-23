"""Explicit adapter from raw DVS-Lip events to the shared dense training contract."""

from __future__ import annotations

from typing import Any

from etsr.data.common import DatasetBundle
from etsr.data.events import EncodedEventDataset
from etsr.dvslip.dataset import DvsLipDataset, load_dvslip_index
from etsr.encoders.count import CountFrameEncoder


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
    train = EncodedEventDataset(
        raw_train,
        encoder,
        horizontal_flip_probability=float(augmentation_config["horizontal_flip_probability"]),
    )
    validation = EncodedEventDataset(raw_validation, encoder)
    return DatasetBundle(
        train=train,
        validation=validation,
        holdout=None,
        classes=raw_train.classes,
    )
