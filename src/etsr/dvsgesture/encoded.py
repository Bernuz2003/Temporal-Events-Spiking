"""DVS-Gesture construction through the shared event representation path."""

from __future__ import annotations

from typing import Any

from etsr.data.common import DatasetBundle
from etsr.data.events import EncodedEventDataset, encoded_augmentation_kwargs
from etsr.dvsgesture.dataset import DvsGestureDataset, load_dvsgesture_index
from etsr.encoders.count import CountFrameEncoder


def build_dvsgesture_bundle(
    dataset_config: dict[str, Any],
    representation_config: dict[str, Any],
    augmentation_config: dict[str, Any],
) -> DatasetBundle:
    if representation_config.get("name") != "count_frames_e0":
        raise ValueError(
            f"Unsupported DVS-Gesture representation: {representation_config.get('name')}"
        )
    flip_probability = float(augmentation_config["horizontal_flip_probability"])
    if flip_probability != 0.0:
        raise ValueError("DVS-Gesture horizontal flip changes left/right gesture labels.")

    dataset_index = load_dvsgesture_index(
        dataset_config["root"],
        dataset_config["validation_subjects"],
    )
    raw_train = DvsGestureDataset(dataset_index, "train")
    raw_validation = DvsGestureDataset(dataset_index, "validation")
    encoder = CountFrameEncoder(
        height=raw_train.height,
        width=raw_train.width,
        window_us=int(representation_config["window_us"]),
        bin_width_us=int(representation_config["bin_width_us"]),
        count_cap=int(representation_config["count_cap"]),
    )
    return DatasetBundle(
        train=EncodedEventDataset(
            raw_train,
            encoder,
            **encoded_augmentation_kwargs(augmentation_config),
        ),
        validation=EncodedEventDataset(raw_validation, encoder),
        holdout=None,
        classes=raw_train.classes,
    )
