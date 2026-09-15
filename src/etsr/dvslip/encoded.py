"""Explicit adapter from raw DVS-Lip events to the shared dense training contract."""

from __future__ import annotations

from typing import Any

from etsr.data.common import DatasetBundle
from etsr.data.events import EncodedEventDataset, encoded_augmentation_kwargs
from etsr.dvslip.dataset import DvsLipDataset, load_dvslip_index
from etsr.encoders.count import (
    CountFrameEncoder,
    MultiGranularCountFrameEncoder,
    PhaseCountFrameEncoder,
    SpikeTemporalBinaryFrameEncoder,
    TemporalBinaryFrameEncoder,
)


def build_dvslip_bundle(
    dataset_config: dict[str, Any],
    representation_config: dict[str, Any],
    augmentation_config: dict[str, Any],
) -> DatasetBundle:
    """Build development train/validation datasets with no official-test holdout."""

    representation_name = representation_config.get("name")
    encoders = {
        "count_frames_e0": CountFrameEncoder,
        "phase_count_frames_e1": PhaseCountFrameEncoder,
        "temporal_binary_frames_tbr": TemporalBinaryFrameEncoder,
        "spike_tbr_lif_paper_aligned": SpikeTemporalBinaryFrameEncoder,
        "multigranular_count_frame": MultiGranularCountFrameEncoder,
    }
    if representation_name not in encoders:
        raise ValueError(f"Unsupported DVS-Lip representation: {representation_config.get('name')}")
    root = dataset_config["root"]
    split_manifest = dataset_config["split_manifest"]
    dataset_index = load_dvslip_index(root, split_manifest)
    raw_train = DvsLipDataset(dataset_index, "train")
    raw_validation = DvsLipDataset(dataset_index, "validation")
    encoder_arguments = dict(
        height=raw_train.height,
        width=raw_train.width,
        window_us=int(representation_config["window_us"]),
        bin_width_us=int(representation_config["bin_width_us"]),
    )
    if representation_name in {"count_frames_e0", "phase_count_frames_e1"}:
        encoder_arguments["count_cap"] = int(representation_config["count_cap"])
    elif representation_name == "multigranular_count_frame":
        encoder_arguments.update(
            micro_bin_width_us=int(representation_config["micro_bin_width_us"]),
            fine_spatial_stride=int(representation_config["fine_spatial_stride"]),
            count_cap=int(representation_config["count_cap"]),
            fine_count_cap=int(representation_config["fine_count_cap"]),
        )
    else:
        encoder_arguments.update(
            micro_bin_width_us=int(representation_config["micro_bin_width_us"]),
            bits=int(representation_config["bits"]),
        )
        if representation_name == "spike_tbr_lif_paper_aligned":
            encoder_arguments.update(
                lif_beta=float(representation_config["lif_beta"]),
                lif_threshold=float(representation_config["lif_threshold"]),
            )
    encoder = encoders[representation_name](**encoder_arguments)
    train = EncodedEventDataset(
        raw_train,
        encoder,
        **encoded_augmentation_kwargs(augmentation_config),
    )
    validation = EncodedEventDataset(raw_validation, encoder)
    return DatasetBundle(
        train=train,
        validation=validation,
        holdout=None,
        classes=raw_train.classes,
    )
