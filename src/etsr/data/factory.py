from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from etsr.data.common import DatasetBundle, IndexedDataset
from etsr.data.dvsgc import create_dvsgc_split
from etsr.data.matched_dvsgc import build_matched_dvsgc_bundle
from etsr.data.synthetic import SyntheticTemporalOrderDataset


def build_dataset_bundle(config: dict[str, Any], seed: int) -> DatasetBundle:
    name = config["name"]
    if name == "matched_dvsgc":
        return build_matched_dvsgc_bundle(config)
    if name == "dvsgc":
        train_raw = create_dvsgc_split(config, "train")
        validation_raw = create_dvsgc_split(config, "validation")
        test_raw = create_dvsgc_split(config, "test")
    elif name == "synthetic_temporal_order":
        common = dict(
            frames_number=int(config["frames_number"]),
            image_size=int(config["image_size"]),
            num_classes=int(config["num_classes"]),
        )
        train_raw = SyntheticTemporalOrderDataset(
            samples=int(config["train_samples"]), seed=seed, **common
        )
        validation_raw = SyntheticTemporalOrderDataset(
            samples=int(config["validation_samples"]), seed=seed + 10_000, **common
        )
        test_raw = SyntheticTemporalOrderDataset(
            samples=int(config["test_samples"]), seed=seed + 20_000, **common
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    wrapper_args = dict(
        input_clip=config.get("input_clip"),
        input_scale=str(config.get("input_scale", "none")),
    )
    train = IndexedDataset(train_raw, **wrapper_args)
    validation = IndexedDataset(validation_raw, **wrapper_args)
    holdout = IndexedDataset(test_raw, **wrapper_args)

    classes = list(getattr(train_raw, "classes", []))
    if not classes:
        class_count = int(config.get("num_classes", 0))
        classes = [str(index) for index in range(class_count)]
    return DatasetBundle(train=train, validation=validation, holdout=holdout, classes=classes)


def build_loader(dataset: Dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        drop_last=False,
        persistent_workers=bool(config.get("num_workers", 0) > 0),
    )
