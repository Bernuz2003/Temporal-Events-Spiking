from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetBundle:
    """The data roles used by training workflows.

    ``holdout`` deliberately avoids names such as ``test`` or ``audit``: the protocol decides how
    and when that partition may be evaluated. It is absent during embargoed DVS-Lip development.
    """

    train: Dataset
    validation: Dataset
    holdout: Dataset | None
    classes: list[str]


class DatasetSubset(Dataset):
    """Index view that preserves the metadata exposed by its source dataset."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)


def balanced_overfit_bundle(
    bundle: DatasetBundle,
    class_count: int,
    samples_per_class: int,
) -> DatasetBundle:
    """Use the same small deterministic train subset for optimization and evaluation."""

    if not 1 <= class_count <= len(bundle.classes):
        raise ValueError(f"overfit class_count must be in [1, {len(bundle.classes)}]")
    if samples_per_class <= 0:
        raise ValueError("overfit samples_per_class must be positive")

    targets = getattr(bundle.train, "targets", None)
    if targets is None or len(targets) != len(bundle.train):
        raise ValueError("training datasets must expose one target per sample")

    selected = {target: [] for target in range(class_count)}
    for index, target in enumerate(targets):
        target = int(target)
        if target in selected and len(selected[target]) < samples_per_class:
            selected[target].append(index)
        if all(len(indices) == samples_per_class for indices in selected.values()):
            break

    missing = {
        target: samples_per_class - len(indices)
        for target, indices in selected.items()
        if len(indices) != samples_per_class
    }
    if missing:
        raise ValueError(f"dataset cannot provide the requested balanced overfit subset: {missing}")

    indices = [index for target in range(class_count) for index in selected[target]]
    subset = DatasetSubset(bundle.train, indices)
    return DatasetBundle(
        train=subset,
        validation=subset,
        holdout=None,
        classes=bundle.classes,
    )


def build_loader(dataset: Dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    """Build the ordinary map-style loader shared by legacy and encoded-event datasets."""

    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        drop_last=False,
        # Recreate workers at epoch boundaries so a restored main RNG state also restores their
        # augmentation seeds after an interrupted run.
        persistent_workers=False,
    )
