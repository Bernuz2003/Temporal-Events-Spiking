"""Single construction seam between training orchestration and the active dataset."""

from __future__ import annotations

from typing import Any

from etsr.data.common import DatasetBundle
from etsr.dvslip.encoded import build_dvslip_bundle


def build_dataset_bundle(config: dict[str, Any]) -> DatasetBundle:
    """Build the active dataset without leaking its details into the runner.

    D017 deliberately postpones a neutral raw-event abstraction until a second concrete dataset
    exists. This small seam is all the genericity needed in the meantime.
    """

    dataset = config["dataset"]
    if dataset["name"] != "dvslip":
        raise ValueError(f"Unsupported dataset: {dataset['name']}")
    return build_dvslip_bundle(
        dataset,
        config["representation"],
        config["augmentation"],
    )
