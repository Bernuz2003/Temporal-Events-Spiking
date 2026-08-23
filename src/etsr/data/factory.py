"""Single construction seam between training orchestration and event datasets."""

from __future__ import annotations

from typing import Any

from etsr.data.common import DatasetBundle
from etsr.dvsgesture.encoded import build_dvsgesture_bundle
from etsr.dvslip.encoded import build_dvslip_bundle


def build_dataset_bundle(config: dict[str, Any]) -> DatasetBundle:
    """Build a configured dataset without leaking source details into the runner."""

    dataset = config["dataset"]
    if dataset["name"] == "dvslip":
        builder = build_dvslip_bundle
    elif dataset["name"] == "dvsgesture":
        builder = build_dvsgesture_bundle
    else:
        raise ValueError(f"Unsupported dataset: {dataset['name']}")
    return builder(dataset, config["representation"], config["augmentation"])
