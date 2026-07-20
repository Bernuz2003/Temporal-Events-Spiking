from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing a required field."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError("The YAML root must be a mapping.")

    config = copy.deepcopy(config)
    _validate(config)

    experiment = config["experiment"]
    experiment["artifact_root"] = os.getenv(
        "ETSR_ARTIFACT_ROOT", experiment.get("artifact_root", "artifacts")
    )
    experiment["checkpoint_root"] = os.getenv(
        "ETSR_CHECKPOINT_ROOT", experiment.get("checkpoint_root", "checkpoints")
    )
    config["_source_path"] = str(config_path.resolve())
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)


def _validate(config: dict[str, Any]) -> None:
    for section in ("experiment", "dataset", "model", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ConfigError(f"Missing configuration section: {section}")

    for field in ("name", "seed"):
        if field not in config["experiment"]:
            raise ConfigError(f"Missing experiment.{field}")

    if "name" not in config["dataset"]:
        raise ConfigError("Missing dataset.name")
    if "name" not in config["model"]:
        raise ConfigError("Missing model.name")
    if int(config["training"].get("epochs", 0)) <= 0:
        raise ConfigError("training.epochs must be positive")
    if "mechanistic_audit" in config:
        _validate_mechanistic_audit(config)


def _validate_mechanistic_audit(config: dict[str, Any]) -> None:
    audit = config["mechanistic_audit"]
    if audit.get("protocol_version") != "temporal_utilization_v1":
        raise ConfigError(
            "mechanistic_audit.protocol_version must be temporal_utilization_v1"
        )
    seeds = [int(value) for value in config["experiment"].get("model_seeds", [])]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ConfigError("At least three distinct model seeds are required")
    if config["dataset"].get("name") != "matched_dvsgc":
        raise ConfigError("The mechanistic audit requires dataset.name=matched_dvsgc")
    if config["dataset"].get("official_split") != "train":
        raise ConfigError("Matched data may only use the official training partition")
    if bool(config["training"].get("evaluate_holdout", True)):
        raise ConfigError("training.evaluate_holdout must be false for the audit protocol")

    primitive_ids = [str(value) for value in config["dataset"].get("primitive_ids", [])]
    if len(primitive_ids) != 3 or len(set(primitive_ids)) != 3:
        raise ConfigError("Exactly three distinct primitives are required")
    if bool(config["dataset"].get("allow_consecutive_repetition", False)):
        raise ConfigError("Consecutive primitive repetitions must be disabled")
    if not bool(config.get("split", {}).get("forbid_official_test", False)):
        raise ConfigError("The configuration must explicitly embargo the official test")

    time_steps = int(config["dataset"]["frames_number"])
    auc_start = int(audit["prefix"]["auc_start_timestep"])
    tail_start = int(audit["prefix"]["tail_start"])
    if not 1 <= auc_start <= tail_start < time_steps:
        raise ConfigError("Prefix boundaries must satisfy 1 <= auc_start <= tail_start < T")
    if any(
        not 1 <= int(value) <= time_steps
        for value in audit["probes"]["prefix_timesteps"]
    ):
        raise ConfigError("Probe prefix timesteps must be in [1, T]")
    ratios = audit["transformations"].get("duration_ratios", [])
    if not ratios or any(
        len(ratio) != 2 or any(float(value) <= 0 for value in ratio) for ratio in ratios
    ):
        raise ConfigError("Positive order-2 duration ratios are required")
    if not audit["probes"].get("regularization_grid"):
        raise ConfigError("The probe regularization grid must not be empty")
    if set(audit["causal"]["regions"]) - {"first_action", "second_action"}:
        raise ConfigError("Unsupported causal region")
