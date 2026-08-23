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
    if config["dataset"]["name"] == "dvslip":
        _validate_dvslip(config)
    if "mechanistic_audit" in config:
        _validate_mechanistic_audit(config)


def _validate_dvslip(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    for field in ("root", "split_manifest"):
        if not isinstance(dataset.get(field), str) or not dataset[field].strip():
            raise ConfigError(f"DVS-Lip requires dataset.{field}")

    representation = config.get("representation")
    if not isinstance(representation, dict):
        raise ConfigError("DVS-Lip requires a representation section")
    if representation.get("name") != "count_frames_e0":
        raise ConfigError(
            "The current DVS-Lip foundation supports representation.name=count_frames_e0"
        )
    for field in ("window_us", "bin_width_us", "count_cap"):
        if type(representation.get(field)) is not int or representation[field] <= 0:
            raise ConfigError(f"DVS-Lip representation.{field} must be a positive integer")
    if representation["window_us"] % representation["bin_width_us"]:
        raise ConfigError("DVS-Lip representation.window_us must divide into exact bins")
    if representation["count_cap"] > 255:
        raise ConfigError("DVS-Lip E0 count_cap must fit uint8")
    model = config["model"]
    if int(model.get("in_channels", 0)) != 2:
        raise ConfigError("DVS-Lip E0 requires model.in_channels=2")
    surrogate_name = str(model.get("surrogate_name", "fast_sigmoid"))
    if surrogate_name not in {"fast_sigmoid", "sigmoid"}:
        raise ConfigError("model.surrogate_name must be fast_sigmoid or sigmoid")
    surrogate_alpha = model.get("surrogate_alpha", 25.0)
    if (
        type(surrogate_alpha) not in (int, float)
        or isinstance(surrogate_alpha, bool)
        or surrogate_alpha <= 0.0
    ):
        raise ConfigError("model.surrogate_alpha must be positive")

    augmentation = config.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ConfigError("DVS-Lip requires an augmentation section")
    unsupported_augmentations = set(augmentation) - {"horizontal_flip_probability"}
    if unsupported_augmentations:
        raise ConfigError(
            f"Unsupported DVS-Lip augmentations: {sorted(unsupported_augmentations)}"
        )
    flip_probability = augmentation.get("horizontal_flip_probability")
    if type(flip_probability) not in (int, float) or not 0.0 <= flip_probability <= 1.0:
        raise ConfigError("augmentation.horizontal_flip_probability must be in [0, 1]")

    training = config["training"]
    if not isinstance(training.get("recipe_id"), str) or not training["recipe_id"].strip():
        raise ConfigError("DVS-Lip requires a non-empty training.recipe_id")
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise ConfigError("The current DVS-Lip recipe supports training.optimizer=adamw")
    if str(training.get("scheduler", "")).lower() != "cosine":
        raise ConfigError("The current DVS-Lip recipe supports training.scheduler=cosine")
    learning_rate = training.get("learning_rate")
    minimum_lr = training.get("min_learning_rate")
    if type(learning_rate) not in (int, float) or learning_rate <= 0.0:
        raise ConfigError("training.learning_rate must be positive")
    if type(minimum_lr) not in (int, float) or not 0.0 <= minimum_lr < learning_rate:
        raise ConfigError("training.min_learning_rate must be in [0, learning_rate)")
    epochs = int(training["epochs"])
    warmup_epochs = training.get("warmup_epochs")
    if type(warmup_epochs) is not int or not 0 <= warmup_epochs < epochs:
        raise ConfigError("training.warmup_epochs must be an integer in [0, epochs)")
    warmup_start_factor = training.get("warmup_start_factor")
    if type(warmup_start_factor) not in (int, float) or not 0.0 < warmup_start_factor <= 1.0:
        raise ConfigError("training.warmup_start_factor must be in (0, 1]")
    accumulation_steps = training.get("gradient_accumulation_steps", 1)
    if type(accumulation_steps) is not int or accumulation_steps <= 0:
        raise ConfigError("training.gradient_accumulation_steps must be a positive integer")
    weight_decay = training.get("weight_decay")
    if type(weight_decay) not in (int, float) or weight_decay < 0.0:
        raise ConfigError("training.weight_decay must be non-negative")
    label_smoothing = training.get("label_smoothing")
    if type(label_smoothing) not in (int, float) or not 0.0 <= label_smoothing < 1.0:
        raise ConfigError("training.label_smoothing must be in [0, 1)")
    gradient_clip_norm = training.get("gradient_clip_norm")
    if type(gradient_clip_norm) not in (int, float) or gradient_clip_norm <= 0.0:
        raise ConfigError("training.gradient_clip_norm must be positive")
    if int(dataset.get("batch_size", 0)) <= 0:
        raise ConfigError("dataset.batch_size must be positive")
    if bool(training.get("evaluate_holdout", True)):
        raise ConfigError("DVS-Lip development requires training.evaluate_holdout=false")


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
