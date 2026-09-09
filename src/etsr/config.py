from __future__ import annotations

import math
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


def validate_config(config: dict[str, Any]) -> None:
    """Validate an in-memory configuration after explicit CLI experiment overrides."""

    _validate(config)


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
    dataset_name = config["dataset"]["name"]
    if dataset_name == "dvslip":
        _validate_dvslip(config)
    elif dataset_name == "dvsgesture":
        _validate_dvsgesture(config)
    else:
        raise ConfigError(f"Unsupported dataset: {dataset_name}")
    _validate_evaluation(config, dataset_name)


def _validate_evaluation(config: dict[str, Any], dataset_name: str) -> None:
    evaluation = config.get("evaluation")
    if evaluation is None:
        return
    if not isinstance(evaluation, dict):
        raise ConfigError("evaluation must be a mapping")
    unsupported = set(evaluation) - {
        "absolute_prefix_times_us",
        "relative_prefix_fractions",
        "class_groups_manifest",
    }
    if unsupported:
        raise ConfigError(f"Unsupported evaluation fields: {sorted(unsupported)}")

    times = evaluation.get("absolute_prefix_times_us")
    window_us = int(config.get("representation", {}).get("window_us", 0))
    if times is not None:
        if (
            not isinstance(times, list)
            or len(times) < 2
            or any(type(value) is not int or value <= 0 for value in times)
            or any(right <= left for left, right in zip(times, times[1:], strict=False))
            or times[-1] != window_us
        ):
            raise ConfigError(
                "evaluation.absolute_prefix_times_us must be strictly increasing positive "
                "integers ending at representation.window_us"
            )
        bin_width_us = int(config["representation"]["bin_width_us"])
        observed_steps = [(int(value) + bin_width_us - 1) // bin_width_us for value in times]
        if len(observed_steps) != len(set(observed_steps)):
            raise ConfigError(
                "evaluation.absolute_prefix_times_us collapse to duplicate encoded steps"
            )

    fractions = evaluation.get("relative_prefix_fractions")
    if fractions is not None:
        if (
            not isinstance(fractions, list)
            or len(fractions) < 2
            or any(
                type(fraction) not in (int, float) or not 0.0 < float(fraction) <= 1.0
                for fraction in fractions
            )
            or any(
                float(right) <= float(left)
                for left, right in zip(fractions, fractions[1:], strict=False)
            )
            or float(fractions[-1]) != 1.0
        ):
            raise ConfigError(
                "evaluation.relative_prefix_fractions must be strictly increasing values in "
                "(0, 1] ending at 1.0"
            )

    class_groups = evaluation.get("class_groups_manifest")
    if class_groups is not None:
        if dataset_name != "dvslip":
            raise ConfigError("evaluation.class_groups_manifest is currently DVS-Lip-specific")
        if not isinstance(class_groups, str) or not class_groups.strip():
            raise ConfigError("evaluation.class_groups_manifest must be a non-empty path")


def _validate_dvslip(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    for field in ("root", "split_manifest"):
        if not isinstance(dataset.get(field), str) or not dataset[field].strip():
            raise ConfigError(f"DVS-Lip requires dataset.{field}")
    if Path(dataset["root"]).name != "train":
        raise ConfigError("DVS-Lip development dataset.root must end in 'train'")

    _validate_event_baseline(config, "DVS-Lip")


def _validate_dvsgesture(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    if not isinstance(dataset.get("root"), str) or not dataset["root"].strip():
        raise ConfigError("DVS-Gesture requires dataset.root")
    if Path(dataset["root"]).name != "train":
        raise ConfigError("DVS-Gesture development dataset.root must end in 'train'")
    validation_subjects = dataset.get("validation_subjects")
    if (
        not isinstance(validation_subjects, list)
        or not validation_subjects
        or any(
            type(subject) is not int or not 1 <= subject <= 23 for subject in validation_subjects
        )
        or len(validation_subjects) != len(set(validation_subjects))
    ):
        raise ConfigError(
            "DVS-Gesture dataset.validation_subjects must be unique official-train subject IDs"
        )

    _validate_event_baseline(config, "DVS-Gesture")
    if float(config["augmentation"]["horizontal_flip_probability"]) != 0.0:
        raise ConfigError("DVS-Gesture horizontal flip changes left/right gesture labels")


def _validate_event_baseline(config: dict[str, Any], dataset_label: str) -> None:
    dataset = config["dataset"]
    representation = config.get("representation")
    if not isinstance(representation, dict):
        raise ConfigError(f"{dataset_label} requires a representation section")
    representation_name = representation.get("name")
    allowed_representations = (
        {
            "count_frames_e0",
            "phase_count_frames_e1",
            "temporal_binary_frames_tbr",
            "spike_tbr_lif_paper_aligned",
        }
        if dataset_label == "DVS-Lip"
        else {"count_frames_e0"}
    )
    if representation_name not in allowed_representations:
        raise ConfigError(f"Unsupported {dataset_label} representation.name={representation_name}")
    for field in ("window_us", "bin_width_us"):
        if type(representation.get(field)) is not int or representation[field] <= 0:
            raise ConfigError(f"{dataset_label} representation.{field} must be a positive integer")
    if representation["window_us"] % representation["bin_width_us"]:
        raise ConfigError(f"{dataset_label} representation.window_us must divide into exact bins")
    if representation_name in {"count_frames_e0", "phase_count_frames_e1"}:
        if type(representation.get("count_cap")) is not int or representation["count_cap"] <= 0:
            raise ConfigError(f"{dataset_label} representation.count_cap must be positive")
        if representation["count_cap"] > 255:
            raise ConfigError(f"{dataset_label} count_cap must fit uint8")
    if representation_name in {"temporal_binary_frames_tbr", "spike_tbr_lif_paper_aligned"}:
        for field in ("micro_bin_width_us", "bits"):
            if type(representation.get(field)) is not int or representation[field] <= 0:
                raise ConfigError(
                    f"{dataset_label} {representation_name} requires positive integer {field}"
                )
        if representation["bits"] > 16:
            raise ConfigError("TBR representations support at most 16 bits")
        if representation["bits"] * representation["micro_bin_width_us"] != representation["bin_width_us"]:
            raise ConfigError("TBR bits * micro_bin_width_us must equal bin_width_us")
    if representation_name == "spike_tbr_lif_paper_aligned":
        beta = representation.get("lif_beta")
        threshold = representation.get("lif_threshold")
        if type(beta) not in (int, float) or isinstance(beta, bool) or not 0.0 <= beta < 1.0:
            raise ConfigError("Spike-TBR representation.lif_beta must be in [0, 1)")
        if (
            type(threshold) not in (int, float)
            or isinstance(threshold, bool)
            or threshold <= 0.0
        ):
            raise ConfigError("Spike-TBR representation.lif_threshold must be positive")
    model = config["model"]
    if model["name"] != "mini_qkformer":
        raise ConfigError("The active baseline requires model.name=mini_qkformer")
    expected_channels = {
        "count_frames_e0": 2,
        "phase_count_frames_e1": 4,
        "temporal_binary_frames_tbr": 1,
        "spike_tbr_lif_paper_aligned": 1,
    }[representation_name]
    if int(model.get("in_channels", 0)) != expected_channels:
        raise ConfigError(
            f"{dataset_label} {representation_name} requires model.in_channels={expected_channels}"
        )
    surrogate_alpha = model.get("surrogate_alpha", 4.0)
    if (
        type(surrogate_alpha) not in (int, float)
        or isinstance(surrogate_alpha, bool)
        or surrogate_alpha <= 0.0
    ):
        raise ConfigError("model.surrogate_alpha must be positive")
    if type(model.get("lif_cross_time", True)) is not bool:
        raise ConfigError("model.lif_cross_time must be boolean")
    if model.get("readout", "mean") not in {"mean", "last", "diagonal_gated"}:
        raise ConfigError("model.readout must be mean, last or diagonal_gated")
    if model.get("readout_time", "fixed_window") not in {"fixed_window", "last_event"}:
        raise ConfigError("model.readout_time must be fixed_window or last_event")
    frontend = model.get("frontend", "baseline")
    if frontend not in {"baseline", "pyramidal"}:
        raise ConfigError("model.frontend must be baseline or pyramidal")
    embed_dim = int(model.get("embed_dim", 128))
    if frontend == "pyramidal" and embed_dim % 16:
        raise ConfigError("model.frontend=pyramidal requires embed_dim divisible by 16")
    if type(model.get("temporal_fir", False)) is not bool:
        raise ConfigError("model.temporal_fir must be boolean")
    if type(model.get("temporal_channel_mixer", False)) is not bool:
        raise ConfigError("model.temporal_channel_mixer must be boolean")
    if type(model.get("learnable_lif_tau", False)) is not bool:
        raise ConfigError("model.learnable_lif_tau must be boolean")
    if model.get("temporal_fir", False) and model.get("temporal_channel_mixer", False):
        raise ConfigError("temporal FIR and channel mixer are mutually exclusive")
    fir_kernel_size = model.get("temporal_fir_kernel_size", 3)
    if type(fir_kernel_size) is not int or fir_kernel_size < 2:
        raise ConfigError("model.temporal_fir_kernel_size must be an integer of at least two")
    fir_dilations = model.get("temporal_fir_dilations", [1, 2])
    if (
        not isinstance(fir_dilations, list)
        or len(fir_dilations) != 2
        or any(type(dilation) is not int or dilation <= 0 for dilation in fir_dilations)
    ):
        raise ConfigError("model.temporal_fir_dilations must contain two positive integers")
    if model.get("temporal_fir", False) and not model.get("lif_cross_time", True):
        raise ConfigError("no-cross-time control cannot include temporal FIR memory")
    mixer_delays = model.get("temporal_channel_mixer_delays", [1, 2, 4])
    if (
        not isinstance(mixer_delays, list)
        or not mixer_delays
        or any(type(delay) is not int or delay <= 0 for delay in mixer_delays)
        or sorted(set(mixer_delays)) != mixer_delays
    ):
        raise ConfigError(
            "model.temporal_channel_mixer_delays must be increasing positive integers"
        )
    if model.get("temporal_channel_mixer", False) and not model.get("lif_cross_time", True):
        raise ConfigError("no-cross-time control cannot include a temporal channel mixer")
    gated_memory = model.get("gated_initial_memory_steps")
    if gated_memory is not None and (
        type(gated_memory) not in (int, float)
        or isinstance(gated_memory, bool)
        or gated_memory <= 1.0
        or not math.isfinite(gated_memory)
    ):
        raise ConfigError("model.gated_initial_memory_steps must be greater than one")

    augmentation = config.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ConfigError(f"{dataset_label} requires an augmentation section")
    supported_augmentations = {
        "horizontal_flip_probability",
        "temporal_mask_count",
        "temporal_mask_max_steps",
        "spatial_erasing_count",
        "spatial_erasing_max_pixels",
    }
    unsupported_augmentations = set(augmentation) - supported_augmentations
    if unsupported_augmentations:
        raise ConfigError(
            f"Unsupported {dataset_label} augmentations: {sorted(unsupported_augmentations)}"
        )
    flip_probability = augmentation.get("horizontal_flip_probability")
    if type(flip_probability) not in (int, float) or not 0.0 <= flip_probability <= 1.0:
        raise ConfigError("augmentation.horizontal_flip_probability must be in [0, 1]")
    for name in ("temporal_mask", "spatial_erasing"):
        count = augmentation.get(f"{name}_count", 0)
        extent_field = "max_steps" if name == "temporal_mask" else "max_pixels"
        extent = augmentation.get(f"{name}_{extent_field}", 0)
        if type(count) is not int or type(extent) is not int or count < 0 or extent < 0:
            raise ConfigError(f"augmentation.{name} values must be non-negative integers")
        if (count == 0) != (extent == 0):
            raise ConfigError(f"augmentation.{name} count and maximum must be enabled together")
    time_steps = int(representation["window_us"]) // int(representation["bin_width_us"])
    if int(augmentation.get("temporal_mask_max_steps", 0)) > time_steps:
        raise ConfigError("augmentation.temporal_mask_max_steps must fit the time axis")

    training = config["training"]
    if not isinstance(training.get("recipe_id"), str) or not training["recipe_id"].strip():
        raise ConfigError(f"{dataset_label} requires a non-empty training.recipe_id")
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise ConfigError(f"The current {dataset_label} recipe supports training.optimizer=adamw")
    if str(training.get("scheduler", "")).lower() != "cosine":
        raise ConfigError(f"The current {dataset_label} recipe supports training.scheduler=cosine")
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
    if training.get("select_metric", "macro_f1") not in ("accuracy", "macro_f1"):
        raise ConfigError("training.select_metric must be accuracy or macro_f1")
