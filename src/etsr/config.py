from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing a required field."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = config_path.resolve()
    if resolved in stack:
        raise ConfigError(f"Configuration inheritance cycle: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("The YAML root must be a mapping.")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    if not isinstance(parent, str) or not parent.strip():
        raise ConfigError("extends must be a non-empty path")
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent configuration not found: {parent_path}")
    return _deep_merge(_load_yaml_with_extends(parent_path, (*stack, resolved)), config)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    config = _load_yaml_with_extends(config_path)

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
            "multigranular_count_frame",
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
    if representation_name in {
        "count_frames_e0",
        "phase_count_frames_e1",
        "multigranular_count_frame",
    }:
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
    if representation_name == "multigranular_count_frame":
        micro_width = representation.get("micro_bin_width_us")
        spatial_stride = representation.get("fine_spatial_stride")
        if type(micro_width) is not int or micro_width <= 0:
            raise ConfigError("MultiGranular requires positive micro_bin_width_us")
        if representation["bin_width_us"] % micro_width:
            raise ConfigError("MultiGranular micro bins must divide the coarse bin")
        if type(spatial_stride) is not int or spatial_stride <= 1:
            raise ConfigError("MultiGranular fine_spatial_stride must exceed one")
        fine_count_cap = representation.get("fine_count_cap")
        if type(fine_count_cap) is not int or not 0 < fine_count_cap <= 65_535:
            raise ConfigError("MultiGranular fine_count_cap must fit the unsigned 16-bit range")
    model = config["model"]
    if model["name"] != "mini_qkformer":
        raise ConfigError("The active baseline requires model.name=mini_qkformer")
    expected_channels = {
        "count_frames_e0": 2,
        "phase_count_frames_e1": 4,
        "temporal_binary_frames_tbr": 1,
        "spike_tbr_lif_paper_aligned": 1,
        "multigranular_count_frame": 2,
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
    stage1_mixer = model.get("stage1_mixer", "token_qk")
    if stage1_mixer not in {"token_qk", "depthwise_conv"}:
        raise ConfigError("model.stage1_mixer must be token_qk or depthwise_conv")
    stage1_kernel = model.get("stage1_depthwise_kernel_size", 3)
    if type(stage1_kernel) is not int or stage1_kernel < 3 or stage1_kernel % 2 == 0:
        raise ConfigError("model.stage1_depthwise_kernel_size must be an odd integer >= 3")
    multigranular = model.get("multigranular", False)
    if type(multigranular) is not bool:
        raise ConfigError("model.multigranular must be boolean")
    is_multigranular = representation_name == "multigranular_count_frame"
    continuation = config.get("continuation")
    predictive = config.get("predictive")
    # First-execution resolved configs kept the objective inside continuation.
    objective_source = (
        predictive
        if isinstance(predictive, dict)
        else continuation
        if isinstance(continuation, dict)
        else {}
    )
    objective_mode = str(objective_source.get("objective", {}).get("mode", "none"))
    training_only_fine_input = objective_mode in {"fine_future", "fine_same"}
    if is_multigranular != multigranular and not (
        is_multigranular and not multigranular and training_only_fine_input
    ):
        raise ConfigError("Multi-granular representation and model configuration must match")
    if is_multigranular:
        fine_channels = model.get("multigranular_fine_channels", 16)
        micro_steps = model.get("multigranular_micro_steps", 8)
        expected_micro_steps = representation["bin_width_us"] // representation["micro_bin_width_us"]
        if type(fine_channels) is not int or fine_channels <= 0:
            raise ConfigError("model.multigranular_fine_channels must be positive")
        if type(micro_steps) is not int or micro_steps != expected_micro_steps:
            raise ConfigError(
                "model.multigranular_micro_steps must match the representation clock ratio"
            )
        if frontend != "pyramidal":
            raise ConfigError("Multi-granular branches require model.frontend=pyramidal")
        mid_channels = model.get("multigranular_fine_mid_channels")
        if mid_channels is not None and (type(mid_channels) is not int or mid_channels <= 0):
            raise ConfigError("model.multigranular_fine_mid_channels must be positive or null")
        if multigranular:
            temporal_groups = model.get("multigranular_temporal_groups")
            if (
                type(temporal_groups) is not int
                or temporal_groups <= 0
                or (embed_dim // 2) % temporal_groups
            ):
                raise ConfigError("model.multigranular_temporal_groups must divide embed_dim/2")
            if model.get("multigranular_fusion") not in {"add", "concat_residual"}:
                raise ConfigError("model.multigranular_fusion must be add or concat_residual")
    if type(model.get("temporal_fir", False)) is not bool:
        raise ConfigError("model.temporal_fir must be boolean")
    if type(model.get("temporal_channel_mixer", False)) is not bool:
        raise ConfigError("model.temporal_channel_mixer must be boolean")
    learnable_delays = model.get("temporal_channel_mixer_learnable_delays", False)
    if type(learnable_delays) is not bool:
        raise ConfigError("model.temporal_channel_mixer_learnable_delays must be boolean")
    if learnable_delays and not model.get("temporal_channel_mixer", False):
        raise ConfigError("Learnable delays require model.temporal_channel_mixer=true")
    dynamic_routing = model.get("temporal_channel_mixer_dynamic_routing", False)
    predictive_auxiliary = model.get("temporal_channel_mixer_predictive_auxiliary", False)
    surprise_routing = model.get("temporal_channel_mixer_surprise_routing", False)
    for name, enabled in (
        ("dynamic_routing", dynamic_routing),
        ("predictive_auxiliary", predictive_auxiliary),
        ("surprise_routing", surprise_routing),
        ("predictive_head", model.get("predictive_head", False)),
    ):
        if type(enabled) is not bool:
            raise ConfigError(f"model.{name} must be boolean")
    if (dynamic_routing or predictive_auxiliary) and not model.get("temporal_channel_mixer", False):
        raise ConfigError("Conditional temporal options require model.temporal_channel_mixer=true")
    if surprise_routing and not predictive_auxiliary:
        raise ConfigError("Surprise routing requires the predictive auxiliary")
    if learnable_delays and (dynamic_routing or predictive_auxiliary):
        raise ConfigError("Conditional routing is defined only for fixed TCAP delays")
    router_pooling = model.get("temporal_channel_mixer_router_pooling", "global")
    if router_pooling not in {"global", "local"}:
        raise ConfigError("model.temporal_channel_mixer_router_pooling must be global or local")
    router_hidden_divisor = model.get("temporal_channel_mixer_router_hidden_divisor")
    if router_hidden_divisor is not None and (
        type(router_hidden_divisor) is not int or router_hidden_divisor <= 0
    ):
        raise ConfigError(
            "model.temporal_channel_mixer_router_hidden_divisor must be positive or null"
        )
    if not (dynamic_routing or surprise_routing) and (
        router_pooling != "global" or router_hidden_divisor is not None
    ):
        raise ConfigError("Temporal router geometry requires dynamic routing")
    routing_stages = model.get("temporal_channel_mixer_routing_stages", [1, 2])
    if (
        not isinstance(routing_stages, list)
        or not routing_stages
        or any(type(stage) is not int or stage not in {1, 2} for stage in routing_stages)
        or sorted(set(routing_stages)) != routing_stages
    ):
        raise ConfigError("model.temporal_channel_mixer_routing_stages must be [1], [2] or [1, 2]")
    predictive_stages = model.get("temporal_channel_mixer_predictive_stages", [1, 2])
    if (
        not isinstance(predictive_stages, list)
        or not predictive_stages
        or any(type(stage) is not int or stage not in {1, 2} for stage in predictive_stages)
        or sorted(set(predictive_stages)) != predictive_stages
    ):
        raise ConfigError(
            "model.temporal_channel_mixer_predictive_stages must be [1], [2] or [1, 2]"
        )
    if not predictive_auxiliary and predictive_stages != [1, 2]:
        raise ConfigError("Predictive stages require the predictive auxiliary")
    if surprise_routing and not set(routing_stages) <= set(predictive_stages):
        raise ConfigError("Surprise routing needs a causal predictor in every routed stage")
    routing_parameterization = model.get(
        "temporal_channel_mixer_routing_parameterization", "independent"
    )
    if routing_parameterization not in {"independent", "amplitude_allocation"}:
        raise ConfigError("Unsupported temporal routing parameterization")
    if not (dynamic_routing or surprise_routing) and (
        routing_stages != [1, 2] or routing_parameterization != "independent"
    ):
        raise ConfigError("Temporal routing controls require an enabled router")
    predictor_groups = model.get("temporal_channel_mixer_predictor_channel_groups")
    if predictor_groups is not None and (
        type(predictor_groups) is not int
        or predictor_groups <= 0
        or (embed_dim // 2) % predictor_groups
        or embed_dim % predictor_groups
    ):
        raise ConfigError(
            "model.temporal_channel_mixer_predictor_channel_groups must divide both TCAP widths"
        )
    predictor_kernel = model.get("temporal_channel_mixer_predictor_spatial_kernel_size", 1)
    if type(predictor_kernel) is not int or predictor_kernel <= 0 or predictor_kernel % 2 == 0:
        raise ConfigError(
            "model.temporal_channel_mixer_predictor_spatial_kernel_size must be a positive odd integer"
        )
    if not predictive_auxiliary and (predictor_groups is not None or predictor_kernel != 1):
        raise ConfigError("Temporal predictor geometry requires predictive auxiliary training")
    predictive_head_kernel = model.get("predictive_head_spatial_kernel_size", 1)
    if (
        type(predictive_head_kernel) is not int
        or predictive_head_kernel <= 0
        or predictive_head_kernel % 2 == 0
    ):
        raise ConfigError("model.predictive_head_spatial_kernel_size must be a positive odd integer")
    predictive_head_hidden = model.get("predictive_head_hidden_channels")
    if predictive_head_hidden is not None and (
        type(predictive_head_hidden) is not int or predictive_head_hidden <= 0
    ):
        raise ConfigError("model.predictive_head_hidden_channels must be positive or null")
    if not model.get("predictive_head", False) and (
        predictive_head_kernel != 1 or predictive_head_hidden is not None
    ):
        raise ConfigError("Predictive-head geometry requires model.predictive_head=true")
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
        "event_mix_probability",
        "event_mix_beta",
        "event_mix_components",
        "event_mix_label_mode",
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
    event_mix_probability = augmentation.get("event_mix_probability", 0.0)
    if (
        type(event_mix_probability) not in (int, float)
        or isinstance(event_mix_probability, bool)
        or not 0.0 <= event_mix_probability <= 1.0
    ):
        raise ConfigError("augmentation.event_mix_probability must be in [0, 1]")
    if event_mix_probability > 0.0:
        beta = augmentation.get("event_mix_beta")
        components = augmentation.get("event_mix_components")
        label_mode = augmentation.get("event_mix_label_mode")
        if type(beta) not in (int, float) or isinstance(beta, bool) or beta <= 0.0:
            raise ConfigError("augmentation.event_mix_beta must be positive")
        if type(components) is not int or components <= 0:
            raise ConfigError("augmentation.event_mix_components must be a positive integer")
        if label_mode != "relative_distance":
            raise ConfigError("augmentation.event_mix_label_mode must be relative_distance")
        if is_multigranular:
            raise ConfigError("EventMix currently requires a single encoded tensor stream")

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
    delay_anneal_epochs = training.get("delay_anneal_epochs")
    if delay_anneal_epochs is not None and (
        not learnable_delays
        or type(delay_anneal_epochs) is not int
        or not 1 <= delay_anneal_epochs <= epochs
    ):
        raise ConfigError("training.delay_anneal_epochs requires learnable delays and must fit epochs")
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
    late_window = training.get("late_summary_window")
    if late_window is not None and (
        type(late_window) is not int or not 1 <= late_window <= epochs
    ):
        raise ConfigError("training.late_summary_window must be an integer within the run")

    if predictive is not None:
        _validate_predictive_section(config, objective_mode)
    elif (
        "runtime" not in config
        and not _is_legacy_resolved_predictive(config)
        and _declares_predictive_modules(model)
    ):
        raise ConfigError(
            "Auxiliary predictors, predictive heads and surprise routing are trained only by a "
            "predictive objective; declare a predictive section with a positive-weight objective"
        )
    if continuation is not None:
        _validate_predictive_continuation(config)


_PREDICTIVE_OBJECTIVE_MODES = {"none", "fine_future", "fine_same", "coarse_future", "late_prefix"}
_AUTHORITY_FIELDS = {"target_ratio", "max_step_factor", "min_weight", "max_weight"}


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(value)


def _is_legacy_resolved_predictive(config: dict[str, Any]) -> bool:
    """First-execution resolved configs kept objective and teacher inside continuation."""

    continuation = config.get("continuation")
    return (
        "runtime" in config
        and "predictive" not in config
        and isinstance(continuation, dict)
        and "objective" in continuation
    )


def _declares_predictive_modules(model: dict[str, Any]) -> bool:
    return bool(
        model.get("temporal_channel_mixer_predictive_auxiliary", False)
        or model.get("predictive_head", False)
        or model.get("temporal_channel_mixer_surprise_routing", False)
    )


def _validate_predictive_continuation(config: dict[str, Any]) -> None:
    continuation = config["continuation"]
    if not isinstance(continuation, dict):
        raise ConfigError("continuation must be a mapping")
    if _is_legacy_resolved_predictive(config):
        return
    if not isinstance(config.get("predictive"), dict):
        raise ConfigError("A continuation belongs to the predictive phase and requires a predictive section")
    for field in ("parent_config", "parent_checkpoint"):
        if not isinstance(continuation.get(field), str) or not continuation[field].strip():
            raise ConfigError(f"continuation.{field} must be a non-empty path")
    for field in (
        "objective",
        "teacher_config",
        "teacher_checkpoint",
        "phase1_audit_report",
        "blocked_reason",
    ):
        if field in continuation:
            raise ConfigError(f"continuation.{field} belongs to the predictive section")
    if continuation.get("freeze_batchnorm_statistics") is not True:
        raise ConfigError("Predictive continuation requires fixed BatchNorm running statistics")
    new_parameter_learning_rate = continuation.get("new_parameter_learning_rate")
    if new_parameter_learning_rate is not None and (
        not _is_number(new_parameter_learning_rate) or new_parameter_learning_rate <= 0.0
    ):
        raise ConfigError("continuation.new_parameter_learning_rate must be positive")


def _validate_predictive_section(config: dict[str, Any], objective_mode: str) -> None:
    predictive = config["predictive"]
    if not isinstance(predictive, dict):
        raise ConfigError("predictive must be a mapping")
    audit_report = predictive.get("phase1_audit_report")
    if audit_report is None and "runtime" not in config:
        raise ConfigError("predictive.phase1_audit_report must name the completed A1-A4 report")
    if audit_report is not None and (not isinstance(audit_report, str) or not audit_report.strip()):
        raise ConfigError("predictive.phase1_audit_report must be a non-empty path")
    blocked_reason = predictive.get("blocked_reason")
    if blocked_reason is not None and (
        not isinstance(blocked_reason, str) or not blocked_reason.strip()
    ):
        raise ConfigError("predictive.blocked_reason must be a non-empty string")
    objective = predictive.get("objective")
    if not isinstance(objective, dict):
        raise ConfigError("predictive.objective must be a mapping")
    if objective_mode not in _PREDICTIVE_OBJECTIVE_MODES:
        raise ConfigError(f"Unsupported predictive objective mode: {objective_mode}")
    weight = objective.get("weight", 0.0)
    if not _is_number(weight) or weight < 0.0:
        raise ConfigError("predictive.objective.weight must be finite and non-negative")
    authority = objective.get("authority")
    if authority is not None:
        if (
            not isinstance(authority, dict)
            or not set(authority) <= _AUTHORITY_FIELDS
            or "target_ratio" not in authority
            or any(not _is_number(value) for value in authority.values())
        ):
            raise ConfigError(
                "predictive.objective.authority needs numeric target_ratio and optional "
                "max_step_factor, min_weight, max_weight"
            )
        low = authority.get("min_weight", 0.0)
        high = authority.get("max_weight", 1.0e6)
        if (
            authority["target_ratio"] <= 0.0
            or authority.get("max_step_factor", 2.0) < 1.0
            or not 0.0 <= low < high
        ):
            raise ConfigError("predictive.objective.authority has invalid bounds")
    minimum_ratio = objective.get("minimum_shared_gradient_ratio")
    if minimum_ratio is not None and (not _is_number(minimum_ratio) or minimum_ratio < 0.0):
        raise ConfigError("predictive.objective.minimum_shared_gradient_ratio must be non-negative")
    auxiliary_declared = objective_mode != "none" or bool(
        config["model"].get("temporal_channel_mixer_predictive_auxiliary", False)
    )
    if auxiliary_declared and float(weight) <= 0.0 and authority is None:
        raise ConfigError(
            "A declared predictive auxiliary objective requires a positive weight or authority"
        )
    if not auxiliary_declared and (float(weight) > 0.0 or authority is not None):
        raise ConfigError("Predictive objective strength is set but no auxiliary is configured")
    ramp = objective.get("ramp_epochs", 0)
    if type(ramp) is not int or ramp < 0 or ramp > int(config["training"]["epochs"]):
        raise ConfigError("predictive.objective.ramp_epochs must fit the training horizon")
    for field, required in (
        ("temporal_region_weights", {"active", "tail"}),
        ("temporal_stage_weights", {"stage1", "stage2"}),
    ):
        values = objective.get(field)
        if values is None:
            continue
        if (
            not isinstance(values, dict)
            or set(values) != required
            or any(not _is_number(value) or value < 0 for value in values.values())
            or sum(values.values()) <= 0
        ):
            raise ConfigError(f"predictive.objective.{field} has invalid weights")
    stage_weights = objective.get("temporal_stage_weights")
    if stage_weights is not None and config["model"].get(
        "temporal_channel_mixer_predictive_auxiliary", False
    ):
        predictor_stages = set(config["model"].get("temporal_channel_mixer_predictive_stages", [1, 2]))
        weighted = {int(stage.removeprefix("stage")) for stage, value in stage_weights.items() if value > 0}
        if not weighted or not weighted <= predictor_stages:
            raise ConfigError(
                "predictive.objective.temporal_stage_weights must weight only stages with a predictor"
            )
    if objective_mode != "none":
        for field in ("teacher_config", "teacher_checkpoint"):
            if not isinstance(predictive.get(field), str) or not predictive[field].strip():
                raise ConfigError(f"predictive.{field} is required by the objective")
    head_required = objective_mode in {"fine_future", "fine_same", "coarse_future"}
    if bool(config["model"].get("predictive_head", False)) != head_required:
        raise ConfigError("model.predictive_head must match the latent predictive objective")
    if head_required:
        if (
            not isinstance(objective.get("normalization_report"), str)
            or not objective["normalization_report"].strip()
        ):
            raise ConfigError("Latent predictive objectives require a normalization_report path")
        if config.get("continuation") is None:
            raise ConfigError(
                "Latent predictive objectives are defined for continuations: their normalization "
                "report is fitted on the continuation parent"
            )
    if (
        objective_mode in {"fine_future", "fine_same"}
        and config["representation"]["name"] != "multigranular_count_frame"
    ):
        raise ConfigError("Fine predictive objectives require multigranular_count_frame input")
    if objective_mode == "late_prefix":
        representation = config["representation"]
        total_steps = int(representation["window_us"]) // int(representation["bin_width_us"])
        steps = objective.get("prefix_steps", [20, 30])
        if (
            not isinstance(steps, list)
            or not steps
            or any(type(step) is not int or not 0 < step <= total_steps for step in steps)
            or sorted(set(steps)) != steps
        ):
            raise ConfigError("predictive.objective.prefix_steps must be increasing steps in the window")
        if objective.get("prefix_readout", "prefix_mean") not in {
            "prefix_mean",
            "fixed_window_denominator",
        }:
            raise ConfigError("predictive.objective.prefix_readout is unsupported")
        temperature = objective.get("temperature", 2.0)
        if not _is_number(temperature) or temperature <= 0.0:
            raise ConfigError("predictive.objective.temperature must be positive")
    if float(config["augmentation"].get("event_mix_probability", 0.0)):
        raise ConfigError("Predictive-phase training does not support EventMix")
