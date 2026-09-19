"""Bounded architectural experiments and checkpoint-only profiling recovery."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from etsr.config import load_config, validate_config
from etsr.runner import profile_checkpoint, train_experiment
from etsr.utils.io import write_json

_AUGMENTATION_FAMILIES = {
    "horizontal_flip": ("horizontal_flip_probability",),
    "temporal_mask": ("temporal_mask_count", "temporal_mask_max_steps"),
    "spatial_erasing": ("spatial_erasing_count", "spatial_erasing_max_pixels"),
    "event_mix": (
        "event_mix_probability",
        "event_mix_beta",
        "event_mix_components",
        "event_mix_label_mode",
    ),
}


def _changed_augmentation_families(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[str, ...]:
    def value(config: dict[str, Any], field: str) -> Any:
        return config.get(field, 0.0 if field == "horizontal_flip_probability" else 0)

    return tuple(
        family
        for family, fields in _AUGMENTATION_FAMILIES.items()
        if any(value(reference, field) != value(candidate, field) for field in fields)
    )


def run_candidate(config: dict[str, Any]) -> dict[str, Any]:
    """A failed overfit never launches a full run; full training starts from fresh weights."""
    validate_config(config)
    dataset_name = config["dataset"]["name"]
    if dataset_name not in {"dvslip", "dvsgesture"} or config["training"].get("overfit"):
        raise ValueError("candidate requires a full DVS-Lip or DVS-Gesture configuration")
    if dataset_name == "dvsgesture":
        reference = load_config("configs/dvsgesture_e0.yaml")
        for section in ("dataset", "representation", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"DVS-Gesture transfer must preserve baseline {section}")
        expected_model = {
            **reference["model"],
            "frontend": "pyramidal",
            "temporal_channel_mixer": True,
            "temporal_channel_mixer_delays": [1, 2, 4, 8],
            "stage1_mixer": "depthwise_conv",
            "stage1_depthwise_kernel_size": 3,
        }
        if config["model"] != expected_model:
            raise ValueError("DVS-Gesture transfer is fixed to the frozen DWC3+TCAP-d8 topology")
        gate_class_count = 11
    else:
        gate_class_count = 16
    reference = load_config("configs/dvslip_e0.yaml")
    representation_name = config["representation"]["name"]
    if dataset_name == "dvsgesture":
        pass
    elif representation_name == "phase_count_frames_e1":
        for section in ("dataset", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"Representation discovery must preserve baseline {section}")
        expected_representation = {**reference["representation"], "name": "phase_count_frames_e1"}
        expected_model = {**reference["model"], "in_channels": 4}
        if config["representation"] != expected_representation or config["model"] != expected_model:
            raise ValueError("E1 discovery may change only phase encoding and input channels")
    elif representation_name in {
        "temporal_binary_frames_tbr",
        "spike_tbr_lif_paper_aligned",
    }:
        reference = load_config("configs/dvslip_f.yaml")
        for section in ("dataset", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"TBR discovery must preserve F {section}")
        expected_model = {**reference["model"], "in_channels": 1}
        if config["model"] != expected_model:
            raise ValueError("TBR discovery may change only F input channels")
        common = {
            "window_us": 2_000_000,
            "bin_width_us": 50_000,
            "micro_bin_width_us": 6_250,
            "bits": 8,
        }
        expected_representation = {"name": representation_name, **common}
        if representation_name == "spike_tbr_lif_paper_aligned":
            expected_representation.update({"lif_beta": 0.9, "lif_threshold": 1.1})
        if config["representation"] != expected_representation:
            raise ValueError("TBR discovery is fixed to the preregistered DVS-Lip paper settings")
    elif config["model"].get("temporal_channel_mixer_learnable_delays") is True:
        reference = load_config("configs/dvslip_f_tcap_stage1_dwc3_d8.yaml")
        for section in ("dataset", "representation", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"Learnable-delay discovery must preserve DWC3+d8 {section}")
        expected_model = {
            **reference["model"],
            "temporal_channel_mixer_learnable_delays": True,
        }
        if config["model"] != expected_model:
            raise ValueError("Learnable-delay discovery may learn only the four existing TCAP taps")
    elif (
        config["model"].get("stage1_mixer") == "depthwise_conv"
        and config["model"].get("temporal_channel_mixer_delays") == [1, 2, 4, 8]
    ):
        reference = load_config("configs/dvslip_f_temporal_capacity.yaml")
        for section in ("dataset", "representation", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"DWC3+d8 discovery must preserve F+TCAP {section}")
        expected_model = {
            **reference["model"],
            "temporal_channel_mixer_delays": [1, 2, 4, 8],
            "stage1_mixer": "depthwise_conv",
            "stage1_depthwise_kernel_size": 3,
        }
        if config["model"] != expected_model:
            raise ValueError("DWC3+d8 discovery may combine only the two registered changes")
    elif (
        representation_name == "count_frames_e0"
        and config["model"].get("frontend") == "pyramidal"
        and config["model"].get("temporal_channel_mixer") is True
        and config["model"].get("temporal_channel_mixer_delays") != [1, 2, 4]
    ):
        reference = load_config("configs/dvslip_f_temporal_capacity.yaml")
        for section in ("dataset", "representation", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"TCAP d8 discovery must preserve F+TCAP {section}")
        expected_model = {
            **reference["model"],
            "temporal_channel_mixer_delays": [1, 2, 4, 8],
        }
        if config["model"] != expected_model:
            raise ValueError("TCAP d8 discovery may add only delay 8")
    elif config["model"].get("stage1_mixer", "token_qk") == "depthwise_conv":
        reference = load_config("configs/dvslip_f_temporal_capacity.yaml")
        for section in ("dataset", "representation", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"High-frequency discovery must preserve F+TCAP {section}")
        expected_model = {
            **reference["model"],
            "stage1_mixer": "depthwise_conv",
            "stage1_depthwise_kernel_size": 3,
        }
        if config["model"] != expected_model:
            raise ValueError("High-frequency discovery is fixed to stage1 depthwise convolution 3x3")
    elif representation_name == "multigranular_count_frame":
        reference = load_config("configs/dvslip_f.yaml")
        for section in ("dataset", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"Multi-granular discovery must preserve F {section}")
        common_representation = {
            "window_us": 2_000_000,
            "bin_width_us": 50_000,
            "micro_bin_width_us": 6_250,
            "count_cap": 255,
            "fine_count_cap": 65_535,
        }
        capacity_model = {
            **reference["model"],
            "multigranular": True,
            "multigranular_fine_channels": 16,
            "multigranular_fine_mid_channels": 32,
            "multigranular_temporal_groups": 1,
            "multigranular_fusion": "concat_residual",
            "multigranular_micro_steps": 8,
        }
        capacity_representation = {
            "name": representation_name,
            **common_representation,
            "fine_spatial_stride": 4,
        }
        allowed_designs = (
            (
                {"name": representation_name, **common_representation, "fine_spatial_stride": 8},
                {
                    **reference["model"],
                    "multigranular": True,
                    "multigranular_fine_channels": 16,
                    "multigranular_temporal_groups": 64,
                    "multigranular_fusion": "add",
                    "multigranular_micro_steps": 8,
                },
            ),
            (capacity_representation, capacity_model),
            (
                capacity_representation,
                {
                    **capacity_model,
                    "temporal_channel_mixer": True,
                    "temporal_channel_mixer_delays": [1, 2, 4],
                },
            ),
        )
        if (config["representation"], config["model"]) not in allowed_designs:
            raise ValueError("Multi-granular discovery is fixed to its preregistered design")
    else:
        for section in ("dataset", "representation", "augmentation", "training", "evaluation"):
            if config.get(section) != reference.get(section):
                raise ValueError(f"Architecture discovery must preserve baseline {section}")
    gate_config = copy.deepcopy(config)
    gate_config["experiment"]["name"] += "_overfit"
    gate_config["training"].update({
        "epochs": 500,
        "amp": False,
        "select_metric": "accuracy",
        "overfit": {
            "class_count": gate_class_count,
            "samples_per_class": 4,
            "stop_on_pass": True,
        },
    })
    gate_config["training"]["recipe_id"] += "_overfit"
    if config["model"].get("temporal_channel_mixer_learnable_delays"):
        # The 500-epoch gate is a ceiling, not a slower annealing recipe than the full run.
        gate_config["training"]["delay_anneal_epochs"] = int(config["training"]["epochs"])
    # Avoid restarting worker processes twice per epoch for just 64 samples.
    gate_config["dataset"]["num_workers"] = 0
    gate_summary = train_experiment(gate_config)
    manifest_path = Path(gate_summary["artifact_dir"]) / "candidate_workflow.json"
    manifest = {"gate": gate_summary, "status": "gate_failed", "official_test_used": False}
    write_json(manifest, manifest_path)
    if not gate_summary["overfit_gate"]["passed"]:
        raise RuntimeError(f"Overfit failed; no full training started. See {manifest_path}")
    manifest["status"] = "training"
    write_json(manifest, manifest_path)
    # train_experiment reseeds and rebuilds the model; gate weights are never reused.
    summary = train_experiment(copy.deepcopy(config))
    manifest.update({"full": summary, "status": "profiling"})
    write_json(manifest, manifest_path)
    artifact_dir = Path(summary["artifact_dir"])
    output = artifact_dir / "hardware_profile_v4.json"
    profile_checkpoint(load_config(artifact_dir / "config_resolved.yaml"), summary["checkpoint"], output)
    manifest.update({"profile": str(output), "status": "complete"})
    write_json(manifest, manifest_path)
    write_json(manifest, artifact_dir / "candidate_workflow.json")
    return manifest


def run_replication(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Repeat one validated configuration from scratch and profile its selected best checkpoint."""

    validate_config(config)
    if config["dataset"]["name"] not in {"dvslip", "dvsgesture"}:
        raise ValueError("replicate supports only DVS-Lip and DVS-Gesture")
    if config["training"].get("overfit"):
        raise ValueError("replicate requires a full training configuration")
    if type(seed) is not int or seed < 0:
        raise ValueError("replication seed must be a non-negative integer")

    replication = copy.deepcopy(config)
    replication["experiment"]["seed"] = seed
    summary = train_experiment(replication)
    artifact_dir = Path(summary["artifact_dir"])
    manifest_path = artifact_dir / "replication_workflow.json"
    manifest = {
        "full": summary,
        "source_experiment": config["experiment"]["name"],
        "seed": seed,
        "status": "profiling",
        "official_test_used": False,
    }
    write_json(manifest, manifest_path)
    output = artifact_dir / "hardware_profile_v4.json"
    profile_checkpoint(load_config(artifact_dir / "config_resolved.yaml"), summary["checkpoint"], output)
    manifest.update({"profile": str(output), "status": "complete"})
    write_json(manifest, manifest_path)
    return manifest


def run_supervised_refinement(config: dict[str, Any]) -> dict[str, Any]:
    """Run a config-declared augmentation screen and profile its selected best."""

    validate_config(config)
    if config["training"].get("overfit"):
        raise ValueError("supervised refinement requires a full training configuration")
    refinement = config.get("refinement")
    if not isinstance(refinement, dict) or set(refinement) != {
        "kind",
        "stage",
        "reference_config",
    }:
        raise ValueError("refinement must declare kind, stage and reference_config")
    if refinement["kind"] != "augmentation":
        raise ValueError("refine currently supports augmentation experiments")
    if refinement["stage"] not in {"single", "combination"}:
        raise ValueError("augmentation refinement stage must be single or combination")
    if not isinstance(refinement["reference_config"], str):
        raise ValueError("refinement.reference_config must be a configuration path")

    reference = load_config(refinement["reference_config"])
    if reference["dataset"]["name"] != config["dataset"]["name"]:
        raise ValueError("augmentation screening cannot change dataset")
    for section in ("dataset", "representation", "evaluation", "model"):
        if config.get(section) != reference.get(section):
            raise ValueError(f"augmentation screening must preserve reference {section}")

    expected_training = {**reference["training"], "recipe_id": config["training"]["recipe_id"]}
    if config["training"] != expected_training or (
        config["training"]["recipe_id"] == reference["training"]["recipe_id"]
    ):
        raise ValueError("augmentation screening may change only the non-empty recipe identifier")
    expected_experiment = {**reference["experiment"], "name": config["experiment"]["name"]}
    if config["experiment"] != expected_experiment or (
        config["experiment"]["name"] == reference["experiment"]["name"]
    ):
        raise ValueError("augmentation screening may change only the experiment name")

    changed_families = _changed_augmentation_families(
        reference["augmentation"], config["augmentation"]
    )
    expected_family_count = 1 if refinement["stage"] == "single" else 2
    if len(changed_families) != expected_family_count:
        requirement = "one family" if refinement["stage"] == "single" else "two families"
        raise ValueError(
            f"{refinement['stage']} augmentation screening requires exactly {requirement}"
        )

    summary = train_experiment(copy.deepcopy(config))
    artifact_dir = Path(summary["artifact_dir"])
    manifest_path = artifact_dir / "refinement_workflow.json"
    manifest = {
        "full": summary,
        "reference_experiment": reference["experiment"]["name"],
        "refinement": refinement,
        "changed_augmentation_families": changed_families,
        "augmentation": config["augmentation"],
        "status": "profiling",
        "official_test_used": False,
    }
    write_json(manifest, manifest_path)
    output = artifact_dir / "hardware_profile_v4.json"
    profile_checkpoint(load_config(artifact_dir / "config_resolved.yaml"), summary["checkpoint"], output)
    manifest.update({"profile": str(output), "status": "complete"})
    write_json(manifest, manifest_path)
    return manifest


def run_predictive_continuation(config: dict[str, Any]) -> dict[str, Any]:
    """Gate, run and profile one preregistered matched continuation."""

    validate_config(config)
    requested = copy.deepcopy(config)
    continuation = requested.get("continuation")
    if not isinstance(continuation, dict):
        raise ValueError("Predictive continuation requires a continuation section.")
    canonical = load_config("configs/dvslip_predictive_continuation_base.yaml")
    for section in ("dataset", "augmentation", "evaluation", "training"):
        if requested[section] != canonical[section]:
            raise ValueError(
                f"Predictive continuation must exactly preserve canonical {section}."
            )
    for field in ("parent_config", "parent_checkpoint", "freeze_batchnorm_statistics"):
        if continuation.get(field) != canonical["continuation"].get(field):
            raise ValueError(f"Predictive continuation must preserve canonical {field}.")

    objective_mode = str(continuation["objective"].get("mode", "none"))
    representation_reference = (
        load_config("configs/dvslip_predictive_fine_future.yaml")["representation"]
        if objective_mode in {"fine_future", "fine_same"}
        else canonical["representation"]
    )
    if requested["representation"] != representation_reference:
        raise ValueError(
            "Predictive continuation must use the preregistered representation for its objective."
        )

    parent = load_config(continuation["parent_config"])
    if parent["experiment"]["name"] != "dvslip_f_tcap_stage1_dwc3_d8":
        raise ValueError("Predictive continuation must start from frozen DWC3+TCAP-d8.")
    if requested["dataset"] != parent["dataset"]:
        raise ValueError("Predictive continuation must preserve the parent dataset protocol.")
    allowed_model_fields = {
        "predictive_head",
        "predictive_head_spatial_kernel_size",
        "predictive_head_hidden_channels",
        "temporal_channel_mixer_dynamic_routing",
        "temporal_channel_mixer_router_pooling",
        "temporal_channel_mixer_router_hidden_divisor",
        "temporal_channel_mixer_predictive_auxiliary",
        "temporal_channel_mixer_predictor_channel_groups",
        "temporal_channel_mixer_predictor_spatial_kernel_size",
        "temporal_channel_mixer_surprise_routing",
    }
    stripped_model = {
        key: value for key, value in requested["model"].items() if key not in allowed_model_fields
    }
    if stripped_model != parent["model"]:
        raise ValueError("Predictive continuation may change only registered training/routing fields.")

    from etsr.evaluation.predictive_diagnostic import run_predictive_preflight

    preflight_output = (
        Path("artifacts")
        / "predictive_preflight"
        / f"{requested['experiment']['name']}__seed{requested['experiment']['seed']}.json"
    )
    preflight = run_predictive_preflight(requested, preflight_output)
    if not preflight["passed"]:
        raise RuntimeError(f"Predictive preflight failed; see {preflight_output}.")

    gate_config = copy.deepcopy(requested)
    gate_config["experiment"]["name"] += "_overfit"
    gate_config["training"].update(
        {
            "epochs": 50,
            "amp": False,
            "select_metric": "accuracy",
            "overfit": {
                "class_count": 16,
                "samples_per_class": 4,
                "stop_on_pass": True,
            },
        }
    )
    gate = train_experiment(gate_config)
    if not gate.get("overfit_gate", {}).get("passed", False):
        raise RuntimeError("Predictive bounded overfit failed; no continuation training was started.")

    full = train_experiment(copy.deepcopy(requested))
    profile_output = Path(full["artifact_dir"]) / "hardware_profile_v4.json"
    profile_checkpoint(
        load_config(Path(full["artifact_dir"]) / "deployment_config_resolved.yaml"),
        full["deployment_checkpoint"],
        profile_output,
    )
    result = {
        "status": "complete",
        "preflight": str(preflight_output.resolve()),
        "gate_artifact_dir": gate["artifact_dir"],
        "artifact_dir": full["artifact_dir"],
        "checkpoint": full["checkpoint"],
        "deployment_checkpoint": full["deployment_checkpoint"],
        "profile": str(profile_output.resolve()),
        "objective": requested["continuation"]["objective"],
        "official_test_used": False,
    }
    write_json(result, Path(full["artifact_dir"]) / "predictive_workflow.json")
    return result


def profile_completed_runs(
    artifact_root: str | Path = "artifacts",
    checkpoint_root: str | Path = "checkpoints",
    max_samples: int = 64,
) -> dict[str, Any]:
    """Reprofile all completed full runs under one schema without altering legacy profiles."""
    root = Path(artifact_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    results = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run_dir = summary_path.parent
        record = {"run_id": run_dir.name}
        try:
            config = load_config(run_dir / "config_resolved.yaml")
            if config["training"].get("overfit"):
                continue
            summary = json.loads(summary_path.read_text())
            checkpoint = Path(checkpoint_root) / run_dir.name / "best.pt"
            if not checkpoint.is_file():
                recorded = Path(summary.get("checkpoint", ""))
                if recorded.is_file():
                    checkpoint = recorded
                else:
                    record.update({"status": "missing_checkpoint", "checkpoint": str(checkpoint)})
                    results.append(record)
                    continue
            output = run_dir / "hardware_profile_v4.json"
            profile = profile_checkpoint(config, checkpoint, output, max_samples=max_samples)
            record.update({
                "status": "complete", "output": str(output),
                "checkpoint_sha256": profile["checkpoint_sha256"],
                "sampling": profile["sampling"],
            })
        except Exception as error:
            record.update({"status": "failed", "error": str(error)})
        results.append(record)
    report = {
        "runs": results,
        "complete": bool(results) and all(row["status"] == "complete" for row in results),
        "official_test_used": False,
    }
    write_json(report, root / "profile_backfill.json")
    return report
