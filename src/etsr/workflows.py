"""Bounded architectural experiments and checkpoint-only profiling recovery."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from etsr.config import load_config, validate_config
from etsr.runner import profile_checkpoint, train_experiment
from etsr.utils.io import write_json


def run_candidate(config: dict[str, Any]) -> dict[str, Any]:
    """A failed overfit never launches a full run; full training starts from fresh weights."""
    validate_config(config)
    if config["dataset"]["name"] != "dvslip" or config["training"].get("overfit"):
        raise ValueError("candidate requires a full DVS-Lip configuration")
    reference = load_config("configs/dvslip_e0.yaml")
    representation_name = config["representation"]["name"]
    if representation_name == "phase_count_frames_e1":
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
            (
                {"name": representation_name, **common_representation, "fine_spatial_stride": 4},
                {
                    **reference["model"],
                    "multigranular": True,
                    "multigranular_fine_channels": 16,
                    "multigranular_fine_mid_channels": 32,
                    "multigranular_temporal_groups": 1,
                    "multigranular_fusion": "concat_residual",
                    "multigranular_micro_steps": 8,
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
        "overfit": {"class_count": 16, "samples_per_class": 4, "stop_on_pass": True},
    })
    gate_config["training"]["recipe_id"] += "_overfit"
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
