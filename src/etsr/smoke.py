from __future__ import annotations

from pathlib import Path
from typing import Any

from etsr.runner import run_temporal_audit, train_experiment
from etsr.utils.io import write_json


class SmokeConfigError(ValueError):
    """Raised when a smoke config could launch more than a bounded synthetic run."""


def _positive_int(mapping: dict[str, Any], field: str) -> int:
    try:
        value = int(mapping[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SmokeConfigError(f"Smoke config requires an integer {field}.") from exc
    if value <= 0:
        raise SmokeConfigError(f"Smoke config requires {field} > 0.")
    return value


def validate_smoke_config(config: dict[str, Any]) -> None:
    """Reject configs that could turn repository validation into a real experiment.

    The smoke path deliberately exercises training, checkpoint restore, holdout evaluation,
    profiling and the behavioral audit. These hard limits keep it dataset-independent and bounded.
    """

    experiment = config.get("experiment", {})
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    training = config.get("training", {})
    profiling = config.get("profiling", {})
    audit = config.get("audit", {})

    if dataset.get("name") != "synthetic_temporal_order":
        raise SmokeConfigError("Smoke runs require dataset.name=synthetic_temporal_order.")
    if not bool(experiment.get("deterministic", False)):
        raise SmokeConfigError("Smoke runs must enable deterministic execution.")
    if _positive_int(training, "epochs") != 1:
        raise SmokeConfigError("Smoke runs must use exactly one epoch.")
    if bool(training.get("amp", False)):
        raise SmokeConfigError("Smoke runs must disable AMP for CPU portability.")
    if not bool(training.get("evaluate_holdout", False)):
        raise SmokeConfigError("Smoke runs must exercise holdout evaluation.")

    limits = {
        "frames_number": 8,
        "image_size": 32,
        "num_classes": 4,
        "train_samples": 64,
        "validation_samples": 32,
        "test_samples": 32,
        "batch_size": 8,
    }
    for field, maximum in limits.items():
        value = _positive_int(dataset, field)
        if value > maximum:
            raise SmokeConfigError(f"Smoke dataset.{field} must be <= {maximum}, got {value}.")
    if _positive_int(dataset, "num_classes") != 4:
        raise SmokeConfigError("The synthetic smoke task requires exactly four classes.")
    try:
        num_workers = int(dataset.get("num_workers", -1))
    except (TypeError, ValueError) as exc:
        raise SmokeConfigError("Smoke dataset.num_workers must be an integer.") from exc
    if num_workers != 0:
        raise SmokeConfigError("Smoke runs require dataset.num_workers=0.")

    if model.get("name") != "mini_qkformer":
        raise SmokeConfigError("The current smoke path validates model.name=mini_qkformer.")
    if _positive_int(model, "in_channels") != 2:
        raise SmokeConfigError("The synthetic smoke model requires exactly two input channels.")
    embed_dim = _positive_int(model, "embed_dim")
    num_heads = _positive_int(model, "num_heads")
    if embed_dim > 32:
        raise SmokeConfigError("Smoke model.embed_dim must be <= 32.")
    if num_heads > 4 or embed_dim % num_heads != 0:
        raise SmokeConfigError("Smoke model.num_heads must be <= 4 and divide embed_dim.")
    try:
        mlp_ratio = float(model.get("mlp_ratio", 0.0))
    except (TypeError, ValueError) as exc:
        raise SmokeConfigError("Smoke model.mlp_ratio must be numeric.") from exc
    if not 0.0 < mlp_ratio <= 2.0:
        raise SmokeConfigError("Smoke model.mlp_ratio must be in (0, 2].")
    if not bool(profiling.get("enabled", False)):
        raise SmokeConfigError("Smoke runs must exercise profiling.")
    if _positive_int(profiling, "max_batches") != 1:
        raise SmokeConfigError("Smoke profiling.max_batches must equal one.")

    perturbations = audit.get("perturbations", [])
    names = [item.get("name") for item in perturbations if isinstance(item, dict)]
    allowed_perturbations = {"original", "reverse_time", "shuffle_time", "reverse_segments"}
    if "original" not in names or len(names) < 2:
        raise SmokeConfigError("Smoke audit requires original and at least one perturbation.")
    if len(names) > 4 or set(names) - allowed_perturbations:
        raise SmokeConfigError("Smoke audit contains too many or unsupported perturbations.")
    try:
        fractions = [float(value) for value in audit.get("prefix_fractions", [])]
    except (TypeError, ValueError) as exc:
        raise SmokeConfigError("Smoke prefix fractions must be numeric.") from exc
    if not 2 <= len(fractions) <= 4 or not all(0.0 < value <= 1.0 for value in fractions):
        raise SmokeConfigError("Smoke audit requires between two and four valid prefix fractions.")


def run_smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    """Run the bounded end-to-end integration workflow and emit one compact result."""

    validate_smoke_config(config)
    training_summary = train_experiment(config)
    audit_summary = run_temporal_audit(config, training_summary["checkpoint"])
    artifact_dir = Path(training_summary["artifact_dir"])
    result = {
        "schema_version": 1,
        "status": "passed",
        "run_id": training_summary["run_id"],
        "config": config.get("_source_path"),
        "artifact_dir": str(artifact_dir),
        "checkpoint": training_summary["checkpoint"],
        "official_test_used": False,
        "training_validation": training_summary["validation"],
        "training_holdout": training_summary.get("test"),
        "audit_summary": str(artifact_dir / "audit" / "audit_summary.json"),
        "audit_conditions": sorted(audit_summary["conditions"]),
    }
    write_json(result, artifact_dir / "smoke_summary.json")
    return result
