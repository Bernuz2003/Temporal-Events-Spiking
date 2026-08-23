#!/usr/bin/env python3
"""Compare two backward surrogates on one checkpoint without updating its weights."""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
from typing import Any

import torch

from etsr.config import load_config
from etsr.dvslip.encoded import EncodedDvsLipDataset, build_dvslip_bundle
from etsr.models.factory import build_model
from etsr.reproducibility import seed_everything
from etsr.training.engine import make_criterion
from etsr.utils.io import write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the checkpoint surrogate with a candidate config on the same balanced "
            "validation batch. No optimizer step is performed."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-count", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        checkpoint = torch.load(path, map_location=device)
    for field in ("model", "config", "num_classes", "epoch", "score"):
        if field not in checkpoint:
            raise ValueError(f"Checkpoint is missing required field: {field}")
    return checkpoint


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def _balanced_batch(
    dataset: EncodedDvsLipDataset,
    class_count: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not 1 <= class_count <= len(dataset.classes):
        raise ValueError(f"class-count must be in [1, {len(dataset.classes)}].")

    selected: dict[int, int] = {}
    raw = dataset.raw_dataset
    for index, sample_id in enumerate(raw.sample_ids):
        target = raw.class_to_idx[Path(sample_id).parts[0]]
        if target < class_count and target not in selected:
            selected[target] = index
        if len(selected) == class_count:
            break
    if len(selected) != class_count:
        raise RuntimeError(
            f"Could select only {len(selected)} of {class_count} validation classes."
        )

    indices = [selected[target] for target in range(class_count)]
    samples = [dataset[index] for index in indices]
    frames = torch.stack([sample[0] for sample in samples])
    targets = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    return frames, targets, indices


def _gradient_summary(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", maxsplit=1)[0]
        entry = totals.setdefault(
            group,
            {"absolute_sum": 0.0, "squared_l2": 0.0, "nonzero": 0.0, "elements": 0.0},
        )
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        gradient_l2 = float(torch.linalg.vector_norm(gradient).item())
        entry["absolute_sum"] += float(torch.linalg.vector_norm(gradient, ord=1).item())
        entry["squared_l2"] += gradient_l2**2
        entry["nonzero"] += float(torch.count_nonzero(gradient).item())
        entry["elements"] += float(gradient.numel())

    return {
        group: {
            "gradient_l2": values["squared_l2"] ** 0.5,
            "gradient_mean_abs": values["absolute_sum"] / max(1.0, values["elements"]),
            "gradient_nonzero_fraction": values["nonzero"] / max(1.0, values["elements"]),
        }
        for group, values in totals.items()
    }


def _run_backward(
    model_config: dict[str, Any],
    checkpoint_state: dict[str, torch.Tensor],
    num_classes: int,
    frames: torch.Tensor,
    targets: torch.Tensor,
    training_config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model = build_model(model_config, num_classes).to(device)
    model.load_state_dict(checkpoint_state)
    model.eval()
    model.zero_grad(set_to_none=True)

    encoded, _ = model._encode(frames)
    pooled = encoded.mean(dim=(0, 3, 4))
    logits = model.head(pooled)
    loss = make_criterion(training_config)(logits, targets)
    loss.backward()

    result = {
        "surrogate_name": str(model_config.get("surrogate_name", "fast_sigmoid")),
        "surrogate_alpha": float(model_config.get("surrogate_alpha", 25.0)),
        "loss": float(loss.detach().item()),
        "parameter_gradients": _gradient_summary(model),
    }
    return result, logits.detach().cpu(), pooled.detach().cpu()


def _model_differences(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    reference = dict(reference)
    candidate = dict(candidate)
    reference.pop("num_classes", None)
    candidate.pop("num_classes", None)
    return {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in sorted(set(reference) | set(candidate))
        if reference.get(key) != candidate.get(key)
    }


def _gradient_ratios(
    reference: dict[str, dict[str, float]], candidate: dict[str, dict[str, float]]
) -> dict[str, float | None]:
    ratios = {}
    for name in sorted(set(reference) & set(candidate)):
        denominator = float(reference[name]["gradient_mean_abs"])
        numerator = float(candidate[name]["gradient_mean_abs"])
        ratios[name] = None if denominator == 0.0 else numerator / denominator
    return ratios


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint)
    candidate_path = Path(args.candidate_config)
    output_path = Path(args.output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _resolve_device(args.device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    reference_config = copy.deepcopy(checkpoint["config"])
    candidate_config = load_config(candidate_path)
    seed_everything(
        int(reference_config["experiment"]["seed"]),
        bool(reference_config["experiment"].get("deterministic", True)),
    )

    differences = _model_differences(
        reference_config["model"], candidate_config["model"]
    )
    unexpected = set(differences) - {"surrogate_name", "surrogate_alpha"}
    if unexpected:
        raise ValueError(
            "Candidate model differs outside the surrogate fields: "
            f"{sorted(unexpected)}"
        )

    bundle = build_dvslip_bundle(
        reference_config["dataset"],
        reference_config["representation"],
        reference_config["augmentation"],
    )
    frames_cpu, targets_cpu, indices = _balanced_batch(
        bundle.validation, args.class_count
    )
    frames = frames_cpu.to(device)
    targets = targets_cpu.to(device)

    reference, reference_logits, reference_pooled = _run_backward(
        reference_config["model"],
        checkpoint["model"],
        int(checkpoint["num_classes"]),
        frames,
        targets,
        reference_config["training"],
        device,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    candidate, candidate_logits, candidate_pooled = _run_backward(
        candidate_config["model"],
        checkpoint["model"],
        int(checkpoint["num_classes"]),
        frames,
        targets,
        reference_config["training"],
        device,
    )

    report = {
        "schema_version": 1,
        "diagnostic_mode": "eval_fp32_balanced_batch_surrogate_comparison",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "candidate_config": str(candidate_path),
        "device": str(device),
        "official_test_used": False,
        "batch": {
            "samples": int(targets.numel()),
            "distinct_targets": int(targets.unique().numel()),
            "targets": targets_cpu.tolist(),
            "validation_indices": indices,
        },
        "model_config_differences": differences,
        "forward_equivalence": {
            "logits_exactly_equal": bool(torch.equal(candidate_logits, reference_logits)),
            "logits_max_abs_difference": float(
                (candidate_logits - reference_logits).abs().max().item()
            ),
            "pooled_exactly_equal": bool(torch.equal(candidate_pooled, reference_pooled)),
            "pooled_max_abs_difference": float(
                (candidate_pooled - reference_pooled).abs().max().item()
            ),
        },
        "reference": reference,
        "candidate": candidate,
        "candidate_over_reference_gradient_mean_abs": _gradient_ratios(
            reference["parameter_gradients"], candidate["parameter_gradients"]
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, output_path)
    print(
        {
            "output": str(output_path.resolve()),
            "checkpoint_epoch": report["checkpoint_epoch"],
            "forward_equivalent": report["forward_equivalence"][
                "logits_exactly_equal"
            ],
            "gradient_ratios": report[
                "candidate_over_reference_gradient_mean_abs"
            ],
            "official_test_used": False,
        }
    )


if __name__ == "__main__":
    main()
