"""Checkpoint-only temporal diagnostics for fixed-window mean-readout models."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from etsr.config import save_config
from etsr.data.common import build_loader
from etsr.evaluation.metrics import (
    ClassificationAccumulator,
    interval_normalized_auc,
    trapezoidal_auc,
)
from etsr.models.spiking import MultiStepLIF
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.reproducibility import collect_environment, git_commit, git_is_dirty
from etsr.training.engine import make_criterion
from etsr.utils.io import ensure_dir, sha256_file, write_csv, write_json

READOUT_SCHEMES = ("prefix_mean", "fixed_horizon_denominator")


def deterministic_prefix_sum(values: torch.Tensor) -> torch.Tensor:
    """Build causal prefix sums without CUDA's nondeterministic ``cumsum`` kernel.

    The diagnostic has only 40 steps. A fixed-order sequence of elementwise additions preserves
    strict deterministic mode at negligible cost relative to the model forward.
    """

    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("values must have a non-empty time dimension")
    running = torch.zeros_like(values[0])
    prefixes = []
    for current in values.unbind(0):
        running = running + current
        prefixes.append(running)
    return torch.stack(prefixes, dim=0)


def temporal_readout_logits(
    encoded: torch.Tensor, head: nn.Module
) -> dict[str, torch.Tensor]:
    """Return logits at every causal prefix under two denominator conventions.

    ``prefix_mean`` matches the existing shortened-sequence evaluation. The second convention
    keeps the trained full-window denominator at every prefix, isolating temporal averaging scale
    from the contribution of post-prefix network activity.
    """

    if encoded.ndim != 5 or encoded.shape[0] == 0:
        raise ValueError("encoded must be a non-empty [T, B, C, H, W] tensor")
    spatial = encoded.mean(dim=(3, 4))
    cumulative = deterministic_prefix_sum(spatial)
    prefix_denominators = torch.arange(
        1, encoded.shape[0] + 1, device=encoded.device, dtype=encoded.dtype
    ).view(-1, 1, 1)
    return {
        "prefix_mean": head(cumulative / prefix_denominators),
        "fixed_horizon_denominator": head(cumulative / encoded.shape[0]),
    }


def gather_temporal_values(values: torch.Tensor, one_based_steps: torch.Tensor) -> torch.Tensor:
    """Select one time row per batch element from a time-major tensor."""

    if values.ndim < 2:
        raise ValueError("values must have shape [T, B, ...]")
    if one_based_steps.ndim != 1 or one_based_steps.shape[0] != values.shape[1]:
        raise ValueError("one_based_steps must contain one entry per batch sample")
    if bool(((one_based_steps < 1) | (one_based_steps > values.shape[0])).any().item()):
        raise ValueError("one_based_steps must fit the time axis")
    batch = torch.arange(values.shape[1], device=values.device)
    return values[one_based_steps - 1, batch]


class _TemporalMetricAccumulator:
    def __init__(self, steps: int, num_classes: int) -> None:
        self.metrics = [ClassificationAccumulator(num_classes) for _ in range(steps)]
        self.scalar_sums = {
            name: np.zeros(steps, dtype=np.float64)
            for name in (
                "top1_confidence",
                "predictive_entropy",
                "top1_margin",
                "true_class_margin",
                "agreement_with_final",
                "stable_through_final",
                "prediction_changed",
            )
        }
        self.samples = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        indices: torch.Tensor,
        criterion: nn.Module,
    ) -> None:
        if logits.ndim != 3:
            raise ValueError("logits must have shape [T, B, classes]")
        probabilities = logits.softmax(dim=-1)
        top_values, top_indices = logits.topk(k=2, dim=-1)
        predictions = top_indices[..., 0]
        final_predictions = predictions[-1]
        agrees = predictions.eq(final_predictions.unsqueeze(0))
        stable = torch.flip(
            torch.cumprod(torch.flip(agrees.to(torch.int64), dims=(0,)), dim=0),
            dims=(0,),
        ).to(torch.bool)
        target_logits = logits.gather(
            2, targets.view(1, -1, 1).expand(logits.shape[0], -1, -1)
        ).squeeze(2)
        competitors = logits.clone()
        competitors.scatter_(
            2,
            targets.view(1, -1, 1).expand(logits.shape[0], -1, -1),
            float("-inf"),
        )
        true_margin = target_logits - competitors.max(dim=2).values
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=2)
        changed = torch.zeros_like(predictions, dtype=torch.bool)
        changed[1:] = predictions[1:].ne(predictions[:-1])

        batch_size = int(targets.numel())
        for step in range(logits.shape[0]):
            loss = criterion(logits[step], targets)
            self.metrics[step].update(logits[step], targets, loss, indices)
        tensors = {
            "top1_confidence": probabilities.max(dim=2).values,
            "predictive_entropy": entropy,
            "top1_margin": top_values[..., 0] - top_values[..., 1],
            "true_class_margin": true_margin,
            "agreement_with_final": agrees,
            "stable_through_final": stable,
            "prediction_changed": changed,
        }
        for name, values in tensors.items():
            self.scalar_sums[name] += values.detach().double().sum(dim=1).cpu().numpy()
        self.samples += batch_size

    def rows(self, bin_width_us: int) -> list[dict[str, float | int]]:
        rows = []
        for index, accumulator in enumerate(self.metrics):
            result = accumulator.compute()
            row: dict[str, float | int] = {
                "time_step": index + 1,
                "time_us": (index + 1) * bin_width_us,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "loss": result.loss,
                "samples": result.samples,
            }
            row.update(
                {name: float(values[index] / max(1, self.samples)) for name, values in self.scalar_sums.items()}
            )
            rows.append(row)
        return rows


class _EventAlignedMetricAccumulator:
    def __init__(self, offsets: list[int], num_classes: int) -> None:
        self.offsets = offsets
        self.metrics = [ClassificationAccumulator(num_classes) for _ in offsets]
        self.confidence_sum = np.zeros(len(offsets), dtype=np.float64)
        self.true_margin_sum = np.zeros(len(offsets), dtype=np.float64)
        self.agreement_sum = np.zeros(len(offsets), dtype=np.float64)
        self.clipped = np.zeros(len(offsets), dtype=np.int64)
        self.effective_offset_sum = np.zeros(len(offsets), dtype=np.int64)
        self.samples = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        indices: torch.Tensor,
        last_event_steps: torch.Tensor,
        criterion: nn.Module,
    ) -> None:
        final_predictions = logits[-1].argmax(dim=1)
        for position, offset in enumerate(self.offsets):
            requested = last_event_steps + offset
            selected_steps = requested.clamp_max(logits.shape[0])
            selected = gather_temporal_values(logits, selected_steps)
            loss = criterion(selected, targets)
            self.metrics[position].update(selected, targets, loss, indices)
            probabilities = selected.softmax(dim=1)
            target_logits = selected.gather(1, targets[:, None]).squeeze(1)
            competitors = selected.clone()
            competitors.scatter_(1, targets[:, None], float("-inf"))
            self.confidence_sum[position] += float(probabilities.max(dim=1).values.sum().item())
            self.true_margin_sum[position] += float(
                (target_logits - competitors.max(dim=1).values).sum().item()
            )
            self.agreement_sum[position] += float(
                selected.argmax(dim=1).eq(final_predictions).sum().item()
            )
            self.clipped[position] += int((requested > logits.shape[0]).sum().item())
            self.effective_offset_sum[position] += int(
                (selected_steps - last_event_steps).sum().item()
            )
        self.samples += int(targets.numel())

    def rows(self, bin_width_us: int) -> list[dict[str, float | int]]:
        rows = []
        for position, (offset, accumulator) in enumerate(zip(self.offsets, self.metrics, strict=True)):
            result = accumulator.compute()
            rows.append(
                {
                    "requested_offset_steps": offset,
                    "requested_offset_us": offset * bin_width_us,
                    "mean_effective_offset_steps": float(
                        self.effective_offset_sum[position] / max(1, self.samples)
                    ),
                    "clipped_fraction": float(self.clipped[position] / max(1, self.samples)),
                    "accuracy": result.accuracy,
                    "macro_f1": result.macro_f1,
                    "loss": result.loss,
                    "top1_confidence": float(
                        self.confidence_sum[position] / max(1, self.samples)
                    ),
                    "true_class_margin": float(
                        self.true_margin_sum[position] / max(1, self.samples)
                    ),
                    "agreement_with_full_window": float(
                        self.agreement_sum[position] / max(1, self.samples)
                    ),
                    "samples": result.samples,
                }
            )
        return rows


def _auc_summary(rows: list[dict[str, float | int]]) -> dict[str, float]:
    points = [float(row["time_us"]) for row in rows]
    return {
        "accuracy_auc_us": trapezoidal_auc(points, [float(row["accuracy"]) for row in rows]),
        "accuracy_auc_normalized": interval_normalized_auc(
            points, [float(row["accuracy"]) for row in rows]
        ),
        "macro_f1_auc_us": trapezoidal_auc(points, [float(row["macro_f1"]) for row in rows]),
        "macro_f1_auc_normalized": interval_normalized_auc(
            points, [float(row["macro_f1"]) for row in rows]
        ),
    }


@torch.no_grad()
def diagnose_checkpoint_temporal_dynamics(
    config: dict[str, Any], checkpoint_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Evaluate every causal prefix and event-aligned tail without changing model weights."""

    # Imported lazily to avoid a runner/evaluation import cycle.
    from etsr.runner import _load_checkpoint_context

    config = copy.deepcopy(config)
    checkpoint_path, checkpoint, bundle, model, device, dataset_metadata = (
        _load_checkpoint_context(config, checkpoint_path)
    )
    if getattr(model, "readout_name", None) != "mean" or getattr(
        model, "readout_time", None
    ) != "fixed_window":
        raise ValueError("Temporal diagnostics require a fixed-window mean-readout checkpoint")

    bin_width_us = int(config["representation"]["bin_width_us"])
    total_steps = int(config["representation"]["window_us"]) // bin_width_us
    offsets = list(range(total_steps))
    curve_accumulators = {
        scheme: _TemporalMetricAccumulator(total_steps, int(checkpoint["num_classes"]))
        for scheme in READOUT_SCHEMES
    }
    aligned_accumulators = {
        scheme: _EventAlignedMetricAccumulator(offsets, int(checkpoint["num_classes"]))
        for scheme in READOUT_SCHEMES
    }
    absolute_activity_sums: dict[str, np.ndarray] = {}
    aligned_activity_sums: dict[str, np.ndarray] = {}
    samples = 0
    observed_last_steps: list[int] = []
    criterion = make_criterion(config["training"])
    loader = build_loader(bundle.validation, config["dataset"], shuffle=False)

    lif_activity: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, MultiStepLIF):
            continue

        def capture(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor, *, layer: str = name) -> None:
            lif_activity[layer] = output.detach().float().flatten(2).mean(dim=2).cpu()

        handles.append(module.register_forward_hook(capture))

    try:
        model.eval()
        for frames, targets, indices in loader:
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            last_steps = model._last_event_steps(frames)
            lif_activity.clear()
            encoded = model._encode(frames)
            if encoded.shape[0] != total_steps:
                raise ValueError("Encoded time dimension differs from the representation contract")
            logits_by_scheme = temporal_readout_logits(encoded, model.head)
            for scheme, logits in logits_by_scheme.items():
                curve_accumulators[scheme].update(logits, targets, indices, criterion)
                aligned_accumulators[scheme].update(
                    logits, targets, indices, last_steps, criterion
                )

            input_activity = frames.detach().ne(0).float().flatten(2).mean(dim=2).transpose(0, 1).cpu()
            encoded_magnitude = encoded.detach().float().abs().flatten(2).mean(dim=2).cpu()
            activities = {
                "input.nonzero_fraction": input_activity,
                **lif_activity,
                "stage2.encoded_abs_mean": encoded_magnitude,
            }
            batch_size = int(targets.numel())
            for name, values in activities.items():
                if values.shape != (total_steps, batch_size):
                    raise ValueError(f"Unexpected temporal activity shape for {name}: {values.shape}")
                absolute_activity_sums.setdefault(
                    name, np.zeros(total_steps, dtype=np.float64)
                )
                aligned_activity_sums.setdefault(
                    name, np.zeros(len(offsets), dtype=np.float64)
                )
                absolute_activity_sums[name] += values.double().sum(dim=1).numpy()
                for position, offset in enumerate(offsets):
                    selected = (last_steps + offset).clamp_max(total_steps).cpu()
                    aligned_activity_sums[name][position] += float(
                        gather_temporal_values(values, selected).double().sum().item()
                    )
            samples += batch_size
            observed_last_steps.extend(last_steps.cpu().tolist())
    finally:
        for handle in handles:
            handle.remove()

    output = ensure_dir(output_dir)
    curve_rows_by_scheme = {
        scheme: accumulator.rows(bin_width_us)
        for scheme, accumulator in curve_accumulators.items()
    }
    absolute_rows = [
        {"readout_scheme": scheme, **row}
        for scheme in READOUT_SCHEMES
        for row in curve_rows_by_scheme[scheme]
    ]
    aligned_rows = [
        {"readout_scheme": scheme, **row}
        for scheme in READOUT_SCHEMES
        for row in aligned_accumulators[scheme].rows(bin_width_us)
    ]
    activity_rows = [
        {
            "layer": name,
            "time_step": step + 1,
            "time_us": (step + 1) * bin_width_us,
            "mean_activity": float(values[step] / max(1, samples)),
        }
        for name, values in absolute_activity_sums.items()
        for step in range(total_steps)
    ]
    aligned_activity_rows = [
        {
            "layer": name,
            "requested_offset_steps": offset,
            "requested_offset_us": offset * bin_width_us,
            "mean_activity": float(values[position] / max(1, samples)),
        }
        for name, values in aligned_activity_sums.items()
        for position, offset in enumerate(offsets)
    ]
    files = {
        "absolute_curve": "temporal_curve_every_bin.csv",
        "event_aligned_curve": "temporal_curve_event_aligned.csv",
        "absolute_activity": "temporal_activity_every_bin.csv",
        "event_aligned_activity": "temporal_activity_event_aligned.csv",
    }
    write_csv(absolute_rows, output / files["absolute_curve"])
    write_csv(aligned_rows, output / files["event_aligned_curve"])
    write_csv(activity_rows, output / files["absolute_activity"])
    write_csv(aligned_activity_rows, output / files["event_aligned_activity"])
    save_config(config, output / "config_evaluated.yaml")
    environment_path = output / "environment.json"
    write_json(collect_environment(device), environment_path)

    last_steps_array = np.asarray(observed_last_steps, dtype=np.int64)
    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_score": float(checkpoint["score"]),
        "checkpoint_git_commit": checkpoint["config"].get("runtime", {}).get("git_commit"),
        "diagnostic_git_commit": git_commit(),
        "diagnostic_git_dirty": git_is_dirty(),
        "device": str(device),
        "samples": samples,
        "time_steps": total_steps,
        "bin_width_us": bin_width_us,
        "readout_schemes": list(READOUT_SCHEMES),
        "prefix_auc": {
            scheme: _auc_summary(rows) for scheme, rows in curve_rows_by_scheme.items()
        },
        "last_event_steps": {
            "minimum": int(last_steps_array.min()),
            "mean": float(last_steps_array.mean()),
            "median": float(np.median(last_steps_array)),
            "p75": float(np.percentile(last_steps_array, 75)),
            "p95": float(np.percentile(last_steps_array, 95)),
            "p99": float(np.percentile(last_steps_array, 99)),
            "maximum": int(last_steps_array.max()),
        },
        "interpretation": {
            "training_distribution": "checkpoint trained only with the full fixed window",
            "prefix_mean": "causal shortened-sequence readout; out of the training horizon",
            "fixed_horizon_denominator": (
                "same causal numerator as the prefix, divided by the trained full-window length"
            ),
            "event_alignment": (
                "oracle use of each sample's final occupied bin for diagnosis only; offsets are "
                "clamped at the fixed horizon"
            ),
            "activity": (
                "mean nonzero spike output per LIF, input nonzero fraction, and final encoded "
                "feature magnitude; membrane trajectories are not inferred"
            ),
        },
        "files": files,
        "environment": str(environment_path.resolve()),
        "environment_sha256": sha256_file(environment_path),
        "official_test_used": False,
        **dataset_metadata,
    }
    write_json(summary, output / "temporal_diagnostic_summary.json")
    return summary


def diagnose_baseline_plif_pair(
    baseline_config: dict[str, Any],
    baseline_checkpoint: str | Path,
    plif_config: dict[str, Any],
    plif_checkpoint: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the preregistered B/PLIF diagnostics sequentially and record AUC deltas."""

    output = ensure_dir(output_dir)
    baseline = diagnose_checkpoint_temporal_dynamics(
        baseline_config, baseline_checkpoint, output / "baseline"
    )
    plif = diagnose_checkpoint_temporal_dynamics(plif_config, plif_checkpoint, output / "plif")
    for field in ("dataset_index_sha256", "split_manifest_sha256", "samples", "time_steps"):
        if baseline.get(field) != plif.get(field):
            raise ValueError(f"Baseline and PLIF diagnostics differ in {field}")
    deltas = {
        scheme: {
            metric: float(plif["prefix_auc"][scheme][metric])
            - float(baseline["prefix_auc"][scheme][metric])
            for metric in baseline["prefix_auc"][scheme]
        }
        for scheme in READOUT_SCHEMES
    }
    summary = {
        "schema_version": 1,
        "baseline": {
            "output": str((output / "baseline").resolve()),
            "checkpoint_sha256": baseline["checkpoint_sha256"],
            "prefix_auc": baseline["prefix_auc"],
        },
        "plif": {
            "output": str((output / "plif").resolve()),
            "checkpoint_sha256": plif["checkpoint_sha256"],
            "prefix_auc": plif["prefix_auc"],
        },
        "plif_minus_baseline_prefix_auc": deltas,
        "samples": baseline["samples"],
        "time_steps": baseline["time_steps"],
        "official_test_used": False,
    }
    write_json(summary, output / "temporal_diagnostic_pair_summary.json")
    return summary


def _tcap_modules(model: nn.Module) -> list[tuple[str, CausalTemporalChannelMixer]]:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, CausalTemporalChannelMixer)
    ]
    if not modules:
        raise ValueError("TCAP tap diagnostics require at least one temporal channel mixer")
    delays = modules[0][1].delays
    if any(module.delays != delays for _, module in modules):
        raise ValueError("All temporal channel mixers must expose the same delay set")
    return modules


def _set_disabled_tcap_delays(
    modules: list[tuple[str, CausalTemporalChannelMixer]],
    original_weights: dict[str, torch.Tensor],
    disabled_delays: tuple[int, ...],
) -> None:
    """Restore learned TCAP weights, then zero complete delay matrices for one ablation."""

    known_delays = modules[0][1].delays
    unknown = sorted(set(disabled_delays) - set(known_delays))
    if unknown:
        raise ValueError(f"Cannot disable delays absent from the checkpoint: {unknown}")
    with torch.no_grad():
        for name, module in modules:
            module.weight.copy_(original_weights[name])
            for delay in disabled_delays:
                module.weight[known_delays.index(delay)].zero_()


@torch.no_grad()
def diagnose_tcap_taps(
    config: dict[str, Any], checkpoint_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Measure the inference-time contribution of every learned TCAP delay.

    This is a checkpoint intervention, not a retrained architectural ablation. Every condition uses
    the same weights and validation order; only complete learned delay matrices are set to zero.
    """

    # Imported lazily to avoid a runner/evaluation import cycle.
    from etsr.runner import _dvslip_group_metrics, _load_checkpoint_context

    config = copy.deepcopy(config)
    checkpoint_path, checkpoint, bundle, model, device, dataset_metadata = (
        _load_checkpoint_context(config, checkpoint_path)
    )
    if config["dataset"]["name"] != "dvslip":
        raise ValueError("TCAP tap diagnostics currently require DVS-Lip")
    if getattr(model, "readout_name", None) != "mean" or getattr(
        model, "readout_time", None
    ) != "fixed_window":
        raise ValueError("TCAP tap diagnostics require fixed-window mean readout")
    modules = _tcap_modules(model)
    delays = modules[0][1].delays
    original_weights = {name: module.weight.detach().clone() for name, module in modules}
    conditions = [("intact", ())]
    conditions.extend((f"without_delay_{delay}", (delay,)) for delay in delays)
    conditions.append(("without_all_history", delays))

    bin_width_us = int(config["representation"]["bin_width_us"])
    total_steps = int(config["representation"]["window_us"]) // bin_width_us
    configured_times = [
        int(value) for value in config.get("evaluation", {}).get("absolute_prefix_times_us", [])
    ]
    configured_steps = {
        max(1, min(total_steps, (time_us + bin_width_us - 1) // bin_width_us)): time_us
        for time_us in configured_times
    }
    if len(configured_steps) < 2:
        raise ValueError("TCAP diagnostics require at least two configured absolute prefixes")

    loader = build_loader(bundle.validation, config["dataset"], shuffle=False)
    criterion = make_criterion(config["training"])
    num_classes = int(checkpoint["num_classes"])
    curve_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    condition_summaries: dict[str, Any] = {}
    prediction_columns: dict[str, dict[str, list[float | int]]] = {}
    model.eval()

    try:
        for condition_name, disabled_delays in conditions:
            _set_disabled_tcap_delays(modules, original_weights, disabled_delays)
            accumulator = _TemporalMetricAccumulator(total_steps, num_classes)
            prediction_columns[condition_name] = {
                "indices": [],
                "targets": [],
                "predictions": [],
                "margins": [],
            }
            for frames, targets, indices in loader:
                if isinstance(frames, dict):
                    frames = {
                        key: value.to(device, non_blocking=True) for key, value in frames.items()
                    }
                else:
                    frames = frames.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                encoded = model._encode(frames)
                logits = temporal_readout_logits(encoded, model.head)["prefix_mean"]
                accumulator.update(logits, targets, indices, criterion)

                final_logits = logits[-1]
                predictions = final_logits.argmax(dim=1)
                true_logits = final_logits.gather(1, targets[:, None]).squeeze(1)
                competitors = final_logits.clone()
                competitors.scatter_(1, targets[:, None], float("-inf"))
                margins = true_logits - competitors.max(dim=1).values
                columns = prediction_columns[condition_name]
                columns["indices"].extend(int(value) for value in indices.tolist())
                columns["targets"].extend(int(value) for value in targets.cpu().tolist())
                columns["predictions"].extend(int(value) for value in predictions.cpu().tolist())
                columns["margins"].extend(float(value) for value in margins.cpu().tolist())

            rows = accumulator.rows(bin_width_us)
            curve_rows.extend(
                {
                    "condition": condition_name,
                    "disabled_delays": ",".join(str(value) for value in disabled_delays),
                    **row,
                }
                for row in rows
            )
            configured_rows = [row for row in rows if int(row["time_step"]) in configured_steps]
            final_result = accumulator.metrics[-1].compute()
            group_metrics = _dvslip_group_metrics(
                config, bundle.classes, final_result.confusion_matrix
            )
            if group_metrics is None:
                raise ValueError("TCAP diagnostics require the DVS-Lip class-group manifest")
            condition_summary = {
                "disabled_delays": list(disabled_delays),
                "validation": {
                    **final_result.to_dict(),
                    "class_group_accuracies": group_metrics,
                },
                "every_bin_prefix_auc": _auc_summary(rows),
                "configured_prefix_auc": _auc_summary(configured_rows),
                "configured_prefixes": [
                    {
                        **row,
                        "requested_us": configured_steps[int(row["time_step"])],
                    }
                    for row in configured_rows
                ],
            }
            condition_summaries[condition_name] = condition_summary
            final_rows.append(
                {
                    "condition": condition_name,
                    "disabled_delays": ",".join(str(value) for value in disabled_delays),
                    "accuracy": final_result.accuracy,
                    "macro_f1": final_result.macro_f1,
                    "loss": final_result.loss,
                    "acc1": group_metrics["metrics"]["Acc1"]["accuracy"],
                    "acc2": group_metrics["metrics"]["Acc2"]["accuracy"],
                    "accuracy_prefix_auc": condition_summary["configured_prefix_auc"][
                        "accuracy_auc_normalized"
                    ],
                    "macro_f1_prefix_auc": condition_summary["configured_prefix_auc"][
                        "macro_f1_auc_normalized"
                    ],
                }
            )
    finally:
        _set_disabled_tcap_delays(modules, original_weights, ())

    intact = condition_summaries["intact"]
    intact_predictions = prediction_columns["intact"]
    prediction_rows = []
    for position, (sample_index, target) in enumerate(
        zip(intact_predictions["indices"], intact_predictions["targets"], strict=True)
    ):
        row: dict[str, Any] = {"sample_index": sample_index, "target": target}
        for condition_name, _ in conditions:
            columns = prediction_columns[condition_name]
            if (
                columns["indices"][position] != sample_index
                or columns["targets"][position] != target
            ):
                raise ValueError("Validation order changed between TCAP ablation conditions")
            row[f"{condition_name}_prediction"] = columns["predictions"][position]
            row[f"{condition_name}_margin"] = columns["margins"][position]
        prediction_rows.append(row)

    ablation_effects: dict[str, Any] = {}
    intact_targets = np.asarray(intact_predictions["targets"], dtype=np.int64)
    intact_pred = np.asarray(intact_predictions["predictions"], dtype=np.int64)
    intact_correct = intact_pred == intact_targets
    for condition_name, disabled_delays in conditions[1:]:
        ablated = condition_summaries[condition_name]
        ablated_pred = np.asarray(
            prediction_columns[condition_name]["predictions"], dtype=np.int64
        )
        ablated_correct = ablated_pred == intact_targets
        ablation_effects[condition_name] = {
            "disabled_delays": list(disabled_delays),
            "intact_minus_ablated": {
                "accuracy": float(
                    intact["validation"]["accuracy"] - ablated["validation"]["accuracy"]
                ),
                "macro_f1": float(
                    intact["validation"]["macro_f1"] - ablated["validation"]["macro_f1"]
                ),
                "acc1": float(
                    intact["validation"]["class_group_accuracies"]["metrics"]["Acc1"][
                        "accuracy"
                    ]
                    - ablated["validation"]["class_group_accuracies"]["metrics"]["Acc1"][
                        "accuracy"
                    ]
                ),
                "acc2": float(
                    intact["validation"]["class_group_accuracies"]["metrics"]["Acc2"][
                        "accuracy"
                    ]
                    - ablated["validation"]["class_group_accuracies"]["metrics"]["Acc2"][
                        "accuracy"
                    ]
                ),
                "macro_f1_prefix_auc": float(
                    intact["configured_prefix_auc"]["macro_f1_auc_normalized"]
                    - ablated["configured_prefix_auc"]["macro_f1_auc_normalized"]
                ),
            },
            "paired_accuracy": {
                "intact_only_correct": int((intact_correct & ~ablated_correct).sum()),
                "ablated_only_correct": int((~intact_correct & ablated_correct).sum()),
                "both_correct": int((intact_correct & ablated_correct).sum()),
                "both_wrong": int((~intact_correct & ~ablated_correct).sum()),
            },
        }

    tap_norm_rows = []
    for name, module in modules:
        weights = original_weights[name].float()
        for index, delay in enumerate(module.delays):
            matrix = weights[index]
            tap_norm_rows.append(
                {
                    "module": name,
                    "delay_steps": delay,
                    "delay_us": delay * bin_width_us,
                    "frobenius_norm": float(matrix.norm().item()),
                    "mean_absolute_weight": float(matrix.abs().mean().item()),
                    "maximum_absolute_weight": float(matrix.abs().max().item()),
                }
            )

    output = ensure_dir(output_dir)
    files = {
        "curve": "tcap_ablation_curve_every_bin.csv",
        "final": "tcap_ablation_final.csv",
        "predictions": "tcap_ablation_predictions.csv",
        "tap_norms": "tcap_tap_norms.csv",
    }
    write_csv(curve_rows, output / files["curve"])
    write_csv(final_rows, output / files["final"])
    write_csv(prediction_rows, output / files["predictions"])
    write_csv(tap_norm_rows, output / files["tap_norms"])
    save_config(config, output / "config_evaluated.yaml")
    environment_path = output / "environment.json"
    write_json(collect_environment(device), environment_path)
    summary = {
        "schema_version": 1,
        "diagnostic": "tcap_delay_matrix_zero_ablation",
        "interpretation": (
            "checkpoint-only intervention: intact learned model versus complete delay matrices "
            "zeroed without retraining; a negative effect does not estimate a retrained architecture"
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_score": float(checkpoint["score"]),
        "checkpoint_git_commit": checkpoint["config"].get("runtime", {}).get("git_commit"),
        "diagnostic_git_commit": git_commit(),
        "diagnostic_git_dirty": git_is_dirty(),
        "device": str(device),
        "samples": len(intact_predictions["targets"]),
        "time_steps": total_steps,
        "bin_width_us": bin_width_us,
        "mixer_count": len(modules),
        "mixer_names": [name for name, _ in modules],
        "delays": list(delays),
        "conditions": condition_summaries,
        "ablation_effects": ablation_effects,
        "tap_norms": tap_norm_rows,
        "files": files,
        "environment": str(environment_path.resolve()),
        "environment_sha256": sha256_file(environment_path),
        "official_test_used": False,
        **dataset_metadata,
    }
    write_json(summary, output / "tcap_ablation_summary.json")
    return summary
