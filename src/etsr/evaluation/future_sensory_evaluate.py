"""One-time internal-holdout evaluation of fitted sensory screens."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from etsr.data.factory import build_dataset_bundle
from etsr.evaluation.future_sensory import (
    FutureDecoder,
    SensoryDataset,
    build_causal_d,
    sensory_loss_parts,
    spatial_e0_count,
)
from etsr.evaluation.future_sensory_fit import (
    _loader,
    screen_code_sha256,
    screen_config,
    stratified_split,
)
from etsr.reproducibility import git_commit, seed_everything
from etsr.utils.io import ensure_dir, sha256_file, write_json

PROBE_EPOCHS = 30
PROBE_BATCH = 64
PROBE_SEED = 20261002
BOOTSTRAP_DRAWS = 2000


def _shift(values: torch.Tensor, delay: int) -> torch.Tensor:
    shifted = torch.zeros_like(values)
    if delay == 0:
        return values
    shifted[:, delay:] = values[:, :-delay]
    return shifted


def _past_profile(older: torch.Tensor, newer: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "fepf2":
        # E0 has no within-bin timestamps; bin centers are the only admissible
        # E0-only timing assumption at phases .25 and .75 of a 100 ms window.
        result = torch.stack(
            (
                .5625 * older + .0625 * newer,
                .375 * older + .375 * newer,
                .0625 * older + .5625 * newer,
            ), dim=3,
        )
    else:
        result = torch.stack((older / 2, older / 2, newer / 2, newer / 2), dim=3)
    count = result.sum(dim=3)
    return result / count.unsqueeze(3).clamp_min(1e-9)


def e0_baselines(
    frames: torch.Tensor, prior: dict, mode: str, *, past: bool = False
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pooled = spatial_e0_count(frames)
    recent = [_shift(pooled, delay) for delay in range(4)]
    mean_n = prior["mean_count"].to(frames.device)[None].expand(frames.shape[0], -1, -1, -1, -1)
    prior_q = prior["q_prior"].to(frames.device)[None].expand(frames.shape[0], -1, -1, -1, -1, -1)
    last100 = recent[0] + recent[1]
    recent200 = (last100 + recent[2] + recent[3]) / 2
    calibration = prior["count_calibration"]
    if not past:
        last_count = (last100 * calibration["last_100ms"]["scale"]
                      + calibration["last_100ms"]["offset"])
        recent_count = (recent200 * calibration["mean_200ms"]["scale"]
                        + calibration["mean_200ms"]["offset"])
    else:
        # Read the past count directly from E0; rare capped pixels can make it
        # lower than the corresponding unbounded raw-event target.
        last_count = last100
        recent_count = recent200
    q_last = _past_profile(recent[1], recent[0], mode)
    q_older = _past_profile(recent[3], recent[2], mode)
    q_last = torch.where(last100.unsqueeze(3) > 0, q_last, prior_q)
    q_older = torch.where((recent[2] + recent[3]).unsqueeze(3) > 0, q_older, prior_q)
    return {
        "mean_field": (mean_n.clamp_min(1e-6), prior_q),
        "last_100ms": (last_count.clamp_min(1e-6), q_last),
        "mean_200ms": (recent_count.clamp_min(1e-6), (q_last + q_older) / 2),
        "zero_alarm": (torch.full_like(last100, 1e-6), prior_q),
    }


def _bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, seed: int = 42) -> list[float]:
    if len(a) != len(b):
        raise ValueError("Unpaired utterance losses")
    rng = np.random.default_rng(seed)
    delta = a - b
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        draws[i] = delta[rng.integers(len(delta), size=len(delta))].mean()
    return np.quantile(draws, [0.025, 0.975]).tolist()


class ClassProbe(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.head = nn.Linear(128, classes)

    def forward(self, x: torch.Tensor, *, steps: int | None = None) -> torch.Tensor:
        if steps is not None:
            x = x[:, :steps]
        batch, time = x.shape[:2]
        feature = self.conv(x.reshape(batch * time, *x.shape[2:])).mean(dim=(-2, -1))
        return self.head(feature.reshape(batch, time, 128)).mean(dim=1)


def _field_input(count: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Expose the physical mass field, never untrained timing logits in empty cells."""
    return (count.unsqueeze(3) * q).flatten(2, 3)


def _cache_arrays(base: Path, count: int, mode: str, *, write: bool) -> dict[str, np.ndarray]:
    components = 3 if mode == "fepf2" else 4
    arrays = {}
    for name, shape in (
        ("encoder", (count, 40, 128, 8, 8)),
        ("predicted", (count, 15, 2 * components, 16, 16)),
        ("oracle", (count, 15, 2 * components, 16, 16)),
        ("label", (count,)),
    ):
        dtype = np.int16 if name == "label" else (np.uint8 if name == "encoder" else np.float16)
        arrays[name] = np.lib.format.open_memmap(
            base / f"{name}.npy", mode="w+" if write else "r", dtype=dtype, shape=shape
        )
    return arrays


def _diagnostic_values(count_hat, q_hat, field, valid, last_us):
    """Descriptive, unbalanced losses by physical region; utterance is the unit."""
    target_count = field.sum(dim=3)
    occupied = target_count > 0
    deviance = 2 * (
        count_hat - target_count
        + torch.where(
            occupied,
            target_count * (target_count.clamp_min(1e-9).log() - count_hat.log()),
            0.0,
        )
    )
    target_q = field / target_count.unsqueeze(3).clamp_min(1e-9)
    kl = torch.where(
        field > 0,
        target_q * (target_q.clamp_min(1e-9).log() - q_hat.clamp_min(1e-9).log()),
        0.0,
    ).sum(dim=3)
    cutoff = (torch.arange(field.shape[1], device=field.device) + 1) * 50_000
    active = (cutoff[None] + 100_000 <= last_us[:, None].to(field.device))
    masks = {
        "all_cells": valid[:, :, None, None, None].expand_as(target_count),
        "occupied_cells": valid[:, :, None, None, None] & occupied,
        "fully_active_window": (valid & active)[:, :, None, None, None].expand_as(target_count),
        "offset_window": (valid & ~active)[:, :, None, None, None].expand_as(target_count),
    }
    for name, polarity in (("off", 0), ("on", 1)):
        polarity_mask = torch.zeros_like(target_count, dtype=torch.bool)
        polarity_mask[:, :, polarity] = True
        masks[name] = masks["all_cells"] & polarity_mask
    output = {}
    for name, mask in masks.items():
        for metric, values, selected in (
            ("count", deviance, mask),
            ("timing", kl, mask & occupied),
        ):
            support = selected.sum(dim=(1, 2, 3, 4))
            mean = (values * selected).sum(dim=(1, 2, 3, 4)) / support.clamp_min(1)
            output[f"{name}_{metric}"] = mean[support > 0].detach().cpu().numpy().tolist()
    count_mask = masks["all_cells"]
    count_support = count_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1)
    count_mean = (count_hat * count_mask).sum(dim=(1, 2, 3, 4)) / count_support
    count_variance = (
        ((count_hat - count_mean[:, None, None, None, None]) ** 2) * count_mask
    ).sum(dim=(1, 2, 3, 4)) / count_support
    timing_mask = masks["occupied_cells"].unsqueeze(3).expand_as(q_hat)
    timing_support = timing_mask.sum(dim=(1, 2, 3, 4, 5)).clamp_min(1)
    timing_mean = (q_hat * timing_mask).sum(dim=(1, 2, 3, 4, 5)) / timing_support
    timing_variance = (
        ((q_hat - timing_mean[:, None, None, None, None, None]) ** 2) * timing_mask
    ).sum(dim=(1, 2, 3, 4, 5)) / timing_support
    output["prediction_count_variance"] = count_variance.detach().cpu().numpy().tolist()
    output["prediction_q_variance_on_occupied_cells"] = (
        timing_variance.detach().cpu().numpy().tolist()
    )
    return output


@torch.no_grad()
def _collect_partition(
    model, decoder, dataset, config, prior, device, cache_path: Path, mode: str,
    *, collect_losses: bool,
) -> tuple[
    dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray
]:
    arrays = _cache_arrays(cache_path, len(dataset), mode, write=True)
    losses: dict[str, dict[str, list[float]]] = {}
    diagnostics: dict[str, dict[str, list[float]]] = {}
    durations: list[int] = []
    endpoints: list[int] = []
    offset = 0
    for frames, field, valid, labels, _, last_us, duration_us in _loader(
        dataset, config, shuffle=False, seed=0
    ):
        frames = frames.to(device)
        field = field.to(device)
        valid = valid.to(device)
        stage2 = model._encode(frames)
        count, q = decoder(stage2)
        count = count.float()
        q = q.float()
        end = offset + frames.shape[0]
        # The final D block is a sum of binary spikes, hence nonnegative exact
        # integers. uint8 caching is lossless and halves this large cache.
        if bool((stage2 < 0).any() or (stage2 > 255).any() or (stage2 != stage2.round()).any()):
            raise RuntimeError("Stage2 features violated the exact uint8 spike-sum cache contract")
        arrays["encoder"][offset:end] = stage2.permute(1, 0, 2, 3, 4).cpu().byte().numpy()
        arrays["predicted"][offset:end] = _field_input(count[:, :15], q[:, :15]).cpu().half().numpy()
        arrays["oracle"][offset:end] = field[:, :15].flatten(2, 3).cpu().half().numpy()
        arrays["label"][offset:end] = labels.numpy()
        durations.extend(duration_us.tolist())
        endpoints.extend(last_us.tolist())
        if collect_losses:
            variants = {
                "model": (count, q),
                **e0_baselines(frames, prior, mode, past=dataset.past),
            }
            for name, (variant_n, variant_q) in variants.items():
                parts = sensory_loss_parts(variant_n, variant_q, field, valid)
                record = losses.setdefault(name, {"count": [], "timing": []})
                record["count"].extend(parts.count.cpu().numpy().tolist())
                record["timing"].extend(parts.timing.cpu().numpy().tolist())
                diagnostic = diagnostics.setdefault(name, {})
                for key, values in _diagnostic_values(
                    variant_n, variant_q, field, valid, last_us
                ).items():
                    diagnostic.setdefault(key, []).extend(values)
        offset = end
    if offset != len(dataset):
        raise RuntimeError("Incomplete field cache")
    for array in arrays.values():
        array.flush()
    loss_arrays = {
        name: {key: np.asarray(values, dtype=np.float64) for key, values in record.items()}
        for name, record in losses.items()
    }
    diagnostic_arrays = {
        name: {key: np.asarray(values, dtype=np.float64) for key, values in record.items()}
        for name, record in diagnostics.items()
    }
    return (
        loss_arrays,
        diagnostic_arrays,
        np.asarray(durations, dtype=np.int64),
        np.asarray(endpoints, dtype=np.int64),
    )


@torch.no_grad()
def _random_encoder_cache(
    config, dataset, device, base: Path, initial_state_path: Path
) -> np.ndarray:
    seed_everything(PROBE_SEED)
    model = build_causal_d(config, 100).to(device).eval()
    model.load_state_dict(
        torch.load(initial_state_path, map_location=device, weights_only=False), strict=True
    )
    array = np.lib.format.open_memmap(
        base / "random_encoder.npy", mode="w+", dtype=np.uint8,
        shape=(len(dataset), 40, 128, 8, 8),
    )
    offset = 0
    for frames, _, _, _, _, _, _ in _loader(dataset, config, shuffle=False, seed=0):
        frames = frames.to(device)
        encoded = model._encode(frames).permute(1, 0, 2, 3, 4)
        if bool((encoded < 0).any() or (encoded > 255).any() or (encoded != encoded.round()).any()):
            raise RuntimeError("Random stage2 features violated uint8 spike-sum contract")
        encoded = encoded.cpu().byte().numpy()
        array[offset : offset + len(encoded)] = encoded
        offset += len(encoded)
    array.flush()
    return array


def _probe_input(arrays: dict, name: str, indices: np.ndarray, prior_q: torch.Tensor) -> torch.Tensor:
    if name.startswith("encoder"):
        key = "encoder" if name == "encoder_ssl" else "random_encoder"
        return torch.from_numpy(np.array(arrays[key][indices], dtype=np.float32))
    key = "predicted" if name.startswith("predicted") else "oracle"
    x = torch.from_numpy(np.array(arrays[key][indices], dtype=np.float32))
    if name.endswith("density"):
        components = prior_q.shape[2]
        reshaped = x.reshape(len(indices), 15, 2, components, 16, 16)
        count = reshaped.sum(dim=3, keepdim=True)
        x = (count * prior_q[:15].unsqueeze(0)).flatten(2, 3)
    return x


def _macro_f1(targets: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    confusion = np.bincount(targets * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    true_positive = np.diag(confusion)
    f1 = 2 * true_positive / np.maximum(confusion.sum(0) + confusion.sum(1), 1)
    return float(f1.mean())


def _probe_train_eval(
    name, train_arrays, holdout_arrays, prior_q, *, classes, device,
    train_indices: np.ndarray | None = None,
    holdout_indices: np.ndarray | None = None,
) -> dict:
    seed_everything(PROBE_SEED)
    channels = 128 if name.startswith("encoder") else 2 * prior_q.shape[2]
    head = ClassProbe(channels, classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    if train_indices is None:
        train_indices = np.arange(len(train_arrays["label"]))
    if holdout_indices is None:
        holdout_indices = np.arange(len(holdout_arrays["label"]))
    targets = np.asarray(train_arrays["label"], dtype=np.int64)
    for epoch in range(PROBE_EPOCHS):
        head.train()
        order = np.random.default_rng(PROBE_SEED + epoch).permutation(train_indices)
        for start in range(0, len(order), PROBE_BATCH):
            indices = order[start : start + PROBE_BATCH]
            x = _probe_input(train_arrays, name, indices, prior_q).to(device)
            y = torch.from_numpy(targets[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y)
            loss.backward()
            optimizer.step()
    head.eval()
    predictions: dict[str, list[np.ndarray]] = {"full": []}
    if name.startswith("encoder"):
        predictions.update({"prefix_1000ms": [], "prefix_1500ms": []})
    with torch.no_grad():
        for start in range(0, len(holdout_indices), PROBE_BATCH):
            indices = holdout_indices[start : start + PROBE_BATCH]
            x = _probe_input(holdout_arrays, name, indices, prior_q).to(device)
            predictions["full"].append(head(x).argmax(dim=1).cpu().numpy())
            if name.startswith("encoder"):
                for key, steps in (("prefix_1000ms", 20), ("prefix_1500ms", 30)):
                    predictions[key].append(head(x, steps=steps).argmax(dim=1).cpu().numpy())
    y_true = np.asarray(holdout_arrays["label"], dtype=np.int64)[holdout_indices]
    result: dict = {"trainable_parameters": sum(p.numel() for p in head.parameters())}
    for key, chunks in predictions.items():
        y_pred = np.concatenate(chunks)
        result[key] = {
            "macro_f1": _macro_f1(y_true, y_pred, classes),
            "accuracy": float((y_pred == y_true).mean()),
            "predictions": y_pred.tolist(),
        }
    return result


def _stratified_resample_indices(
    groups: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Resample utterances within each class, keeping the paired comparison intact."""
    return np.concatenate([
        group[rng.integers(len(group), size=len(group))] for group in groups
    ])


def _active_field_indices(last_event_us: np.ndarray) -> np.ndarray:
    """A fixed 15-step cohort; no endpoint-dependent mask enters the probe."""
    return np.flatnonzero(last_event_us > 15 * 50_000)


def _bootstrap_f1_delta(y, c, b, classes, seed=42):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        indices = _stratified_resample_indices(groups, rng)
        draws[i] = _macro_f1(y[indices], c[indices], classes) - _macro_f1(y[indices], b[indices], classes)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def _first_pass_gate(skills: dict, comparisons: dict) -> bool:
    return bool(
        skills["count"]["passed"] and skills["timing"]["passed"]
        and comparisons["predicted_vs_density"]["delta_macro_f1"] >= .01
        and comparisons["predicted_vs_density"]["bootstrap_95"][0] > 0
        and comparisons["encoder_vs_random"]["delta_macro_f1"] >= .01
        and comparisons["oracle_vs_density"]["delta_macro_f1"] > 0
        and comparisons["oracle_vs_density"]["bootstrap_95"][0] > 0
    )


def _screen_one(
    fit_dir: Path, config: dict, bundle, split: dict, output: Path,
    device: torch.device, random_cache: dict[str, dict[str, np.ndarray]],
):
    fit_report = json.loads((fit_dir / "fit_report.json").read_text())
    if fit_report["split_sha256"] != sha256_file(fit_dir / "train_only_split.json"):
        raise ValueError("Fit split was modified")
    if fit_report["config_resolved_sha256"] != sha256_file(fit_dir / "config_resolved.yaml"):
        raise ValueError("Fit resolved configuration was modified")
    initial_state_path = fit_dir / "initial_encoder_state.pt"
    initial_sha256 = sha256_file(initial_state_path)
    if fit_report["initial_encoder_sha256"] != initial_sha256:
        raise ValueError("Fit initial encoder checkpoint was modified")
    if yaml.safe_load((fit_dir / "config_resolved.yaml").read_text()) != {
        key: value for key, value in config.items() if not key.startswith("_")
    }:
        raise ValueError("Evaluation configuration differs from the fitted D trunk")
    if json.loads((fit_dir / "train_only_split.json").read_text()) != split:
        raise ValueError("Fit used a different train-only split")
    mode = fit_report["mode"]
    past = fit_report["past"]
    model = build_causal_d(config, len(bundle.classes)).to(device).eval()
    decoder = FutureDecoder(3 if mode == "fepf2" else 4).to(device).eval()
    state = torch.load(fit_dir / "last_state.pt", map_location=device, weights_only=False)
    if state["epoch"] != fit_report["epochs"]:
        raise ValueError("Fit report and checkpoint epoch differ")
    model.load_state_dict(state["model"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    prior = torch.load(fit_dir / "fit_prior.pt", map_location="cpu", weights_only=False)
    fit_data = SensoryDataset(bundle.train, split["fit_indices"], mode, past)
    held_data = SensoryDataset(bundle.train, split["holdout_indices"], mode, past)
    base = ensure_dir(output / f"{mode}_{'past' if past else 'future'}_seed{fit_report['seed']}")
    train_dir = ensure_dir(base / "fit_cache")
    held_dir = ensure_dir(base / "holdout_cache")
    _, _, _, fit_endpoints = _collect_partition(
        model, decoder, fit_data, config, prior, device, train_dir, mode,
        collect_losses=False,
    )
    loss_arrays, diagnostic_arrays, durations, endpoints = _collect_partition(
        model, decoder, held_data, config, prior, device, held_dir, mode, collect_losses=True
    )
    losses = {
        name: {part: float(values.mean()) for part, values in record.items()}
        for name, record in loss_arrays.items()
    }
    duration_bands = {}
    for label, selected in (
        ("under_1000ms", durations < 1_000_000),
        ("1000_to_1500ms", (durations >= 1_000_000) & (durations < 1_500_000)),
        ("at_least_1500ms", durations >= 1_500_000),
    ):
        duration_bands[label] = {
            "utterances": int(selected.sum()),
            "model_count": float(loss_arrays["model"]["count"][selected].mean()) if selected.any() else None,
            "model_timing": float(loss_arrays["model"]["timing"][selected].mean()) if selected.any() else None,
        }
    baselines = ("mean_field", "last_100ms", "mean_200ms")
    skills = {}
    for part in ("count", "timing"):
        best = min(losses[name][part] for name in baselines)
        intervals = {
            name: _bootstrap_mean_difference(loss_arrays[name][part], loss_arrays["model"][part])
            for name in baselines
        }
        skills[part] = {
            "relative_to_best_baseline": 1 - losses["model"][part] / best,
            "paired_gain_intervals_vs_each": intervals,
            "passed": (1 - losses["model"][part] / best >= .05)
            and all(bounds[0] > 0 for bounds in intervals.values()),
        }
    train_arrays = _cache_arrays(train_dir, len(fit_data), mode, write=False)
    held_arrays = _cache_arrays(held_dir, len(held_data), mode, write=False)
    # No endpoint enters a probe input or changes its number of steps. The
    # primary field comparison is conditional on every fixed cutoff having
    # received training loss; full-population losses and coverage are descriptive.
    fit_active = _active_field_indices(fit_endpoints)
    held_active = _active_field_indices(endpoints)
    fit_labels = np.asarray(train_arrays["label"], dtype=np.int64)[fit_active]
    held_labels = np.asarray(held_arrays["label"], dtype=np.int64)[held_active]
    if not len(fit_active) or not len(held_active):
        raise RuntimeError("The fixed-cutoff active probe has no eligible utterances")
    fit_class_support = np.bincount(fit_labels, minlength=len(bundle.classes))
    held_class_support = np.bincount(held_labels, minlength=len(bundle.classes))
    if initial_sha256 not in random_cache:
        shared = ensure_dir(output / "paired_initial_encoders" / initial_sha256)
        random_cache[initial_sha256] = {
            "fit": _random_encoder_cache(
                config, fit_data, device, ensure_dir(shared / "fit"), initial_state_path
            ),
            "holdout": _random_encoder_cache(
                config, held_data, device, ensure_dir(shared / "holdout"), initial_state_path
            ),
        }
    train_arrays["random_encoder"] = random_cache[initial_sha256]["fit"]
    held_arrays["random_encoder"] = random_cache[initial_sha256]["holdout"]
    probe_names = (
        "oracle", "oracle_density", "predicted", "predicted_density",
        "encoder_ssl", "encoder_random",
    )
    probes = {
        name: _probe_train_eval(
            name, train_arrays, held_arrays, prior["q_prior"],
            classes=len(bundle.classes), device=device,
            train_indices=(fit_active if name in {
                "oracle", "oracle_density", "predicted", "predicted_density"
            } else None),
            holdout_indices=(held_active if name in {
                "oracle", "oracle_density", "predicted", "predicted_density"
            } else None),
        )
        for name in probe_names
    }
    y = np.asarray(held_arrays["label"], dtype=np.int64)
    comparisons = {}
    for left, right, key in (
        ("predicted", "predicted_density", "predicted_vs_density"),
        ("encoder_ssl", "encoder_random", "encoder_vs_random"),
        ("oracle", "oracle_density", "oracle_vs_density"),
    ):
        left_predictions = np.asarray(probes[left]["full"]["predictions"])
        right_predictions = np.asarray(probes[right]["full"]["predictions"])
        paired_targets = held_labels if key in {
            "predicted_vs_density", "oracle_vs_density"
        } else y
        comparisons[key] = {
            "delta_macro_f1": probes[left]["full"]["macro_f1"] - probes[right]["full"]["macro_f1"],
            "bootstrap_95": _bootstrap_f1_delta(
                paired_targets, left_predictions, right_predictions, len(bundle.classes)
            ),
        }
    gate = (_first_pass_gate(skills, comparisons)
            and bool((fit_class_support > 0).all() and (held_class_support > 0).all()))
    result = {
        "fit_dir": str(fit_dir), "mode": mode, "past": past,
        "fit_checkpoint_sha256": sha256_file(fit_dir / "last_state.pt"),
        "initial_encoder_sha256": initial_sha256,
        "code_sha256": fit_report["code_sha256"],
        "seed": fit_report["seed"], "epochs": fit_report["epochs"],
        "losses": losses, "skills": skills, "probes": probes,
        "baseline_contract": {
            "count_weighted_mean_and_history_calibration_fit_only": True,
            "past_last_100ms_count_direct_from_e0_subject_to_cap": bool(past),
        },
        "descriptive_breakdown": {
            name: {key: float(values.mean()) if len(values) else None for key, values in record.items()}
            for name, record in diagnostic_arrays.items()
        },
        "duration_bands": duration_bands,
        "field_probe_population": {
            "primary": "last_event_strictly_after_750ms; fixed 15-step readout",
            "fit_utterances": int(len(fit_active)),
            "holdout_utterances": int(len(held_active)),
            "fit_classes_covered": int((fit_class_support > 0).sum()),
            "holdout_classes_covered": int((held_class_support > 0).sum()),
            "requires_all_classes_for_promotion": True,
            "full_population_losses_and_post_end_fraction_are_descriptive": True,
        },
        "fraction_finished_at_fixed_probe_cutoffs": {
            str(step * 50_000): float((endpoints <= step * 50_000).mean())
            for step in range(1, 16)
        },
        "comparisons": comparisons, "first_pass_gate": gate,
        "holdout_indices_sha256": split["holdout_id_sha256"],
        "development_validation_used": False, "official_test_used": False,
    }
    write_json(result, base / "evaluation.json")
    return result


def evaluate_future_sensory(
    config: dict, fit_dirs: list[str], output_dir: str | Path,
    *, previous_evaluation: str | Path | None = None,
) -> dict:
    """Refuse the holdout until all initially compared arms meet the epoch policy."""
    if len(fit_dirs) < 2:
        raise ValueError("Evaluate the two preregistered target arms together")
    config = screen_config(config)
    reports = [json.loads((Path(directory) / "fit_report.json").read_text()) for directory in fit_dirs]
    initial = [report for report in reports if report["seed"] == 42 and not report["past"]]
    if {report["mode"] for report in initial} != {"fepf2", "voxel4"}:
        raise ValueError("Initial FEPF-2 and Voxel-4 fits must both be present")
    if any(report["extension_requested_by_this_arm"] for report in initial) and any(
        report["epochs"] != 80 for report in initial
    ):
        raise RuntimeError("At least one arm was still falling >5%; extend both to 80 before opening holdout")
    if any(report["epochs"] != initial[0]["epochs"] for report in initial):
        raise ValueError("Unequal initial target training budgets")
    if any(report["epochs"] != initial[0]["epochs"] for report in reports):
        raise ValueError("All future replicas and matched-past controls need the initial 40/80 budget")
    current_commit = git_commit()
    current_code_sha256 = screen_code_sha256()
    if any(
        report["git_commit"] != current_commit
        or report["code_sha256"] != current_code_sha256
        for report in reports
    ):
        raise ValueError("Evaluation requires the exact fit commit and executable source")
    output = ensure_dir(output_dir)
    if (output / "evaluation.json").exists():
        raise FileExistsError("The internal holdout has already been evaluated at this output")
    bundle = build_dataset_bundle(config)
    split = stratified_split(bundle.train.targets, bundle.train.sample_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    random_cache: dict[str, dict[str, np.ndarray]] = {}
    for directory, fit_report in zip(fit_dirs, reports, strict=True):
        suffix = f"{fit_report['mode']}_{'past' if fit_report['past'] else 'future'}_seed{fit_report['seed']}"
        previous_path = (
            Path(previous_evaluation) / suffix / "evaluation.json"
            if previous_evaluation is not None else None
        )
        if previous_path is not None and previous_path.exists():
            arm = json.loads(previous_path.read_text())
            if arm["holdout_indices_sha256"] != split["holdout_id_sha256"] or (
                arm["mode"], arm["past"], arm["seed"]
            ) != (fit_report["mode"], fit_report["past"], fit_report["seed"]):
                raise ValueError(f"Previous evaluation disagrees with current fit: {previous_path}")
            if arm["fit_checkpoint_sha256"] != sha256_file(Path(directory) / "last_state.pt"):
                raise ValueError(f"Previous evaluation used another checkpoint: {previous_path}")
            if arm["code_sha256"] != current_code_sha256 or (
                arm["initial_encoder_sha256"]
                != sha256_file(Path(directory) / "initial_encoder_state.pt")
            ):
                raise ValueError(f"Previous evaluation used another implementation: {previous_path}")
            results.append(arm)
        else:
            results.append(
                _screen_one(Path(directory), config, bundle, split, output, device, random_cache)
            )
    future_initial = {
        arm["mode"]: arm for arm in results if arm["seed"] == 42 and not arm["past"]
    }
    initial_passes = [arm for arm in future_initial.values() if arm["first_pass_gate"]]
    winner = None
    if initial_passes:
        winner = max(
            initial_passes,
            key=lambda arm: (
                arm["probes"]["encoder_ssl"]["full"]["macro_f1"],
                arm["skills"]["timing"]["relative_to_best_baseline"],
                arm["mode"] == "fepf2",
            ),
        )["mode"]
    promotion = {"initial_winner": winner, "three_seed_gate": None}
    if winner is not None:
        future = {
            arm["seed"]: arm for arm in results
            if arm["mode"] == winner and not arm["past"]
        }
        past = {
            arm["seed"]: arm for arm in results
            if arm["mode"] == winner and arm["past"]
        }
        if set(future) >= {42, 43, 44} and set(past) >= {42, 43, 44}:
            seed_gates = {
                seed: bool(future[seed]["first_pass_gate"])
                for seed in (42, 43, 44)
            }
            paired_deltas = {
                seed: (
                    future[seed]["probes"]["encoder_ssl"]["full"]["macro_f1"]
                    - past[seed]["probes"]["encoder_ssl"]["full"]["macro_f1"]
                )
                for seed in (42, 43, 44)
            }
            promotion["three_seed_gate"] = {
                "each_future_seed_passes": seed_gates,
                "future_minus_fine_past_encoder_f1": paired_deltas,
                "passed": all(seed_gates.values())
                and sum(delta > 0 for delta in paired_deltas.values()) >= 2
                and np.mean(list(paired_deltas.values())) >= .01,
            }
    report = {
        "arms": [
            {
                "mode": arm["mode"], "past": arm["past"], "seed": arm["seed"],
                "first_pass_gate": arm["first_pass_gate"],
                "count_skill": arm["skills"]["count"]["relative_to_best_baseline"],
                "timing_skill": arm["skills"]["timing"]["relative_to_best_baseline"],
                "encoder_f1": arm["probes"]["encoder_ssl"]["full"]["macro_f1"],
            }
            for arm in results
        ],
        "promotion": promotion,
        "temporary_feature_caches_removed_after_success": True,
        "development_validation_used": False, "official_test_used": False,
    }
    write_json(report, output / "evaluation.json")
    # Stage2 spike sums and fields are reproducible from the checkpoints. They
    # are temporary probe inputs, not scientific results, and can occupy tens
    # of gigabytes for the two initial targets.
    for cache_file in output.rglob("*.npy"):
        cache_file.unlink()
    return report
