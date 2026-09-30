"""Frozen-D probe of learned temporal expectation versus matched history controls.

The fit and validation commands are deliberately separate. All architecture, training-budget,
and seed choices are fixed in the fit artifact before development validation is opened.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from etsr.data.common import build_loader
from etsr.data.events import move_encoded_input
from etsr.data.factory import build_dataset_bundle
from etsr.evaluation.metrics import classification_metrics
from etsr.models.factory import build_model
from etsr.reproducibility import git_commit, seed_everything
from etsr.training.checkpointing import load_model_state
from etsr.training.predictive import last_occupied_steps
from etsr.utils.io import ensure_dir, sha256_file, write_json

DELAYS = (1, 2, 4, 8)
SEEDS = (42, 43, 44, 45, 46)
Q_EPOCHS = 30
HEAD_EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
INTERNAL_Q_FIT_FRACTION = 0.8
T_CRITICAL_DF4_95 = 2.7764451051977987
PRIMARY_MODES = ("b", "b_rand", "c")
SECONDARY_MODES = ("linear_refit", "a", "b_flat", "r_q", "r_p")
ALL_MODES = (*PRIMARY_MODES, *SECONDARY_MODES)


def causal_history(z: torch.Tensor) -> torch.Tensor:
    """Four causal taps; unavailable history is zero, for every probe arm."""

    if z.ndim != 3:
        raise ValueError("Expected features [sample, time, channel].")
    zeros = torch.zeros_like(z)
    taps = []
    for delay in DELAYS:
        tap = zeros.clone()
        tap[:, delay:] = z[:, :-delay]
        taps.append(tap)
    return torch.cat(taps, dim=-1)


class HistoryEncoder(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * channels, 256), nn.GELU(), nn.Linear(256, channels)
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.net(history)


class StepHead(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ProbeClassifier(nn.Module):
    """Identical B/C/B_rand graph; only history-encoder training differs."""

    def __init__(
        self,
        mode: str,
        classes: int,
        *,
        encoder: HistoryEncoder | None = None,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown probe mode: {mode}")
        self.mode = mode
        self.encoder = encoder
        if head is None:
            if mode in PRIMARY_MODES:
                head = StepHead(256, 256, classes)
            elif mode == "b_flat":
                head = StepHead(640, 345, classes)
            elif mode == "linear_refit":
                head = nn.Linear(128, classes)
            else:
                head = StepHead(128, 256, classes)
        self.head = head
        if mode in (*PRIMARY_MODES, "r_q") and encoder is None:
            raise ValueError(f"{mode} requires a history encoder")
        if mode in ("c", "b_rand", "r_q"):
            assert self.encoder is not None
            self.encoder.requires_grad_(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode in PRIMARY_MODES:
            assert self.encoder is not None
            features = torch.cat((z, self.encoder(causal_history(z))), dim=-1)
        elif self.mode == "b_flat":
            features = torch.cat((z, causal_history(z)), dim=-1)
        elif self.mode == "r_q":
            assert self.encoder is not None
            features = z - self.encoder(causal_history(z))
        elif self.mode == "r_p":
            previous = torch.zeros_like(z)
            previous[:, 1:] = z[:, :-1]
            features = z - previous
        else:
            features = z
        return self.head(features).mean(dim=1)


def _probe_config(config: dict[str, Any]) -> dict[str, Any]:
    if config["dataset"]["name"] != "dvslip":
        raise ValueError("The innovation probe is preregistered for DVS-Lip only.")
    if int(config["experiment"]["seed"]) != 42 or config["experiment"]["name"] != "dvslip_predictive_dynamic_tcap":
        raise ValueError("The innovation probe is preregistered for D seed 42 only.")
    if not config["model"].get("temporal_channel_mixer_dynamic_routing") or tuple(
        config["model"].get("temporal_channel_mixer_routing_stages", ())
    ) != (2,) or config["model"].get("temporal_channel_mixer_routing_parameterization") != "amplitude_allocation":
        raise ValueError("The innovation probe requires the stage2 dynamic-TCAP architecture.")
    if config["model"].get("readout", "mean") != "mean" or config["model"].get(
        "readout_time", "fixed_window"
    ) != "fixed_window":
        raise ValueError("The exact D-head anchor requires mean + fixed_window readout.")
    result = copy.deepcopy(config)
    result["augmentation"] = {}
    return result


def _slice_time(frames: torch.Tensor | dict[str, torch.Tensor], steps: int):
    if isinstance(frames, torch.Tensor):
        return frames[:, :steps]
    ratio = frames["fine"].shape[1] // frames["coarse"].shape[1]
    return {"coarse": frames["coarse"][:, :steps], "fine": frames["fine"][:, : steps * ratio]}


@torch.no_grad()
def _features_for_partition(model, dataset, config, device) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        "z": [], "target": [], "index": [], "endpoint": []
    }
    checked = False
    for frames, target, index in build_loader(dataset, config["dataset"], shuffle=False):
        frames = move_encoded_input(frames, device)
        encoded = model._encode(frames)
        z = encoded.mean(dim=(3, 4)).permute(1, 0, 2).float()
        if not checked:
            # The unchanged trained head must reproduce D exactly at the chosen probe point.
            original = model(frames)
            torch.testing.assert_close(model.head(z.mean(1)), original, atol=2e-5, rtol=2e-5)
            small = frames[:1] if isinstance(frames, torch.Tensor) else {
                key: value[:1] for key, value in frames.items()
            }
            prefix = model._encode(_slice_time(small, 20)).mean(dim=(3, 4))
            torch.testing.assert_close(prefix, encoded[:20, :1].mean(dim=(3, 4)), atol=2e-5, rtol=2e-5)
            checked = True
        chunks["z"].append(z.cpu())
        chunks["target"].append(target.cpu())
        chunks["index"].append(index.cpu())
        chunks["endpoint"].append(last_occupied_steps(frames).cpu())
    if not checked:
        raise ValueError("Empty dataset partition")
    return {key: torch.cat(value) for key, value in chunks.items()}


def _normalization(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = z.mean(dim=(0, 1))
    std = z.std(dim=(0, 1), unbiased=False).clamp_min(1e-5)
    return mean, std


def _batches(count: int, seed: int, epoch: int):
    generator = torch.Generator().manual_seed(seed * 100_000 + epoch)
    order = torch.randperm(count, generator=generator)
    yield from order.split(BATCH_SIZE)


def _fit_q(
    encoder: HistoryEncoder,
    z: torch.Tensor,
    fit_indices: torch.Tensor,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int | str]]:
    encoder.to(device).train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rows = []
    for epoch in range(1, Q_EPOCHS + 1):
        total = 0.0
        count = 0
        for positions in _batches(len(fit_indices), seed, epoch):
            samples = z[fit_indices[positions]].to(device)
            prediction = encoder(causal_history(samples))
            loss = F.mse_loss(prediction, samples)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(samples)
            count += len(samples)
        rows.append({"seed": seed, "mode": "q", "epoch": epoch, "train_loss": total / count})
    encoder.eval().requires_grad_(False)
    return rows


def _fit_classifier(
    probe: ProbeClassifier,
    z: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int | str]]:
    probe.to(device).train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in probe.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    rows = []
    for epoch in range(1, HEAD_EPOCHS + 1):
        total_loss = 0.0
        total_correct = 0
        for positions in _batches(len(z), seed, epoch):
            samples = z[positions].to(device)
            labels = targets[positions].to(device)
            logits = probe(samples)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(samples)
            total_correct += int((logits.argmax(1) == labels).sum().item())
        rows.append(
            {
                "seed": seed,
                "mode": probe.mode,
                "epoch": epoch,
                "train_loss": total_loss / len(z),
                "train_accuracy": total_correct / len(z),
            }
        )
    probe.eval()
    return rows


@torch.no_grad()
def _q_skill(encoder, z, endpoints, device) -> dict[str, dict[str, float]]:
    channels = z.shape[-1]
    aggregates = {
        name: {
            "errors": [0.0, 0.0, 0.0],
            "elements": 0,
            "rows": 0,
            "pred_sum": torch.zeros(channels, dtype=torch.float64),
            "target_sum": torch.zeros(channels, dtype=torch.float64),
            "pred_square": torch.zeros(channels, dtype=torch.float64),
            "target_square": torch.zeros(channels, dtype=torch.float64),
        }
        for name in ("all", "active", "tail")
    }
    for start in range(0, len(z), BATCH_SIZE):
        samples = z[start : start + BATCH_SIZE].to(device)
        prediction = encoder(causal_history(samples))
        persistence = torch.zeros_like(samples)
        persistence[:, 1:] = samples[:, :-1]
        recent = causal_history(samples).reshape(*samples.shape[:2], 4, samples.shape[2]).mean(2)
        time = torch.arange(samples.shape[1], device=device)[None, :]
        active = time < endpoints[start : start + len(samples)].to(device)[:, None]
        for name, mask in (("all", torch.ones_like(active)), ("active", active), ("tail", ~active)):
            if not bool(mask.any()):
                continue
            pred_rows, target_rows = prediction[mask], samples[mask]
            persistent_rows, recent_rows = persistence[mask], recent[mask]
            stat = aggregates[name]
            stat["errors"][0] += float((pred_rows - target_rows).square().sum().item())
            stat["errors"][1] += float((persistent_rows - target_rows).square().sum().item())
            stat["errors"][2] += float((recent_rows - target_rows).square().sum().item())
            stat["elements"] += target_rows.numel()
            stat["rows"] += len(target_rows)
            stat["pred_sum"] += pred_rows.sum(0).double().cpu()
            stat["target_sum"] += target_rows.sum(0).double().cpu()
            stat["pred_square"] += pred_rows.square().sum(0).double().cpu()
            stat["target_square"] += target_rows.square().sum(0).double().cpu()
    report = {}
    for name, stat in aggregates.items():
        error, persistence, recent = stat["errors"]
        elements = stat["elements"]
        if elements == 0:
            continue
        pred_square = float(stat["pred_square"].sum().item())
        target_square = float(stat["target_square"].sum().item())
        pred_variance = (
            stat["pred_square"] / stat["rows"]
            - (stat["pred_sum"] / stat["rows"]).square()
        ).clamp_min(0).mean()
        target_variance = (
            stat["target_square"] / stat["rows"]
            - (stat["target_sum"] / stat["rows"]).square()
        ).clamp_min(0).mean()
        report[name] = {
            "mse": error / elements,
            "persistence_mse": persistence / elements,
            "recent_mean_mse": recent / elements,
            "skill_vs_persistence": 1.0 - error / max(persistence, 1e-12),
            "skill_vs_recent_mean": 1.0 - error / max(recent, 1e-12),
            "prediction_rms_over_target_rms": math.sqrt(pred_square / max(target_square, 1e-12)),
            "prediction_std_over_target_std": math.sqrt(
                float(pred_variance / target_variance.clamp_min(1e-12))
            ),
            "elements": elements,
        }
    return report


def _initial_probes(classes: int, seed: int) -> dict[str, ProbeClassifier]:
    seed_everything(seed, True)
    encoder = HistoryEncoder()
    joint_head = StepHead(256, 256, classes)
    state_head = StepHead(128, 256, classes)
    probes = {
        mode: ProbeClassifier(mode, classes, encoder=copy.deepcopy(encoder), head=copy.deepcopy(joint_head))
        for mode in PRIMARY_MODES
    }
    probes["linear_refit"] = ProbeClassifier("linear_refit", classes)
    probes["a"] = ProbeClassifier("a", classes, head=copy.deepcopy(state_head))
    probes["b_flat"] = ProbeClassifier("b_flat", classes)
    probes["r_q"] = ProbeClassifier(
        "r_q", classes, encoder=copy.deepcopy(encoder), head=copy.deepcopy(state_head)
    )
    probes["r_p"] = ProbeClassifier("r_p", classes, head=copy.deepcopy(state_head))
    return probes


def _parameter_counts(probes: dict[str, ProbeClassifier]) -> dict[str, dict[str, int]]:
    return {
        mode: {
            "total": sum(p.numel() for p in probe.parameters()),
            "classification_trainable": sum(p.numel() for p in probe.parameters() if p.requires_grad),
        }
        for mode, probe in probes.items()
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_innovation_probe(config: dict[str, Any], checkpoint_path: str | Path, output_dir: str | Path):
    """Train all arms on frozen D features; never open development validation."""

    config = _probe_config(config)
    output = ensure_dir(output_dir)
    fit_file = Path(output) / "fit.pt"
    if fit_file.exists():
        raise FileExistsError(f"Fit artifact already exists: {fit_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEEDS[0], True)
    bundle = build_dataset_bundle(config)
    if len(bundle.classes) != 100:
        raise ValueError("The DVS-Lip probe requires the 100-class development split.")
    model = build_model(config["model"], len(bundle.classes)).to(device)
    checkpoint = load_model_state(checkpoint_path, model, device)
    if int(checkpoint["config"]["experiment"]["seed"]) != 42 or checkpoint["config"]["experiment"]["name"] != "dvslip_predictive_dynamic_tcap":
        raise ValueError("The supplied checkpoint is not D seed 42.")
    model.eval().requires_grad_(False)
    train = _features_for_partition(model, bundle.train, config, device)
    if train["z"].shape[1:] != (40, 128):
        raise ValueError(f"Expected D features [sample,40,128], found {tuple(train['z'].shape)}")
    mean, std = _normalization(train["z"])
    z = (train["z"] - mean) / std
    rows = []
    internal = {}
    states = {}
    counts = None
    for seed in SEEDS:
        probes = _initial_probes(len(bundle.classes), seed)
        if counts is None:
            counts = _parameter_counts(probes)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(z), generator=generator)
        split = int(len(z) * INTERNAL_Q_FIT_FRACTION)
        fit_indices, holdout_indices = order[:split], order[split:]
        q = probes["c"].encoder
        assert q is not None
        q.requires_grad_(True)
        rows.extend(_fit_q(q, z, fit_indices, device, seed))
        internal[str(seed)] = _q_skill(q, z[holdout_indices], train["endpoint"][holdout_indices], device)
        probes["r_q"].encoder.load_state_dict(q.state_dict())
        # B and B_rand keep the exact same initial encoder and head weights as C.
        for mode in ALL_MODES:
            rows.extend(_fit_classifier(probes[mode], z, train["target"], device, seed))
        states[str(seed)] = {
            mode: {name: value.detach().cpu() for name, value in probe.state_dict().items()}
            for mode, probe in probes.items()
        }
    payload = {
        "schema_version": 1,
        "config_source": config.get("_source_path"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_score": float(checkpoint["score"]),
        "git_commit": git_commit(),
        "seeds": SEEDS,
        "q_epochs": Q_EPOCHS,
        "head_epochs": HEAD_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "normalization": {"mean": mean, "std": std},
        "q_internal_skill_by_seed": internal,
        "probe_config": {
            key: config[key] for key in ("dataset", "representation", "model") if key in config
        },
        "split_manifest_sha256": sha256_file(config["dataset"]["split_manifest"]),
        "num_classes": len(bundle.classes),
        "states": states,
    }
    torch.save(payload, fit_file)
    _write_rows(Path(output) / "training_curves.csv", rows)
    q_qualified_seeds = sum(
        internal[str(seed)]["active"]["skill_vs_persistence"] > 0
        and internal[str(seed)]["active"]["skill_vs_recent_mean"] > 0
        and internal[str(seed)]["active"]["prediction_std_over_target_std"] >= 0.1
        for seed in SEEDS
    )
    report = {
        "schema_version": 1,
        "fit_artifact": str(fit_file.resolve()),
        "fit_artifact_sha256": sha256_file(fit_file),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "development_train_samples": len(z),
        "q_internal_holdout_fraction": 1.0 - INTERNAL_Q_FIT_FRACTION,
        "q_internal_skill_by_seed": internal,
        "q_internal_qualified_seeds": q_qualified_seeds,
        "q_internal_gate_passed": q_qualified_seeds >= 4,
        "parameter_counts": counts,
        "training_budget": {
            "q_epochs": Q_EPOCHS, "head_epochs": HEAD_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        },
        "development_validation_opened": False,
        "official_test_used": False,
    }
    write_json(report, Path(output) / "fit_report.json")
    return report


@torch.no_grad()
def _predict_logits(probe: ProbeClassifier, z: torch.Tensor, device: torch.device, steps: int):
    chunks = []
    for start in range(0, len(z), BATCH_SIZE):
        chunks.append(probe(z[start : start + BATCH_SIZE, :steps].to(device)).cpu())
    return torch.cat(chunks)


def _scores(target: np.ndarray, predicted: np.ndarray, classes: int) -> dict[str, float]:
    matrix = np.bincount(target * classes + predicted, minlength=classes**2).reshape(
        classes, classes
    )
    true_positive = np.diag(matrix).astype(np.float64)
    actual = matrix.sum(1)
    predicted_count = matrix.sum(0)
    precision = np.divide(true_positive, predicted_count, out=np.zeros(classes), where=predicted_count > 0)
    recall = np.divide(true_positive, actual, out=np.zeros(classes), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=precision + recall > 0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(true_positive.sum() / len(target))}


def _mcnemar_exact(target: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left_only = int(np.sum((left == target) & (right != target)))
    right_only = int(np.sum((left != target) & (right == target)))
    discordant = left_only + right_only
    if discordant == 0:
        return {"left_only_correct": 0, "right_only_correct": 0, "two_sided_p": 1.0}
    tail = min(left_only, right_only)
    log_probabilities = [
        math.lgamma(discordant + 1) - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1) - discordant * math.log(2)
        for k in range(tail + 1)
    ]
    maximum = max(log_probabilities)
    probability = math.exp(maximum) * sum(math.exp(value - maximum) for value in log_probabilities)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "two_sided_p": min(1.0, 2.0 * probability),
    }


def _paired_sample_bootstrap(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    classes: int,
    seed: int,
    repetitions: int = 1000,
) -> dict[str, float | int]:
    """Stratified utterance uncertainty conditional on this pair of fitted heads."""

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(target == label) for label in range(classes)]
    deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups if len(group)])
        deltas[index] = (
            _scores(target[sample], left[sample], classes)["macro_f1"]
            - _scores(target[sample], right[sample], classes)["macro_f1"]
        )
    return {
        "repetitions": repetitions,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "conditional_on_fitted_seed_pair": True,
    }


def _seed_gate(deltas: list[float]) -> dict[str, Any]:
    if len(deltas) != len(SEEDS):
        raise ValueError("The preregistered gate requires exactly five seed-pair differences.")
    mean = float(np.mean(deltas))
    standard_deviation = float(np.std(deltas, ddof=1))
    lower = mean - T_CRITICAL_DF4_95 * standard_deviation / math.sqrt(len(deltas))
    positives = sum(delta > 0 for delta in deltas)
    return {
        "delta_macro_f1_by_seed": deltas,
        "mean_delta_macro_f1": mean,
        "standard_deviation_across_seeds": standard_deviation,
        "t95_lower": lower,
        "t95_upper": mean + T_CRITICAL_DF4_95 * standard_deviation / math.sqrt(len(deltas)),
        "positive_seed_pairs": positives,
        "pass": bool(lower > 0 and positives >= 4),
        "uncertainty_axis": "five_paired_initialization_seeds",
    }


def _group_summary(config: dict[str, Any], classes: list[str], target, predicted):
    from etsr.runner import _dvslip_group_metrics

    metrics = classification_metrics(target, predicted, len(classes))
    groups = _dvslip_group_metrics(
        config, classes, torch.tensor(metrics["confusion_matrix"], dtype=torch.long)
    )
    return {
        "Acc1": groups["metrics"]["Acc1"] if groups else None,
        "Acc2": groups["metrics"]["Acc2"] if groups else None,
        "visually_confusable_pair_errors": (
            {
                key: groups["visually_confusable_pair_errors"][key]
                for key in ("paired_class_samples", "paired_class_errors", "within_pair_errors", "within_pair_share_of_paired_class_errors")
            }
            if groups else None
        ),
    }


def evaluate_innovation_probe(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    fit_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Open validation once after the complete five-seed fit artifact has been fixed."""

    config = _probe_config(config)
    output = ensure_dir(output_dir)
    report_file = Path(output) / "evaluation.json"
    opened_file = Path(output) / "validation_opened.json"
    if report_file.exists() or opened_file.exists():
        raise FileExistsError(f"Validation has already been opened in: {output}")
    fit_file = Path(fit_dir) / "fit.pt"
    fit = torch.load(fit_file, map_location="cpu", weights_only=False)
    if fit["git_commit"] != git_commit():
        raise ValueError("Probe evaluation requires the same code commit as the train-only fit.")
    if fit["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("Frozen D checkpoint differs from the fitted probe's checkpoint.")
    probe_config = {key: config[key] for key in ("dataset", "representation", "model") if key in config}
    if fit["probe_config"] != probe_config or fit["split_manifest_sha256"] != sha256_file(
        config["dataset"]["split_manifest"]
    ):
        raise ValueError("Dataset split, representation or model config differs from the fit.")
    if tuple(fit["seeds"]) != SEEDS or fit["num_classes"] != 100:
        raise ValueError("Fit artifact does not match the preregistered DVS-Lip protocol.")
    q_passes = sum(
        fit["q_internal_skill_by_seed"][str(seed)]["active"]["skill_vs_persistence"] > 0
        and fit["q_internal_skill_by_seed"][str(seed)]["active"]["skill_vs_recent_mean"] > 0
        and fit["q_internal_skill_by_seed"][str(seed)]["active"]["prediction_std_over_target_std"] >= 0.1
        for seed in SEEDS
    )
    if q_passes < 4:
        raise ValueError(
            f"Q passed the train-internal active-region skill/noncollapse gate on only {q_passes}/5 seeds; validation remains unopened."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataset_bundle(config)
    model = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint_path, model, device)
    model.eval().requires_grad_(False)
    write_json(
        {"fit_artifact_sha256": sha256_file(fit_file), "validation_opened": True},
        opened_file,
    )
    validation = _features_for_partition(model, bundle.validation, config, device)
    if validation["z"].shape[1:] != (40, 128):
        raise ValueError("Validation features do not match the preregistered [40,128] D probe.")
    target = validation["target"].numpy()
    raw_z = validation["z"]
    mean, std = fit["normalization"]["mean"], fit["normalization"]["std"]
    z = (raw_z - mean) / std
    classes = len(bundle.classes)
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    metrics: dict[str, Any] = {}

    @torch.no_grad()
    def anchor_logits(steps: int) -> torch.Tensor:
        return model.head(raw_z[:, :steps].mean(1).to(device)).cpu()

    anchor = anchor_logits(40)
    anchor_pred = anchor.argmax(1).numpy()
    metrics["d_exact"] = {
        **_scores(target, anchor_pred, classes),
        **_group_summary(config, bundle.classes, target, anchor_pred),
    }
    outputs["d_exact"] = {"full": anchor, "prefix_1000ms": anchor_logits(20), "prefix_1500ms": anchor_logits(30)}

    q_validation = {}
    for seed in SEEDS:
        key = str(seed)
        probes = _initial_probes(classes, seed)
        outputs[key] = {}
        metrics[key] = {}
        q = probes["c"].encoder
        assert q is not None
        q.load_state_dict({name.removeprefix("encoder."): value for name, value in fit["states"][key]["c"].items() if name.startswith("encoder.")})
        q.eval().requires_grad_(False).to(device)
        q_validation[key] = _q_skill(q, z, validation["endpoint"], device)
        for mode in ALL_MODES:
            probe = probes[mode].to(device)
            probe.load_state_dict(fit["states"][key][mode])
            probe.eval()
            full = _predict_logits(probe, z, device, 40)
            prefix_1000 = _predict_logits(probe, z, device, 20)
            prefix_1500 = _predict_logits(probe, z, device, 30)
            predicted = full.argmax(1).numpy()
            metrics[key][mode] = {
                **_scores(target, predicted, classes),
                **_group_summary(config, bundle.classes, target, predicted),
                "prefix_1000ms": _scores(target, prefix_1000.argmax(1).numpy(), classes),
                "prefix_1500ms": _scores(target, prefix_1500.argmax(1).numpy(), classes),
            }
            outputs[key][mode] = full
    comparisons = {}
    for control in ("b", "b_rand"):
        deltas = [
            metrics[str(seed)]["c"]["macro_f1"] - metrics[str(seed)][control]["macro_f1"]
            for seed in SEEDS
        ]
        per_seed = {}
        for seed in SEEDS:
            key = str(seed)
            c_pred = outputs[key]["c"].argmax(1).numpy()
            b_pred = outputs[key][control].argmax(1).numpy()
            per_seed[key] = {
                "delta_macro_f1": metrics[key]["c"]["macro_f1"] - metrics[key][control]["macro_f1"],
                "delta_accuracy": metrics[key]["c"]["accuracy"] - metrics[key][control]["accuracy"],
                "utterance_bootstrap": _paired_sample_bootstrap(target, c_pred, b_pred, classes, seed),
                "mcnemar_accuracy": _mcnemar_exact(target, c_pred, b_pred),
            }
        comparisons[f"c_minus_{control}"] = {"seed_gate": _seed_gate(deltas), "per_seed": per_seed}

    # Descriptive only: a five-head ensemble is not the single-head candidate governed above.
    ensembles = {}
    for mode in ALL_MODES:
        averaged = torch.stack([outputs[str(seed)][mode] for seed in SEEDS]).mean(0)
        predicted = averaged.argmax(1).numpy()
        ensembles[mode] = {
            **_scores(target, predicted, classes),
            **_group_summary(config, bundle.classes, target, predicted),
        }
    residual_comparison = {
        str(seed): (
            metrics[str(seed)]["r_q"]["macro_f1"] - metrics[str(seed)]["r_p"]["macro_f1"]
        )
        for seed in SEEDS
    }

    prediction_file = Path(output) / "validation_predictions.pt"
    torch.save({"targets": validation["target"], "indices": validation["index"], "logits": outputs}, prediction_file)
    report = {
        "schema_version": 1,
        "frozen_checkpoint_sha256": fit["checkpoint_sha256"],
        "fit_artifact_sha256": sha256_file(fit_file),
        "validation_samples": len(target),
        "official_test_used": False,
        "exact_anchor": metrics["d_exact"],
        "q_validation_skill_by_seed": q_validation,
        "head_metrics_by_seed": {key: metrics[key] for key in (str(seed) for seed in SEEDS)},
        "primary_comparisons": comparisons,
        "descriptive_five_seed_ensembles": ensembles,
        "descriptive_r_q_minus_r_p_macro_f1_by_seed": residual_comparison,
        "promotion_gate_passed": all(value["seed_gate"]["pass"] for value in comparisons.values()),
        "promotion_interpretation": "A passing frozen-feature screen authorizes one end-to-end run; it is not an end-to-end gain.",
        "validation_predictions": str(prediction_file.resolve()),
        "validation_read_count": 1,
    }
    write_json(report, report_file)
    return report
