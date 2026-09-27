"""Train-only frozen-feature screen for the stage-2 predictive bottleneck."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from etsr.data.common import build_loader
from etsr.data.events import move_encoded_input
from etsr.data.factory import build_dataset_bundle
from etsr.evaluation.predictive_diagnostic import _diagnostic_split
from etsr.models.factory import build_model
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.reproducibility import seed_everything
from etsr.training.checkpointing import load_model_state
from etsr.training.predictive import last_occupied_steps
from etsr.utils.io import ensure_dir, sha256_file, write_json


def _active_errors(
    sequence: torch.Tensor,
    endpoints: torch.Tensor,
    mixer: CausalTemporalChannelMixer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history = torch.cat((sequence.new_zeros((mixer.max_delay, *sequence.shape[1:])), sequence))
    prediction = mixer._causal_prediction(sequence, history)
    delayed = [
        history[mixer.max_delay - delay : mixer.max_delay - delay + len(sequence)]
        for delay in mixer.delays
    ]
    mask = torch.arange(len(sequence), device=sequence.device)[:, None] < endpoints[None, :]

    def active_loss(candidate: torch.Tensor) -> torch.Tensor:
        per_step = F.smooth_l1_loss(candidate, sequence, reduction="none").flatten(2).mean(2)
        return ((per_step * mask).sum(0) / mask.sum(0).clamp_min(1)).mean()

    return (
        active_loss(prediction),
        active_loss(delayed[0]),
        active_loss(torch.stack(delayed).mean(0)),
    )


def _extract_features(
    model, loader, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    mixer = dict(model.named_modules()).get("patch_embed2.down.temporal_channel_mixer")
    if not isinstance(mixer, CausalTemporalChannelMixer):
        raise ValueError("C0 must have the registered stage-2 TCAP mixer")
    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().to(device="cpu", dtype=torch.float16))

    handle = mixer.register_forward_pre_hook(capture)
    result = []
    try:
        with torch.no_grad():
            for frames, _targets, _indices in loader:
                frames = move_encoded_input(frames, device)
                endpoint = last_occupied_steps(frames).cpu()
                model(frames)
                if len(captured) != 1:
                    raise RuntimeError("Expected exactly one stage-2 feature capture per batch")
                result.append((captured.pop(), endpoint))
    finally:
        handle.remove()
    return result


@torch.no_grad()
def _score(
    mixer: CausalTemporalChannelMixer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    mixer.eval()
    sums = torch.zeros(3, dtype=torch.float64)
    samples = 0
    target_variance = 0.0
    for cached, endpoints in batches:
        sequence = cached.to(device=device, dtype=torch.float32)
        errors = _active_errors(sequence, endpoints.to(device), mixer)
        size = sequence.shape[1]
        sums += torch.tensor([float(error) for error in errors], dtype=torch.float64) * size
        mask = torch.arange(len(sequence), device=device)[:, None] < endpoints.to(device)[None, :]
        spatial_variance = sequence.flatten(2).var(dim=2, unbiased=False)
        target_variance += float(
            ((spatial_variance * mask).sum(0) / mask.sum(0).clamp_min(1)).sum()
        )
        samples += size
    prediction, persistence, delay_mean = (sums / samples).tolist()
    return {
        "prediction_loss": prediction,
        "persistence_loss": persistence,
        "delay_mean_loss": delay_mean,
        "skill_vs_persistence": 1.0 - prediction / max(persistence, 1e-12),
        "skill_vs_delay_mean": 1.0 - prediction / max(delay_mean, 1e-12),
        "target_variance": target_variance / samples,
    }


def run_bottleneck_feature_probe(
    config: dict[str, Any],
    checkpoint: str | Path,
    output: str | Path,
    *,
    fit_samples: int = 256,
    holdout_samples: int = 256,
    steps: int = 600,
) -> dict[str, Any]:
    """Fit predictor heads on frozen C0 features; never uses development validation."""

    if min(fit_samples, holdout_samples, steps) <= 0:
        raise ValueError("Probe sample counts and training steps must be positive")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, True)
    no_augmentation = copy.deepcopy(config)
    no_augmentation["augmentation"] = {"horizontal_flip_probability": 0.0}
    bundle = build_dataset_bundle(no_augmentation)
    fit_dataset, holdout_dataset = _diagnostic_split(
        bundle.train, fit_samples, holdout_samples, seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], len(bundle.classes)).to(device)
    load_model_state(checkpoint, model, device)
    model.eval().requires_grad_(False)
    fit = _extract_features(
        model, build_loader(fit_dataset, no_augmentation["dataset"], shuffle=False), device
    )
    holdout = _extract_features(
        model, build_loader(holdout_dataset, no_augmentation["dataset"], shuffle=False), device
    )
    reference = dict(model.named_modules())["patch_embed2.down.temporal_channel_mixer"]
    assert isinstance(reference, CausalTemporalChannelMixer)
    channels, delays = reference.channels, reference.delays
    candidates = [("s0_linear", None, None)] + [
        (f"k{rank}_{'linear' if hidden is None else 'joint_gelu256'}", rank, hidden)
        for rank in (32, 64, channels)
        for hidden in (None, 256)
    ]
    results: dict[str, Any] = {}
    for name, rank, hidden in candidates:
        seed_everything(seed, True)
        mixer = CausalTemporalChannelMixer(
            channels,
            delays,
            predictive_auxiliary=True,
            predictor_channel_groups=1 if rank is None else None,
            predictor_spatial_kernel_size=3,
            predictor_rank=rank,
            predictor_hidden_channels=hidden,
        ).to(device)
        trainable = [
            parameter
            for parameter_name, parameter in mixer.named_parameters()
            if parameter_name.startswith(
                ("predictor_spatial.", "predictor_projections.", "predictor_bottleneck.")
            )
        ]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        training_losses = []
        for step in range(steps):
            cached, endpoints = fit[step % len(fit)]
            sequence = cached.to(device=device, dtype=torch.float32)
            mixer.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _persistence, _mean = _active_errors(sequence, endpoints.to(device), mixer)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach()))
        results[name] = {
            "rank": rank if rank is not None else channels,
            "joint_nonlinear": hidden is not None,
            "predictor_parameters": sum(parameter.numel() for parameter in trainable),
            "previous_50_step_training_loss": (
                sum(training_losses[-100:-50]) / 50 if steps >= 100 else None
            ),
            "last_50_step_training_loss": (
                sum(training_losses[-50:]) / min(50, steps)
            ),
            "fit": _score(mixer, fit, device),
            "holdout": _score(mixer, holdout, device),
        }
    report = {
        "schema_version": 1,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "feature_source": "patch_embed2.down.temporal_channel_mixer input",
        "data_source": "development_train_only_disjoint_utterances",
        "fit_samples": fit_samples,
        "holdout_samples": holdout_samples,
        "optimizer_steps_per_candidate": steps,
        "selection_note": (
            "This probe measures frozen-feature predictability, not classification benefit. "
            "Read both causal skills and fit-holdout gaps; do not auto-select rank."
        ),
        "candidates": results,
        "official_test_used": False,
    }
    path = Path(output)
    ensure_dir(path.parent)
    write_json(report, path)
    return report
