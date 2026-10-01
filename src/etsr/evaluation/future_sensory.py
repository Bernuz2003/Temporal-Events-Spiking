"""Causal, train-only future-event targets and losses for the sensory screen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from etsr.data.events import EncodedEventDataset, EventSample
from etsr.models.factory import build_model

BIN_US = 50_000
HORIZON_US = 100_000
STEPS = 40
GRID = 16
MODES = ("fepf2", "voxel4")


def spatial_e0_count(frames: torch.Tensor) -> torch.Tensor:
    """Sum each E0 polarity frame into nonoverlapping 8×8 target-grid cells."""
    batch, time, polarity = frames.shape[:3]
    return F.avg_pool2d(frames.reshape(-1, 1, 128, 128), 8, stride=8).reshape(
        batch, time, polarity, GRID, GRID
    ) * 64


def target_field(sample: EventSample, mode: str, *, past: bool = False) -> np.ndarray:
    """Return [40,2,J,16,16]; no clipping, endpoint or label enters the target."""
    if mode not in MODES:
        raise ValueError(f"Unsupported sensory target: {mode}")
    components = 3 if mode == "fepf2" else 4
    time = np.asarray(sample.t_us, dtype=np.int64)
    polarity = np.asarray(sample.polarity, dtype=np.int64)
    cell = (np.asarray(sample.y, dtype=np.int64) // 8) * GRID + (
        np.asarray(sample.x, dtype=np.int64) // 8
    )
    event_bin = time // BIN_US
    result = np.zeros((STEPS, 2, components, GRID, GRID), dtype=np.float32)
    flat = result.reshape(-1)
    for relative in (0, 1):
        # A future event falls into the two overlapping 100 ms windows ending
        # one or two E0 bins later. The matched-past target uses those same
        # events at the corresponding later cutoffs.
        cutoff_bin = event_bin + (1 + relative if past else -relative)
        valid = (cutoff_bin >= 1) & (cutoff_bin <= 38)
        if not np.any(valid):
            continue
        indices = np.flatnonzero(valid)
        cutoff = cutoff_bin[indices]
        start = (cutoff - 2 if past else cutoff) * BIN_US
        phase = (time[indices] - start).astype(np.float64) / HORIZON_US
        if np.any((phase < 0) | (phase >= 1)):
            raise AssertionError("Event assigned outside its half-open target interval")
        prefix = ((cutoff - 1) * 2 + polarity[indices]) * components * GRID * GRID
        prefix += cell[indices]
        if mode == "fepf2":
            weights = ((1 - phase) ** 2, 2 * phase * (1 - phase), phase**2)
            for component, weight in enumerate(weights):
                np.add.at(flat, prefix + component * GRID * GRID, weight)
        else:
            component = np.minimum((phase * 4).astype(np.int64), 3)
            np.add.at(flat, prefix + component * GRID * GRID, 1)
    return result


class SensoryDataset(Dataset):
    def __init__(self, encoded: EncodedEventDataset, indices: list[int], mode: str, past: bool):
        if any(
            getattr(encoded, name)
            for name in (
                "horizontal_flip_probability",
                "temporal_mask_count",
                "spatial_erasing_count",
            )
        ):
            raise ValueError("The screen requires unaugmented E0 inputs")
        self.encoded = encoded
        self.indices = tuple(indices)
        self.mode = mode
        self.past = past

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        sample = self.encoded.raw_dataset[index]
        frames = self.encoded.encoder(sample).tensor.to(torch.float32)
        field = torch.from_numpy(target_field(sample, self.mode, past=self.past))
        # Model output at index c-1 has seen E0 frames [0,c); only cutoffs
        # inside the observed utterance and with a complete 100 ms horizon count.
        cutoffs = torch.arange(1, STEPS + 1, dtype=torch.int64) * BIN_US
        valid = (cutoffs < int(sample.t_us[-1])) & (cutoffs + HORIZON_US <= STEPS * BIN_US)
        return (
            frames, field, valid, int(sample.target), int(index),
            int(sample.t_us[-1]), int(sample.duration_us),
        )


def replace_temporal_batchnorm(module: nn.Module) -> int:
    """GroupNorm on each flattened (time,batch) row; no inter-time leakage."""
    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.BatchNorm1d | nn.BatchNorm2d):
            channels = child.num_features
            if channels == 8:
                groups = 2
            elif channels == 16:
                groups = 4
            elif channels >= 32:
                groups = 8
            else:
                raise ValueError(f"No registered causal GroupNorm rule for {channels} channels")
            if channels % groups:
                raise ValueError(f"No registered causal GroupNorm rule for {channels} channels")
            setattr(module, name, nn.GroupNorm(groups, channels, affine=True))
            replaced += 1
        else:
            replaced += replace_temporal_batchnorm(child)
    return replaced


def build_causal_d(config: dict, classes: int) -> nn.Module:
    model_config = config["model"]
    if (
        config["representation"]["name"] != "count_frames_e0"
        or tuple(model_config["temporal_channel_mixer_delays"]) != (1, 2, 4, 8)
        or not model_config["temporal_channel_mixer_dynamic_routing"]
        or tuple(model_config["temporal_channel_mixer_routing_stages"]) != (2,)
        or model_config["temporal_channel_mixer_routing_parameterization"] != "amplitude_allocation"
        or int(model_config.get("temporal_channel_mixer_router_groups", 1)) != 1
    ):
        raise ValueError("Future sensory screen requires the frozen D G=1 E0 topology")
    model = build_model(model_config, classes)
    replaced = replace_temporal_batchnorm(model)
    if replaced == 0 or any(isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d) for m in model.modules()):
        raise RuntimeError("Failed to remove all time-mixing BatchNorm modules")
    return model


class FutureDecoder(nn.Module):
    """Stage2-only, per-step learned 8-to-16 upsampling and factorized count/time output."""

    def __init__(self, components: int):
        super().__init__()
        self.components = components
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 2 * (components + 1), 1),
        )

    def forward(self, stage2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps, batch = stage2.shape[:2]
        dense = self.net(stage2.flatten(0, 1)).reshape(
            time_steps, batch, 2, self.components + 1, GRID, GRID
        )
        dense = dense.permute(1, 0, 2, 3, 4, 5)
        count = F.softplus(dense[:, :, :, 0]) + 1e-6
        q = dense[:, :, :, 1:].softmax(dim=3)
        return count, q


@dataclass
class LossParts:
    count: torch.Tensor  # [B], utterance means
    timing: torch.Tensor  # [B], utterance means


def balanced_count_weights(count: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Exact cell weights of the balanced count loss, including utterance averaging."""
    positive = count > 0
    spatial_dims = (2, 3, 4)
    pos_count = positive.sum(dim=spatial_dims).clamp_min(1).to(count.dtype)
    empty_count = (~positive).sum(dim=spatial_dims).clamp_min(1).to(count.dtype)
    groups = (positive.any(dim=spatial_dims).to(count.dtype)
              + (~positive).any(dim=spatial_dims).to(count.dtype))
    weights = torch.where(
        positive,
        pos_count[:, :, None, None, None].reciprocal(),
        empty_count[:, :, None, None, None].reciprocal(),
    ) / groups[:, :, None, None, None]
    return weights * valid[:, :, None, None, None] / valid.sum(dim=1).clamp_min(1)[:, None, None, None, None]


def sensory_loss_parts(
    count_hat: torch.Tensor, q_hat: torch.Tensor, field: torch.Tensor, valid: torch.Tensor
) -> LossParts:
    """Balanced positive/empty cells, then valid cutoffs, then utterances."""
    target_count = field.sum(dim=3)
    positive = target_count > 0
    deviance = 2 * (
        count_hat - target_count
        + torch.where(
            positive,
            target_count * (target_count.clamp_min(1e-9).log() - count_hat.log()),
            0.0,
        )
    )
    q_target = field / target_count.unsqueeze(3).clamp_min(1e-9)
    divergence = torch.where(
        field > 0,
        q_target * (q_target.clamp_min(1e-9).log() - q_hat.clamp_min(1e-9).log()),
        0.0,
    ).sum(dim=3)
    spatial_dims = (2, 3, 4)
    pos_count = positive.sum(dim=spatial_dims)
    kl = (divergence * positive).sum(dim=spatial_dims) / pos_count.clamp_min(1)
    if not bool(valid.any(dim=1).all()):
        raise ValueError("Every utterance needs at least one valid active cutoff")
    count_per_utterance = (deviance * balanced_count_weights(target_count, valid)).sum(
        dim=(1, 2, 3, 4)
    )
    # A cutoff without occupied cells contributes only count loss. Its zero
    # KL must not dilute timing on other valid/occupied cutoffs.
    timing_valid = valid & (pos_count > 0)
    timing_per_utterance = (kl * timing_valid).sum(dim=1) / timing_valid.sum(dim=1).clamp_min(1)
    return LossParts(count_per_utterance, timing_per_utterance)


def target_to_count_q(field: torch.Tensor, q_fallback: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = field.sum(dim=3)
    q = field / count.unsqueeze(3).clamp_min(1e-9)
    q = torch.where(count.unsqueeze(3) > 0, q, q_fallback)
    return count, q
