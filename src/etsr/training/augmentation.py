"""Batch-level augmentation for encoded event tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from etsr.data.events import EncodedInput


@dataclass(frozen=True)
class MixedBatch:
    """One mixed input batch and the two targets contributing to every sample."""

    frames: EncodedInput
    secondary_targets: torch.Tensor
    primary_weights: torch.Tensor


class EventMix:
    """EventMix using the paper defaults and the authors' public GMM realization.

    The public BrainCog code supplies the details omitted by the paper: component means are
    sampled on the encoded grid, diagonal scales follow ``0.5 * max(U(0, 1), 0.1)`` per axis,
    and unnormalised Gaussian kernels define a full-resolution mask.  We retain the paper's
    relative-distance labels, which outperform count- and area-based labels in its ablation.
    """

    def __init__(
        self,
        *,
        probability: float,
        beta: float,
        components: int,
    ) -> None:
        self.probability = float(probability)
        self.beta = float(beta)
        self.components = int(components)
        self._grid_cache: dict[tuple[int, int, int, str, int | None], tuple[torch.Tensor, ...]] = {}

    def _coordinate_grid(
        self, time_steps: int, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (time_steps, height, width, device.type, device.index)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        temporal = torch.arange(time_steps, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        vertical = torch.arange(height, device=device, dtype=torch.float32).view(1, 1, -1, 1)
        horizontal = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, 1, -1)
        self._grid_cache[key] = (temporal, vertical, horizontal)
        return temporal, vertical, horizontal

    def _gmm_mask(
        self,
        samples: int,
        time_steps: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        temporal, vertical, horizontal = self._coordinate_grid(time_steps, height, width, device)
        axis_sizes = torch.tensor([time_steps, height, width], device=device, dtype=torch.float32)
        means = torch.stack(
            [
                torch.randint(time_steps, (samples, self.components), device=device),
                torch.randint(height, (samples, self.components), device=device),
                torch.randint(width, (samples, self.components), device=device),
            ],
            dim=-1,
        ).to(torch.float32)
        scales = (
            torch.rand(samples, self.components, 3, device=device).clamp_min(0.1) * axis_sizes * 0.5
        )
        weights = torch.rand(samples, self.components, device=device)

        # Accumulate one component at a time.  This preserves the full T x H x W mask without
        # materialising an additional samples x components x T x H x W tensor.
        scores = torch.full(
            (samples, time_steps, height, width),
            -torch.inf,
            device=device,
            dtype=torch.float32,
        )
        for component in range(self.components):
            squared_distance = (
                (
                    (temporal - means[:, component, 0, None, None, None])
                    / scales[:, component, 0, None, None, None]
                )
                .square()
                .add(
                    (
                        (vertical - means[:, component, 1, None, None, None])
                        / scales[:, component, 1, None, None, None]
                    ).square()
                )
                .add(
                    (
                        (horizontal - means[:, component, 2, None, None, None])
                        / scales[:, component, 2, None, None, None]
                    ).square()
                )
            )
            component_scores = (
                -0.5 * squared_distance
                + weights[:, component, None, None, None]
                .clamp_min(torch.finfo(torch.float32).tiny)
                .log()
            )
            scores = torch.logaddexp(scores, component_scores)

        concentration = torch.full((samples,), self.beta, device=device)
        fractions = torch.distributions.Beta(concentration, concentration).sample()
        flattened = scores.flatten(1)
        primary_counts = (fractions * flattened.shape[1]).round().long()
        primary_counts.clamp_(1, flattened.shape[1] - 1)
        masks = []
        for sample_index, primary_count in enumerate(primary_counts.tolist()):
            # CUDA kthvalue is incompatible with the repository's deterministic mode. topk
            # still avoids a complete argsort and selects the requested area exactly.
            selected = flattened[sample_index].topk(primary_count, sorted=False).indices
            mask = torch.zeros_like(flattened[sample_index], dtype=torch.bool)
            mask.scatter_(0, selected, True)
            masks.append(mask)
        return torch.stack(masks).reshape_as(scores)

    @staticmethod
    def _partner_indices(batch_size: int, device: torch.device) -> torch.Tensor:
        order = torch.randperm(batch_size, device=device)
        partners = torch.empty_like(order)
        partners[order] = order.roll(1)
        return partners

    def _relative_distance_weights(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor,
        mixed: torch.Tensor,
    ) -> torch.Tensor:
        def spatial_average(frames: torch.Tensor) -> torch.Tensor:
            batch_size, time_steps, channels, height, width = frames.shape
            flattened = frames.reshape(batch_size * time_steps, channels, height, width)
            averaged = functional.avg_pool2d(flattened, kernel_size=3, stride=1, padding=1)
            return averaged.reshape(batch_size, time_steps, channels, height, width)

        pooled_primary = spatial_average(primary)
        pooled_secondary = spatial_average(secondary)
        pooled_mixed = spatial_average(mixed)
        dimensions = tuple(range(1, pooled_mixed.ndim))
        distance_primary = (pooled_primary - pooled_mixed).square().mean(dim=dimensions)
        distance_secondary = (pooled_secondary - pooled_mixed).square().mean(dim=dimensions)
        numerator = distance_secondary.square()
        denominator = distance_primary.square() + numerator
        fallback = torch.full_like(denominator, 0.5)
        return torch.where(
            denominator > torch.finfo(denominator.dtype).eps, numerator / denominator, fallback
        )

    @torch.no_grad()
    def __call__(self, frames: EncodedInput, targets: torch.Tensor) -> MixedBatch:
        if not isinstance(frames, torch.Tensor):
            raise ValueError("EventMix currently requires one encoded tensor stream")
        batch_size, time_steps, _channels, height, width = frames.shape
        if batch_size < 2:
            return MixedBatch(frames, targets, torch.ones(batch_size, device=frames.device))

        partners = self._partner_indices(batch_size, frames.device)
        secondary = frames.index_select(0, partners)
        apply = torch.rand(batch_size, device=frames.device) < self.probability
        mixed = frames.clone()
        primary_weights = torch.ones(batch_size, device=frames.device, dtype=frames.dtype)
        selected = apply.nonzero(as_tuple=False).flatten()
        if selected.numel():
            mask = self._gmm_mask(int(selected.numel()), time_steps, height, width, frames.device)[
                :, :, None
            ]
            selected_primary = frames.index_select(0, selected)
            selected_secondary = secondary.index_select(0, selected)
            selected_mixed = torch.where(mask, selected_primary, selected_secondary)
            mixed.index_copy_(0, selected, selected_mixed)
            primary_weights.index_copy_(
                0,
                selected,
                self._relative_distance_weights(
                    selected_primary, selected_secondary, selected_mixed
                ).to(primary_weights.dtype),
            )
        return MixedBatch(mixed, targets.index_select(0, partners), primary_weights)


def build_batch_augmentation(config: dict[str, Any]) -> EventMix | None:
    """Build the configured batch transform once per training run."""

    probability = float(config.get("event_mix_probability", 0.0))
    if probability == 0.0:
        return None
    return EventMix(
        probability=probability,
        beta=float(config["event_mix_beta"]),
        components=int(config["event_mix_components"]),
    )
