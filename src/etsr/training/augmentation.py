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
    """Paper-aligned EventMix with an explicit diagonal-GMM reconstruction.

    The paper fixes probability, beta area distribution, component count and distance-based
    labels, but does not fully specify how random GMM covariance or pooling resolution are drawn.
    Those choices are therefore explicit configuration values in this implementation.
    """

    def __init__(
        self,
        *,
        probability: float,
        beta: float,
        components: int,
        gmm_scale_min: float,
        gmm_scale_max: float,
        mask_grid_size: int,
        distance_spatial_pool: int,
    ) -> None:
        self.probability = float(probability)
        self.beta = float(beta)
        self.components = int(components)
        self.gmm_scale_min = float(gmm_scale_min)
        self.gmm_scale_max = float(gmm_scale_max)
        self.mask_grid_size = int(mask_grid_size)
        self.distance_spatial_pool = int(distance_spatial_pool)
        self._grid_cache: dict[tuple[int, int, str, int | None], tuple[torch.Tensor, ...]] = {}

    def _coordinate_grid(
        self, time_steps: int, grid_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (time_steps, grid_size, device.type, device.index)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        temporal = torch.linspace(0.0, 1.0, time_steps, device=device).view(1, 1, -1, 1, 1)
        vertical = torch.linspace(0.0, 1.0, grid_size, device=device).view(1, 1, 1, -1, 1)
        horizontal = torch.linspace(0.0, 1.0, grid_size, device=device).view(1, 1, 1, 1, -1)
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
        grid_size = min(self.mask_grid_size, height, width)
        temporal, vertical, horizontal = self._coordinate_grid(time_steps, grid_size, device)
        means = torch.rand(samples, self.components, 3, device=device)
        scales = self.gmm_scale_min + (
            self.gmm_scale_max - self.gmm_scale_min
        ) * torch.rand(samples, self.components, 3, device=device)
        weights = torch.rand(samples, self.components, device=device)
        weights = weights / weights.sum(dim=1, keepdim=True)

        squared_distance = (
            ((temporal - means[:, :, 0, None, None, None]) / scales[:, :, 0, None, None, None])
            .square()
            .add(
                ((vertical - means[:, :, 1, None, None, None]) / scales[:, :, 1, None, None, None])
                .square()
            )
            .add(
                ((horizontal - means[:, :, 2, None, None, None]) / scales[:, :, 2, None, None, None])
                .square()
            )
        )
        component_log_scores = (
            -0.5 * squared_distance
            - scales.prod(dim=2).log()[:, :, None, None, None]
            + weights.log()[:, :, None, None, None]
        )
        scores = torch.logsumexp(component_log_scores, dim=1)

        concentration = torch.full((samples,), self.beta, device=device)
        fractions = torch.distributions.Beta(concentration, concentration).sample()
        flattened = scores.flatten(1)
        ordered_indices = flattened.argsort(dim=1)
        selected_counts = (fractions * flattened.shape[1]).round().long()
        selected_counts.clamp_(1, flattened.shape[1] - 1)
        ranked_mask = torch.arange(flattened.shape[1], device=device)[None] < selected_counts[:, None]
        low_resolution = torch.zeros_like(flattened, dtype=torch.bool)
        low_resolution.scatter_(1, ordered_indices, ranked_mask)
        low_resolution = low_resolution.reshape_as(scores)
        return functional.interpolate(
            low_resolution[:, None].to(torch.float32),
            size=(time_steps, height, width),
            mode="nearest",
        ).squeeze(1).bool()

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
        pool = min(self.distance_spatial_pool, primary.shape[-2], primary.shape[-1])

        def spatial_average(frames: torch.Tensor) -> torch.Tensor:
            channels_first = frames.permute(0, 2, 1, 3, 4)
            return functional.avg_pool3d(
                channels_first,
                kernel_size=(1, pool, pool),
                stride=(1, pool, pool),
                ceil_mode=True,
            )

        pooled_primary = spatial_average(primary)
        pooled_secondary = spatial_average(secondary)
        pooled_mixed = spatial_average(mixed)
        dimensions = tuple(range(1, pooled_mixed.ndim))
        distance_primary = (pooled_primary - pooled_mixed).square().mean(dim=dimensions)
        distance_secondary = (pooled_secondary - pooled_mixed).square().mean(dim=dimensions)
        numerator = distance_secondary.square()
        denominator = distance_primary.square() + numerator
        fallback = torch.full_like(denominator, 0.5)
        return torch.where(denominator > torch.finfo(denominator.dtype).eps, numerator / denominator, fallback)

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
            mask = self._gmm_mask(
                int(selected.numel()), time_steps, height, width, frames.device
            )[:, :, None]
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
        gmm_scale_min=float(config["event_mix_gmm_scale_min"]),
        gmm_scale_max=float(config["event_mix_gmm_scale_max"]),
        mask_grid_size=int(config["event_mix_mask_grid_size"]),
        distance_spatial_pool=int(config["event_mix_distance_spatial_pool"]),
    )
