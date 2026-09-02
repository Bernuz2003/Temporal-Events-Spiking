"""Shared boundary between dataset-specific event readers and representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EventSample:
    """One validated event sample in physical microseconds."""

    x: np.ndarray
    y: np.ndarray
    t_us: np.ndarray
    polarity: np.ndarray
    target: int
    sample_id: str
    speaker_id: str | int | None
    duration_us: int
    metadata: dict[str, Any]


class EncodedEventDataset(Dataset):
    """Encode a raw event dataset on access for the dense training engine."""

    def __init__(
        self,
        raw_dataset: Dataset,
        encoder: Any,
        horizontal_flip_probability: float = 0.0,
        temporal_mask_count: int = 0,
        temporal_mask_max_steps: int = 0,
        spatial_erasing_count: int = 0,
        spatial_erasing_max_pixels: int = 0,
    ) -> None:
        if (raw_dataset.height, raw_dataset.width) != (encoder.height, encoder.width):
            raise ValueError("Dataset and encoder sensor sizes differ.")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")
        for name, count, maximum in (
            ("temporal mask", temporal_mask_count, temporal_mask_max_steps),
            ("spatial erasing", spatial_erasing_count, spatial_erasing_max_pixels),
        ):
            if type(count) is not int or type(maximum) is not int or count < 0 or maximum < 0:
                raise ValueError(f"{name} count and maximum must be non-negative integers.")
            if (count == 0) != (maximum == 0):
                raise ValueError(f"{name} count and maximum must be enabled together.")
        if temporal_mask_max_steps > encoder.time_steps:
            raise ValueError("temporal_mask_max_steps cannot exceed the encoded time steps.")
        if spatial_erasing_max_pixels > min(encoder.height, encoder.width):
            raise ValueError("spatial_erasing_max_pixels cannot exceed the sensor dimensions.")

        self.raw_dataset = raw_dataset
        self.encoder = encoder
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.temporal_mask_count = temporal_mask_count
        self.temporal_mask_max_steps = temporal_mask_max_steps
        self.spatial_erasing_count = spatial_erasing_count
        self.spatial_erasing_max_pixels = spatial_erasing_max_pixels
        self.classes = raw_dataset.classes
        self.class_to_idx = raw_dataset.class_to_idx
        self.sample_ids = raw_dataset.sample_ids
        self.targets = raw_dataset.targets
        self.runtime_metadata = getattr(raw_dataset, "runtime_metadata", {})
        self.representation_metadata = {
            "name": encoder.name,
            "parameters": encoder.parameters,
            "state_profile": encoder.state_profile,
        }

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        encoded = self.encoder(self.raw_dataset[index])
        frames = encoded.tensor
        if self.horizontal_flip_probability and bool(
            torch.rand(()) < self.horizontal_flip_probability
        ):
            frames = torch.flip(frames, dims=(-1,))
        for _ in range(self.temporal_mask_count):
            length = int(torch.randint(1, self.temporal_mask_max_steps + 1, ()).item())
            start = int(torch.randint(0, frames.shape[0] - length + 1, ()).item())
            frames[start : start + length] = 0
        for _ in range(self.spatial_erasing_count):
            size = int(torch.randint(1, self.spatial_erasing_max_pixels + 1, ()).item())
            top = int(torch.randint(0, frames.shape[-2] - size + 1, ()).item())
            left = int(torch.randint(0, frames.shape[-1] - size + 1, ()).item())
            frames[..., top : top + size, left : left + size] = 0
        return frames.to(torch.float32), encoded.target, int(index)
