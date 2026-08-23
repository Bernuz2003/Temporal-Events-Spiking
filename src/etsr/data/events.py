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
    ) -> None:
        if (raw_dataset.height, raw_dataset.width) != (encoder.height, encoder.width):
            raise ValueError("Dataset and encoder sensor sizes differ.")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")

        self.raw_dataset = raw_dataset
        self.encoder = encoder
        self.horizontal_flip_probability = float(horizontal_flip_probability)
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
        return frames.to(torch.float32), encoded.target, int(index)
