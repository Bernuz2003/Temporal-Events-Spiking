"""Shared boundary between dataset-specific event readers and representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

EncodedInput = torch.Tensor | dict[str, torch.Tensor]


def move_encoded_input(frames: EncodedInput, device: torch.device) -> EncodedInput:
    if isinstance(frames, torch.Tensor):
        return frames.to(device, non_blocking=True)
    return {name: value.to(device, non_blocking=True) for name, value in frames.items()}


def encoded_batch_size(frames: EncodedInput) -> int:
    if isinstance(frames, torch.Tensor):
        return int(frames.shape[0])
    return int(frames["coarse"].shape[0])


def encoded_time_steps(frames: EncodedInput) -> int:
    if isinstance(frames, torch.Tensor):
        return int(frames.shape[1])
    return int(frames["coarse"].shape[1])


def slice_encoded_batch(frames: EncodedInput, stop: int) -> EncodedInput:
    if isinstance(frames, torch.Tensor):
        return frames[:stop]
    return {name: value[:stop] for name, value in frames.items()}


def select_encoded_batch(frames: EncodedInput, index: int) -> EncodedInput:
    if isinstance(frames, torch.Tensor):
        return frames[index : index + 1]
    return {name: value[index : index + 1] for name, value in frames.items()}


def slice_encoded_time(frames: EncodedInput, steps: int) -> EncodedInput:
    if isinstance(frames, torch.Tensor):
        return frames[:, :steps]
    coarse_steps = int(frames["coarse"].shape[1])
    fine_steps = int(frames["fine"].shape[1])
    if fine_steps % coarse_steps:
        raise ValueError("Fine time axis must be an integer multiple of the coarse axis.")
    ratio = fine_steps // coarse_steps
    return {"coarse": frames["coarse"][:, :steps], "fine": frames["fine"][:, : steps * ratio]}


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

    def __getitem__(self, index: int) -> tuple[EncodedInput, int, int]:
        encoded = self.encoder(self.raw_dataset[index])
        frames = encoded.tensor
        if self.horizontal_flip_probability and bool(
            torch.rand(()) < self.horizontal_flip_probability
        ):
            if isinstance(frames, torch.Tensor):
                frames = torch.flip(frames, dims=(-1,))
            else:
                frames = {name: torch.flip(value, dims=(-1,)) for name, value in frames.items()}
        for _ in range(self.temporal_mask_count):
            length = int(torch.randint(1, self.temporal_mask_max_steps + 1, ()).item())
            time_steps = self.encoder.time_steps
            start = int(torch.randint(0, time_steps - length + 1, ()).item())
            if isinstance(frames, torch.Tensor):
                frames[start : start + length] = 0
            else:
                ratio = frames["fine"].shape[0] // frames["coarse"].shape[0]
                frames["coarse"][start : start + length] = 0
                frames["fine"][start * ratio : (start + length) * ratio] = 0
        for _ in range(self.spatial_erasing_count):
            size = int(torch.randint(1, self.spatial_erasing_max_pixels + 1, ()).item())
            top = int(torch.randint(0, self.encoder.height - size + 1, ()).item())
            left = int(torch.randint(0, self.encoder.width - size + 1, ()).item())
            if isinstance(frames, torch.Tensor):
                frames[..., top : top + size, left : left + size] = 0
            else:
                frames["coarse"][..., top : top + size, left : left + size] = 0
                stride = self.encoder.fine_spatial_stride
                fine_top, fine_left = top // stride, left // stride
                fine_bottom = (top + size + stride - 1) // stride
                fine_right = (left + size + stride - 1) // stride
                frames["fine"][..., fine_top:fine_bottom, fine_left:fine_right] = 0
        if isinstance(frames, torch.Tensor):
            output: EncodedInput = frames.to(torch.float32)
        else:
            output = {name: value.to(torch.float32) for name, value in frames.items()}
        return output, encoded.target, int(index)
