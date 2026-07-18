from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    """Adds a stable sample index and standardizes frames to [T, C, H, W]."""

    def __init__(self, dataset: Dataset, input_clip: float | None = None, input_scale: str = "none"):
        self.dataset = dataset
        self.input_clip = input_clip
        self.input_scale = input_scale
        self.classes = getattr(dataset, "classes", None)
        self.class_to_idx = getattr(dataset, "class_to_idx", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        frames, target = self.dataset[index]
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        frames = torch.as_tensor(frames, dtype=torch.float32)

        if frames.ndim != 4:
            raise ValueError(f"Expected 4-D frames, got shape {tuple(frames.shape)}")

        # Official DVS datasets normally return [T, C, H, W].
        # This fallback handles [C, T, H, W] when C is clearly the polarity axis.
        if frames.shape[0] in (1, 2) and frames.shape[1] > 2:
            frames = frames.permute(1, 0, 2, 3).contiguous()

        if self.input_clip is not None:
            frames = frames.clamp(max=float(self.input_clip))

        if self.input_scale == "sample_max":
            maximum = frames.amax()
            if maximum > 0:
                frames = frames / maximum
        elif self.input_scale != "none":
            raise ValueError(f"Unsupported input_scale: {self.input_scale}")

        return frames, int(target), int(index)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)
