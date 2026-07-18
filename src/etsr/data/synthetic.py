from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticTemporalOrderDataset(Dataset):
    """Small deterministic dataset used only for smoke tests.

    Each class is defined by the order of two moving polarity patterns. The dataset is deliberately
    simple: its purpose is to exercise the complete research pipeline without downloading data.
    """

    def __init__(
        self,
        samples: int,
        frames_number: int,
        image_size: int,
        num_classes: int,
        seed: int,
    ) -> None:
        self.samples = samples
        self.frames_number = frames_number
        self.image_size = image_size
        self.num_classes = num_classes
        self.seed = seed
        self.classes = [str(index) for index in range(num_classes)]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(self.seed + index)
        target = index % self.num_classes
        frames = torch.zeros(self.frames_number, 2, self.image_size, self.image_size)

        half = max(1, self.frames_number // 2)
        patterns = (target % 2, (target // 2) % 2)
        for time_index in range(self.frames_number):
            segment = 0 if time_index < half else 1
            direction = patterns[segment]
            local_t = time_index if segment == 0 else time_index - half
            span = max(1, half - 1)
            if direction == 0:
                x = 3 + int((self.image_size - 7) * local_t / span)
            else:
                x = self.image_size - 4 - int((self.image_size - 7) * local_t / span)
            y = self.image_size // 3 if segment == 0 else 2 * self.image_size // 3
            polarity = (target + segment) % 2
            frames[time_index, polarity, y - 1 : y + 2, x - 1 : x + 2] = 1.0

        noise = torch.rand(frames.shape, generator=generator) < 0.002
        frames = torch.maximum(frames, noise.to(frames.dtype))
        return frames, target
