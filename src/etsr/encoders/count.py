"""Physical-time, polarity-separated count frames for the E0 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from etsr.dvslip.dataset import EventSample


@dataclass(frozen=True)
class EncodedRepresentation:
    """Tensor plus the timing, representation and state semantics needed to interpret it."""

    tensor: torch.Tensor
    target: int
    sample_id: str
    time_axis: int
    time_bin_edges_us: tuple[int, ...]
    time_unit: str
    representation_name: str
    representation_parameters: dict[str, Any]
    state_profile: dict[str, int | bool]
    metadata: dict[str, Any]


class CountFrameEncoder:
    """Accumulate exact ON/OFF event counts into fixed physical-time bins."""

    name = "count_frames_e0"

    def __init__(
        self,
        *,
        height: int,
        width: int,
        window_us: int,
        bin_width_us: int,
        count_cap: int,
    ) -> None:
        for field, value in (
            ("height", height),
            ("width", width),
            ("window_us", window_us),
            ("bin_width_us", bin_width_us),
            ("count_cap", count_cap),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer.")
        if window_us % bin_width_us:
            raise ValueError("window_us must be exactly divisible by bin_width_us.")
        if count_cap > np.iinfo(np.uint8).max:
            raise ValueError("E0 count_cap must fit in uint8 storage.")

        self.height = height
        self.width = width
        self.window_us = window_us
        self.bin_width_us = bin_width_us
        self.count_cap = count_cap
        self.time_steps = window_us // bin_width_us
        self.time_bin_edges_us = tuple(range(0, window_us + bin_width_us, bin_width_us))

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "window_us": self.window_us,
            "bin_width_us": self.bin_width_us,
            "time_steps": self.time_steps,
            "channels": {"OFF": 0, "ON": 1},
            "count_cap": self.count_cap,
            "logical_count_bits": math.ceil(math.log2(self.count_cap + 1)),
            "storage_dtype": "uint8",
            "overflow_policy": "error",
        }

    @property
    def state_profile(self) -> dict[str, int | bool]:
        return {
            "stateful": False,
            "persistent_state_elements": 0,
            "persistent_state_bits": 0,
        }

    def __call__(self, sample: EventSample) -> EncodedRepresentation:
        lengths = {
            len(sample.t_us),
            len(sample.x),
            len(sample.y),
            len(sample.polarity),
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError(f"Event fields are empty or misaligned for {sample.sample_id}.")

        timestamps = np.asarray(sample.t_us, dtype=np.int64)
        x_coords = np.asarray(sample.x, dtype=np.int64)
        y_coords = np.asarray(sample.y, dtype=np.int64)
        polarities = np.asarray(sample.polarity, dtype=np.int64)
        if timestamps[0] < 0 or timestamps[-1] >= self.window_us:
            raise ValueError(
                f"Sample {sample.sample_id} timestamps must fit [0,{self.window_us}) us; "
                f"found [{int(timestamps[0])},{int(timestamps[-1])}]."
            )
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError(f"Sample {sample.sample_id} timestamps are not monotonic.")
        if np.any((x_coords < 0) | (x_coords >= self.width)) or np.any(
            (y_coords < 0) | (y_coords >= self.height)
        ):
            raise ValueError(f"Sample {sample.sample_id} coordinates exceed the encoder sensor.")
        if np.any((polarities < 0) | (polarities > 1)):
            raise ValueError(f"Sample {sample.sample_id} polarity must use OFF=0 and ON=1.")

        time_bins = timestamps // self.bin_width_us
        linear_indices = (
            (time_bins * 2 + polarities) * self.height + y_coords
        ) * self.width + x_coords
        occupied_indices, occupied_counts = np.unique(linear_indices, return_counts=True)
        maximum_count = int(occupied_counts.max())
        if maximum_count > self.count_cap:
            raise ValueError(
                f"Sample {sample.sample_id} needs count {maximum_count}, exceeding the explicit "
                f"E0 uint8 cap {self.count_cap}; revise the representation instead of clipping."
            )

        flat_counts = np.zeros(self.time_steps * 2 * self.height * self.width, dtype=np.uint8)
        flat_counts[occupied_indices] = occupied_counts.astype(np.uint8)
        tensor = torch.from_numpy(flat_counts.reshape(self.time_steps, 2, self.height, self.width))
        return EncodedRepresentation(
            tensor=tensor,
            target=sample.target,
            sample_id=sample.sample_id,
            time_axis=0,
            time_bin_edges_us=self.time_bin_edges_us,
            time_unit="microsecond",
            representation_name=self.name,
            representation_parameters=self.parameters,
            state_profile=self.state_profile,
            metadata={
                "source_event_count": len(timestamps),
                "encoded_event_count": int(occupied_counts.sum()),
                "maximum_voxel_count": maximum_count,
                "timestamp_normalized": False,
            },
        )
