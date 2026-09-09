"""Physical-time, polarity-separated count frames for the E0 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from etsr.data.events import EventSample


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
        # EventSample is the validated boundary; the encoder enforces only E0-specific limits.
        timestamps = np.asarray(sample.t_us)
        x_coords = np.asarray(sample.x)
        y_coords = np.asarray(sample.y)
        polarities = np.asarray(sample.polarity)
        if timestamps[0] < 0 or timestamps[-1] >= self.window_us:
            raise ValueError(
                f"Sample {sample.sample_id} timestamps must fit [0,{self.window_us}) us; "
                f"found [{int(timestamps[0])},{int(timestamps[-1])}]."
            )
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


class PhaseCountFrameEncoder(CountFrameEncoder):
    """Count-preserving two-basis encoding of event phase inside each physical-time bin.

    Every event contributes ``1-phase`` and ``phase`` to two channels of its polarity. Summing
    those channels reconstructs the E0 count exactly up to floating-point roundoff, while their
    ratio exposes the first temporal moment inside the 50 ms bin without increasing model steps.
    """

    name = "phase_count_frames_e1"

    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters.update(
            {
                "channels": {
                    "OFF_phase_start": 0,
                    "OFF_phase_end": 1,
                    "ON_phase_start": 2,
                    "ON_phase_end": 3,
                },
                "phase_basis": ["1 - local_phase", "local_phase"],
                "storage_dtype": "float32",
                "count_reconstruction": "sum each polarity's two phase channels",
            }
        )
        return parameters

    def __call__(self, sample: EventSample) -> EncodedRepresentation:
        timestamps = np.asarray(sample.t_us)
        x_coords = np.asarray(sample.x)
        y_coords = np.asarray(sample.y)
        polarities = np.asarray(sample.polarity)
        if timestamps[0] < 0 or timestamps[-1] >= self.window_us:
            raise ValueError(
                f"Sample {sample.sample_id} timestamps must fit [0,{self.window_us}) us; "
                f"found [{int(timestamps[0])},{int(timestamps[-1])}]."
            )
        time_bins = timestamps // self.bin_width_us
        count_indices = (
            (time_bins * 2 + polarities) * self.height + y_coords
        ) * self.width + x_coords
        _, occupied_counts = np.unique(count_indices, return_counts=True)
        maximum_count = int(occupied_counts.max())
        if maximum_count > self.count_cap:
            raise ValueError(
                f"Sample {sample.sample_id} needs count {maximum_count}, exceeding the explicit "
                f"E1 count cap {self.count_cap}; revise the representation instead of clipping."
            )

        local_phase = (timestamps % self.bin_width_us).astype(np.float32) / float(
            self.bin_width_us
        )
        first_channels = polarities * 2
        first_indices = (
            (time_bins * 4 + first_channels) * self.height + y_coords
        ) * self.width + x_coords
        second_indices = first_indices + self.height * self.width
        flat = np.zeros(self.time_steps * 4 * self.height * self.width, dtype=np.float32)
        np.add.at(flat, first_indices, 1.0 - local_phase)
        np.add.at(flat, second_indices, local_phase)
        tensor = torch.from_numpy(flat.reshape(self.time_steps, 4, self.height, self.width))
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
                "encoded_event_mass": float(tensor.sum().item()),
                "maximum_voxel_count_before_phase_split": maximum_count,
                "timestamp_normalized": False,
                "endpoint_knowledge": "none",
            },
        )


class TemporalBinaryFrameEncoder:
    """Canonical TBR occupancy encoding with fixed physical macro-windows.

    The polarity-agnostic event occupancy of ``bits`` consecutive micro-bins is packed into one
    scalar in [0, 1]. The latest micro-bin is the most significant bit, as specified by TBR.
    """

    name = "temporal_binary_frames_tbr"

    def __init__(
        self,
        *,
        height: int,
        width: int,
        window_us: int,
        bin_width_us: int,
        micro_bin_width_us: int,
        bits: int,
    ) -> None:
        for field, value in (
            ("height", height),
            ("width", width),
            ("window_us", window_us),
            ("bin_width_us", bin_width_us),
            ("micro_bin_width_us", micro_bin_width_us),
            ("bits", bits),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer.")
        if window_us % bin_width_us:
            raise ValueError("window_us must be exactly divisible by bin_width_us.")
        if bits * micro_bin_width_us != bin_width_us:
            raise ValueError("bits * micro_bin_width_us must equal bin_width_us.")
        if bits > 16:
            raise ValueError("TBR currently supports at most 16 bits per macro-frame.")
        self.height = height
        self.width = width
        self.window_us = window_us
        self.bin_width_us = bin_width_us
        self.micro_bin_width_us = micro_bin_width_us
        self.bits = bits
        self.time_steps = window_us // bin_width_us
        self.micro_time_steps = self.time_steps * bits
        self.time_bin_edges_us = tuple(range(0, window_us + bin_width_us, bin_width_us))

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "window_us": self.window_us,
            "bin_width_us": self.bin_width_us,
            "time_steps": self.time_steps,
            "micro_bin_width_us": self.micro_bin_width_us,
            "micro_time_steps": self.micro_time_steps,
            "bits": self.bits,
            "channels": {"polarity_agnostic_TBR": 0},
            "occupancy": "binary_per_pixel_per_micro_bin",
            "bit_order": "latest_micro_bin_is_most_significant",
            "normalization_divisor": (1 << self.bits) - 1,
            "polarity_policy": "discard",
            "multiplicity_policy": "discard_within_micro_bin",
            "storage_dtype": "float32",
            "endpoint_knowledge": "none",
        }

    @property
    def state_profile(self) -> dict[str, int | bool]:
        return {
            "stateful": True,
            "persistent_state_elements": self.height * self.width,
            "persistent_state_bits": self.height * self.width * self.bits,
        }

    def _validate_events(self, sample: EventSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamps = np.asarray(sample.t_us)
        if timestamps[0] < 0 or timestamps[-1] >= self.window_us:
            raise ValueError(
                f"Sample {sample.sample_id} timestamps must fit [0,{self.window_us}) us; "
                f"found [{int(timestamps[0])},{int(timestamps[-1])}]."
            )
        return timestamps, np.asarray(sample.x), np.asarray(sample.y)

    def _micro_counts(self, sample: EventSample) -> tuple[np.ndarray, int]:
        timestamps, x_coords, y_coords = self._validate_events(sample)
        micro_bins = timestamps // self.micro_bin_width_us
        indices = (micro_bins * self.height + y_coords) * self.width + x_coords
        occupied, counts = np.unique(indices, return_counts=True)
        flat = np.zeros(self.micro_time_steps * self.height * self.width, dtype=np.uint16)
        if int(counts.max()) > np.iinfo(np.uint16).max:
            raise ValueError("TBR micro-bin event multiplicity exceeds uint16 preprocessing storage.")
        flat[occupied] = counts.astype(np.uint16)
        return flat.reshape(self.micro_time_steps, self.height, self.width), len(occupied)

    def _pack_binary(self, binary: np.ndarray) -> torch.Tensor:
        packed = np.zeros((self.time_steps, self.height, self.width), dtype=np.uint16)
        for bit in range(self.bits):
            packed |= binary[bit :: self.bits].astype(np.uint16) << bit
        normalized = packed.astype(np.float32) / float((1 << self.bits) - 1)
        return torch.from_numpy(normalized[:, None])

    def __call__(self, sample: EventSample) -> EncodedRepresentation:
        micro_counts, occupied = self._micro_counts(sample)
        tensor = self._pack_binary(micro_counts > 0)
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
                "source_event_count": len(sample.t_us),
                "occupied_micro_voxels": occupied,
                "micro_bin_collisions": len(sample.t_us) - occupied,
                "timestamp_normalized": False,
                "endpoint_knowledge": "none",
            },
        )


class SpikeTemporalBinaryFrameEncoder(TemporalBinaryFrameEncoder):
    """Paper-aligned fixed-LIF Spike-TBR reconstruction for DVS-Lip.

    The published paper specifies beta=0.9 and threshold=1.1 for its DVS-Lip LIF result but does
    not release code or fully define polarity weights and cross-window membrane handling. This
    implementation records the explicit local choices: polarity is discarded, event counts drive
    one LIF per pixel at 6.25 ms resolution, and the membrane is reset at each 50 ms TBR window.
    """

    name = "spike_tbr_lif_paper_aligned"

    def __init__(self, *, lif_beta: float, lif_threshold: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= lif_beta < 1.0:
            raise ValueError("lif_beta must be in [0, 1).")
        if lif_threshold <= 0.0:
            raise ValueError("lif_threshold must be positive.")
        self.lif_beta = float(lif_beta)
        self.lif_threshold = float(lif_threshold)

    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters.update(
            {
                "lif_beta": self.lif_beta,
                "lif_threshold": self.lif_threshold,
                "lif_input": "polarity_agnostic_event_count_per_micro_bin",
                "lif_reset": "hard_zero_after_spike",
                "membrane_window_policy": "reset_each_macro_window",
                "fidelity": "paper_aligned_reconstruction_not_official_code_replication",
            }
        )
        return parameters

    @property
    def state_profile(self) -> dict[str, int | bool]:
        pixels = self.height * self.width
        return {
            "stateful": True,
            "persistent_state_elements": 2 * pixels,
            "persistent_state_bits": pixels * (32 + self.bits),
        }

    def __call__(self, sample: EventSample) -> EncodedRepresentation:
        micro_counts, occupied = self._micro_counts(sample)
        spikes = np.zeros_like(micro_counts, dtype=np.bool_)
        membrane = np.zeros((self.height, self.width), dtype=np.float32)
        for macro in range(self.time_steps):
            membrane.fill(0.0)
            for bit in range(self.bits):
                step = macro * self.bits + bit
                membrane *= self.lif_beta
                membrane += micro_counts[step]
                fired = membrane >= self.lif_threshold
                spikes[step] = fired
                membrane[fired] = 0.0
        tensor = self._pack_binary(spikes)
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
                "source_event_count": len(sample.t_us),
                "occupied_micro_voxels": occupied,
                "emitted_spike_voxels": int(spikes.sum()),
                "timestamp_normalized": False,
                "endpoint_knowledge": "none",
            },
        )
