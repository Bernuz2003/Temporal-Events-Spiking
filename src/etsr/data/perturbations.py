from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    seed: int = 0
    segments: int = 2
    target_mode: str = "keep"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerturbationSpec:
        return cls(
            name=str(data["name"]),
            seed=int(data.get("seed", 0)),
            segments=int(data.get("segments", 2)),
            target_mode=str(data.get("target_mode", "keep")),
        )


class PerturbedDataset(Dataset):
    """Applies a deterministic temporal perturbation at evaluation time."""

    def __init__(self, dataset: Dataset, spec: PerturbationSpec):
        self.dataset = dataset
        self.spec = spec
        self.classes = getattr(dataset, "classes", None)
        self.class_to_idx = getattr(dataset, "class_to_idx", None)
        self._reverse_target_map = self._build_reverse_target_map()
        self._sample_targets: list[int] | None = None
        self._reverse_sample_map = self._build_reverse_sample_map()
        if self.spec.name == "reverse_actions":
            self.resolved_method = "paired_reversed_action_sample"
        elif self.spec.name == "reverse_segments":
            self.resolved_method = "equal_temporal_chunks"
        else:
            self.resolved_method = self.spec.name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.spec.name == "reverse_actions":
            if (
                self._reverse_sample_map is None
                or self._reverse_target_map is None
                or self._sample_targets is None
            ):
                raise RuntimeError("reverse_actions requires paired samples and reversible classes.")
            counterpart_index = self._reverse_sample_map[int(index)]
            frames, counterpart_target, _counterpart_stable_index = self.dataset[counterpart_index]
            target = self._sample_targets[int(index)]
            stable_index = int(index)
            if int(counterpart_target) != self._reverse_target_map[int(target)]:
                raise RuntimeError("The paired reverse-action sample has an unexpected target.")
        else:
            frames, target, stable_index = self.dataset[index]
            frames = apply_temporal_perturbation(frames, self.spec, stable_index)

        if self.spec.target_mode == "reverse_class":
            if self._reverse_target_map is None:
                raise RuntimeError(
                    "reverse_class requires string class names whose reversed sequence is present."
                )
            target = self._reverse_target_map[int(target)]
        elif self.spec.target_mode != "keep":
            raise ValueError(f"Unsupported target_mode: {self.spec.target_mode}")
        return frames, int(target), int(stable_index)

    def _build_reverse_target_map(self) -> dict[int, int] | None:
        if not self.classes:
            return None
        mapping = {str(name): index for index, name in enumerate(self.classes)}
        result: dict[int, int] = {}
        for index, name in enumerate(self.classes):
            reversed_name = str(name)[::-1]
            if reversed_name not in mapping:
                return None
            result[index] = mapping[reversed_name]
        return result

    def _build_reverse_sample_map(self) -> dict[int, int] | None:
        if self.spec.name != "reverse_actions":
            return None
        if self._reverse_target_map is None:
            raise RuntimeError(
                "reverse_actions requires string class names whose reversed sequence is present."
            )

        samples = getattr(self.dataset, "samples", None)
        if not samples:
            raise RuntimeError(
                "reverse_actions requires DatasetFolder-style samples with source file paths."
            )
        if len(samples) != len(self.dataset):
            raise RuntimeError("reverse_actions requires one samples entry per dataset item.")

        lookup: dict[tuple[int, str], int] = {}
        self._sample_targets = []
        for sample_index, (sample_path, target) in enumerate(samples):
            target = int(target)
            self._sample_targets.append(target)
            key = (target, Path(sample_path).name)
            if key in lookup:
                raise RuntimeError(f"Duplicate reverse-action pairing key: {key}")
            lookup[key] = sample_index

        result: dict[int, int] = {}
        for sample_index, (sample_path, target) in enumerate(samples):
            reverse_target = self._reverse_target_map[int(target)]
            reverse_key = (reverse_target, Path(sample_path).name)
            if reverse_key not in lookup:
                raise RuntimeError(
                    "Missing paired reverse-action sample for "
                    f"class={target}, file={Path(sample_path).name}."
                )
            result[sample_index] = lookup[reverse_key]
        return result


def apply_temporal_perturbation(
    frames: torch.Tensor, spec: PerturbationSpec, sample_index: int
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError("Temporal perturbations expect [T, C, H, W].")

    if spec.name == "original":
        return frames.clone()
    if spec.name == "reverse_time":
        return frames.flip(0)
    if spec.name == "shuffle_time":
        generator = torch.Generator().manual_seed(spec.seed + int(sample_index))
        permutation = torch.randperm(frames.shape[0], generator=generator)
        return frames[permutation]
    if spec.name == "reverse_segments":
        if spec.segments < 2:
            raise ValueError("reverse_segments requires at least two segments.")
        chunks = torch.tensor_split(frames, spec.segments, dim=0)
        return torch.cat(list(reversed(chunks)), dim=0)
    raise ValueError(f"Unsupported perturbation: {spec.name}")


def count_preserving_resample(frames: torch.Tensor, new_length: int) -> torch.Tensor:
    """Re-bin time while preserving the total count at every channel and pixel."""
    if frames.ndim != 4:
        raise ValueError("Expected frames [T, C, H, W].")
    old_length = int(frames.shape[0])
    if old_length <= 0 or new_length <= 0:
        raise ValueError("Temporal lengths must be positive.")
    if new_length == old_length:
        return frames.clone()

    old_edges = torch.linspace(0.0, 1.0, old_length + 1, device=frames.device)
    new_edges = torch.linspace(0.0, 1.0, new_length + 1, device=frames.device)
    weights = frames.new_zeros((new_length, old_length))
    for new_index in range(new_length):
        for old_index in range(old_length):
            overlap = torch.minimum(
                new_edges[new_index + 1], old_edges[old_index + 1]
            ) - torch.maximum(new_edges[new_index], old_edges[old_index])
            weights[new_index, old_index] = overlap.clamp_min(0.0) * old_length
    return torch.einsum("nt,tchw->nchw", weights, frames)


def redistribute_segment_durations(
    frames: torch.Tensor,
    segment_lengths: list[int],
    target_ratios: list[float],
) -> tuple[torch.Tensor, list[int]]:
    if len(segment_lengths) != len(target_ratios):
        raise ValueError("Segment lengths and target ratios must be aligned.")
    if sum(segment_lengths) != frames.shape[0]:
        raise ValueError("Segment lengths do not cover the sequence.")
    if any(length <= 0 for length in segment_lengths) or any(
        ratio <= 0 for ratio in target_ratios
    ):
        raise ValueError("Segment lengths and ratios must be positive.")

    total = int(frames.shape[0])
    ratio_sum = sum(target_ratios)
    target_lengths = [
        max(1, int(round(total * ratio / ratio_sum))) for ratio in target_ratios
    ]
    # Rounding is assigned to the final segment so that the overall sequence length never changes.
    target_lengths[-1] += total - sum(target_lengths)
    if target_lengths[-1] <= 0:
        raise ValueError("Target ratios produce an empty final segment.")

    chunks = torch.split(frames, segment_lengths, dim=0)
    transformed = [
        count_preserving_resample(chunk, target_length)
        for chunk, target_length in zip(chunks, target_lengths, strict=True)
    ]
    return torch.cat(transformed, dim=0), target_lengths
