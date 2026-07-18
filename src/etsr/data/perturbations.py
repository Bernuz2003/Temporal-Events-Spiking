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
    def from_dict(cls, data: dict[str, Any]) -> "PerturbationSpec":
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
        frames, target, stable_index = self.dataset[index]
        if self.spec.name == "reverse_actions":
            if self._reverse_sample_map is None or self._reverse_target_map is None:
                raise RuntimeError("reverse_actions requires paired samples and reversible classes.")
            counterpart_index = self._reverse_sample_map[int(index)]
            frames, counterpart_target, _counterpart_stable_index = self.dataset[counterpart_index]
            if int(counterpart_target) != self._reverse_target_map[int(target)]:
                raise RuntimeError("The paired reverse-action sample has an unexpected target.")
        else:
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

        lookup: dict[tuple[int, str], int] = {}
        for sample_index, (sample_path, target) in enumerate(samples):
            key = (int(target), Path(sample_path).name)
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
