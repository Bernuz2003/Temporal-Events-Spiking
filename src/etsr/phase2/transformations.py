from __future__ import annotations

import torch


def count_preserving_resample(frames: torch.Tensor, new_length: int) -> torch.Tensor:
    """Re-bin frame counts along time while preserving every C/H/W total."""
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
    if any(length <= 0 for length in segment_lengths) or any(ratio <= 0 for ratio in target_ratios):
        raise ValueError("Segment lengths and ratios must be positive.")

    total = int(frames.shape[0])
    normalized = [ratio / sum(target_ratios) for ratio in target_ratios]
    target_lengths = [max(1, int(round(total * ratio))) for ratio in normalized]
    target_lengths[-1] += total - sum(target_lengths)
    if target_lengths[-1] <= 0:
        raise ValueError("Target ratios produce an empty final segment.")

    chunks = torch.split(frames, segment_lengths, dim=0)
    transformed = [
        count_preserving_resample(chunk, target_length)
        for chunk, target_length in zip(chunks, target_lengths, strict=True)
    ]
    return torch.cat(transformed, dim=0), target_lengths
