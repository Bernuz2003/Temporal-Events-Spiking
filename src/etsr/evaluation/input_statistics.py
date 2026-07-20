from __future__ import annotations

from typing import Any

import numpy as np
import torch


def input_statistics(
    frames: torch.Tensor, metadata: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Return tidy timestep rows plus temporal and order-invariant shortcut features."""
    if frames.ndim != 5 or frames.shape[0] != len(metadata):
        raise ValueError("Expected frames [B, T, C, H, W] aligned with metadata.")
    values = frames.detach().cpu().float()
    batch, time, channels, height, width = values.shape
    flat_counts = values.sum(dim=(3, 4))
    frame_difference = torch.zeros((batch, time), dtype=values.dtype)
    if time > 1:
        frame_difference[:, 1:] = (values[:, 1:] - values[:, :-1]).abs().sum(dim=(2, 3, 4))

    y_coordinates = torch.linspace(0.0, 1.0, height).view(1, 1, 1, height, 1)
    x_coordinates = torch.linspace(0.0, 1.0, width).view(1, 1, 1, 1, width)
    total = values.sum(dim=(2, 3, 4)).clamp_min(1e-12)
    centroid_x = (values * x_coordinates).sum(dim=(2, 3, 4)) / total
    centroid_y = (values * y_coordinates).sum(dim=(2, 3, 4)) / total

    rows = []
    for sample_index, item in enumerate(metadata):
        transition = int(item["transition_indices"][0])
        for timestep in range(time):
            on_count = float(flat_counts[sample_index, timestep, 0])
            off_count = float(flat_counts[sample_index, timestep, 1]) if channels > 1 else 0.0
            count = on_count + off_count
            rows.append(
                {
                    "sample_index": sample_index,
                    "source_filename": item["source_filename"],
                    "class_name": item["class_name"],
                    "timestep": timestep + 1,
                    "normalized_time": (timestep + 1) / time,
                    "segment": 0 if timestep < transition else 1,
                    "event_count": count,
                    "on_count": on_count,
                    "off_count": off_count,
                    "polarity_ratio": on_count / count if count > 0 else 0.0,
                    "frame_difference_l1": float(frame_difference[sample_index, timestep]),
                    "centroid_x": float(centroid_x[sample_index, timestep]),
                    "centroid_y": float(centroid_y[sample_index, timestep]),
                }
            )

    temporal_features = torch.cat(
        [
            flat_counts.reshape(batch, -1),
            frame_difference,
            centroid_x,
            centroid_y,
        ],
        dim=1,
    ).numpy()
    order_invariant_features = torch.cat(
        [
            flat_counts.sum(dim=1),
            centroid_x.mean(dim=1, keepdim=True),
            centroid_y.mean(dim=1, keepdim=True),
        ],
        dim=1,
    ).numpy()
    return rows, temporal_features, order_invariant_features


def collect_input_audit(
    loader, dataset
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    temporal_parts = []
    invariant_parts = []
    target_parts = []
    sample_offset = 0
    raw_dataset = getattr(dataset, "dataset", dataset)
    for frames, targets, indices in loader:
        metadata = [raw_dataset.metadata_for_index(int(index)) for index in indices]
        batch_rows, temporal, invariant = input_statistics(frames, metadata)
        for row in batch_rows:
            row["sample_index"] += sample_offset
        sample_offset += frames.shape[0]
        rows.extend(batch_rows)
        temporal_parts.append(temporal)
        invariant_parts.append(invariant)
        target_parts.append(targets.numpy())
    return (
        rows,
        np.concatenate(temporal_parts),
        np.concatenate(invariant_parts),
        np.concatenate(target_parts),
    )
