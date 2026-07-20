from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class TraceCollection:
    indices: np.ndarray
    targets: np.ndarray
    cumulative_logits: np.ndarray
    per_timestep_logits: np.ndarray
    evidence_without_bias: np.ndarray
    head_bias: np.ndarray
    features: dict[str, np.ndarray]
    metadata: list[dict[str, Any]]

    @property
    def predictions(self) -> np.ndarray:
        return self.cumulative_logits[:, -1].argmax(axis=1)


@torch.no_grad()
def collect_traces(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    layers: list[str],
) -> TraceCollection:
    if not hasattr(model, "forward_with_trace"):
        raise TypeError("Phase 2 tracing requires a model.forward_with_trace method.")
    model.eval()
    indices_parts = []
    target_parts = []
    cumulative_parts = []
    timestep_parts = []
    evidence_parts = []
    feature_parts: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}
    head_bias = None

    for frames, targets, indices in loader:
        result = model.forward_with_trace(frames.to(device, non_blocking=True))
        features = result["features"]
        missing = [layer for layer in layers if layer not in features]
        if missing:
            raise KeyError(f"Trace is missing configured layers: {missing}")
        indices_parts.append(indices.cpu().numpy())
        target_parts.append(targets.cpu().numpy())
        cumulative_parts.append(result["cumulative_logits"].detach().cpu().numpy())
        timestep_parts.append(result["per_timestep_logits"].detach().cpu().numpy())
        evidence_parts.append(result["evidence_without_bias"].detach().cpu().numpy())
        for layer in layers:
            feature_parts[layer].append(features[layer].detach().cpu().numpy())
        if head_bias is None:
            bias = result["head_bias"]
            head_bias = (
                np.zeros(result["cumulative_logits"].shape[-1], dtype=np.float32)
                if bias is None
                else bias.detach().cpu().numpy()
            )

    indices = np.concatenate(indices_parts)
    order = np.argsort(indices)
    dataset = loader.dataset
    raw_dataset = getattr(dataset, "dataset", dataset)
    if not hasattr(raw_dataset, "metadata_for_index"):
        raise TypeError("Phase 2 tracing requires dataset metadata_for_index().")
    metadata = [raw_dataset.metadata_for_index(int(index)) for index in indices[order]]
    return TraceCollection(
        indices=indices[order],
        targets=np.concatenate(target_parts)[order],
        cumulative_logits=np.concatenate(cumulative_parts)[order],
        per_timestep_logits=np.concatenate(timestep_parts)[order],
        evidence_without_bias=np.concatenate(evidence_parts)[order],
        head_bias=np.asarray(head_bias),
        features={layer: np.concatenate(parts)[order] for layer, parts in feature_parts.items()},
        metadata=metadata,
    )


def save_trace_npz(trace: TraceCollection, path: str) -> None:
    payload: dict[str, np.ndarray] = {
        "indices": trace.indices,
        "targets": trace.targets,
        "cumulative_logits": trace.cumulative_logits,
        "per_timestep_logits": trace.per_timestep_logits,
        "evidence_without_bias": trace.evidence_without_bias,
        "head_bias": trace.head_bias,
        "source_filenames": np.asarray([item["source_filename"] for item in trace.metadata]),
        "class_names": np.asarray([item["class_name"] for item in trace.metadata]),
        "segment_lengths": np.asarray(
            [item["segment_lengths"] for item in trace.metadata], dtype=np.int64
        ),
    }
    for layer, features in trace.features.items():
        payload[f"features__{layer}"] = features
    np.savez_compressed(path, **payload)
