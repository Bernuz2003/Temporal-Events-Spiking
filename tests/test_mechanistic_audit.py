import numpy as np
import torch
from torch import nn

from etsr.evaluation.causal import (
    patch_with_counterfactual_activations,
    prediction_intervention_rates,
    segment_aligned_time_indices,
)
from etsr.evaluation.input_statistics import input_statistics


class _ToyTemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tap = nn.Identity()

    def forward(self, frames):
        time_major = self.tap(frames.transpose(0, 1))
        return time_major.mean(dim=0)


def test_segment_alignment_never_crosses_semantic_boundaries():
    recipient = torch.tensor([[3, 5], [5, 3]])
    donor = torch.tensor([[5, 3], [3, 5]])

    indices = segment_aligned_time_indices(recipient, donor)

    assert torch.all(indices[0, :3] < 5)
    assert torch.all(indices[0, 3:] >= 5)
    assert torch.all(indices[1, :5] < 3)
    assert torch.all(indices[1, 5:] >= 3)


def test_counterfactual_patch_respects_mask_and_aligned_indices():
    model = _ToyTemporalModel()
    original = torch.zeros(1, 4, 2)
    donor = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    mask = torch.tensor([[True, True, False, False]])
    indices = torch.tensor([[1, 0, 2, 3]], dtype=torch.long)

    logits = patch_with_counterfactual_activations(
        model, "tap", original, donor, mask, donor_time_indices=indices
    )

    expected = (donor[0, 1] + donor[0, 0]) / 4
    assert torch.allclose(logits[0], expected)


def test_prediction_intervention_separates_inverse_presence_from_change():
    baseline = torch.tensor([1, 2, 0, 3])
    patched = torch.tensor([1, 0, 2, 4])
    inverse_targets = torch.tensor([1, 0, 5, 4])

    result = prediction_intervention_rates(baseline, patched, inverse_targets)

    assert result["prediction_changed_rate"] == 0.75
    assert result["inverse_prediction_rate"] == 0.75
    assert result["prediction_changed_to_inverse_rate"] == 0.5


def test_order_invariant_shortcut_features_do_not_encode_frame_order():
    frames = torch.rand(1, 6, 2, 4, 4)
    metadata = [
        {
            "source_filename": "source.npz",
            "class_name": "13",
            "transition_indices": [3],
        }
    ]

    _rows, temporal, invariant = input_statistics(frames, metadata)
    _reversed_rows, reversed_temporal, reversed_invariant = input_statistics(
        frames.flip(1), metadata
    )

    assert not np.array_equal(temporal, reversed_temporal)
    assert np.allclose(invariant, reversed_invariant)
