import torch
from torch import nn

from etsr.phase2.causal import (
    patch_with_counterfactual_activations,
    segment_aligned_time_indices,
)


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
