import torch

from etsr.phase2.transformations import (
    count_preserving_resample,
    redistribute_segment_durations,
)


def test_count_preserving_rebin_preserves_every_spatial_polarity_total():
    frames = torch.rand(5, 2, 4, 4)

    transformed = count_preserving_resample(frames, 8)

    assert transformed.shape == (8, 2, 4, 4)
    assert torch.allclose(transformed.sum(dim=0), frames.sum(dim=0), atol=1e-5)


def test_duration_redistribution_uses_known_boundaries_and_preserves_counts():
    frames = torch.rand(8, 2, 4, 4)

    transformed, lengths = redistribute_segment_durations(frames, [3, 5], [0.75, 0.25])

    assert lengths == [6, 2]
    assert transformed.shape == frames.shape
    assert torch.allclose(transformed.sum(dim=0), frames.sum(dim=0), atol=1e-5)
