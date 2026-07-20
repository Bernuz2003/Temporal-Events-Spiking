import torch
from torch.utils.data import Dataset

from etsr.data.perturbations import (
    PerturbationSpec,
    PerturbedDataset,
    apply_temporal_perturbation,
    count_preserving_resample,
    redistribute_segment_durations,
)


class PairedActionDataset(Dataset):
    classes = ["13", "31"]
    class_to_idx = {"13": 0, "31": 1}
    samples = [("/dataset/13/sample.npz", 0), ("/dataset/31/sample.npz", 1)]

    def __init__(self):
        self.read_indices = []

    def __len__(self):
        return 2

    def __getitem__(self, index):
        self.read_indices.append(index)
        _path, target = self.samples[index]
        frames = torch.full((4, 1, 1, 1), float(target + 1))
        return frames, target, index


class IncompletePairedActionDataset(PairedActionDataset):
    samples = [("/dataset/13/sample.npz", 0), ("/dataset/31/other.npz", 1)]


def test_reverse_time():
    frames = torch.arange(4).reshape(4, 1, 1, 1).float()
    result = apply_temporal_perturbation(frames, PerturbationSpec("reverse_time"), 0)
    assert result[:, 0, 0, 0].tolist() == [3, 2, 1, 0]


def test_shuffle_is_deterministic_per_sample():
    frames = torch.arange(8).reshape(8, 1, 1, 1).float()
    spec = PerturbationSpec("shuffle_time", seed=9)
    first = apply_temporal_perturbation(frames, spec, 3)
    second = apply_temporal_perturbation(frames, spec, 3)
    assert torch.equal(first, second)
    assert sorted(first[:, 0, 0, 0].tolist()) == list(range(8))


def test_reverse_segments_preserves_internal_order():
    frames = torch.arange(6).reshape(6, 1, 1, 1).float()
    spec = PerturbationSpec("reverse_segments", segments=2)
    result = apply_temporal_perturbation(frames, spec, 0)
    assert result[:, 0, 0, 0].tolist() == [3, 4, 5, 0, 1, 2]


def test_reverse_actions_uses_paired_sample_and_preserves_original_index():
    source = PairedActionDataset()
    dataset = PerturbedDataset(source, PerturbationSpec("reverse_actions", target_mode="keep"))

    frames, target, stable_index = dataset[0]

    assert frames.unique().item() == 2.0
    assert target == 0
    assert stable_index == 0
    assert dataset.resolved_method == "paired_reversed_action_sample"
    assert source.read_indices == [1]


def test_reverse_actions_can_remap_target():
    dataset = PerturbedDataset(
        PairedActionDataset(),
        PerturbationSpec("reverse_actions", target_mode="reverse_class"),
    )

    frames, target, stable_index = dataset[0]

    assert frames.unique().item() == 2.0
    assert target == 1
    assert stable_index == 0


def test_reverse_actions_fails_if_a_source_pair_is_missing():
    try:
        PerturbedDataset(IncompletePairedActionDataset(), PerturbationSpec("reverse_actions"))
    except RuntimeError as error:
        assert "Missing paired reverse-action sample" in str(error)
    else:
        raise AssertionError("reverse_actions accepted an incomplete source pairing")


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
