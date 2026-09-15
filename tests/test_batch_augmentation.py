import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from etsr.config import ConfigError, load_config, validate_config
from etsr.training.augmentation import EventMix, build_batch_augmentation
from etsr.training.engine import train_one_epoch


class _DisabledScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, _optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def _event_mix(probability: float = 1.0) -> EventMix:
    return EventMix(
        probability=probability,
        beta=1.0,
        components=3,
        gmm_scale_min=0.08,
        gmm_scale_max=0.35,
        mask_grid_size=8,
        distance_spatial_pool=2,
    )


def test_event_mix_uses_deranged_partners_and_shared_spatiotemporal_masks():
    torch.manual_seed(42)
    frames = torch.stack(
        [torch.full((4, 2, 8, 8), float(index + 1)) for index in range(4)]
    )
    targets = torch.arange(4)

    mixed = _event_mix()(frames, targets)

    assert mixed.frames.shape == frames.shape
    assert torch.all(mixed.secondary_targets != targets)
    assert torch.all((0.0 <= mixed.primary_weights) & (mixed.primary_weights <= 1.0))
    assert torch.any(mixed.frames != frames)
    for index in range(frames.shape[0]):
        allowed = torch.tensor(
            [float(index + 1), float(mixed.secondary_targets[index] + 1)]
        )
        assert torch.isin(mixed.frames[index], allowed).all()
        assert torch.isin(allowed, mixed.frames[index]).all()
        # The GMM mask is shared by both polarity channels.
        assert torch.equal(mixed.frames[index, :, 0], mixed.frames[index, :, 1])


def test_event_mix_integrates_soft_targets_with_label_smoothing_and_backward():
    torch.manual_seed(7)
    frames = torch.randn(4, 2, 1, 2, 2)
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(frames, targets, torch.arange(4)), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original = model[1].weight.detach().clone()

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(label_smoothing=0.1),
        torch.device("cpu"),
        _DisabledScaler(),
        False,
        1.0,
        1,
        _event_mix(),
    )

    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert not torch.equal(model[1].weight, original)


def test_event_mix_config_is_explicit_and_zero_probability_disables_it():
    reference = load_config("configs/dvsgesture_f_tcap_stage1_dwc3_d8.yaml")
    assert build_batch_augmentation(reference["augmentation"]) is None

    candidate = load_config("configs/dvsgesture_f_tcap_stage1_dwc3_d8_event_mix.yaml")
    assert isinstance(build_batch_augmentation(candidate["augmentation"]), EventMix)

    invalid = copy.deepcopy(candidate)
    invalid["augmentation"]["event_mix_gmm_scale_max"] = 1.5
    with pytest.raises(ConfigError, match="GMM scales"):
        validate_config(invalid)

    invalid = copy.deepcopy(candidate)
    invalid["augmentation"]["event_mix_label_mode"] = "area"
    with pytest.raises(ConfigError, match="relative_distance"):
        validate_config(invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SMILIES CUDA runtime")
def test_event_mix_runs_on_cuda_at_dvsgesture_shape():
    torch.manual_seed(11)
    frames = torch.rand(2, 100, 2, 128, 128, device="cuda")
    targets = torch.tensor([0, 1], device="cuda")

    mixed = EventMix(
        probability=1.0,
        beta=1.0,
        components=3,
        gmm_scale_min=0.08,
        gmm_scale_max=0.35,
        mask_grid_size=32,
        distance_spatial_pool=4,
    )(frames, targets)

    assert mixed.frames.shape == frames.shape
    assert mixed.frames.device.type == "cuda"
    assert torch.isfinite(mixed.primary_weights).all()
    assert torch.all(mixed.secondary_targets != targets)
