from __future__ import annotations

import numpy as np
import pytest
import torch

from etsr.config import load_config
from etsr.data.events import EventSample
from etsr.evaluation.future_sensory import (
    FutureDecoder,
    build_causal_d,
    sensory_loss_parts,
    target_field,
)
from etsr.evaluation.future_sensory_evaluate import (
    ClassProbe,
    _field_input,
    _past_profile,
    _probe_input,
    _stratified_resample_indices,
)
from etsr.evaluation.future_sensory_fit import (
    _batch_loss,
    _router_parameters,
    _screen_parameters,
    stratified_split,
    train_causality_preflight,
)


def _sample() -> EventSample:
    return EventSample(
        x=np.array([0, 0, 8, 8], dtype=np.int8),
        y=np.array([0, 0, 8, 8], dtype=np.int8),
        t_us=np.array([50_000, 75_000, 100_000, 199_999], dtype=np.int64),
        polarity=np.array([0, 0, 1, 1], dtype=np.int8),
        target=0,
        sample_id="test/sample.npy",
        speaker_id=None,
        duration_us=149_999,
        metadata={},
    )


def test_future_targets_preserve_count_and_half_open_boundaries():
    sample = _sample()
    fepf = target_field(sample, "fepf2")
    voxel = target_field(sample, "voxel4")
    assert fepf.shape == (40, 2, 3, 16, 16)
    assert voxel.shape == (40, 2, 4, 16, 16)
    np.testing.assert_allclose(fepf.sum(axis=2), voxel.sum(axis=2), atol=1e-6)
    assert voxel[0, 0, 0, 0, 0] == 1  # 50 ms belongs to [50,150), not earlier.
    assert voxel[0, 0, 1, 0, 0] == 1
    assert voxel[0, 1, 2, 1, 1] == 1
    assert voxel[1, 1, 3, 1, 1] == 1


def test_past_target_never_uses_future():
    sample = _sample()
    original = target_field(sample, "fepf2", past=True)
    shorter = EventSample(
        x=sample.x[:-1], y=sample.y[:-1], t_us=sample.t_us[:-1],
        polarity=sample.polarity[:-1], target=0, sample_id=sample.sample_id,
        speaker_id=None, duration_us=50_000, metadata={},
    )
    trimmed = target_field(shorter, "fepf2", past=True)
    np.testing.assert_array_equal(original[:2], trimmed[:2])


def test_balanced_loss_uses_utterances_as_units_and_skips_empty_kl():
    field = torch.zeros(2, 2, 2, 3, 1, 2)
    field[0, 0, 0, 0, 0, 0] = 2
    field[1, 0, 1, 2, 0, 1] = 1
    count = torch.ones(2, 2, 2, 1, 2)
    q = torch.full((2, 2, 2, 3, 1, 2), 1 / 3)
    valid = torch.tensor([[True, False], [True, True]])
    loss = sensory_loss_parts(count, q, field, valid)
    assert loss.count.shape == (2,)
    assert loss.timing.shape == (2,)
    torch.testing.assert_close(loss.timing, torch.full((2,), np.log(3.0)))


def test_e0_profile_keeps_physical_clock_and_zero_past():
    old = torch.ones(1, 1, 2, 1, 1)
    new = torch.zeros_like(old)
    fepf = _past_profile(old, new, "fepf2")
    voxel = _past_profile(old, new, "voxel4")
    torch.testing.assert_close(fepf.flatten(), torch.tensor([.5625, .375, .0625] * 2))
    torch.testing.assert_close(voxel.flatten(), torch.tensor([.5, .5, 0, 0] * 2))


def test_stratified_split_is_reproducible_and_disjoint():
    targets = [0] * 10 + [1] * 10
    identities = [f"class/{i}.npy" for i in range(20)]
    split = stratified_split(targets, identities)
    assert split == stratified_split(targets, identities)
    assert len(split["fit_indices"]) == 16
    assert len(split["holdout_indices"]) == 4
    assert not set(split["fit_indices"]) & set(split["holdout_indices"])


def test_causal_d_replaces_batchnorm_and_preserves_prefix_gradients():
    config = load_config("configs/dvslip_future_sensory_screen.yaml")
    model = build_causal_d(config, 100)
    assert not any(
        isinstance(module, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d)
        for module in model.modules()
    )
    groups_by_channels = {
        channel: {module.num_groups for module in model.modules()
                  if isinstance(module, torch.nn.GroupNorm) and module.num_channels == channel}
        for channel in (8, 16, 32)
    }
    assert groups_by_channels[8] == {2}
    assert groups_by_channels[16] == {4}
    assert groups_by_channels[32] == {8}
    result = train_causality_preflight(model, FutureDecoder(3), torch.device("cpu"))
    assert result["train"]["prefix_gradient_norm"] > 0
    assert result["eval"]["gradient_max_abs"] == 0


def test_class_probe_averages_logits_after_nonlinear_per_step_processing():
    probe = ClassProbe(8, 100)
    inputs = torch.randn(2, 15, 8, 16, 16)
    with torch.no_grad():
        full = probe(inputs)
        prefix = probe(inputs, steps=5)
    assert full.shape == prefix.shape == (2, 100)


def test_field_probe_uses_mass_and_suppresses_empty_cell_timing():
    count = torch.zeros(1, 15, 2, 16, 16)
    count[..., 0, 0] = 2.0
    q = torch.full((1, 15, 2, 3, 16, 16), 1 / 3)
    q[..., 0, 1, 1] = 1.0
    q[..., 1:, 1, 1] = 0.0
    mass = _field_input(count, q)
    assert mass.shape == (1, 15, 6, 16, 16)
    torch.testing.assert_close(mass.reshape(1, 15, 2, 3, 16, 16).sum(dim=3), count)
    assert torch.count_nonzero(mass[..., 1, 1]) == 0
    prior = torch.full((40, 2, 3, 16, 16), 1 / 3)
    density = _probe_input({"predicted": mass.numpy()}, "predicted_density", np.array([0]), prior)
    torch.testing.assert_close(
        density.reshape(1, 15, 2, 3, 16, 16).sum(dim=3), count
    )
    assert torch.count_nonzero(density[..., 1, 1]) == 0
    oracle = _probe_input({"oracle": mass.numpy()}, "oracle", np.array([0]), prior)
    torch.testing.assert_close(oracle, mass)


def test_stratified_bootstrap_keeps_class_support_and_pairs_predictions():
    targets = np.array([0, 0, 1, 1, 1, 2, 2])
    groups = [np.flatnonzero(targets == label) for label in np.unique(targets)]
    rng = np.random.default_rng(42)
    for _ in range(10):
        indices = _stratified_resample_indices(groups, rng)
        np.testing.assert_array_equal(np.bincount(targets[indices]), [2, 3, 2])
        assert len(indices) == len(targets)


def test_sensory_loss_reaches_router_after_zero_initialized_tcap_updates():
    torch.manual_seed(0)
    config = load_config("configs/dvslip_future_sensory_screen.yaml")
    model = build_causal_d(config, 100)
    decoder = FutureDecoder(3)
    optimizer = torch.optim.AdamW(_screen_parameters(model, decoder), lr=1e-3)
    frames = torch.rand(1, 6, 2, 128, 128)
    field = torch.rand(1, 6, 2, 3, 16, 16)
    valid = torch.ones(1, 6, dtype=torch.bool)
    batch = (frames, field, valid, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    prior = {"scale_n": 1.0, "scale_q": 1.0}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _batch_loss(model, decoder, batch, prior, torch.device("cpu"))
        loss.backward()
        router_gradient = sum(
            float(parameter.grad.norm())
            for parameter in _router_parameters(model)
            if parameter.grad is not None
        )
        optimizer.step()
    assert router_gradient > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SMILIES CUDA runtime")
def test_future_screen_cuda_amp_preflight_and_backward():
    torch.manual_seed(0)
    device = torch.device("cuda")
    config = load_config("configs/dvslip_future_sensory_screen.yaml")
    model = build_causal_d(config, 100).to(device)
    decoder = FutureDecoder(3).to(device)
    report = train_causality_preflight(model, decoder, device, amp=True)
    assert report["train"]["prefix_gradient_norm"] > 0
    frames = torch.rand(2, 6, 2, 128, 128, device=device)
    field = torch.rand(2, 6, 2, 3, 16, 16, device=device)
    valid = torch.ones(2, 6, dtype=torch.bool, device=device)
    batch = (frames, field, valid, torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2))
    optimizer = torch.optim.AdamW(_screen_parameters(model, decoder), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, _ = _batch_loss(
                model, decoder, batch, {"scale_n": 1.0, "scale_q": 1.0}, device
            )
        assert bool(torch.isfinite(loss))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        router_gradient = sum(
            float(parameter.grad.norm())
            for parameter in _router_parameters(model)
            if parameter.grad is not None
        )
        scaler.step(optimizer)
        scaler.update()
    assert router_gradient > 0
