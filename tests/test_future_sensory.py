from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from etsr.config import load_config
from etsr.data.events import EventSample
from etsr.evaluation import future_sensory_evaluate as sensory_evaluation
from etsr.evaluation.future_sensory import (
    FutureDecoder,
    balanced_count_weights,
    build_causal_d,
    sensory_loss_parts,
    target_field,
)
from etsr.evaluation.future_sensory_evaluate import (
    ClassProbe,
    _active_field_indices,
    _field_input,
    _first_pass_gate,
    _past_profile,
    _probe_input,
    _probe_train_eval,
    _random_encoder_cache,
    _stratified_resample_indices,
    e0_baselines,
    evaluate_future_sensory,
)
from etsr.evaluation.future_sensory_fit import (
    _batch_loss,
    _fit_count_calibration,
    _router_parameters,
    _screen_parameters,
    fit_prior,
    stratified_split,
    train_causality_preflight,
)
from etsr.models.temporal import CausalTemporalChannelMixer


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


def test_fit_count_baselines_use_the_same_balanced_cell_weights_as_the_loss():
    count = torch.tensor([[[[[10., 0., 0., 0.]]]], [[[[1., 1., 0., 0.]]]]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    weights = balanced_count_weights(count, valid)
    torch.testing.assert_close(weights.sum(dim=(1, 2, 3, 4)), torch.ones(2))
    weighted_mean = (weights * count).sum(dim=0) / weights.sum(dim=0)
    assert weighted_mean[0, 0, 0, 0] == pytest.approx(7.0)
    calibration = _fit_count_calibration(
        np.array([10., 10., 10.]), np.array([5., 25., 45.]), history_divisor=1
    )
    assert calibration["scale"] == pytest.approx(2.0, abs=1e-3)
    assert calibration["offset"] == pytest.approx(0.5, abs=1e-3)


def test_fit_prior_calibrates_count_and_timing_without_holdout():
    frames = torch.zeros(2, 40, 2, 128, 128)
    frames[0, 0, 0, 0, 0] = 2
    field = torch.zeros(2, 40, 2, 3, 16, 16)
    field[0, 0, 0, 0, 0, 0] = 10
    field[1, 0, 0, 1, 0, 0] = 1
    field[1, 0, 0, 1, 0, 1] = 1
    valid = torch.zeros(2, 40, dtype=torch.bool)
    valid[:, 0] = True
    batch = (frames, field, valid, None, None, None, None)
    prior = fit_prior([batch], 3, torch.device("cpu"))
    assert prior["mean_count"][0, 0, 0, 0] == pytest.approx(7.0)
    assert prior["q_prior"][0, 0, 0, 0, 0] == pytest.approx(7 / 15, abs=1e-6)
    assert prior["scale_n"] > 0 and prior["scale_q"] > 0
    baselines = e0_baselines(frames, prior, "fepf2", past=True)
    assert baselines["last_100ms"][0][0, 1, 0, 0, 0] == pytest.approx(2.0)
    assert baselines["last_100ms"][0][1, 1, 0, 0, 0] == pytest.approx(1e-6)


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
    with torch.no_grad():
        mixer = next(
            module for module in model.modules()
            if isinstance(module, CausalTemporalChannelMixer)
            and module.content_router is not None and 8 in module.delays
        )
        mixer.weight[mixer.delays.index(8), 0, 0] = .25
    active_d8 = train_causality_preflight(model, FutureDecoder(3), torch.device("cpu"))
    assert active_d8["train"]["gradient_max_abs"] == 0


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


def test_first_gate_requires_real_and_predicted_timing_and_paired_encoder_gain():
    skills = {name: {"passed": True} for name in ("count", "timing")}
    comparisons = {
        "predicted_vs_density": {"delta_macro_f1": .02, "bootstrap_95": [.001, .03]},
        "encoder_vs_random": {"delta_macro_f1": .02},
        "oracle_vs_density": {"delta_macro_f1": .01, "bootstrap_95": [.001, .02]},
    }
    assert _first_pass_gate(skills, comparisons)
    comparisons["oracle_vs_density"]["bootstrap_95"][0] = -.001
    assert not _first_pass_gate(skills, comparisons)
    comparisons["oracle_vs_density"]["bootstrap_95"][0] = .001
    comparisons["predicted_vs_density"]["delta_macro_f1"] = .005
    assert not _first_pass_gate(skills, comparisons)
    comparisons["predicted_vs_density"]["delta_macro_f1"] = .02
    comparisons["encoder_vs_random"]["delta_macro_f1"] = .005
    assert not _first_pass_gate(skills, comparisons)


def test_fixed_field_probe_cohort_excludes_unsupervised_post_end_cutoffs():
    endpoints = np.array([749_999, 750_000, 750_001, 1_100_000])
    np.testing.assert_array_equal(_active_field_indices(endpoints), [2, 3])


def test_field_probe_trains_and_evaluates_on_the_fixed_active_cohort(monkeypatch):
    monkeypatch.setattr(sensory_evaluation, "PROBE_EPOCHS", 1)
    mass = np.zeros((4, 15, 6, 16, 16), dtype=np.float32)
    mass[:, :, 0, 0, 0] = np.arange(1, 5)[:, None]
    arrays = {"predicted": mass, "label": np.array([0, 1, 0, 1])}
    prior = torch.full((40, 2, 3, 16, 16), 1 / 3)
    result = _probe_train_eval(
        "predicted", arrays, arrays, prior, classes=2, device=torch.device("cpu"),
        train_indices=np.array([0, 1]), holdout_indices=np.array([2, 3]),
    )
    assert len(result["full"]["predictions"]) == 2


def test_evaluation_rejects_unmatched_replica_budget_before_reading_holdout(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sensory_evaluation, "screen_config", lambda config: config)
    fit_dirs = []
    for mode, seed, epochs in (("fepf2", 42, 80), ("voxel4", 42, 80), ("fepf2", 43, 40)):
        directory = tmp_path / f"{mode}_{seed}"
        directory.mkdir()
        (directory / "fit_report.json").write_text(json.dumps({
            "mode": mode, "seed": seed, "past": False, "epochs": epochs,
            "extension_requested_by_this_arm": False,
        }))
        fit_dirs.append(str(directory))
    with pytest.raises(ValueError, match="40/80 budget"):
        evaluate_future_sensory({}, fit_dirs, tmp_path / "unopened")


def test_evaluation_rejects_code_mismatch_before_reading_holdout(monkeypatch, tmp_path):
    monkeypatch.setattr(sensory_evaluation, "screen_config", lambda config: config)
    fit_dirs = []
    for mode in ("fepf2", "voxel4"):
        directory = tmp_path / mode
        directory.mkdir()
        (directory / "fit_report.json").write_text(json.dumps({
            "mode": mode, "seed": 42, "past": False, "epochs": 40,
            "extension_requested_by_this_arm": False,
            "git_commit": "other-commit", "code_sha256": "other-code",
        }))
        fit_dirs.append(str(directory))
    with pytest.raises(ValueError, match="exact fit commit"):
        evaluate_future_sensory({}, fit_dirs, tmp_path / "unopened")


def test_random_encoder_cache_loads_exact_pre_ssl_state(monkeypatch, tmp_path):
    class DummyEncoder(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(float(value)))

        def _encode(self, frames):
            return torch.ones(40, frames.shape[0], 128, 8, 8) * self.weight

    initial = DummyEncoder(3)
    initial_path = tmp_path / "initial_encoder_state.pt"
    torch.save(initial.state_dict(), initial_path)
    monkeypatch.setattr(sensory_evaluation, "build_causal_d", lambda *_: DummyEncoder(9))
    frames = torch.zeros(1, 40, 2, 128, 128)
    monkeypatch.setattr(
        sensory_evaluation, "_loader", lambda *args, **kwargs:
        [(frames, None, None, None, None, None, None)],
    )
    cached = _random_encoder_cache(
        {}, [None], torch.device("cpu"), tmp_path, initial_path
    )
    assert cached.shape == (1, 40, 128, 8, 8)
    assert np.all(cached == 3)


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
    trained_report = train_causality_preflight(model, decoder, device, amp=True)
    assert trained_report["train"]["prefix_gradient_norm"] > 0
    assert trained_report["eval"]["gradient_max_abs"] == 0
