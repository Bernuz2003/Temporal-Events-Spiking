from __future__ import annotations

import json

import pytest
import torch

from etsr.config import load_config
from etsr.evaluation.predictive_diagnostic import (
    _fit_convex_tcap_predictors,
    _fit_dense_ridge,
    _region_weights,
)
from etsr.models.factory import build_model
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.reproducibility import git_commit
from etsr.training.predictive import PredictiveTrainingObjective, _balanced_masked_loss
from etsr.utils.io import sha256_file


def test_predictive_configs_inherit_the_frozen_continuation_recipe():
    r0 = load_config("configs/dvslip_predictive_r0.yaml")
    future = load_config("configs/dvslip_predictive_fine_future.yaml")
    dynamic = load_config("configs/dvslip_predictive_dynamic_tcap.yaml")
    surprise = load_config("configs/dvslip_predictive_s1.yaml")
    assert r0["training"]["epochs"] == 64
    assert r0["training"]["recipe_id"] == "dvslip_predictive_continuation_64_v1"
    assert r0["training"]["learning_rate"] == 1e-4
    assert r0["model"]["temporal_channel_mixer_delays"] == [1, 2, 4, 8]
    assert future["representation"]["name"] == "multigranular_count_frame"
    assert not future["model"].get("multigranular", False)
    assert future["model"]["predictive_head"]
    assert future["model"]["predictive_head_spatial_kernel_size"] == 3
    assert future["model"]["predictive_head_hidden_channels"] == 128
    assert dynamic["model"]["temporal_channel_mixer_router_pooling"] == "local"
    assert dynamic["model"]["temporal_channel_mixer_router_hidden_divisor"] == 2
    assert surprise["model"]["temporal_channel_mixer_predictor_channel_groups"] == 1
    assert surprise["model"]["temporal_channel_mixer_predictor_spatial_kernel_size"] == 3


def test_dynamic_tcap_starts_as_exact_fixed_tcap_and_is_causal():
    torch.manual_seed(4)
    fixed = CausalTemporalChannelMixer(3, (1, 2, 4))
    dynamic = CausalTemporalChannelMixer(3, (1, 2, 4), dynamic_routing=True)
    with torch.no_grad():
        fixed.weight.normal_()
        dynamic.weight.copy_(fixed.weight)
    sequence = torch.randn(7, 2, 3, 2, 2)
    expected = fixed(sequence)
    actual = dynamic(sequence)
    torch.testing.assert_close(actual, expected)

    altered = sequence.clone()
    altered[5:] = torch.randn_like(altered[5:]) * 100
    torch.testing.assert_close(dynamic(sequence)[:5], dynamic(altered)[:5])


def test_dynamic_model_router_remains_unit_after_global_initialization():
    fixed = _tiny_model(multigranular=False).eval()
    dynamic = MiniQKFormer(
        in_channels=2,
        num_classes=5,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2),
        temporal_channel_mixer_dynamic_routing=True,
        stage1_mixer="depthwise_conv",
    ).eval()
    incompatible = dynamic.load_state_dict(fixed.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all("content_router" in key for key in incompatible.missing_keys)
    frames = torch.randn(1, 4, 2, 32, 32)
    torch.testing.assert_close(dynamic(frames), fixed(frames))


def test_surprise_tcap_has_finite_predictor_and_router_gradients():
    torch.manual_seed(3)
    mixer = CausalTemporalChannelMixer(
        4,
        (1, 2),
        dynamic_routing=True,
        predictive_auxiliary=True,
        surprise_routing=True,
    )
    with torch.no_grad():
        mixer.weight.normal_(std=0.1)
    sequence = torch.randn(6, 2, 4, 3, 3, requires_grad=True)
    output = mixer(sequence)
    auxiliary = mixer.auxiliary_loss()
    assert auxiliary is not None and torch.isfinite(auxiliary)
    (output.square().mean() + auxiliary).backward()
    assert mixer.predictor_logits.grad is not None
    assert mixer.surprise_router.weight.grad is not None
    assert mixer.content_router is not None
    assert mixer.content_router.weight.grad is not None
    assert torch.isfinite(mixer.predictor_logits.grad).all()
    assert torch.isfinite(mixer.surprise_router.weight.grad).all()


def test_local_nonlinear_router_starts_as_fixed_tcap_and_varies_spatially():
    torch.manual_seed(13)
    fixed = CausalTemporalChannelMixer(4, (1, 2))
    local = CausalTemporalChannelMixer(
        4,
        (1, 2),
        dynamic_routing=True,
        router_pooling="local",
        router_hidden_divisor=2,
    )
    with torch.no_grad():
        fixed.weight.normal_()
        local.weight.copy_(fixed.weight)
    sequence = torch.randn(6, 2, 4, 3, 3)
    torch.testing.assert_close(local(sequence), fixed(sequence))
    assert local.last_routing_statistics is not None
    torch.testing.assert_close(
        local.last_routing_statistics["gate_mean_by_delay"], torch.ones(2)
    )


def test_spatial_mimo_predictor_is_causal_and_all_new_families_receive_gradients():
    torch.manual_seed(17)
    mixer = CausalTemporalChannelMixer(
        4,
        (1, 2),
        predictive_auxiliary=True,
        predictor_channel_groups=1,
        predictor_spatial_kernel_size=3,
        surprise_routing=True,
    )
    with torch.no_grad():
        mixer.weight.normal_(std=0.1)
    sequence = torch.randn(6, 2, 4, 3, 3, requires_grad=True)
    altered = sequence.detach().clone()
    altered[4:] = torch.randn_like(altered[4:]) * 100
    torch.testing.assert_close(mixer(sequence)[:4], mixer(altered)[:4])
    output = mixer(sequence)
    auxiliary = mixer.auxiliary_loss()
    assert auxiliary is not None
    (output.square().mean() + auxiliary).backward()
    families = (
        mixer.predictor_spatial,
        mixer.predictor_projections,
        mixer.surprise_router,
    )
    for family in families:
        assert family is not None
        gradients = [parameter.grad for parameter in family.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_surprise_tcap_eval_computes_routing_without_auxiliary_loss():
    mixer = CausalTemporalChannelMixer(
        4,
        (1, 2),
        predictive_auxiliary=True,
        surprise_routing=True,
    ).eval()
    sequence = torch.randn(6, 2, 4, 3, 3)
    output = mixer(sequence)
    assert output.shape == sequence.shape
    assert mixer.last_routing_statistics is not None
    assert torch.isfinite(mixer.last_routing_statistics["surprise_mean"])
    assert mixer.auxiliary_loss() is None
    assert mixer.auxiliary_error() is None


def test_matched_predictive_mask_uses_context_indices_only():
    prediction = torch.zeros(5, 1, 2, 1, 1)
    target = torch.ones_like(prediction)
    loss, coverage = _balanced_masked_loss(
        prediction,
        target,
        valid_context_steps=torch.tensor([3]),
    )
    assert torch.isfinite(loss)
    assert coverage == pytest.approx(3 / 5)


def test_dense_ridge_matches_a_shared_per_position_affine_map():
    context = torch.randn(128, 3)
    expected_weight = torch.tensor(
        [[2.0, 0.0, 0.5], [0.0, -1.0, 0.25], [1.0, 0.5, 0.0]]
    )
    expected_bias = torch.tensor([0.2, -0.3, 0.4])
    target = context @ expected_weight + expected_bias
    weights, _sum, _square_sum, rows, samples = _fit_dense_ridge(
        iter(((context, target, target, 1),)),
        channels=3,
        ridge=1e-8,
    )
    assert rows == 128 and samples == 1
    torch.testing.assert_close(weights[:-1], expected_weight, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(weights[-1], expected_bias, atol=1e-4, rtol=1e-4)


def test_normalization_report_provenance_is_strict(tmp_path):
    parent = tmp_path / "parent.pt"
    teacher_checkpoint = tmp_path / "teacher.pt"
    parent.write_bytes(b"parent")
    teacher_checkpoint.write_bytes(b"teacher")
    representation = {"name": "multigranular_count_frame", "bin_width_us": 50_000}
    report_path = tmp_path / "probe.json"
    report = {
        "schema_version": 2,
        "mode": "fine_future",
        "horizon_steps": 2,
        "alignment_horizon_steps": 2,
        "probe_geometry": "shared_dense_affine_1x1_per_spatial_position",
        "probe_git_commit": git_commit(),
        "student_checkpoint_sha256": sha256_file(parent),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "representation": representation,
        "target_standardization": {
            "mean_by_channel": [0.0] * 16,
            "std_by_channel": [1.0] * 16,
            "minimum_std": 1e-5,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = {
        "mode": "fine_future",
        "weight": 0.1,
        "ramp_epochs": 4,
        "horizon_steps": 2,
        "alignment_horizon_steps": 2,
        "normalization_report": str(report_path),
    }
    provenance = {
        "parent_checkpoint": str(parent),
        "teacher_checkpoint": str(teacher_checkpoint),
        "representation": representation,
    }
    PredictiveTrainingObjective(config, _tiny_model(multigranular=True), provenance)

    report["mode"] = "coarse_future"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        PredictiveTrainingObjective(config, _tiny_model(multigranular=True), provenance)


def test_tcap_probe_fits_convex_tap_and_balances_active_tail():
    weights = _region_weights(6, torch.tensor([2, 6]))
    torch.testing.assert_close(weights[:, 0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(weights[:2, 0].sum(), torch.tensor(0.5))
    torch.testing.assert_close(weights[2:, 0].sum(), torch.tensor(0.5))
    torch.testing.assert_close(weights[:, 1].sum(), torch.tensor(1.0))

    statistics = {
        "mixer": {
            "gram": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64),
            "cross": torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            "target_sum": torch.zeros(1, dtype=torch.float64),
            "target_square_sum": torch.ones(1, dtype=torch.float64),
            "target_count": torch.tensor(1.0, dtype=torch.float64),
        }
    }
    coefficients = _fit_convex_tcap_predictors(statistics, steps=100)["mixer"]
    assert coefficients[0, 0] > 0.9
    torch.testing.assert_close(coefficients.sum(0), torch.ones(1))


def _tiny_model(*, multigranular: bool, predictive_head: bool = False) -> MiniQKFormer:
    return MiniQKFormer(
        in_channels=2,
        num_classes=5,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2),
        stage1_mixer="depthwise_conv",
        multigranular=multigranular,
        multigranular_fine_channels=4,
        multigranular_temporal_groups=16,
        multigranular_micro_steps=2,
        predictive_head=predictive_head,
    )


def test_cross_resolution_objective_trains_only_student_predictor():
    torch.manual_seed(8)
    student = _tiny_model(multigranular=False, predictive_head=True)
    teacher = _tiny_model(multigranular=True)
    teacher.eval().requires_grad_(False)
    frames = {
        "coarse": torch.randn(2, 5, 2, 32, 32),
        "fine": torch.randn(2, 10, 2, 4, 4),
    }
    objective = PredictiveTrainingObjective(
        {
            "mode": "fine_future",
            "weight": 0.1,
            "ramp_epochs": 4,
            "horizon_steps": 1,
            "alignment_horizon_steps": 1,
        },
        teacher,
    )
    result = objective(
        student,
        frames,
        torch.tensor([1, 2]),
        torch.nn.CrossEntropyLoss(),
        epoch=4,
    )
    assert objective.effective_weight(1) == 0.0
    assert objective.effective_weight(4) == 0.1
    assert result.logits.shape == (2, 5)
    assert result.metrics["predictive_pair_coverage"] > 0
    result.total_loss.backward()
    assert student.predictive_head is not None
    assert student.predictive_head.weight.grad is not None
    assert torch.isfinite(student.predictive_head.weight.grad).all()
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_training_only_predictor_is_absent_from_deployment_forward():
    torch.manual_seed(11)
    model = _tiny_model(multigranular=False, predictive_head=True).eval()
    frames = torch.randn(1, 4, 2, 32, 32)
    reference = model(frames)
    with torch.no_grad():
        assert model.predictive_head is not None
        model.predictive_head.weight.fill_(1000)
        model.predictive_head.bias.fill_(-1000)
    torch.testing.assert_close(model(frames), reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SMILIES CUDA runtime")
def test_cuda_amp_cross_resolution_objective_at_dvslip_shape():
    student_config = load_config("configs/dvslip_predictive_fine_future.yaml")["model"]
    teacher_config = load_config(
        "configs/dvslip_f_multigranular_temporal_capacity.yaml"
    )["model"]
    student = build_model(student_config, 100).cuda().train()
    teacher = build_model(teacher_config, 100).cuda().eval().requires_grad_(False)
    frames = {
        "coarse": torch.rand(1, 40, 2, 128, 128, device="cuda"),
        "fine": torch.rand(1, 320, 2, 32, 32, device="cuda"),
    }
    objective = PredictiveTrainingObjective(
        {
            "mode": "fine_future",
            "weight": 0.1,
            "ramp_epochs": 4,
            "horizon_steps": 2,
            "alignment_horizon_steps": 2,
        },
        teacher,
    )
    with torch.autocast("cuda", dtype=torch.float16):
        result = objective(
            student,
            frames,
            torch.tensor([3], device="cuda"),
            torch.nn.CrossEntropyLoss(),
            epoch=4,
        )
    result.total_loss.backward()
    assert student.predictive_head is not None
    assert student.predictive_head.weight.grad is not None
    assert torch.isfinite(student.predictive_head.weight.grad).all()
