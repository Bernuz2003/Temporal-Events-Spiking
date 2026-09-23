from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from etsr.config import ConfigError, load_config, validate_config
from etsr.evaluation.predictive_diagnostic import (
    _fit_convex_tcap_predictors,
    _fit_dense_ridge,
    _region_weights,
)
from etsr.models.factory import build_model
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.reproducibility import git_commit
from etsr.training.engine import evaluate
from etsr.training.predictive import (
    PredictiveTrainingObjective,
    _balanced_masked_loss,
    validate_predictive_training_authorization,
)
from etsr.utils.io import sha256_file


def test_predictive_configs_inherit_the_frozen_continuation_recipe():
    r0 = load_config("configs/dvslip_predictive_r0.yaml")
    future = load_config("configs/dvslip_predictive_fine_future.yaml")
    dynamic = load_config("configs/dvslip_predictive_dynamic_tcap.yaml")
    surprise = load_config("configs/dvslip_predictive_s1.yaml")
    assert r0["training"]["epochs"] == 64
    assert r0["training"]["recipe_id"] == "dvslip_predictive_continuation_64"
    assert r0["training"]["learning_rate"] == 1e-5
    assert r0["continuation"]["new_parameter_learning_rate"] == 1e-4
    assert r0["continuation"]["phase1_audit_report"].endswith("phase1_audit.json")
    assert r0["model"]["temporal_channel_mixer_delays"] == [1, 2, 4, 8]
    assert future["representation"]["name"] == "multigranular_count_frame"
    assert not future["model"].get("multigranular", False)
    assert future["model"]["predictive_head"]
    assert future["model"]["predictive_head_spatial_kernel_size"] == 3
    assert future["model"]["predictive_head_hidden_channels"] == 128
    assert dynamic["model"]["temporal_channel_mixer_router_pooling"] == "local"
    assert dynamic["model"]["temporal_channel_mixer_router_hidden_divisor"] == 2
    assert dynamic["model"]["temporal_channel_mixer_routing_stages"] == [2]
    assert (
        dynamic["model"]["temporal_channel_mixer_routing_parameterization"]
        == "amplitude_allocation"
    )
    assert surprise["model"]["temporal_channel_mixer_predictor_channel_groups"] == 1
    assert surprise["model"]["temporal_channel_mixer_predictor_spatial_kernel_size"] == 3

    stable_s0 = load_config("configs/dvslip_predictive_s0.yaml")
    stable_s1 = load_config("configs/dvslip_predictive_s1.yaml")
    stable_late_prefix = load_config("configs/dvslip_predictive_late_prefix.yaml")
    for candidate in (stable_s0, stable_s1, stable_late_prefix):
        assert candidate["training"]["recipe_id"] == "dvslip_predictive_continuation_64"
        assert candidate["training"]["learning_rate"] == 1e-5
        assert candidate["continuation"]["new_parameter_learning_rate"] == 1e-4
    assert not stable_s0["model"].get("temporal_channel_mixer_dynamic_routing", False)
    assert not stable_s1["model"].get("temporal_channel_mixer_dynamic_routing", False)
    assert stable_s1["model"]["temporal_channel_mixer_surprise_routing"]
    assert stable_s1["model"]["temporal_channel_mixer_router_pooling"] == "local"
    assert stable_s1["model"]["temporal_channel_mixer_routing_stages"] == [2]
    assert future["continuation"]["blocked_reason"]

    r0["continuation"].pop("phase1_audit_report")
    with pytest.raises(ConfigError, match="must name the completed A1-A4 report"):
        validate_config(r0)


def test_predictive_selection_excludes_the_complete_auxiliary_ramp():
    objective = PredictiveTrainingObjective(
        {"mode": "none", "weight": 0.4, "ramp_epochs": 4}
    )
    assert [objective.selection_eligible(epoch) for epoch in range(1, 6)] == [
        False,
        False,
        False,
        True,
        True,
    ]
    control = PredictiveTrainingObjective(
        {"mode": "none", "weight": 0.0, "ramp_epochs": 4}
    )
    assert all(control.selection_eligible(epoch) for epoch in range(1, 6))


def test_phase1_training_authorization_checks_report_and_parent_hash(tmp_path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"checkpoint")
    report = tmp_path / "phase1_audit.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "sections": [
                    "A1_gradient_authority",
                    "A2_discriminative_probes",
                    "A3_representation_movement",
                    "A4_tail_margin",
                ],
                "official_test_used": False,
                "checkpoints": {"c0": {"sha256": sha256_file(parent)}},
            }
        )
    )
    continuation = {
        "parent_checkpoint": str(parent),
        "phase1_audit_report": str(report),
    }
    assert validate_predictive_training_authorization(continuation) == report

    invalid = json.loads(report.read_text())
    invalid["checkpoints"]["c0"]["sha256"] = "wrong"
    report.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="different C0 checkpoints"):
        validate_predictive_training_authorization(continuation)

    continuation["blocked_reason"] = "requires a different causal target"
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        validate_predictive_training_authorization(continuation)


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


def test_temporal_variation_is_detached_and_only_recorded_for_predictive_mixers():
    sequence = torch.randn(7, 2, 3, 2, 2, requires_grad=True)
    fixed = CausalTemporalChannelMixer(3, (1, 2))
    fixed(sequence).sum().backward()
    assert fixed.last_temporal_variation is None

    predictive_sequence = torch.randn(7, 2, 3, 2, 2, requires_grad=True)
    predictive = CausalTemporalChannelMixer(3, (1, 2), predictive_auxiliary=True)
    predictive(predictive_sequence).sum().backward()
    variation = predictive.last_temporal_variation
    assert variation is not None
    assert not variation.requires_grad
    assert variation.grad_fn is None


def test_amplitude_allocation_router_starts_as_exact_fixed_tcap():
    torch.manual_seed(41)
    fixed = CausalTemporalChannelMixer(3, (1, 2, 4))
    routed = CausalTemporalChannelMixer(
        3,
        (1, 2, 4),
        dynamic_routing=True,
        router_pooling="local",
        routing_parameterization="amplitude_allocation",
    )
    with torch.no_grad():
        fixed.weight.normal_()
        routed.weight.copy_(fixed.weight)
    sequence = torch.randn(7, 2, 3, 2, 2)
    torch.testing.assert_close(routed(sequence), fixed(sequence))
    statistics = routed.last_routing_statistics
    assert statistics is not None
    torch.testing.assert_close(statistics["amplitude_mean"], torch.tensor(1.0))
    torch.testing.assert_close(
        statistics["allocation_mean_by_delay"], torch.full((3,), 1 / 3)
    )


def test_stage2_only_surprise_routing_keeps_stage1_fixed():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=5,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2),
        temporal_channel_mixer_predictive_auxiliary=True,
        temporal_channel_mixer_predictor_channel_groups=1,
        temporal_channel_mixer_surprise_routing=True,
        temporal_channel_mixer_router_pooling="local",
        temporal_channel_mixer_routing_stages=(2,),
        temporal_channel_mixer_routing_parameterization="amplitude_allocation",
        stage1_mixer="depthwise_conv",
    )
    mixers = [
        module
        for module in model.modules()
        if isinstance(module, CausalTemporalChannelMixer)
    ]
    assert len(mixers) == 2
    assert mixers[0].surprise_router is None
    assert mixers[1].surprise_router is not None


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
    total_variance = local.last_routing_statistics["gate_std_by_delay"].square()
    between_sample_variance = (
        local.last_routing_statistics["gate_sample_mean_second_moment_by_delay"]
        - local.last_routing_statistics["gate_mean_by_delay"].square()
    )
    within_sample_variance = local.last_routing_statistics[
        "gate_within_sample_variance_mean_by_delay"
    ]
    torch.testing.assert_close(
        total_variance,
        between_sample_variance + within_sample_variance,
        atol=1e-6,
        rtol=1e-5,
    )


def test_evaluation_collects_fixed_weight_input_dependence_statistics():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=5,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2),
        temporal_channel_mixer_dynamic_routing=True,
        temporal_channel_mixer_router_pooling="local",
        temporal_channel_mixer_router_hidden_divisor=2,
        temporal_channel_mixer_predictive_auxiliary=True,
        temporal_channel_mixer_surprise_routing=True,
        stage1_mixer="depthwise_conv",
    )
    frames = torch.randn(2, 4, 2, 32, 32)
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(frames, targets, torch.arange(2)), batch_size=1)
    routing_statistics = []

    evaluate(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        5,
        routing_statistics=routing_statistics,
    )

    assert len(routing_statistics) == 4
    assert {row["delay"] for row in routing_statistics} == {1, 2}
    assert all(row["gate_mean"] == pytest.approx(1.0) for row in routing_statistics)
    assert all(row["gate_std"] == pytest.approx(0.0) for row in routing_statistics)
    assert all(row["gate_cv"] == pytest.approx(0.0) for row in routing_statistics)
    assert all(row["sample_count"] == 2 for row in routing_statistics)
    assert all(row["normalized_surprise_mean"] >= 0.0 for row in routing_statistics)
    assert all(row["normalized_surprise_std"] >= 0.0 for row in routing_statistics)


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
    assert mixer.prediction_diagnostics() is not None
    assert mixer.auxiliary_loss() is None
    assert mixer.auxiliary_error() is None


def test_temporal_predictor_reports_region_skill_against_causal_references():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=5,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2),
        temporal_channel_mixer_predictive_auxiliary=True,
        temporal_channel_mixer_predictor_channel_groups=1,
        temporal_channel_mixer_predictor_spatial_kernel_size=3,
        stage1_mixer="depthwise_conv",
    ).eval()
    frames = torch.randn(2, 4, 2, 32, 32)
    frames[0, 2:] = 0
    frames[1, 3:] = 0
    loader = DataLoader(
        TensorDataset(frames, torch.tensor([0, 1]), torch.arange(2)), batch_size=2
    )
    statistics = []
    evaluate(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        5,
        temporal_prediction_statistics=statistics,
    )
    metrics = statistics[0]

    expected = {
        "temporal_prediction_active_loss",
        "temporal_prediction_tail_loss",
        "temporal_persistence_active_loss",
        "temporal_persistence_tail_loss",
        "temporal_delay_mean_active_loss",
        "temporal_delay_mean_tail_loss",
        "temporal_target_active_variance",
        "temporal_target_tail_variance",
        "temporal_prediction_active_skill_vs_persistence",
        "temporal_prediction_tail_skill_vs_persistence",
        "temporal_prediction_active_skill_vs_delay_mean",
        "temporal_prediction_tail_skill_vs_delay_mean",
    }
    assert expected <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(metrics[name])) for name in expected)


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
    predictor_gradients = [
        parameter.grad for parameter in student.predictive_head.parameters()
    ]
    assert predictor_gradients
    assert all(gradient is not None for gradient in predictor_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in predictor_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in predictor_gradients) > 0.0
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
    predictor_gradients = [
        parameter.grad for parameter in student.predictive_head.parameters()
    ]
    assert predictor_gradients
    assert all(gradient is not None for gradient in predictor_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in predictor_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in predictor_gradients) > 0.0
