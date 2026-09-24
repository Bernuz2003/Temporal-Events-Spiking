from __future__ import annotations

import copy
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from etsr.config import ConfigError, load_config, validate_config
from etsr.data.common import DatasetSubset
from etsr.evaluation.predictive_diagnostic import (
    _fit_convex_tcap_predictors,
    _fit_dense_ridge,
    _region_weights,
)
from etsr.models.factory import build_model
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.reproducibility import capture_random_state, git_commit, restore_random_state
from etsr.training.engine import evaluate
from etsr.training.predictive import (
    PredictiveTrainingObjective,
    _balanced_masked_loss,
    base_model_config,
    class_stratified_indices,
    dataset_targets,
    fixed_window_prefix_logits,
    load_backbone_state,
    validate_predictive_training_authorization,
)
from etsr.utils.io import sha256_file


def test_predictive_configs_follow_the_audited_regimes():
    c0 = load_config("configs/dvslip_f_tcap_stage1_dwc3_d8.yaml")
    r0 = load_config("configs/dvslip_predictive_r0.yaml")
    late_prefix = load_config("configs/dvslip_predictive_late_prefix.yaml")
    future = load_config("configs/dvslip_predictive_fine_future.yaml")
    same = load_config("configs/dvslip_predictive_fine_same.yaml")
    coarse = load_config("configs/dvslip_predictive_coarse_future.yaml")
    dynamic = load_config("configs/dvslip_predictive_dynamic_tcap.yaml")
    s0 = load_config("configs/dvslip_predictive_s0.yaml")
    s1 = load_config("configs/dvslip_predictive_s1.yaml")

    # Continuations from frozen C0: R0 and L15, plus the recorded P designs.
    for continuation in (r0, late_prefix, future, same, coarse):
        assert continuation["training"]["recipe_id"] == "dvslip_predictive_continuation_64"
        assert continuation["training"]["epochs"] == 64
        assert continuation["training"]["learning_rate"] == 1e-5
        assert continuation["continuation"]["new_parameter_learning_rate"] == 1e-4
        assert "objective" not in continuation["continuation"]
        assert continuation["predictive"]["phase1_audit_report"].endswith("phase1_audit.json")
        validate_config(continuation)
    objective = late_prefix["predictive"]["objective"]
    assert objective["prefix_steps"] == [30]
    assert objective["prefix_readout"] == "fixed_window_denominator"
    assert future["predictive"]["blocked_reason"].startswith("Closed")
    assert same["predictive"]["blocked_reason"].startswith("Closed")
    assert coarse["predictive"]["blocked_reason"].startswith("Suspended")
    assert future["representation"]["name"] == "multigranular_count_frame"
    assert future["model"]["predictive_head_hidden_channels"] == 128

    # From-scratch branches: frozen C0 recipe and topology plus registered phase fields only.
    for scratch in (dynamic, s0, s1):
        assert "continuation" not in scratch
        for section in ("dataset", "representation", "augmentation", "training", "evaluation"):
            assert scratch[section] == c0[section]
        assert base_model_config(scratch["model"]) == c0["model"]
        validate_config(scratch)
    assert dynamic["model"]["temporal_channel_mixer_routing_stages"] == [2]
    assert dynamic["model"]["temporal_channel_mixer_routing_parameterization"] == "amplitude_allocation"
    assert dynamic["predictive"]["objective"]["weight"] == 0.0
    s0_objective = s0["predictive"]["objective"]
    assert s0_objective["temporal_region_weights"] == {"active": 1.0, "tail": 0.0}
    assert s0_objective["temporal_stage_weights"] == {"stage1": 0.0, "stage2": 1.0}
    assert s0_objective["authority"]["target_ratio"] == 0.25
    assert s1["predictive"]["objective"] == s0_objective
    assert s1["model"]["temporal_channel_mixer_surprise_routing"]
    assert s1["model"]["temporal_channel_mixer_router_pooling"] == "local"
    assert s1["model"]["temporal_channel_mixer_routing_stages"] == [2]
    assert not s1["model"].get("temporal_channel_mixer_dynamic_routing", False)

    r0["predictive"].pop("phase1_audit_report")
    with pytest.raises(ConfigError, match="must name the completed A1-A4 report"):
        validate_config(r0)


def test_predictive_modules_cannot_train_without_an_objective():
    s0 = load_config("configs/dvslip_predictive_s0.yaml")
    s0.pop("predictive")
    with pytest.raises(ConfigError, match="trained only by a predictive objective"):
        validate_config(s0)
    s1 = load_config("configs/dvslip_predictive_s1.yaml")
    s1["predictive"]["objective"]["weight"] = 0.0
    s1["predictive"]["objective"].pop("authority")
    with pytest.raises(ConfigError, match="positive weight or authority"):
        validate_config(s1)
    r0 = load_config("configs/dvslip_predictive_r0.yaml")
    r0["continuation"]["objective"] = {"mode": "none", "weight": 0.0}
    with pytest.raises(ConfigError, match="belongs to the predictive section"):
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
                "schema_version": 2,
                "complete": True,
                "sections": [
                    "A1_gradient_authority",
                    "A2_discriminative_probes",
                    "A3_representation_movement",
                    "A4_tail_margin",
                ],
                "official_test_used": False,
                "checkpoints": {"c0": {"sha256": sha256_file(parent)}},
                "A1_gradient_authority": {
                    "batch_selection": "class_stratified_disjoint_batches",
                    "batches": 4,
                    "distinct_classes": 64,
                },
                "A2_discriminative_probes": {"fit_samples": 8192, "holdout_samples": 2048},
            }
        )
    )
    predictive = {"phase1_audit_report": str(report)}
    continuation = {"parent_checkpoint": str(parent)}
    assert validate_predictive_training_authorization(predictive, continuation) == report
    # A from-scratch branch has no parent: the report must still be complete and valid.
    assert validate_predictive_training_authorization(predictive, None) == report

    invalid = json.loads(report.read_text())
    invalid["A1_gradient_authority"]["batch_selection"] = "first_unshuffled_batch"
    report.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="Regenerate the stratified A1"):
        validate_predictive_training_authorization(predictive, None)

    invalid["A1_gradient_authority"]["batch_selection"] = "class_stratified_disjoint_batches"
    report.write_text(json.dumps(invalid))

    invalid = json.loads(report.read_text())
    invalid["checkpoints"]["c0"]["sha256"] = "wrong"
    report.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="different C0 checkpoints"):
        validate_predictive_training_authorization(predictive, continuation)

    predictive["blocked_reason"] = "requires a different causal target"
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        validate_predictive_training_authorization(predictive, None)
    with pytest.raises(ValueError, match="requires a predictive section"):
        validate_predictive_training_authorization(None, None)


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


def _tiny_stage2_predictive_model(*, surprise: bool = False) -> MiniQKFormer:
    return MiniQKFormer(
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
        temporal_channel_mixer_predictive_stages=(2,),
        temporal_channel_mixer_surprise_routing=surprise,
        temporal_channel_mixer_router_pooling="local" if surprise else "global",
        temporal_channel_mixer_routing_stages=(2,) if surprise else (1, 2),
        temporal_channel_mixer_routing_parameterization=(
            "amplitude_allocation" if surprise else "independent"
        ),
        stage1_mixer="depthwise_conv",
    )


_AUTHORITY_OBJECTIVE = {
    "mode": "none",
    "weight": 0.1,
    "ramp_epochs": 4,
    "temporal_region_weights": {"active": 1.0, "tail": 0.0},
    "temporal_stage_weights": {"stage1": 0.0, "stage2": 1.0},
    "authority": {"target_ratio": 0.25, "max_step_factor": 2.0},
}


def test_fixed_window_prefix_logits_equal_the_full_window_readout_of_the_prefix():
    torch.manual_seed(21)
    model = _tiny_model(multigranular=False).eval()
    frames = torch.rand(2, 6, 2, 32, 32)
    total = frames.shape[1]
    with torch.no_grad():
        spatial = model._encode(frames).mean(dim=(3, 4))
        for steps in (3, 5):
            corrected = fixed_window_prefix_logits(model, model(frames[:, :steps]), steps, total)
            expected = model.head(spatial[:steps].sum(0) / total)
            torch.testing.assert_close(corrected, expected)
        full = model(frames)
        torch.testing.assert_close(fixed_window_prefix_logits(model, full, total, total), full)


def test_full_window_prefix_distillation_vanishes_when_student_equals_teacher():
    torch.manual_seed(22)
    student = _tiny_model(multigranular=False).eval()
    teacher = _tiny_model(multigranular=False)
    teacher.load_state_dict(student.state_dict())
    teacher.eval().requires_grad_(False)
    objective = PredictiveTrainingObjective(
        {
            "mode": "late_prefix",
            "weight": 0.1,
            "prefix_steps": (6,),
            "prefix_readout": "fixed_window_denominator",
        },
        teacher,
    )
    result = objective(
        student, torch.rand(2, 6, 2, 32, 32), torch.tensor([0, 1]), torch.nn.CrossEntropyLoss(), 1
    )
    assert float(result.auxiliary_loss) == pytest.approx(0.0, abs=1e-6)
    assert result.metrics["prefix_kl_6"] == pytest.approx(0.0, abs=1e-6)


def test_authority_calibration_sets_the_shared_ratio_without_side_effects():
    torch.manual_seed(23)
    model = _tiny_stage2_predictive_model()
    frames = torch.rand(4, 6, 2, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    objective = PredictiveTrainingObjective(copy.deepcopy(_AUTHORITY_OBJECTIVE))
    parameters = {name: value.clone() for name, value in model.state_dict().items()}
    model.train()
    rng = torch.get_rng_state()

    record = objective.calibrate(
        model, frames, targets, torch.nn.CrossEntropyLoss(), epoch=0,
        freeze_batchnorm_statistics=False,
    )

    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(parameters[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training
    assert record["updated"] and record["unit_ratio"] > 0.0
    assert record["nominal_shared_ratio"] == pytest.approx(0.25, rel=1e-6)
    assert objective.weight == pytest.approx(0.25 / record["unit_ratio"])

    # After the first calibration the weight moves by at most max_step_factor per epoch.
    objective.weight *= 10.0
    bounded = objective.calibrate(
        model, frames, targets, torch.nn.CrossEntropyLoss(), epoch=2,
        freeze_batchnorm_statistics=False,
    )
    assert bounded["weight"] == pytest.approx(bounded["previous_weight"] / 2.0)
    # Selection eligibility follows ramp progress, not the calibrated weight.
    assert [objective.selection_eligible(epoch) for epoch in range(1, 6)] == [
        False, False, False, True, True,
    ]
    restored = PredictiveTrainingObjective(copy.deepcopy(_AUTHORITY_OBJECTIVE))
    restored.load_state_dict(objective.state_dict())
    assert restored.weight == objective.weight
    assert restored.calibrated and len(restored.calibration_history) == 2


def test_stage2_predictive_stages_leave_no_untrained_predictor_in_stage1():
    model = _tiny_stage2_predictive_model(surprise=True)
    mixers = [module for module in model.modules() if isinstance(module, CausalTemporalChannelMixer)]
    assert [mixer.predictive_auxiliary for mixer in mixers] == [False, True]
    assert [mixer.surprise_router is not None for mixer in mixers] == [False, True]
    # V_delta is still recorded in stage1, which receives the stage2 loss by backpropagation.
    assert all(mixer.record_temporal_variation for mixer in mixers)
    with pytest.raises(ValueError, match="predictor in every routed stage"):
        MiniQKFormer(
            in_channels=2, num_classes=5, embed_dim=32, num_heads=4, frontend="pyramidal",
            temporal_channel_mixer=True, temporal_channel_mixer_delays=(1, 2),
            temporal_channel_mixer_predictive_auxiliary=True,
            temporal_channel_mixer_predictive_stages=(2,),
            temporal_channel_mixer_surprise_routing=True,
            temporal_channel_mixer_routing_stages=(1, 2),
            stage1_mixer="depthwise_conv",
        )


def test_class_stratified_indices_span_classes_in_sorted_and_nested_datasets():
    targets = [target for target in range(5) for _ in range(4)]  # class-sorted, like DVS-Lip
    dataset = TensorDataset(torch.zeros(len(targets), 1), torch.tensor(targets))
    dataset.targets = targets
    assert sorted(targets[index] for index in class_stratified_indices(dataset, 3)) == [0, 2, 4]
    everything = class_stratified_indices(dataset, 7)
    assert len(set(everything)) == 7
    assert {targets[index] for index in everything} == set(range(5))
    subset = DatasetSubset(dataset, list(range(8, 20)))  # classes 2, 3 and 4
    assert dataset_targets(subset) == targets[8:20]
    chosen = class_stratified_indices(subset, 3)
    assert sorted(dataset_targets(subset)[index] for index in chosen) == [2, 3, 4]


def test_scratch_branch_reuses_the_c0_topology_initialization_and_data_stream():
    c0 = load_config("configs/dvslip_f_tcap_stage1_dwc3_d8.yaml")["model"]
    s1 = load_config("configs/dvslip_predictive_s1.yaml")["model"]
    assert base_model_config(s1) == c0

    torch.manual_seed(42)
    reference = build_model(c0, 100)
    reference_draw = torch.rand(8)

    torch.manual_seed(42)
    base = build_model(base_model_config(s1), 100)
    state = capture_random_state()
    model = build_model(s1, 100)
    new_names = load_backbone_state(model, base.state_dict())
    restore_random_state(state)

    assert torch.equal(torch.rand(8), reference_draw)
    model_state = model.state_dict()
    assert all(torch.equal(model_state[name], value) for name, value in reference.state_dict().items())
    assert new_names and all(
        "surprise_router" in name or ".predictor_" in name for name in new_names
    )
