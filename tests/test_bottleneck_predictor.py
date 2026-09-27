from __future__ import annotations

import copy

import pytest
import torch

from etsr.config import load_config
from etsr.evaluation.bottleneck_probe import _active_errors
from etsr.evaluation.predictive_diagnostic import _bottleneck_gradient_contract
from etsr.models.factory import build_model
from etsr.models.temporal import CausalTemporalChannelMixer
from etsr.training.checkpointing import save_deployment_checkpoint


@pytest.mark.parametrize("hidden", [None, 256])
def test_bottleneck_predictor_is_causal_and_restricts_history_gradient(hidden):
    mixer = CausalTemporalChannelMixer(
        8,
        (1, 2, 4),
        predictive_auxiliary=True,
        predictor_spatial_kernel_size=3,
        predictor_rank=3,
        predictor_hidden_channels=hidden,
    )
    sequence = torch.randn(9, 2, 8, 3, 3)
    history = torch.cat((torch.zeros(4, 2, 8, 3, 3), sequence))
    first = mixer._causal_prediction(sequence, history)
    altered = sequence.clone()
    altered[6:] = torch.randn_like(altered[6:])
    altered_history = torch.cat((torch.zeros_like(history[:4]), altered))
    second = mixer._causal_prediction(altered, altered_history)
    assert torch.allclose(first[:6], second[:6], atol=1e-6)

    class Holder(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    contract = _bottleneck_gradient_contract(Holder(mixer))
    assert contract is not None and contract["passed"]
    assert contract["relative_outside_row_gradient"] < 1e-5

    loss, persistence, delay_mean = _active_errors(sequence, torch.tensor([7, 8]), mixer)
    assert all(torch.isfinite(value) for value in (loss, persistence, delay_mean))
    loss.backward()
    assert mixer.predictor_bottleneck is not None
    assert mixer.predictor_bottleneck.projection.weight.grad is not None
    assert mixer.predictor_bottleneck.projection.weight.grad.norm() > 0
    if hidden is not None:
        assert mixer.predictor_bottleneck.joint_head is not None
        assert mixer.predictor_bottleneck.joint_head[-1].weight.grad.norm() > 0

    mixer.eval()
    mixer(sequence)
    assert mixer.last_projected_diagnostics is not None
    assert mixer.last_projected_features is not None
    assert torch.isfinite(mixer.last_projected_diagnostics["row_variation"]).all()


def test_bottleneck_gradient_contract_at_selected_rank():
    mixer = CausalTemporalChannelMixer(
        128,
        (1, 2, 4, 8),
        predictive_auxiliary=True,
        predictor_spatial_kernel_size=3,
        predictor_rank=64,
        predictor_hidden_channels=256,
    )

    class Holder(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    contract = _bottleneck_gradient_contract(Holder(mixer))
    assert contract is not None
    assert contract["arithmetic"] == "cpu_float64_reference"
    assert contract["passed"]
    assert contract["relative_outside_row_gradient"] < 1e-8


def test_static_router_is_input_independent_but_trainable():
    mixer = CausalTemporalChannelMixer(
        8,
        (1, 2, 4),
        dynamic_routing=True,
        router_pooling="constant",
        routing_parameterization="amplitude_allocation",
    )
    left = torch.randn(5, 2, 8, 3, 3)
    right = torch.randn_like(left)
    state = mixer.reset_state(left[0])
    left_gates, _ = mixer._conditional_gates(left, torch.cat((state, left)))
    right_gates, _ = mixer._conditional_gates(right, torch.cat((state, right)))
    assert left_gates is not None and right_gates is not None
    assert torch.allclose(left_gates, right_gates)
    assert torch.allclose(left_gates, torch.ones_like(left_gates))
    (left_gates * torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3)).sum().backward()
    assert mixer.content_router is not None
    assert mixer.content_router.weight.grad is not None
    assert mixer.content_router.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("config_path", "expected_parameters"),
    [
        ("configs/dvslip_predictive_s0_bottleneck_k32.yaml", 501_028),
        ("configs/dvslip_predictive_dynamic_tcap_bottleneck_k32.yaml", 509_609),
    ],
)
def test_bottleneck_deployment_strictly_loads_without_predictor(
    tmp_path, config_path, expected_parameters
):
    config = load_config(config_path)
    model = build_model(config["model"], 100)
    deployment = copy.deepcopy(config)
    deployment.pop("predictive")
    for name in (
        "temporal_channel_mixer_predictive_auxiliary",
        "temporal_channel_mixer_predictive_stages",
        "temporal_channel_mixer_predictor_channel_groups",
        "temporal_channel_mixer_predictor_spatial_kernel_size",
        "temporal_channel_mixer_predictor_rank",
        "temporal_channel_mixer_predictor_hidden_channels",
    ):
        deployment["model"].pop(name, None)
    checkpoint_path = tmp_path / "deployment.pt"
    save_deployment_checkpoint(checkpoint_path, model, 1, 0.0, deployment, 100)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    assert not any("predictor_bottleneck" in name for name in state)
    deployed = build_model(deployment["model"], 100)
    deployed.load_state_dict(state, strict=True)
    assert sum(parameter.numel() for parameter in deployed.parameters()) == expected_parameters
