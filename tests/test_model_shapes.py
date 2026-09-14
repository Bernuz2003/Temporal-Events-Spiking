from pathlib import Path

import pytest
import torch
from torch import nn

from etsr.config import load_config
from etsr.models.factory import build_model
from etsr.models.layers import (
    ConvBNLIF2d,
    FineTemporalBranch,
    InitialPatchEmbedding,
    PatchEmbeddingStage,
    PyramidalPatchEmbedding,
    SpikingBlock,
    SpikingDepthwiseLocalMixer,
)
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.readout import DiagonalGatedReadout
from etsr.models.spiking import MultiStepLIF
from etsr.models.temporal import CausalTemporalChannelMixer, CausalTemporalFIR
from etsr.training.engine import make_optimizer


def test_mini_qkformer_output_shape_and_backward():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=6,
        embed_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
    )
    frames = torch.rand(2, 4, 2, 32, 32)
    logits = model(frames)
    assert logits.shape == (2, 6)
    logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_causal_temporal_fir_is_identity_initialized_and_streamable():
    fir = CausalTemporalFIR(channels=3, kernel_size=3, dilation=2)
    sequence = torch.rand(7, 2, 3, 4, 4, requires_grad=True)

    full, full_state = fir.forward_sequence(sequence)
    first, state = fir.forward_sequence(sequence[:3])
    second, state = fir.forward_sequence(sequence[3:], state)

    assert torch.equal(full, sequence)
    assert torch.equal(torch.cat((first, second)), full)
    assert torch.equal(state, full_state)
    assert state.shape == (4, 2, 3, 4, 4)
    full.mean().backward()
    assert sequence.grad is not None
    assert fir.weight.grad is not None


def test_causal_temporal_fir_cannot_propagate_a_future_perturbation_backward_in_time():
    fir = CausalTemporalFIR(channels=2, kernel_size=3, dilation=1)
    with torch.no_grad():
        fir.weight.copy_(torch.tensor([[0.5, 0.25, -0.5], [1.0, -0.25, 0.75]]))
    original = torch.rand(6, 1, 2, 2, 2)
    perturbed = original.clone()
    perturbed[4:] += 10

    assert torch.equal(fir(original)[:4], fir(perturbed)[:4])


def test_temporal_channel_mixer_is_identity_initialized_causal_and_streamable():
    mixer = CausalTemporalChannelMixer(3, delays=(1, 2, 4))
    sequence = torch.rand(7, 2, 3, 2, 2, requires_grad=True)
    identity, _ = mixer.forward_sequence(sequence)
    assert torch.equal(identity, sequence)
    with torch.no_grad():
        mixer.weight.copy_(torch.arange(27).reshape(3, 3, 3) / 50)
    full, full_state = mixer.forward_sequence(sequence)
    first, state = mixer.forward_sequence(sequence[:3])
    second, state = mixer.forward_sequence(sequence[3:], state)
    assert torch.allclose(torch.cat((first, second)), full)
    assert torch.equal(state, full_state)
    perturbed = sequence.detach().clone()
    perturbed[5:] += 10
    assert torch.equal(mixer(sequence)[:5], mixer(perturbed)[:5])
    full.sum().backward()
    assert sequence.grad is not None
    assert mixer.weight.grad is not None and torch.count_nonzero(mixer.weight.grad) > 0


def test_learnable_delay_tcap_has_causal_soft_gradients_and_hard_streaming():
    mixer = CausalTemporalChannelMixer(3, delays=(1, 2, 4, 8), learnable_delays=True)
    sequence = torch.randn(12, 2, 3, 2, 2, requires_grad=True)
    assert torch.equal(mixer(sequence), sequence)  # Zero MIMO weights preserve identity.
    with torch.no_grad():
        mixer.weight.copy_(torch.randn_like(mixer.weight) * 0.2)

    mixer.set_delay_progress(1, 128)
    assert mixer.delay_temperature == pytest.approx(4.0)
    mixer.set_delay_progress(65, 129)
    assert mixer.delay_temperature == pytest.approx(0.501 + (4.0 - 0.501) / 4)
    mixer.set_delay_progress(1, 128)
    probabilities = mixer.delay_distribution()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4, 3))
    soft = mixer(sequence)
    soft_first, soft_state = mixer.forward_sequence(sequence[:5])
    soft_second, _ = mixer.forward_sequence(sequence[5:], soft_state)
    assert torch.allclose(torch.cat((soft_first, soft_second)), soft, atol=1e-6)
    soft.square().mean().backward()
    assert mixer.delay_centers.grad is not None
    assert torch.isfinite(mixer.delay_centers.grad).all()
    assert torch.count_nonzero(mixer.delay_centers.grad) > 0

    fixed = CausalTemporalChannelMixer(3, delays=(1, 2, 4, 8)).eval()
    with torch.no_grad():
        fixed.weight.copy_(mixer.weight)
    mixer.eval()
    assert torch.equal(mixer(sequence.detach()), fixed(sequence.detach()))
    mixer.train()

    with torch.no_grad():
        mixer.delay_centers[0, 0] = -1
        mixer.delay_centers[3, 2] = 20
    mixer.project_delay_centers_()
    assert mixer.delay_centers.min() >= 1
    assert mixer.delay_centers.max() <= 8
    mixer.set_delay_progress(128, 128)
    assert mixer.delay_temperature == pytest.approx(0.501)

    mixer.eval()
    assert torch.equal(mixer.delay_distribution().sum(dim=1), torch.ones(4, 3))
    full, full_state = mixer.forward_sequence(sequence.detach())
    first, state = mixer.forward_sequence(sequence[:5].detach())
    second, state = mixer.forward_sequence(sequence[5:].detach(), state)
    assert torch.allclose(torch.cat((first, second)), full, atol=1e-6)
    assert torch.equal(state, full_state)
    perturbed = sequence.detach().clone()
    perturbed[9:] += 20
    assert torch.equal(mixer(sequence.detach())[:9], mixer(perturbed)[:9])
    state = mixer.reset_state(sequence[0].detach())
    step_outputs = []
    for step in sequence.detach():
        output, state = mixer.transition(step, state)
        step_outputs.append(output)
    assert torch.allclose(torch.stack(step_outputs), full, atol=1e-6)
    restored = CausalTemporalChannelMixer(3, delays=(1, 2, 4, 8), learnable_delays=True)
    restored.load_state_dict(mixer.state_dict())
    restored.eval()
    assert torch.equal(restored.discrete_delays(), mixer.discrete_delays())
    assert torch.equal(restored(sequence.detach()), full)


def test_learnable_delay_tcap_hard_mode_selects_per_channel_integer_history():
    mixer = CausalTemporalChannelMixer(2, delays=(1, 4), learnable_delays=True).eval()
    with torch.no_grad():
        mixer.delay_centers.copy_(torch.tensor([[1.1, 3.9], [2.6, 1.4]]))
        mixer.weight.copy_(torch.tensor([[[0.2, 0.3], [0.4, 0.5]], [[0.7, 0.8], [0.9, 1.0]]]))
    assert mixer.discrete_delays().tolist() == [[1, 4], [3, 1]]
    sequence = torch.arange(12, dtype=torch.float32).reshape(6, 1, 2, 1, 1)
    observed = mixer(sequence)
    expected = sequence.clone()
    for t in range(6):
        for branch in range(2):
            inputs = []
            for channel in range(2):
                source = t - int(mixer.discrete_delays()[branch, channel])
                inputs.append(sequence[source, 0, channel, 0, 0] if source >= 0 else 0.0)
            expected[t, 0, :, 0, 0] += mixer.weight[branch] @ torch.tensor(inputs)
    assert torch.allclose(observed, expected)
    assert "delay_centers" not in CausalTemporalChannelMixer(2).state_dict()


def test_zero_initialized_mimo_does_not_prevent_later_delay_learning():
    torch.manual_seed(5)
    mixer = CausalTemporalChannelMixer(2, delays=(1, 2, 4, 8), learnable_delays=True)
    initial = mixer.delay_centers.detach().clone()
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=0.01, weight_decay=0.0)
    sequence = torch.randn(12, 1, 2, 2, 2)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        mixer(sequence).square().mean().backward()
        optimizer.step()
        mixer.project_delay_centers_()
    assert not torch.equal(mixer.delay_centers, initial)


def test_learnable_delay_candidate_adds_only_768_parameters_to_fixed_dwc3_d8():
    fixed = load_config("configs/dvslip_f_tcap_stage1_dwc3_d8.yaml")
    adaptive = load_config("configs/dvslip_f_tcap_stage1_dwc3_learnable_delays.yaml")
    assert adaptive["model"] == {
        **fixed["model"],
        "temporal_channel_mixer_learnable_delays": True,
    }
    fixed_count = sum(parameter.numel() for parameter in build_model(fixed["model"], 100).parameters())
    adaptive_count = sum(
        parameter.numel() for parameter in build_model(adaptive["model"], 100).parameters()
    )
    assert fixed_count == 501_028
    assert adaptive_count == fixed_count + 4 * (64 + 128)

    small = MiniQKFormer(
        2,
        4,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2, 4, 8),
        temporal_channel_mixer_learnable_delays=True,
        stage1_mixer="depthwise_conv",
    ).train()
    optimizer = make_optimizer(small, {"learning_rate": 3e-4, "weight_decay": 5e-4})
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    with torch.no_grad():
        for module in small.modules():
            if isinstance(module, CausalTemporalChannelMixer):
                module.weight.normal_(std=0.01)
    loss = small(torch.rand(2, 10, 2, 32, 32)).square().mean()
    loss.backward()
    delay_grads = [
        module.delay_centers.grad
        for module in small.modules()
        if isinstance(module, CausalTemporalChannelMixer)
    ]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in delay_grads)
    assert sum(torch.count_nonzero(grad) for grad in delay_grads) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_temporal_channel_mixer_preserves_activation_dtype(dtype):
    mixer = CausalTemporalChannelMixer(2)
    sequence = torch.rand(5, 1, 2, 2, 2, dtype=dtype)
    assert mixer(sequence).dtype == dtype


def test_structural_candidates_preserve_shape_backward_and_expected_parameter_budget():
    expected = (("pyramidal", False, 431_076), ("pyramidal", True, 431_652))
    for frontend, temporal_fir, expected_parameters in expected:
        model = MiniQKFormer(
            in_channels=2,
            num_classes=100,
            embed_dim=128,
            num_heads=8,
            frontend=frontend,
            temporal_fir=temporal_fir,
        )
        assert isinstance(model.patch_embed1, PyramidalPatchEmbedding)
        assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters

    small = MiniQKFormer(
        in_channels=2,
        num_classes=6,
        embed_dim=16,
        num_heads=4,
        frontend="pyramidal",
        temporal_fir=True,
    )
    logits = small(torch.rand(2, 4, 2, 16, 16))
    assert logits.shape == (2, 6)
    logits.mean().backward()
    assert all(parameter.grad is not None for parameter in small.parameters())

    combined = MiniQKFormer(
        in_channels=2,
        num_classes=100,
        embed_dim=128,
        num_heads=8,
        frontend="pyramidal",
        temporal_channel_mixer=True,
    )
    assert sum(parameter.numel() for parameter in combined.parameters()) == 492_516


def test_f_tcap_starts_as_the_exact_pyramidal_frontend():
    torch.manual_seed(23)
    frontend = MiniQKFormer(
        2, 5, embed_dim=16, num_heads=4, frontend="pyramidal"
    ).eval()
    torch.manual_seed(23)
    combined = MiniQKFormer(
        2,
        5,
        embed_dim=16,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
    ).eval()
    frames = torch.rand(2, 5, 2, 16, 16)

    with torch.no_grad():
        assert torch.equal(frontend(frames), combined(frames))


def test_temporal_fir_keeps_the_initial_model_function_unchanged():
    torch.manual_seed(7)
    baseline = MiniQKFormer(2, 5, embed_dim=16, num_heads=4).eval()
    torch.manual_seed(7)
    temporal = MiniQKFormer(2, 5, embed_dim=16, num_heads=4, temporal_fir=True).eval()
    frames = torch.rand(2, 5, 2, 16, 16)

    with torch.no_grad():
        assert torch.equal(baseline(frames), temporal(frames))


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"temporal_channel_mixer": True, "temporal_channel_mixer_delays": (1, 2, 4)},
        {"learnable_lif_tau": True},
    ],
)
def test_temporal_probes_keep_initial_baseline_logits_exact(model_kwargs):
    torch.manual_seed(17)
    baseline = MiniQKFormer(2, 5, embed_dim=16, num_heads=4).eval()
    torch.manual_seed(17)
    probe = MiniQKFormer(2, 5, embed_dim=16, num_heads=4, **model_kwargs).eval()
    common = set(baseline.state_dict()) & set(probe.state_dict())
    assert all(torch.equal(baseline.state_dict()[key], probe.state_dict()[key]) for key in common)
    frames = torch.rand(2, 6, 2, 16, 16)
    with torch.no_grad():
        assert torch.equal(baseline(frames), probe(frames))


@pytest.mark.parametrize(
    "model_kwargs,parameter_selector",
    [
        (
            {"temporal_channel_mixer": True},
            lambda module: isinstance(module, CausalTemporalChannelMixer),
        ),
        (
            {"learnable_lif_tau": True},
            lambda module: isinstance(module, MultiStepLIF) and module.learnable_tau,
        ),
    ],
)
def test_temporal_probe_parameters_receive_finite_nonzero_gradients(
    model_kwargs, parameter_selector
):
    torch.manual_seed(2)
    model = MiniQKFormer(2, 4, embed_dim=16, num_heads=4, **model_kwargs).train()
    logits = model(torch.rand(2, 6, 2, 32, 32) * 5)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    parameters = [
        parameter
        for module in model.modules()
        if parameter_selector(module)
        for parameter in module.parameters(recurse=False)
    ]
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert sum(torch.count_nonzero(parameter.grad) for parameter in parameters) > 0


def test_mini_qkformer_propagates_surrogate_alpha_to_every_lif():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=6,
        embed_dim=32,
        num_heads=4,
        surrogate_alpha=3.0,
    )

    lif_modules = [module for module in model.modules() if isinstance(module, MultiStepLIF)]
    assert lif_modules
    assert all(module.surrogate_alpha == 3.0 for module in lif_modules)


def test_mini_qkformer_supports_controlled_temporal_modes_and_readouts():
    frames = torch.rand(2, 4, 2, 32, 32)
    for readout in ("mean", "last", "diagonal_gated"):
        model = MiniQKFormer(
            in_channels=2,
            num_classes=6,
            embed_dim=32,
            num_heads=4,
            lif_cross_time=False,
            readout=readout,
        )
        assert model(frames).shape == (2, 6)
        assert all(
            not module.cross_time for module in model.modules() if isinstance(module, MultiStepLIF)
        )


def test_no_cross_time_with_mean_readout_is_bin_order_invariant_at_inference():
    model = MiniQKFormer(
        in_channels=2,
        num_classes=3,
        embed_dim=16,
        num_heads=4,
        lif_cross_time=False,
        readout="mean",
    ).eval()
    frames = torch.rand(2, 4, 2, 16, 16)

    with torch.no_grad():
        original = model(frames)
        permuted = model(frames[:, [2, 0, 3, 1]])

    assert torch.allclose(original, permuted)


def test_diagonal_gated_readout_has_linear_state_and_parameter_cost():
    readout = DiagonalGatedReadout(8)
    sequence = torch.rand(5, 3, 8, requires_grad=True)

    output = readout(sequence)
    assert output.shape == (3, 8)
    assert sum(parameter.numel() for parameter in readout.parameters()) == 6 * 8
    output.mean().backward()
    assert sequence.grad is not None
    assert all(parameter.grad is not None for parameter in readout.parameters())

    step_state = readout.reset_state(sequence[0])
    for current in sequence:
        step_state = readout.step(current, step_state)
    chunk_state = readout.forward_sequence(sequence[:2])
    chunk_state = readout.forward_sequence(sequence[2:], chunk_state)

    assert torch.allclose(readout(sequence), step_state)
    assert torch.allclose(readout(sequence), chunk_state)
    assert readout.detach_state(step_state).grad_fn is None


def test_diagonal_gated_readout_initialization_preserves_long_range_gradient():
    readout = DiagonalGatedReadout(1, initial_memory_steps=20)
    sequence = torch.zeros(40, 1, 1, requires_grad=True)

    readout(sequence).sum().backward()

    gradients = sequence.grad[:, 0, 0]
    assert torch.sigmoid(readout.gate_bias).item() == pytest.approx(0.05)
    assert readout.gate_input.item() == 0.0
    assert gradients[0].abs() / gradients[-1].abs() > 0.1
    assert (gradients[0] / gradients[-1]).item() == pytest.approx(0.95**39)
    legacy_sequence = torch.zeros(40, 1, 1, requires_grad=True)
    DiagonalGatedReadout(1)(legacy_sequence).sum().backward()
    legacy_gradient = legacy_sequence.grad[:, 0, 0]
    assert (legacy_gradient[0] / legacy_gradient[-1]).item() == pytest.approx(0.5**39, abs=1e-18)


def test_fir_backbone_prefix_is_causal_in_eval_mode():
    model = MiniQKFormer(
        2, 3, embed_dim=16, num_heads=4, frontend="pyramidal", temporal_fir=True
    ).eval()
    for module in model.modules():
        if isinstance(module, CausalTemporalFIR):
            with torch.no_grad():
                module.weight[:, 1:] = 0.5
    frames = torch.rand(1, 7, 2, 32, 32) * 20
    changed = frames.clone()
    changed[:, 4:] += 100
    with torch.no_grad():
        torch.testing.assert_close(model._encode(frames)[:4], model._encode(changed)[:4])


def test_gated_v2_can_load_historical_weights_without_changing_predictions():
    legacy = DiagonalGatedReadout(3)
    corrected = DiagonalGatedReadout(3, initial_memory_steps=20)
    assert torch.equal(legacy.gate_input, torch.ones(3))
    assert torch.equal(legacy.gate_bias, torch.zeros(3))
    corrected.load_state_dict(legacy.state_dict(), strict=True)
    sequence = torch.rand(40, 2, 3)
    assert torch.equal(legacy(sequence), corrected(sequence))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fir_learned_taps_match_step_execution_and_gradients_without_dtype_promotion(dtype):
    fir = CausalTemporalFIR(2, kernel_size=3, dilation=2)
    with torch.no_grad():
        fir.weight.copy_(torch.tensor([[0.5, -0.2, 0.7], [0.8, 0.3, -0.6]]))
    sequence = torch.rand(7, 2, 2, 3, 3, dtype=dtype, requires_grad=True)
    output, full_state = fir.forward_sequence(sequence)
    assert output.dtype == dtype
    reference_state = fir.reset_state(sequence[0])
    reference = []
    for step in sequence:
        value, reference_state = fir.transition(step, reference_state)
        reference.append(value)
    assert torch.equal(output, torch.stack(reference))
    first, state = fir.forward_sequence(sequence[:1])
    second, state = fir.forward_sequence(sequence[1:], state)
    assert torch.equal(output, torch.cat((first, second)))
    assert torch.equal(full_state, state)
    grads = torch.autograd.grad(output.float().sum(), (sequence, fir.weight), retain_graph=True)
    ref_grads = torch.autograd.grad(torch.stack(reference).float().sum(), (sequence, fir.weight))
    for grad, ref_grad in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(grad, ref_grad, rtol=0.02, atol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SMILIES CUDA runtime")
@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"frontend": "pyramidal"},
        {"frontend": "pyramidal", "temporal_fir": True},
        {"readout": "diagonal_gated", "gated_initial_memory_steps": 20},
        {"temporal_channel_mixer": True},
        {
            "frontend": "pyramidal",
            "temporal_channel_mixer": True,
            "stage1_mixer": "depthwise_conv",
        },
        {
            "frontend": "pyramidal",
            "temporal_channel_mixer": True,
            "temporal_channel_mixer_delays": (1, 2, 4, 8),
            "temporal_channel_mixer_learnable_delays": True,
            "stage1_mixer": "depthwise_conv",
        },
        {"learnable_lif_tau": True},
    ],
)
def test_cuda_amp_candidate_backward_at_dvslip_shape(model_kwargs):
    model = MiniQKFormer(2, 100, **model_kwargs).cuda().train()
    if model_kwargs.get("temporal_channel_mixer_learnable_delays"):
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, CausalTemporalChannelMixer):
                    module.weight.normal_(std=0.01)
    frames = torch.rand(1, 40, 2, 128, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(frames)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3], device="cuda"))
    assert logits.shape == (1, 100)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SMILIES CUDA runtime")
@pytest.mark.parametrize(
    "model_kwargs,fine_size",
    [
        (
            {
                "multigranular": True,
                "multigranular_fine_channels": 16,
                "multigranular_temporal_groups": 64,
                "multigranular_fusion": "add",
                "multigranular_micro_steps": 8,
            },
            16,
        ),
        (
            {
                "multigranular": True,
                "multigranular_fine_channels": 16,
                "multigranular_fine_mid_channels": 32,
                "multigranular_temporal_groups": 1,
                "multigranular_fusion": "concat_residual",
                "multigranular_micro_steps": 8,
            },
            32,
        ),
    ],
)
def test_cuda_amp_multigranular_backward_at_dvslip_shape(model_kwargs, fine_size):
    model = MiniQKFormer(2, 100, frontend="pyramidal", **model_kwargs).cuda().train()
    frames = {
        "coarse": torch.rand(1, 40, 2, 128, 128, device="cuda"),
        "fine": torch.rand(1, 320, 2, fine_size, fine_size, device="cuda"),
    }
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(frames)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3], device="cuda"))
    loss.backward()

    assert logits.shape == (1, 100)
    assert torch.isfinite(loss)
    fine_gradients = [
        parameter.grad for parameter in model.fine_temporal_branch.parameters()
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in fine_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in fine_gradients) > 0


def test_mean_and_last_readouts_have_explicit_temporal_semantics():
    encoded = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4, 1, 1)
    mean_model = MiniQKFormer(2, 2, embed_dim=4, num_heads=1, readout="mean")
    last_model = MiniQKFormer(2, 2, embed_dim=4, num_heads=1, readout="last")

    assert torch.equal(mean_model._readout(encoded), encoded.mean(dim=(0, 3, 4)))
    assert torch.equal(last_model._readout(encoded), encoded[-1, :, :, 0, 0])


def test_last_event_readouts_exclude_only_the_trailing_silent_bins():
    frames = torch.zeros(2, 4, 1, 1, 1)
    frames[0, :2] = 1
    frames[1, :3] = 1
    valid_steps = MiniQKFormer._last_event_steps(frames)
    encoded = torch.arange(4 * 2 * 4, dtype=torch.float32).reshape(4, 2, 4, 1, 1)

    mean_model = MiniQKFormer(2, 2, embed_dim=4, num_heads=1, readout="mean")
    last_model = MiniQKFormer(2, 2, embed_dim=4, num_heads=1, readout="last")

    assert torch.equal(valid_steps, torch.tensor([2, 3]))
    assert torch.allclose(
        mean_model._readout(encoded, valid_steps),
        torch.stack(
            (
                encoded[:2, 0].mean(dim=(0, 2, 3)),
                encoded[:3, 1].mean(dim=(0, 2, 3)),
            )
        ),
    )
    assert torch.equal(
        last_model._readout(encoded, valid_steps),
        torch.stack((encoded[1, 0, :, 0, 0], encoded[2, 1, :, 0, 0])),
    )


def test_gated_readout_can_freeze_each_sample_at_its_last_event():
    readout = DiagonalGatedReadout(4)
    sequence = torch.rand(5, 3, 4)
    valid_steps = torch.tensor([2, 5, 3])

    batched = readout(sequence, valid_steps)
    individual = torch.cat(
        [
            readout(sequence[: int(steps.item()), index : index + 1])
            for index, steps in enumerate(valid_steps)
        ],
        dim=0,
    )

    assert torch.allclose(batched, individual)


def test_spiking_block_uses_direct_residual_additions():
    block = SpikingBlock(nn.Identity(), dim=4, mlp_ratio=1.0, tau=2.0, threshold=1.0)
    block.mlp = nn.Identity()
    x = torch.rand(2, 1, 4, 2, 2)

    assert torch.equal(block(x), 4.0 * x)


def test_stage1_depthwise_local_mixer_preserves_shape_and_has_expected_budget():
    model = MiniQKFormer(
        2,
        100,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        temporal_channel_mixer_delays=(1, 2, 4),
        stage1_mixer="depthwise_conv",
        stage1_depthwise_kernel_size=3,
    )
    assert isinstance(model.stage1.attention, SpikingDepthwiseLocalMixer)
    assert model.stage1.attention.conv.groups == 64
    assert sum(parameter.numel() for parameter in model.parameters()) == 480_548

    small = MiniQKFormer(
        2,
        6,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        temporal_channel_mixer=True,
        stage1_mixer="depthwise_conv",
    )
    frames = torch.rand(2, 4, 2, 32, 32)
    logits = small(frames)
    logits.square().mean().backward()
    assert logits.shape == (2, 6)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in small.stage1.attention.parameters()
    )


def test_speds_shortcuts_spike_each_branch_before_direct_addition():
    initial = InitialPatchEmbedding(in_channels=2, embed_dim=16, tau=2.0, threshold=1.0)
    stage = PatchEmbeddingStage(in_channels=8, out_channels=16, tau=2.0, threshold=1.0)

    assert isinstance(initial.shortcut, ConvBNLIF2d)
    assert not hasattr(initial, "output_lif")
    assert isinstance(stage.down, ConvBNLIF2d)
    assert isinstance(stage.shortcut, ConvBNLIF2d)
    assert not hasattr(stage, "output_lif")


def test_dvslip_capacity_scan_changes_only_model_width():
    root = Path(__file__).parents[1]
    baseline = load_config(root / "configs" / "dvslip_e0.yaml")
    candidates = (
        ("dvslip_e0_capacity_1m.yaml", 192, 1_113_508),
        ("dvslip_e0_capacity_2m.yaml", 256, 1_967_972),
    )

    for filename, width, expected_parameters in candidates:
        config = load_config(root / "configs" / filename)
        assert config["dataset"] == baseline["dataset"]
        assert config["representation"] == baseline["representation"]
        assert config["augmentation"] == baseline["augmentation"]
        assert config["evaluation"] == baseline["evaluation"]
        assert config["training"] == baseline["training"]
        assert config["model"] == {**baseline["model"], "embed_dim": width}
        model = build_model(config["model"], num_classes=100)
        assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters


def test_dvslip_candidate_configs_change_only_the_declared_architecture():
    root = Path(__file__).parents[1]
    baseline = load_config(root / "configs" / "dvslip_e0.yaml")
    candidates = {
        "dvslip_gated_v2.yaml": {
            "readout": "diagonal_gated",
            "readout_time": "fixed_window",
            "gated_initial_memory_steps": 20.0,
        },
        "dvslip_b_temporal_capacity.yaml": {
            "temporal_channel_mixer": True,
            "temporal_channel_mixer_delays": [1, 2, 4],
        },
        "dvslip_b_plif.yaml": {"learnable_lif_tau": True},
        "dvslip_f.yaml": {"frontend": "pyramidal"},
        "dvslip_f_t.yaml": {
            "frontend": "pyramidal",
            "temporal_fir": True,
            "temporal_fir_kernel_size": 3,
            "temporal_fir_dilations": [1, 2],
        },
        "dvslip_f_temporal_capacity.yaml": {
            "frontend": "pyramidal",
            "temporal_channel_mixer": True,
            "temporal_channel_mixer_delays": [1, 2, 4],
        },
        "dvslip_f_temporal_capacity_d8.yaml": {
            "frontend": "pyramidal",
            "temporal_channel_mixer": True,
            "temporal_channel_mixer_delays": [1, 2, 4, 8],
        },
        "dvslip_f_tcap_stage1_dwc3.yaml": {
            "frontend": "pyramidal",
            "temporal_channel_mixer": True,
            "temporal_channel_mixer_delays": [1, 2, 4],
            "stage1_mixer": "depthwise_conv",
            "stage1_depthwise_kernel_size": 3,
        },
        "dvslip_f_tcap_stage1_dwc3_d8.yaml": {
            "frontend": "pyramidal",
            "temporal_channel_mixer": True,
            "temporal_channel_mixer_delays": [1, 2, 4, 8],
            "stage1_mixer": "depthwise_conv",
            "stage1_depthwise_kernel_size": 3,
        },
        "dvslip_f_tcap_stage1_dwc3_learnable_delays.yaml": {
            "frontend": "pyramidal",
            "temporal_channel_mixer": True,
            "temporal_channel_mixer_delays": [1, 2, 4, 8],
            "temporal_channel_mixer_learnable_delays": True,
            "stage1_mixer": "depthwise_conv",
            "stage1_depthwise_kernel_size": 3,
        },
        "dvslip_b_t.yaml": {
            "temporal_fir": True,
            "temporal_fir_kernel_size": 3,
            "temporal_fir_dilations": [1, 2],
        },
    }

    for filename, model_delta in candidates.items():
        config = load_config(root / "configs" / filename)
        for section in ("dataset", "representation", "augmentation", "evaluation", "training"):
            assert config[section] == baseline[section]
        assert config["model"] == {**baseline["model"], **model_delta}


def test_dvslip_phase_representation_changes_only_input_measurement_and_channels():
    root = Path(__file__).parents[1]
    baseline = load_config(root / "configs" / "dvslip_e0.yaml")
    phase = load_config(root / "configs" / "dvslip_b_phase_e1.yaml")

    for section in ("dataset", "augmentation", "evaluation", "training"):
        assert phase[section] == baseline[section]
    assert phase["representation"] == {
        **baseline["representation"],
        "name": "phase_count_frames_e1",
    }
    assert phase["model"] == {**baseline["model"], "in_channels": 4}
    model = build_model(phase["model"], num_classes=100)
    assert sum(parameter.numel() for parameter in model.parameters()) == 501_284


@pytest.mark.parametrize(
    "filename,name,extra",
    [
        ("dvslip_f_tbr.yaml", "temporal_binary_frames_tbr", {}),
        (
            "dvslip_f_spike_tbr_lif.yaml",
            "spike_tbr_lif_paper_aligned",
            {"lif_beta": 0.9, "lif_threshold": 1.1},
        ),
    ],
)
def test_dvslip_tbr_candidates_change_only_f_representation_and_input_channels(
    filename, name, extra
):
    root = Path(__file__).parents[1]
    frontend = load_config(root / "configs" / "dvslip_f.yaml")
    candidate = load_config(root / "configs" / filename)

    for section in ("dataset", "augmentation", "evaluation", "training"):
        assert candidate[section] == frontend[section]
    assert candidate["representation"] == {
        "name": name,
        "window_us": 2_000_000,
        "bin_width_us": 50_000,
        "micro_bin_width_us": 6_250,
        "bits": 8,
        **extra,
    }
    assert candidate["model"] == {**frontend["model"], "in_channels": 1}
    model = build_model(candidate["model"], num_classes=100)
    assert sum(parameter.numel() for parameter in model.parameters()) == 431_004


def test_multigranular_lite_keeps_main_clock_and_accepts_aligned_two_rate_input():
    model = MiniQKFormer(
        2,
        5,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        multigranular=True,
        multigranular_fine_channels=4,
        multigranular_temporal_groups=16,
        multigranular_fusion="add",
        multigranular_micro_steps=8,
    )
    frames = {
        "coarse": torch.rand(2, 4, 2, 32, 32),
        "fine": torch.rand(2, 32, 2, 4, 4),
    }
    encoded = model._encode(frames)
    logits = model(frames)
    logits.square().sum().backward()

    assert encoded.shape[:3] == (4, 2, 32)
    assert logits.shape == (2, 5)
    assert model.fine_temporal_branch is not None
    assert model.fine_temporal_branch.temporal_reduce.groups == 16
    torch.testing.assert_close(
        model.fine_temporal_branch.temporal_reduce.weight,
        torch.full_like(model.fine_temporal_branch.temporal_reduce.weight, 1 / 8),
    )
    gradients = [parameter.grad for parameter in model.fine_temporal_branch.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_fine_temporal_branch_cannot_propagate_future_microsteps_to_past_macros():
    branch = FineTemporalBranch(
        in_channels=2,
        hidden_channels=4,
        mid_channels=8,
        out_channels=16,
        micro_steps_per_macro=8,
        temporal_groups=1,
        tau=2.0,
        threshold=1.0,
    ).eval()
    original = torch.rand(1, 32, 2, 8, 8)
    perturbed = original.clone()
    perturbed[:, 16:] += 100

    with torch.no_grad():
        before = branch(original)
        after = branch(perturbed)

    assert torch.equal(before[:2], after[:2])


def test_multigranular_capacity_uses_learned_spatial_temporal_and_fusion_paths():
    model = MiniQKFormer(
        2,
        5,
        embed_dim=32,
        num_heads=4,
        frontend="pyramidal",
        multigranular=True,
        multigranular_fine_channels=4,
        multigranular_fine_mid_channels=8,
        multigranular_temporal_groups=1,
        multigranular_fusion="concat_residual",
        multigranular_micro_steps=8,
    )
    frames = {
        "coarse": torch.rand(2, 4, 2, 32, 32),
        "fine": torch.rand(2, 32, 2, 8, 8),
    }
    logits = model(frames)
    logits.square().sum().backward()

    assert logits.shape == (2, 5)
    assert model.fine_temporal_branch is not None
    assert model.fine_temporal_branch.spatial_down is not None
    assert model.fine_temporal_branch.temporal_reduce.groups == 1
    assert model.multigranular_fusion is not None
    expected_reducer = torch.zeros_like(model.fine_temporal_branch.temporal_reduce.weight)
    diagonal = torch.arange(expected_reducer.shape[0])
    expected_reducer[diagonal, diagonal] = 1 / 8
    torch.testing.assert_close(
        model.fine_temporal_branch.temporal_reduce.weight, expected_reducer
    )
    gradients = [parameter.grad for parameter in model.fine_temporal_branch.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_dvslip_multigranular_config_changes_only_preregistered_f_fields():
    root = Path(__file__).parents[1]
    frontend = load_config(root / "configs" / "dvslip_f.yaml")
    candidate = load_config(root / "configs" / "dvslip_f_multigranular_lite.yaml")

    for section in ("dataset", "augmentation", "evaluation", "training"):
        assert candidate[section] == frontend[section]
    assert candidate["representation"] == {
        "name": "multigranular_count_frame",
        "window_us": 2_000_000,
        "bin_width_us": 50_000,
        "micro_bin_width_us": 6_250,
        "fine_spatial_stride": 8,
        "count_cap": 255,
        "fine_count_cap": 65_535,
    }
    assert candidate["model"] == {
        **frontend["model"],
        "multigranular": True,
        "multigranular_fine_channels": 16,
        "multigranular_temporal_groups": 64,
        "multigranular_fusion": "add",
        "multigranular_micro_steps": 8,
    }


def test_dvslip_multigranular_capacity_config_stays_below_baseline_parameter_budget():
    root = Path(__file__).parents[1]
    frontend = load_config(root / "configs" / "dvslip_f.yaml")
    candidate = load_config(root / "configs" / "dvslip_f_multigranular_capacity.yaml")

    for section in ("dataset", "augmentation", "evaluation", "training"):
        assert candidate[section] == frontend[section]
    assert candidate["representation"]["fine_spatial_stride"] == 4
    assert candidate["model"] == {
        **frontend["model"],
        "multigranular": True,
        "multigranular_fine_channels": 16,
        "multigranular_fine_mid_channels": 32,
        "multigranular_temporal_groups": 1,
        "multigranular_fusion": "concat_residual",
        "multigranular_micro_steps": 8,
    }
    model = build_model(candidate["model"], num_classes=100)
    assert sum(parameter.numel() for parameter in model.parameters()) == 480_036


def test_dvslip_multigranular_tcap_combines_only_validated_components():
    root = Path(__file__).parents[1]
    multigranular = load_config(root / "configs" / "dvslip_f_multigranular_capacity.yaml")
    temporal = load_config(root / "configs" / "dvslip_f_temporal_capacity.yaml")
    combined = load_config(
        root / "configs" / "dvslip_f_multigranular_temporal_capacity.yaml"
    )

    for section in ("dataset", "augmentation", "evaluation", "training"):
        assert combined[section] == multigranular[section]
    assert combined["representation"] == multigranular["representation"]
    assert combined["model"] == {
        **multigranular["model"],
        "temporal_channel_mixer": temporal["model"]["temporal_channel_mixer"],
        "temporal_channel_mixer_delays": temporal["model"][
            "temporal_channel_mixer_delays"
        ],
    }
    model = build_model(combined["model"], num_classes=100)
    assert sum(parameter.numel() for parameter in model.parameters()) == 541_476
