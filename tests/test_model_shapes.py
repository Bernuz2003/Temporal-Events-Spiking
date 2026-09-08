from pathlib import Path

import pytest
import torch
from torch import nn

from etsr.config import load_config
from etsr.models.factory import build_model
from etsr.models.layers import (
    ConvBNLIF2d,
    InitialPatchEmbedding,
    PatchEmbeddingStage,
    PyramidalPatchEmbedding,
    SpikingBlock,
)
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.readout import DiagonalGatedReadout
from etsr.models.spiking import MultiStepLIF
from etsr.models.temporal import CausalTemporalFIR


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


def test_temporal_fir_keeps_the_initial_model_function_unchanged():
    torch.manual_seed(7)
    baseline = MiniQKFormer(2, 5, embed_dim=16, num_heads=4).eval()
    torch.manual_seed(7)
    temporal = MiniQKFormer(2, 5, embed_dim=16, num_heads=4, temporal_fir=True).eval()
    frames = torch.rand(2, 5, 2, 16, 16)

    with torch.no_grad():
        assert torch.equal(baseline(frames), temporal(frames))


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
    model = MiniQKFormer(2, 3, embed_dim=16, num_heads=4,
                        frontend="pyramidal", temporal_fir=True).eval()
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
@pytest.mark.parametrize("frontend,temporal,readout", [
    ("pyramidal", False, "mean"), ("pyramidal", True, "mean"),
    ("baseline", False, "diagonal_gated"),
])
def test_cuda_amp_candidate_backward_at_dvslip_shape(frontend, temporal, readout):
    model = MiniQKFormer(2, 100, frontend=frontend, temporal_fir=temporal,
                        readout=readout, gated_initial_memory_steps=20).cuda().train()
    frames = torch.rand(1, 40, 2, 128, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(frames)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3], device="cuda"))
    assert logits.shape == (1, 100)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


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
            "readout": "diagonal_gated", "readout_time": "fixed_window",
            "gated_initial_memory_steps": 20.0,
        },
        "dvslip_f.yaml": {"frontend": "pyramidal"},
        "dvslip_f_t.yaml": {
            "frontend": "pyramidal",
            "temporal_fir": True,
            "temporal_fir_kernel_size": 3,
            "temporal_fir_dilations": [1, 2],
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
