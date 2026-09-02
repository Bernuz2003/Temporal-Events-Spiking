from pathlib import Path

import torch
from torch import nn

from etsr.config import load_config
from etsr.models.factory import build_model
from etsr.models.layers import (
    ConvBNLIF2d,
    InitialPatchEmbedding,
    PatchEmbeddingStage,
    SpikingBlock,
)
from etsr.models.mini_qkformer import MiniQKFormer
from etsr.models.readout import DiagonalGatedReadout
from etsr.models.spiking import MultiStepLIF


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
