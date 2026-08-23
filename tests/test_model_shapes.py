import torch
from torch import nn

from etsr.models.layers import (
    ConvBNLIF2d,
    InitialPatchEmbedding,
    PatchEmbeddingStage,
    SpikingBlock,
)
from etsr.models.mini_qkformer import MiniQKFormer
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
